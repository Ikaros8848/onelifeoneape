"""Trainable stencil-offset gating for SCAMP sequential schedule co-design.

The gate is shared by all sixteen convolution filters for a kernel offset. A
hard export removes an offset only when its group gate is zero, which is the
condition required to remove a sequential tap from the PPA schedule.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.hardware_aware import HardwareAwareCNN, ternary_quantize


class OffsetGatedHardwareAwareCNN(HardwareAwareCNN):
    def __init__(self, *args, target_retained_offsets=14, gate_temperature=1.0,
                 gate_mode="soft", **kwargs):
        super().__init__(*args, **kwargs)
        if target_retained_offsets < 1 or target_retained_offsets > 16:
            raise ValueError("target_retained_offsets must be in [1,16]")
        self.target_retained_offsets = int(target_retained_offsets)
        self.gate_temperature = float(gate_temperature)
        self.gate_mode = gate_mode
        self.offset_logits = nn.Parameter(torch.full((4, 4), 4.0))

    def offset_probabilities(self):
        return torch.sigmoid(self.offset_logits / max(self.gate_temperature, 1e-6))

    def offset_mask(self, hard=None):
        hard = self.gate_mode == "hard" if hard is None else hard
        p = self.offset_probabilities()
        if not hard:
            return p
        # Straight-through top-k mask: forward uses exactly k taps while the
        # backward path retains sigmoid gradients for ranking.
        flat = p.flatten()
        k = self.target_retained_offsets
        top = torch.zeros_like(flat)
        top[torch.topk(flat, k=k).indices] = 1.0
        return p + (top.view_as(p) - p).detach()

    def regularization(self):
        """L1 group penalty and a soft target for the requested tap budget."""
        p = self.offset_probabilities()
        return p.sum(), (p.sum() - self.target_retained_offsets).pow(2)

    def forward(self, x, return_info=False):
        gates = self.offset_mask()
        if self.quantize and self.ternary_conv:
            q, signs, alpha, mask = ternary_quantize(
                self.conv_weight, self.threshold_ratio, self.per_channel)
            q = q * gates.view(1, 1, 4, 4)
            pos, neg = (q > 0).to(q.dtype), (q < 0).to(q.dtype)
            y = F.conv2d(F.pad(x, (1, 2, 1, 2)), pos * q.abs(), None)
            y = y - F.conv2d(F.pad(x, (1, 2, 1, 2)), neg * q.abs(), None)
            y = y.clamp(-self.accumulator_limit, self.accumulator_limit) if self.accumulator_limit is not None else y
            y = y + self.conv_bias.view(1, -1, 1, 1)
            signs = signs * (gates > 0.5).to(signs.dtype).view(1, 1, 4, 4)
            mask = mask & (gates > 0.5).view(1, 1, 4, 4)
        else:
            w = self.conv_weight * gates.view(1, 1, 4, 4)
            y = F.conv2d(F.pad(x, (1, 2, 1, 2)), w, self.conv_bias)
            signs = alpha = mask = None
        y = F.relu(y)
        y = y.clamp(max=self.activation_limit) if self.activation_limit is not None else y
        flat = F.max_pool2d(y, 4, 4).flatten(1)
        if self.quantize and self.ternary_fc:
            qfc, fsigns, falpha, fmask = ternary_quantize(
                self.fc_weight, self.threshold_ratio, self.per_channel)
            logits = F.linear(flat, (qfc > 0).to(qfc.dtype) * qfc.abs())
            logits = logits - F.linear(flat, (qfc < 0).to(qfc.dtype) * qfc.abs())
            logits = logits.clamp(-self.accumulator_limit, self.accumulator_limit) if self.accumulator_limit is not None else logits
            logits = logits + self.fc_bias
        else:
            logits = F.linear(flat, self.fc_weight, self.fc_bias)
            fsigns = falpha = fmask = None
        if not return_info:
            return logits
        return logits, {
            "conv_sign": signs, "conv_alpha": alpha, "conv_mask": mask,
            "fc_sign": fsigns, "fc_alpha": falpha, "fc_mask": fmask,
            "offset_probabilities": self.offset_probabilities().detach(),
            "offset_mask": gates.detach(),
        }


def load_offset_gated_checkpoint(path, **kwargs):
    model = OffsetGatedHardwareAwareCNN(quantize=True, **kwargs)
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Loading a regular QAT checkpoint is useful for warm-start screening.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected}")
    return model, missing


def export_deploy_state(model):
    """Return a plain HardwareAwareCNN state with the hard mask baked in.

    Quantized values, rather than raw latent values, are exported so the
    per-channel scale remains identical after the gate module is removed.
    """
    with torch.no_grad():
        qconv, _, _, _ = ternary_quantize(
            model.conv_weight, model.threshold_ratio, model.per_channel)
        qfc, _, _, _ = ternary_quantize(
            model.fc_weight, model.threshold_ratio, model.per_channel)
        hard = model.offset_mask(hard=True).view(1, 1, 4, 4)
        return {
            "conv_weight": (qconv * hard).detach().cpu(),
            "conv_bias": model.conv_bias.detach().cpu(),
            "fc_weight": qfc.detach().cpu(),
            "fc_bias": model.fc_bias.detach().cpu(),
        }
