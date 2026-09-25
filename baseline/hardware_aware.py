"""SCAMP-aware layers and models.

The simulator keeps latent FP32 parameters for optimization, but its hardware
forward path uses ternary weights and explicitly computes positive and negative
accumulations.  This is functional emulation, not cycle-accurate SCAMP code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary_quantize(w, threshold_ratio=0.05, per_channel=True):
    dims = tuple(range(1, w.ndim)) if per_channel and w.ndim > 1 else tuple(range(w.ndim))
    threshold = threshold_ratio * w.detach().abs().amax(dim=dims, keepdim=True)
    mask = w.abs() > threshold
    count = mask.sum(dim=dims, keepdim=True).clamp_min(1)
    alpha = (w.detach().abs() * mask).sum(dim=dims, keepdim=True) / count
    hard = w.sign() * mask * alpha
    # Straight-through estimator: forward=hard, backward approximately identity.
    return w + (hard - w).detach(), w.sign() * mask, alpha.detach(), mask


def _clip(x, limit):
    return x.clamp(-limit, limit) if limit is not None else x


def ternary_conv2d(x, w, bias, padding=(1, 2), threshold_ratio=0.05,
                   per_channel=True, accumulator_limit=None):
    q, qsign, alpha, mask = ternary_quantize(w, threshold_ratio, per_channel)
    # q is alpha*{-1,0,+1}; split the accumulation to expose SCAMP's add/subtract path.
    pos = (q > 0).to(q.dtype)
    neg = (q < 0).to(q.dtype)
    # F.pad's four-tuple is left, right, top, bottom; this matches the
    # asymmetric padding used by the existing FP32 reference graph.
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    y = F.conv2d(F.pad(x, padding), pos * q.abs(), None)
    y = y - F.conv2d(F.pad(x, padding), neg * q.abs(), None)
    y = _clip(y, accumulator_limit)
    if bias is not None:
        y = y + bias.view(1, -1, 1, 1)
    return y, qsign, alpha, mask


class HardwareAwareCNN(nn.Module):
    def __init__(self, quantize=False, threshold_ratio=0.05,
                 per_channel=True, activation_limit=None,
                 accumulator_limit=None, ternary_conv=True,
                 ternary_fc=True):
        super().__init__()
        self.conv_weight = nn.Parameter(torch.empty(16, 1, 4, 4))
        self.conv_bias = nn.Parameter(torch.zeros(16))
        self.fc_weight = nn.Parameter(torch.empty(10, 4096))
        self.fc_bias = nn.Parameter(torch.zeros(10))
        nn.init.kaiming_uniform_(self.conv_weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.fc_weight, a=5 ** 0.5)
        self.quantize = quantize
        self.threshold_ratio = threshold_ratio
        self.per_channel = per_channel
        self.activation_limit = activation_limit
        self.accumulator_limit = accumulator_limit
        self.ternary_conv = ternary_conv
        self.ternary_fc = ternary_fc

    def load_from_fp32(self, model):
        self.conv_weight.data.copy_(model.conv.weight.data)
        self.conv_bias.data.copy_(model.conv.bias.data)
        self.fc_weight.data.copy_(model.fc.weight.data)
        self.fc_bias.data.copy_(model.fc.bias.data)
        return self

    def _weights(self, w, enabled):
        if self.quantize and enabled:
            return ternary_quantize(w, self.threshold_ratio, self.per_channel)
        return w, None, None, torch.ones_like(w, dtype=torch.bool)

    def forward(self, x, return_info=False):
        if self.quantize and self.ternary_conv:
            y, qs_conv, alpha_conv, mask_conv = ternary_conv2d(
                x, self.conv_weight, self.conv_bias,
                threshold_ratio=self.threshold_ratio,
                per_channel=self.per_channel,
                accumulator_limit=self.accumulator_limit)
        else:
            y = F.conv2d(F.pad(x, (1, 2, 1, 2)), self.conv_weight, self.conv_bias)
            y = _clip(y, self.accumulator_limit)
            qs_conv = alpha_conv = mask_conv = None
        y = _clip(F.relu(y), self.activation_limit)
        y = F.max_pool2d(y, 4, 4)
        flat = y.flatten(1)
        if self.quantize and self.ternary_fc:
            qfc, qs_fc, alpha_fc, mask_fc = ternary_quantize(
                self.fc_weight, self.threshold_ratio, self.per_channel)
            pos = (qfc > 0).to(qfc.dtype)
            neg = (qfc < 0).to(qfc.dtype)
            logits = F.linear(flat, pos * qfc.abs()) - F.linear(flat, neg * qfc.abs())
            logits = _clip(logits, self.accumulator_limit) + self.fc_bias
        else:
            logits = F.linear(flat, self.fc_weight, self.fc_bias)
            logits = _clip(logits, self.accumulator_limit)
            qs_fc = alpha_fc = mask_fc = None
        if not return_info:
            return logits
        return logits, {
            "conv_sign": qs_conv, "conv_alpha": alpha_conv,
            "conv_mask": mask_conv, "fc_sign": qs_fc,
            "fc_alpha": alpha_fc, "fc_mask": mask_fc,
        }


def load_hardware_from_checkpoint(path, **kwargs):
    from .torch_model import build_model
    m = HardwareAwareCNN(**kwargs)
    fp = build_model()
    fp.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.load_from_fp32(fp)
