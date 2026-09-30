"""Transparent offset-by-offset convolution schedule simulator.

This is a CPU reference implementation. It intentionally does not claim
SCAMP instruction timing or CPU/GPU acceleration. Its purpose is to make the
logical retained-tap schedule executable and auditable.
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


Offset = Tuple[int, int]


def retained_offsets(signs: torch.Tensor) -> List[Offset]:
    if signs.ndim != 4:
        raise ValueError("signs must have shape [out_channels,in_channels,kh,kw]")
    active = signs.ne(0).any(dim=(0, 1))
    return [(i, j) for i in range(active.shape[0])
            for j in range(active.shape[1]) if bool(active[i, j])]


def build_schedule(
    signs: torch.Tensor,
    retained: Optional[Iterable[Offset]] = None,
    output_hw: Tuple[int, int] = (64, 64),
) -> List[Dict[str, int]]:
    """Return one auditable record per executed kernel offset."""
    signs = signs.detach().cpu()
    offsets = list(retained) if retained is not None else retained_offsets(signs)
    h, w = output_hw
    records = []
    for i, j in offsets:
        plus = int((signs[:, :, i, j] > 0).sum())
        minus = int((signs[:, :, i, j] < 0).sum())
        zero = int((signs[:, :, i, j] == 0).sum())
        records.append({
            "offset_i": int(i), "offset_j": int(j),
            "positive_weight_terms": plus,
            "negative_weight_terms": minus,
            "zero_weight_terms": zero,
            "active_operations": (plus + minus) * h * w,
            "skipped_operations": zero * h * w,
            "candidate_operations": signs.shape[0] * signs.shape[1] * h * w,
            "local_movement_elements": h * w,
            "register_local_operations": (plus + minus) * h * w + h * w,
        })
    return records


def sparse_reference_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    retained: Optional[Sequence[Offset]] = None,
    padding: Tuple[int, int, int, int] = (1, 2, 1, 2),
) -> torch.Tensor:
    """Execute only selected offsets using explicit per-offset accumulation.

    `weight` may contain channel scales and ternary signs. Zero weights inside
    a retained group contribute no arithmetic, while the output remains
    mathematically identical to a dense convolution with the same masked
    weight. Padding follows torch's (left, right, top, bottom) convention.
    """
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("x must be BCHW and weight must be OCKK")
    left, right, top, bottom = padding
    padded = F.pad(x, (left, right, top, bottom))
    out_h = padded.shape[-2] - weight.shape[-2] + 1
    out_w = padded.shape[-1] - weight.shape[-1] + 1
    offsets = list(retained) if retained is not None else retained_offsets(weight)
    y = x.new_zeros((x.shape[0], weight.shape[0], out_h, out_w))
    for i, j in offsets:
        patch = padded[:, :, i:i + out_h, j:j + out_w]
        coeff = weight[:, :, i, j]
        # This is a transparent reference contraction for one tap. It does
        # not call a full 4x4 convolution kernel.
        y = y + torch.einsum("bchw,oc->bohw", patch, coeff)
    if bias is not None:
        y = y + bias.view(1, -1, 1, 1)
    return y


def summarize_schedule(records: Sequence[Dict[str, int]]) -> Dict[str, int]:
    keys = ("active_operations", "skipped_operations", "candidate_operations")
    return {
        "schedule_length": len(records),
        **{key: sum(int(r[key]) for r in records) for key in keys},
    }
