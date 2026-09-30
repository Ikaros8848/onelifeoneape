"""Analytical SCAMP-aware cost model for the single-layer MNIST network.

The report deliberately separates scalar activity from sequential PPA schedule
depth. It is a repeatable software proxy, not a cycle-accurate simulator.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .config import COMPOSITE_PROFILES, HardwareProxyConfig


Offset = Tuple[int, int]


@dataclass
class CostReport:
    name: str
    mapping: str
    parameters: int
    positive_terms: int
    negative_terms: int
    active_terms: int
    skipped_terms: int
    dense_terms: int
    comparisons: int
    retained_offsets: int
    total_offsets: int
    offset_route_depth: int
    fc_mask_passes: int
    external_movement_bits: int
    local_movement_elements: int
    feature_footprint_elements: int
    analog_live_planes: int
    digital_live_planes: int
    register_pressure: float
    register_excess: float
    latency_proxy: float
    energy_proxy: float
    assumptions: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _counts(signs: torch.Tensor) -> Tuple[int, int, int]:
    signs = signs.detach().cpu()
    return (int((signs > 0).sum()), int((signs < 0).sum()),
            int((signs == 0).sum()))


def _offsets_from_conv(conv_signs: torch.Tensor) -> List[Offset]:
    if conv_signs.ndim != 4:
        raise ValueError("conv_signs must have shape [out_channels,in_channels,kh,kw]")
    active = conv_signs.ne(0).any(dim=(0, 1))
    return [(y, x) for y in range(active.shape[0])
            for x in range(active.shape[1]) if bool(active[y, x])]


def route_depth(offsets: Sequence[Offset], start: Offset = (0, 0)) -> int:
    """Manhattan routing proxy for a supplied sequential offset order."""
    depth = 0
    prev = start
    for current in offsets:
        depth += abs(current[0] - prev[0]) + abs(current[1] - prev[1])
        prev = current
    return depth


def shortest_route_depth(offsets: Sequence[Offset], start: Offset = (0, 0)) -> int:
    """Exact shortest Manhattan route through at most 16 stencil offsets."""
    points = tuple(offsets)
    if not points:
        return 0
    if len(points) > 16:
        raise ValueError("exact route search is limited to a 4x4 stencil")

    @lru_cache(maxsize=None)
    def visit(mask: int, last: int) -> int:
        if mask == (1 << len(points)) - 1:
            return 0
        origin = start if last == len(points) else points[last]
        best = 10**9
        for nxt, point in enumerate(points):
            if mask & (1 << nxt):
                continue
            dist = abs(origin[0] - point[0]) + abs(origin[1] - point[1])
            best = min(best, dist + visit(mask | (1 << nxt), nxt))
        return best

    return visit(0, len(points))


class ScampCostModel:
    def __init__(self, config: Optional[HardwareProxyConfig] = None):
        self.config = config or HardwareProxyConfig()

    def analyze_ternary(
        self,
        name: str,
        conv_signs: torch.Tensor,
        fc_signs: torch.Tensor,
        parameters: int = 41242,
        offset_order: Optional[Iterable[Offset]] = None,
        analog_live_planes: Optional[int] = None,
        digital_live_planes: Optional[int] = None,
    ) -> CostReport:
        cfg = self.config
        conv_plus, conv_minus, conv_zero = _counts(conv_signs)
        fc_plus, fc_minus, fc_zero = _counts(fc_signs)
        out_pixels = 64 * 64
        conv_pos_terms = conv_plus * out_pixels
        conv_neg_terms = conv_minus * out_pixels
        conv_skip_terms = conv_zero * out_pixels
        positive_terms = conv_pos_terms + fc_plus
        negative_terms = conv_neg_terms + fc_minus
        skipped_terms = conv_skip_terms + fc_zero
        dense_terms = int(conv_signs.numel()) * out_pixels + int(fc_signs.numel())
        active_terms = positive_terms + negative_terms

        active_offsets = _offsets_from_conv(conv_signs)
        if offset_order is None:
            scheduled_offsets = active_offsets
        else:
            active_set = set(active_offsets)
            scheduled_offsets = [tuple(o) for o in offset_order if tuple(o) in active_set]
            if set(scheduled_offsets) != active_set:
                raise ValueError("offset_order must contain every retained offset exactly once")
        offset_depth = shortest_route_depth(scheduled_offsets)

        # One positive and/or negative sparse-global-sum mask pass per class.
        fc_flat = fc_signs.reshape(fc_signs.shape[0], -1)
        fc_mask_passes = int((fc_flat > 0).any(1).sum() + (fc_flat < 0).any(1).sum())

        pooled_elements = 16 * 16 * 16
        conv_elements = 16 * 64 * 64
        input_elements = 64 * 64
        output_elements = 10
        comparisons = 16 * 16 * 16 * (4 * 4 - 1)

        # External/configuration movement is kept separate from local NEWS-like
        # motion. Conv maps are assumed to remain local until pooling.
        weight_bits = (int(conv_signs.numel()) + int(fc_signs.numel())) * cfg.ternary_bits
        scale_bias_count = conv_signs.shape[0] + fc_signs.shape[0] + 26
        external_bits = (
            input_elements * cfg.input_bits
            + pooled_elements * cfg.activation_bits
            + output_elements * cfg.activation_bits
            + weight_bits
            + scale_bias_count * cfg.metadata_bits
        )
        local_move_elements = offset_depth * out_pixels + comparisons

        analog_live = analog_live_planes or cfg.base_analog_live_planes
        digital_live = digital_live_planes or cfg.base_digital_live_planes
        analog_pressure = analog_live / cfg.analog_register_planes
        digital_pressure = digital_live / cfg.digital_register_planes
        register_pressure = max(analog_pressure, digital_pressure)
        register_excess = (
            max(0.0, analog_pressure - 1.0) + max(0.0, digital_pressure - 1.0)
        )

        latency = (
            len(active_offsets) * cfg.latency_tap_stage
            + offset_depth * cfg.latency_shift_stage
            + cfg.pool_schedule_depth * cfg.latency_pool_stage
            + fc_mask_passes * cfg.latency_fc_mask_stage
            + output_elements * cfg.latency_readout_stage
            + register_excess * cfg.latency_spill_stage
        )
        register_accesses = active_terms + comparisons + local_move_elements
        control_stages = len(active_offsets) + offset_depth + cfg.pool_schedule_depth + fc_mask_passes
        energy = (
            positive_terms * cfg.energy_add
            + negative_terms * cfg.energy_subtract
            + comparisons * cfg.energy_compare
            + local_move_elements * cfg.energy_local_move
            + external_bits * cfg.energy_external_bit
            + register_accesses * cfg.energy_register_access
            + control_stages * cfg.energy_control_stage
        )

        return CostReport(
            name=name, mapping="ternary_scamp_proxy", parameters=parameters,
            positive_terms=positive_terms, negative_terms=negative_terms,
            active_terms=active_terms, skipped_terms=skipped_terms,
            dense_terms=dense_terms, comparisons=comparisons,
            retained_offsets=len(active_offsets),
            total_offsets=int(conv_signs.shape[-2] * conv_signs.shape[-1]),
            offset_route_depth=offset_depth, fc_mask_passes=fc_mask_passes,
            external_movement_bits=external_bits,
            local_movement_elements=local_move_elements,
            feature_footprint_elements=max(input_elements, conv_elements, pooled_elements),
            analog_live_planes=analog_live, digital_live_planes=digital_live,
            register_pressure=register_pressure, register_excess=register_excess,
            latency_proxy=latency, energy_proxy=energy,
            assumptions=cfg.to_dict(),
        )


def _safe_ratio(value: float, reference: float) -> float:
    if reference == 0:
        return 0.0 if value == 0 else float("inf")
    return value / reference


def compare_reports(
    report: CostReport, reference: CostReport,
    profiles: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    """Normalize a report to one immutable reference and score profiles."""
    profiles = profiles or COMPOSITE_PROFILES
    normalized = {
        "compute": _safe_ratio(report.active_terms, reference.active_terms),
        "movement": _safe_ratio(
            report.external_movement_bits + report.local_movement_elements,
            reference.external_movement_bits + reference.local_movement_elements),
        "storage": _safe_ratio(
            report.feature_footprint_elements * report.register_pressure,
            reference.feature_footprint_elements * reference.register_pressure),
        "latency": _safe_ratio(report.latency_proxy, reference.latency_proxy),
        "energy": _safe_ratio(report.energy_proxy, reference.energy_proxy),
    }
    composite = {
        name: sum(weights[key] * normalized[key] for key in normalized)
        for name, weights in profiles.items()
    }
    improvement = {key: (1.0 - value) * 100.0 for key, value in normalized.items()}
    improvement.update({f"composite_{k}": (1.0 - v) * 100.0 for k, v in composite.items()})
    return {
        "reference": reference.name,
        "normalized": normalized,
        "composite_cost": composite,
        "improvement_percent": improvement,
        "profiles": profiles,
    }
