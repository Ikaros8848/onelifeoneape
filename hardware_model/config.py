"""Transparent assumptions for normalized SCAMP cost accounting."""
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class HardwareProxyConfig:
    # Documented capacities; used only as denominators for pressure scores.
    analog_register_planes: int = 7
    digital_register_planes: int = 13

    # Accounting widths, not measured analogue precision.
    input_bits: int = 8
    activation_bits: int = 8
    ternary_bits: int = 2
    metadata_bits: int = 16

    # Live-plane schedule assumptions.
    base_analog_live_planes: int = 3
    base_digital_live_planes: int = 4
    pool_schedule_depth: int = 4

    # Dimensionless energy-proxy coefficients.
    energy_add: float = 1.0
    energy_subtract: float = 1.0
    energy_compare: float = 0.5
    energy_local_move: float = 0.25
    energy_external_bit: float = 0.01
    energy_register_access: float = 0.10
    energy_control_stage: float = 32.0

    # Latency-proxy coefficients are sequential-stage equivalents.
    latency_tap_stage: float = 1.0
    latency_shift_stage: float = 1.0
    latency_pool_stage: float = 1.0
    latency_fc_mask_stage: float = 1.0
    latency_readout_stage: float = 1.0
    latency_spill_stage: float = 4.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# These weights are fixed before experiments. Metrics are normalized to the
# same reference report, so profiles expose preference sensitivity rather than
# changing metric definitions.
COMPOSITE_PROFILES = {
    "balanced": {
        "compute": 0.20, "movement": 0.20, "storage": 0.20,
        "latency": 0.20, "energy": 0.20,
    },
    "compute_heavy": {
        "compute": 0.50, "movement": 0.125, "storage": 0.125,
        "latency": 0.125, "energy": 0.125,
    },
    "movement_heavy": {
        "compute": 0.125, "movement": 0.50, "storage": 0.125,
        "latency": 0.125, "energy": 0.125,
    },
    "schedule_heavy": {
        "compute": 0.125, "movement": 0.125, "storage": 0.125,
        "latency": 0.50, "energy": 0.125,
    },
    "storage_heavy": {
        "compute": 0.125, "movement": 0.125, "storage": 0.50,
        "latency": 0.125, "energy": 0.125,
    },
}
