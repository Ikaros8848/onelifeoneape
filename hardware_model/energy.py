"""Energy-proxy variants and ablations.

All outputs are dimensionless software estimates. Proxy A reproduces the
historical formula; Proxy B removes arithmetic from register accesses.
"""
from typing import Dict

from .config import HardwareProxyConfig
from .cost_model import CostReport


def energy_components(report: CostReport, config: HardwareProxyConfig | None = None,
                      orthogonal_registers: bool = False) -> Dict[str, float]:
    cfg = config or HardwareProxyConfig()
    arithmetic = (report.positive_terms * cfg.energy_add
                   + report.negative_terms * cfg.energy_subtract)
    comparison = report.comparisons * cfg.energy_compare
    movement = (report.local_movement_elements * cfg.energy_local_move
                + report.external_movement_bits * cfg.energy_external_bit)
    register_accesses = report.comparisons + report.local_movement_elements
    if not orthogonal_registers:
        register_accesses += report.active_terms
    register = register_accesses * cfg.energy_register_access
    control_stages = (report.retained_offsets + report.offset_route_depth
                      + cfg.pool_schedule_depth + report.fc_mask_passes)
    control = control_stages * cfg.energy_control_stage
    return {
        "arithmetic": arithmetic,
        "comparison": comparison,
        "movement": movement,
        "register": register,
        "control": control,
        "total": arithmetic + comparison + movement + register + control,
        "register_accesses": float(register_accesses),
        "control_stages": float(control_stages),
    }


def energy_ablation(report: CostReport, config: HardwareProxyConfig | None = None) -> Dict[str, float]:
    cfg = config or HardwareProxyConfig()
    c = energy_components(report, cfg, orthogonal_registers=False)
    b = energy_components(report, cfg, orthogonal_registers=True)
    operation_only = c["arithmetic"]
    operation_movement = c["arithmetic"] + c["comparison"] + c["movement"]
    operation_movement_register = operation_movement + b["register"]
    return {
        "A_arithmetic_only": operation_only,
        "B_operation_plus_movement": operation_movement,
        "C_operation_movement_plus_orthogonal_register": operation_movement_register,
        "D_full_current_proxy": c["total"],
        "B_orthogonal_total": b["total"],
    }
