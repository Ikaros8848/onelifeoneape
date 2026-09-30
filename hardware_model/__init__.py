"""SCAMP-aware analytical cost proxies.

These modules do not simulate analogue physics or predict real chip latency/energy.
"""
from .config import HardwareProxyConfig, COMPOSITE_PROFILES
from .cost_model import ScampCostModel, CostReport, compare_reports

__all__ = [
    "HardwareProxyConfig",
    "COMPOSITE_PROFILES",
    "ScampCostModel",
    "CostReport",
    "compare_reports",
]
