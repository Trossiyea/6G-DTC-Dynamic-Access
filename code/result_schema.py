"""
Serializable schema for simulation results.

Keeps a structured view of the primary KPIs while allowing passthrough of
scenario-specific extras for downstream consumers (e.g., web visualizations).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


_CORE_KEYS = {
    "avg_se_baseline_default",
    "avg_se_radiomap",
    "improvement_vs_default_pct",
    "system_bandwidth_hz",
    "total_throughput_baseline_bps",
    "total_throughput_radiomap_bps",
    "avg_ue_throughput_baseline_bps",
    "avg_ue_throughput_radiomap_bps",
}


@dataclass
class SimulationResult:
    avg_se_baseline_default: float
    avg_se_radiomap: float
    improvement_vs_default_pct: float
    system_bandwidth_hz: Optional[float] = None
    total_throughput_baseline_bps: Optional[float] = None
    total_throughput_radiomap_bps: Optional[float] = None
    avg_ue_throughput_baseline_bps: Optional[float] = None
    avg_ue_throughput_radiomap_bps: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationResult":
        extras = {k: v for k, v in data.items() if k not in _CORE_KEYS}
        return cls(
            avg_se_baseline_default=float(data["avg_se_baseline_default"]),
            avg_se_radiomap=float(data["avg_se_radiomap"]),
            improvement_vs_default_pct=float(data["improvement_vs_default_pct"]),
            system_bandwidth_hz=data.get("system_bandwidth_hz"),
            total_throughput_baseline_bps=data.get("total_throughput_baseline_bps"),
            total_throughput_radiomap_bps=data.get("total_throughput_radiomap_bps"),
            avg_ue_throughput_baseline_bps=data.get("avg_ue_throughput_baseline_bps"),
            avg_ue_throughput_radiomap_bps=data.get("avg_ue_throughput_radiomap_bps"),
            extras=extras,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "avg_se_baseline_default": self.avg_se_baseline_default,
            "avg_se_radiomap": self.avg_se_radiomap,
            "improvement_vs_default_pct": self.improvement_vs_default_pct,
            "system_bandwidth_hz": self.system_bandwidth_hz,
            "total_throughput_baseline_bps": self.total_throughput_baseline_bps,
            "total_throughput_radiomap_bps": self.total_throughput_radiomap_bps,
            "avg_ue_throughput_baseline_bps": self.avg_ue_throughput_baseline_bps,
            "avg_ue_throughput_radiomap_bps": self.avg_ue_throughput_radiomap_bps,
        }
        # Preserve optional values only when present to keep JSON clean
        clean = {k: v for k, v in payload.items() if v is not None}
        clean.update(self.extras)
        return clean


def to_serializable_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw result dict from run_once/run_constellation into a
    schema-validated, JSON-ready dict. Non-core keys are preserved in extras.
    """
    return SimulationResult.from_dict(data).to_dict()
