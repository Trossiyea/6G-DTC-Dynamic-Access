"""
Trace data structures for UI playback and debugging.

This module defines small, NumPy-friendly containers that hold per-TTI
time series needed by an interactive UI (e.g., PySide6 dashboard) without
coupling UI code into the simulator core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SchedulerTrace:
    """Per-scheduler time series trace.

    Notes on units:
    - ue_thr_* arrays store "sum spectral efficiency across allocated PRBs"
      per UE per TTI (compatible with scheduler internal accounting).
    - sum_se_* arrays store average SE per PRB for each TTI.
    """

    assignments: Optional[np.ndarray] = None  # [T, Z] winner UE per PRB
    ue_thr_scheduled: Optional[np.ndarray] = None  # [T, N_UE] scheduled (instant) SE sum
    ue_thr_acked: Optional[np.ndarray] = None  # [T, N_UE] ACKed SE sum (HARQ)
    sum_se_scheduled_per_prb: Optional[np.ndarray] = None  # [T]
    sum_se_acked_per_prb: Optional[np.ndarray] = None  # [T]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments": self.assignments,
            "ue_thr_scheduled": self.ue_thr_scheduled,
            "ue_thr_acked": self.ue_thr_acked,
            "sum_se_scheduled_per_prb": self.sum_se_scheduled_per_prb,
            "sum_se_acked_per_prb": self.sum_se_acked_per_prb,
        }


@dataclass
class OALSTrace:
    """OALS per-TTI debug/visualization series."""

    phi: np.ndarray  # [T, N_UE]
    trend: np.ndarray  # [T, N_UE]
    urgency: np.ndarray  # [T, N_UE]
    correction: np.ndarray  # [T, N_UE]
    update_ttis: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phi": self.phi,
            "trend": self.trend,
            "urgency": self.urgency,
            "correction": self.correction,
            "update_ttis": list(self.update_ttis),
        }


@dataclass
class SimulationTrace:
    """Unified trace container for UI playback."""

    mode: str  # "single" or "constellation"
    tti: np.ndarray  # [T]
    baseline: SchedulerTrace
    radiomap: SchedulerTrace
    oals: Optional[OALSTrace] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "mode": self.mode,
            "tti": self.tti,
            "baseline": self.baseline.to_dict(),
            "radiomap": self.radiomap.to_dict(),
            "extra": dict(self.extra),
        }
        if self.oals is not None:
            out["oals"] = self.oals.to_dict()
        return out

