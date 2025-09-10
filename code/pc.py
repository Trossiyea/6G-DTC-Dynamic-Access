"""
NR PUSCH fractional power control utilities (TS 38.213 inspired).

This initial drop exposes an open-loop calculation that mirrors the
existing behavior, so we can later plug in per-TTI PRB counts (M) and
closed-loop TPC without changing other code.
"""

from typing import Optional, Union
import numpy as np


def pusch_open_loop_power(
    P_cmax_dbm: float,
    P0_dbm: float,
    alpha: float,
    PL_db: Union[float, np.ndarray],
    M_prb: int = 1,
    delta_tf_db: float = 0.0,
) -> np.ndarray:
    """
    Open-loop part of NR PUSCH power control (simplified):
    P = min(P_cmax, 10*log10(M*12) + P0 + alpha*PL + Δ_TF)
    - M_prb: number of PRBs scheduled (placeholder; we keep constant for now)
    - 12 subcarriers per PRB assumed
    Returns array matching PL_db shape.
    """
    PL_db = np.asarray(PL_db, dtype=float)
    M = max(1, int(M_prb))
    term_prb = 10.0 * np.log10(M * 12.0)
    p = term_prb + float(P0_dbm) + float(alpha) * PL_db + float(delta_tf_db)
    return np.minimum(p, float(P_cmax_dbm))


class ClosedLoopTPC:
    """
    Skeleton for closed-loop ΔTPC. Not used yet; kept for future steps.
    """

    def __init__(self, step_db: float = 1.0):
        self.step_db = float(step_db)
        self.delta_db = 0.0

    def get(self) -> float:
        return self.delta_db

    def apply_command(self, cmd: int) -> None:
        # Placeholder: cmd in {-1, 0, +1} to decrease/hold/increase
        self.delta_db += float(cmd) * self.step_db

