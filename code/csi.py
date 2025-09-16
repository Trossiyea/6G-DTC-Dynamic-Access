"""
CSI utilities for NR-NTN simulation.

Initial step: provide a stable wrapper for SINR->SE mapping so we can
later swap in 3GPP TS 38.214 MCS tables, OLLA, and CSI delay without
changing call sites across the codebase.

This first version keeps the existing behavior (legacy thresholds)
to avoid regressions.
"""

from typing import Dict, Optional
import numpy as np


# 3GPP TS 38.214 §5.2.2.1 CQI-to-SE entries (DL).
# Each tuple: (SINR threshold for CQI k at 10% BLER, modulation order Qm, R_x1024)
_NR_CQI_TABLE: Dict[str, np.ndarray] = {
    "nr_64qam": np.array([
        (-6.7, 2, 78),   # CQI 1
        (-4.7, 2, 120),
        (-2.3, 2, 193),
        (0.2,  2, 308),
        (2.4,  2, 449),
        (4.3,  2, 602),
        (5.9,  4, 378),
        (8.1,  4, 490),
        (10.3, 4, 616),
        (11.7, 6, 466),
        (14.1, 6, 567),
        (16.3, 6, 666),
        (18.7, 6, 772),
        (21.0, 6, 873),
        (22.7, 6, 948),
    ], dtype=float),
    # For 256QAM-capable UE the CQI-to-SE mapping reuses the same thresholds
    # while allowing higher-order MCS beyond CQI 15 through adaptive MCS.
    "nr_256qam": np.array([
        (-6.7, 2, 78),
        (-4.7, 2, 120),
        (-2.3, 2, 193),
        (0.2,  2, 308),
        (2.4,  2, 449),
        (4.3,  2, 602),
        (5.9,  4, 378),
        (8.1,  4, 490),
        (10.3, 4, 616),
        (11.7, 6, 466),
        (14.1, 6, 567),
        (16.3, 6, 666),
        (18.7, 6, 772),
        (21.0, 6, 873),
        (22.7, 6, 948),
    ], dtype=float),
}


def _resolve_table(table: str) -> np.ndarray:
    key = str(table or "nr_64qam").lower()
    if key == "legacy":
        key = "nr_64qam"
    if key not in _NR_CQI_TABLE:
        raise ValueError(f"Unknown CQI table: {table}")
    return _NR_CQI_TABLE[key]


def get_nr_cqi_table(table: str = "nr_64qam") -> np.ndarray:
    """Return a copy of the CQI table entries (threshold_dB, Qm, R_x1024)."""
    return _resolve_table(table).copy()


def sinr_to_cqi(sinr_db: np.ndarray, table: str = "nr_64qam") -> np.ndarray:
    """Map SINR (dB) to CQI index (0..15) using 38.214 thresholds."""
    sinr_db = np.asarray(sinr_db, dtype=float)
    tbl = _resolve_table(table)
    thr = tbl[:, 0]
    idx = np.searchsorted(thr, sinr_db, side="right")
    return np.clip(idx, 0, tbl.shape[0])


def cqi_to_se(cqi: np.ndarray, table: str = "nr_64qam") -> np.ndarray:
    """Convert CQI (0..15) into spectral efficiency (bits/s/Hz)."""
    cqi = np.asarray(cqi, dtype=int)
    tbl = _resolve_table(table)
    se_vals = (tbl[:, 1] * tbl[:, 2]) / 1024.0
    out = np.zeros_like(cqi, dtype=float)
    mask = cqi > 0
    idx = np.clip(cqi[mask] - 1, 0, se_vals.size - 1)
    out[mask] = se_vals[idx]
    return out


def sinr_to_se_mcs(sinr_db: np.ndarray, table: str = "legacy") -> np.ndarray:
    """Map SINR to spectral efficiency via CQI tables."""
    sinr_db = np.asarray(sinr_db, dtype=float)
    tbl_key = str(table or "nr_64qam").lower()
    if tbl_key not in ("legacy", "nr_64qam", "nr_256qam"):
        raise ValueError(f"Unknown MCS table: {table}")
    if tbl_key == "legacy":
        tbl_key = "nr_64qam"
    cqi = sinr_to_cqi(sinr_db, table=tbl_key)
    return cqi_to_se(cqi, table=tbl_key)


class OLLA:
    """
    Outer Loop Link Adaptation placeholder (no-op for now).
    Later we will adapt target BLER by nudging SINR offset.
    """

    def __init__(self, step_db: float = 0.1, init_offset_db: float = 0.0):
        self.step_db = float(step_db)
        self.offset_db = float(init_offset_db)

    def apply(self, sinr_db: np.ndarray) -> np.ndarray:
        return np.asarray(sinr_db, dtype=float) + self.offset_db

    def update(self, ack: Optional[np.ndarray] = None) -> None:
        # No-op in this initial drop; will adjust offset based on ACK/NACK later.
        return


def effective_sinr_eesm(sinr_db: np.ndarray, beta_db: float = 1.0, axis: int = -1) -> np.ndarray:
    """
    Exponential Effective SINR Mapping (EESM):
    SINR_eff(dB) = -β * ln( mean( exp( -SINR_i/β ) ) )
    - sinr_db: array of SINR values in dB (per-subcarrier/PRB samples along `axis`).
    - beta_db: calibration parameter β in dB domain (typical 1..6 dB depending on MCS).
    - axis: axis along which to aggregate.
    Returns an array with `axis` reduced.
    """
    x = np.asarray(sinr_db, dtype=float)
    beta = float(beta_db)
    # Guard for empty axis
    if x.size == 0:
        return np.array(0.0, dtype=float)
    # Numeric handling: exp(-x/beta) can under/overflow at extremes; clip beta>0
    beta = max(beta, 1e-6)
    t = np.exp(-x / beta)
    m = np.mean(t, axis=axis)
    # Avoid log(0)
    m = np.maximum(m, 1e-30)
    sinr_eff_db = -beta * np.log(m)
    return sinr_eff_db
