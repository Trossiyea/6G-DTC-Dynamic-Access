"""
CSI utilities for NR-NTN simulation.

Initial step: provide a stable wrapper for SINR->SE mapping so we can
later swap in 3GPP TS 38.214 MCS tables, OLLA, and CSI delay without
changing call sites across the codebase.

This first version keeps the existing behavior (legacy thresholds)
to avoid regressions.
"""

from typing import Optional, Tuple
import numpy as np


def _legacy_sinr_to_se_mcs(sinr_db: np.ndarray) -> np.ndarray:
    """
    Legacy mapping: approximate LTE-like CQI thresholds to spectral efficiency.
    Maintains existing behavior in main.py before we introduce 38.214 tables.
    """
    thr_db = np.array([
        -6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7
    ], dtype=float)
    se_vals = np.array([
        0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063,
        2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547
    ], dtype=float)
    idx = np.searchsorted(thr_db, sinr_db, side='right') - 1
    idx = np.clip(idx, 0, len(se_vals) - 1)
    se = se_vals[idx]
    se = np.where(sinr_db < thr_db[0], 0.0, se)
    return se


def _nr_cqi_table(table: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (thresholds_dB[15], se_vals[15]) for CQI 1..15 at 10% BLER AWGN.
    - For now, thresholds are the common industry set used for 64QAM (legacy LTE-like),
      which is a reasonable baseline for NR Table 1 (64QAM).
    - For Table 2 (256QAM), we reuse thresholds initially and keep SE equal to Table 1
      until calibrated values are provided.
    """
    thr_db = np.array([
        -6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7
    ], dtype=float)
    if table in ("legacy", "nr_64qam"):
        se_vals = np.array([
            0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063,
            2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547
        ], dtype=float)
        return thr_db, se_vals
    if table == "nr_256qam":
        # Placeholder: same thresholds and SE as 64QAM until calibrated values are supplied.
        se_vals = np.array([
            0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063,
            2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547
        ], dtype=float)
        return thr_db, se_vals
    raise ValueError(f"Unknown CQI table: {table}")


def sinr_to_cqi(sinr_db: np.ndarray, table: str = "nr_64qam") -> np.ndarray:
    """
    Map SINR to CQI (0..15), with 0 indicating out of coverage.
    """
    sinr_db = np.asarray(sinr_db, dtype=float)
    thr_db, _ = _nr_cqi_table(table)
    idx = np.searchsorted(thr_db, sinr_db, side='right')
    cqi = np.clip(idx, 0, 15)
    return cqi


def cqi_to_se(cqi: np.ndarray, table: str = "nr_64qam") -> np.ndarray:
    """
    Map CQI (0..15) to spectral efficiency (bits/s/Hz). CQI=0 -> SE=0.
    """
    cqi = np.asarray(cqi, dtype=int)
    _, se_vals = _nr_cqi_table(table)
    # cqi 1..15 map to se_vals[0..14]
    se = np.zeros_like(cqi, dtype=float)
    mask = (cqi > 0)
    se[mask] = se_vals[np.clip(cqi[mask] - 1, 0, len(se_vals) - 1)]
    return se


def sinr_to_se_mcs(sinr_db: np.ndarray, table: str = "legacy") -> np.ndarray:
    """
    Public API: map SINR (dB) to spectral efficiency (bits/s/Hz).
    - table: "legacy" keeps current behavior; future options will include
      standardized 38.214 tables (e.g., "nr_64qam", "nr_256qam").
    """
    sinr_db = np.asarray(sinr_db, dtype=float)
    if table == "legacy":
        return _legacy_sinr_to_se_mcs(sinr_db)
    if table in ("nr_64qam", "nr_256qam"):
        cqi = sinr_to_cqi(sinr_db, table="nr_64qam" if table == "nr_64qam" else "nr_256qam")
        return cqi_to_se(cqi, table=table)
    raise ValueError(f"Unknown MCS table: {table}")


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


def _default_eesm_beta_table(table: str = "nr_64qam") -> np.ndarray:
    """
    Returns a simple per-CQI EESM beta lookup (dB) for CQI 0..15.
    Placeholder calibration: small-to-moderate beta growing with CQI.
    """
    # CQI 0..15; CQI=0 unused
    if table in ("legacy", "nr_64qam", "nr_256qam"):
        return np.array([
            1.0,  # CQI 0 (unused)
            1.0, 1.0, 1.0, 1.2, 1.2, 1.4, 1.5, 1.7,
            1.8, 2.0, 2.2, 2.4, 2.6, 3.0, 3.5, 4.0
        ], dtype=float)
    return np.full(16, 1.0, dtype=float)


def pick_eesm_beta_from_cqi(cqi: np.ndarray,
                            table: str = "nr_64qam",
                            custom_table: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Map CQI (0..15) to an EESM beta (dB). If custom_table is provided (len>=16), use it;
    otherwise use the default placeholder table above.
    """
    cqi = np.asarray(cqi, dtype=int)
    if custom_table is not None:
        bt = np.asarray(custom_table, dtype=float)
        if bt.size < 16:
            # pad/repeat to length 16
            pad = np.full(16, float(bt.flat[0]), dtype=float)
            pad[:bt.size] = bt
            bt = pad
    else:
        bt = _default_eesm_beta_table(table)
    idx = np.clip(cqi, 0, 15)
    return bt[idx]
