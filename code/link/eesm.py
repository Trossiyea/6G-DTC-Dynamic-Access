# -*- coding: utf-8 -*-
"""
EESM (Exponential Effective SINR Mapping) for NR Link Adaptation.

Provides:
- Per-PRB to effective SINR mapping
- Soft combining across HARQ transmissions
"""

import numpy as np


def effective_sinr_eesm(
    sinr_db: np.ndarray,
    beta_db: float = 1.0,
    axis: int = -1
) -> np.ndarray:
    """Exponential Effective SINR Mapping (EESM).

    SINR_eff(dB) = -beta * ln( mean( exp( -SINR_i / beta ) ) )

    Args:
        sinr_db: SINR values in dB (per-subcarrier/PRB samples along axis)
        beta_db: Calibration parameter beta in dB domain (typical 1..6 dB)
        axis: Axis along which to aggregate

    Returns:
        Effective SINR in dB with axis reduced
    """
    x = np.asarray(sinr_db, dtype=float)
    beta = float(beta_db)

    # Guard for empty axis
    if x.size == 0:
        return np.array(0.0, dtype=float)

    # Numeric handling: clip beta > 0
    beta = max(beta, 1e-6)
    t = np.exp(-x / beta)
    m = np.mean(t, axis=axis)
    # Avoid log(0)
    m = np.maximum(m, 1e-30)
    sinr_eff_db = -beta * np.log(m)
    return sinr_eff_db


def eff_sinr_eesm_db(sinr_db_vec: np.ndarray, beta_db: float = 1.0) -> float:
    """Convenience wrapper around per-PRB EESM to single scalar (dB).

    Args:
        sinr_db_vec: 1D array of SINR values in dB
        beta_db: EESM beta parameter

    Returns:
        Scalar effective SINR in dB
    """
    x = np.asarray(sinr_db_vec, dtype=float)
    beta = max(1e-6, float(beta_db))
    t = np.exp(-x / beta)
    m = np.maximum(np.mean(t, axis=-1), 1e-30)
    return float(-beta * np.log(m))


def combine_eff_sinr_db(prev_eff_db: float, curr_eff_db: float) -> float:
    """Approximate soft-combining by adding linear SNRs of effective values.

    Used for HARQ combining across retransmissions.

    Args:
        prev_eff_db: Previous combined effective SINR (dB), or None
        curr_eff_db: Current transmission effective SINR (dB)

    Returns:
        Combined effective SINR in dB
    """
    if prev_eff_db is None:
        return float(curr_eff_db)
    a = 10.0 ** (float(prev_eff_db) / 10.0)
    b = 10.0 ** (float(curr_eff_db) / 10.0)
    return 10.0 * np.log10(a + b)
