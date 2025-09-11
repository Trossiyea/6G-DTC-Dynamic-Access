"""
NTN uplink frequency pre-compensation and CFO/ICI modeling utilities.

Key elements:
- Decompose residual CFO into:
  (i) Doppler prediction error |f_d - f_pred|,
  (ii) LO mismatch (ppm) between UE and gNB,
  (iii) small constant residual CFO (residual_freq_hz).
- Apply a simple PTRS/DMRS tracking capability that removes up to
  `ptrs_cfo_track_hz` from the total CFO, leaving only the untrackable residual.
- Convert residual CFO to ICI penalty factor using a first-order model.
"""

from typing import Optional
import numpy as np


def lo_mismatch_hz(fc_hz: float, ue_lo_ppm: float, gnb_lo_ppm: float,
                   override_mismatch_ppm: Optional[float] = None) -> float:
    """Compute LO mismatch CFO in Hz: |Δppm| * f_c / 1e6."""
    if override_mismatch_ppm is not None:
        dppm = float(override_mismatch_ppm)
    else:
        dppm = abs(float(ue_lo_ppm) - float(gnb_lo_ppm))
    return abs(float(fc_hz)) * dppm * 1e-6


def compute_residual_cfo_hz(fd_true_hz: np.ndarray,
                             f_pred_hz: np.ndarray,
                             fc_hz: float,
                             ue_lo_ppm: float,
                             gnb_lo_ppm: float,
                             residual_freq_hz: float = 0.0,
                             ptrs_cfo_track_hz: float = 0.0,
                             override_mismatch_ppm: Optional[float] = None) -> np.ndarray:
    """
    Total residual CFO per-UE (Hz) after precomp + tracking.
    total_pretrack = |fd_true - f_pred| + LO_mismatch + residual_freq_hz
    eps_f = max(0, total_pretrack - ptrs_cfo_track_hz)
    """
    fd_true_hz = np.asarray(fd_true_hz, dtype=float)
    f_pred_hz = np.asarray(f_pred_hz, dtype=float)
    doppler_resid = np.abs(fd_true_hz - f_pred_hz)
    lo_hz = lo_mismatch_hz(fc_hz, ue_lo_ppm, gnb_lo_ppm, override_mismatch_ppm)
    total_pretrack = doppler_resid + float(abs(residual_freq_hz)) + float(lo_hz)
    eps = np.maximum(0.0, total_pretrack - float(ptrs_cfo_track_hz))
    return eps


def ici_factor_from_cfo(eps_f_hz: np.ndarray, scs_khz: float) -> np.ndarray:
    """
    First-order ICI penalty factor: 1 + (2π·ε_f·T_sym)^2.
    """
    T_sym = 1.0 / (max(float(scs_khz), 1e-9) * 1e3)
    eps = np.asarray(eps_f_hz, dtype=float)
    return 1.0 + (2.0 * np.pi * eps * T_sym) ** 2

