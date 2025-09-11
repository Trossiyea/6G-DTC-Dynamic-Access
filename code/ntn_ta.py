"""
NTN Timing Advance (TA) utilities aligned with NR numerology.

Provides:
- NR base time unit Ts and TA step ΔTA = 16 * Ts * 2^μ
- Effective CP duration (approx.) for FR1 normal CP across SCS
- Helpers to quantize TA and compute misalignment penalty

Notes:
- We focus on FR1 normal CP. Extended CP and exact per-symbol CP tables are
  simplified via a scale with SCS to keep the system model light-weight.
"""

from typing import Tuple
import numpy as np
import math


def nr_Ts_s() -> float:
    """NR base time unit Ts = 1 / (15 kHz * 2048)."""
    return 1.0 / (15000.0 * 2048.0)


def scs_to_mu(scs_khz: float) -> int:
    """Numerology μ from SCS in kHz (μ = log2(scs/15))."""
    scs = max(float(scs_khz), 1e-9)
    mu = int(round(math.log2(scs / 15.0)))
    return max(0, mu)


def ta_step_us(scs_khz: float) -> float:
    """
    TA granularity ΔTA = 16 * Ts * 2^μ, returned in microseconds.
    """
    Ts = nr_Ts_s()
    mu = scs_to_mu(scs_khz)
    step_s = 16.0 * Ts * (2 ** mu)
    return step_s * 1e6


def effective_cp_us(scs_khz: float, cp_type: str = "normal", symbol_index: int = 1) -> float:
    """
    Approximate effective CP duration for a typical PUSCH demod symbol, in microseconds.
    - cp_type: 'normal' (FR1); extended not explicitly modeled (falls back to scaled normal CP).
    - symbol_index: 0 for first symbol in slot (longer CP); >0 for others (shorter CP).
    Baseline values for 30 kHz: first ≈ 2.86 μs; others ≈ 2.34 μs.
    For other SCS, scale approximately inversely with SCS.
    """
    scs = max(float(scs_khz), 1e-9)
    if cp_type.lower() != "normal":
        # Simple fallback: scale normal CP numbers
        base_first, base_other = 2.86, 2.34
    else:
        base_first, base_other = 2.86, 2.34
    scale = 30.0 / scs
    if symbol_index <= 0:
        return base_first * scale
    return base_other * scale


def quantize_ta_s(tau_s: np.ndarray, step_us: float) -> np.ndarray:
    """Quantize propagation delay to nearest TA step (seconds)."""
    s = np.asarray(tau_s, dtype=float)
    q = max(float(step_us), 1e-9) * 1e-6
    return np.round(s / q) * q


def misalignment_penalty(e_us: np.ndarray, cp_us: float, margin_us: float,
                         drop_if_exceed: bool = True, exponent: float = 2.0) -> np.ndarray:
    """
    Convert timing error (μs) to an SNR scaling factor in [0,1].
    - If drop_if_exceed: factor=0 when e > (cp - margin)
    - Else: smooth decay with exponent over the excess beyond threshold.
    """
    e = np.asarray(e_us, dtype=float)
    thr = max(0.0, float(cp_us) - float(margin_us))
    if drop_if_exceed:
        return np.where(e > thr, 0.0, 1.0)
    # Smooth penalty
    excess = np.maximum(0.0, e - thr)
    denom = max(float(cp_us), 1e-6)
    expn = max(float(exponent), 1.0)
    fac = np.clip(1.0 - (excess / denom), 0.0, 1.0) ** expn
    return fac

