# -*- coding: utf-8 -*-
"""
Unit conversion and basic physical computation utilities.

This module provides commonly used conversion functions for RF/wireless
simulation, including power unit conversions and thermal noise calculations.
"""

from typing import Union
import numpy as np

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, float, list]


def dbm_to_mw(dbm: ArrayLike) -> np.ndarray:
    """
    Convert power from dBm to milliwatts.

    Args:
        dbm: Power value(s) in dBm

    Returns:
        Power in milliwatts (same shape as input)
    """
    return 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


def mw_to_dbm(mw: ArrayLike) -> np.ndarray:
    """
    Convert power from milliwatts to dBm.

    Args:
        mw: Power value(s) in milliwatts

    Returns:
        Power in dBm (same shape as input)

    Note:
        Values <= 0 are clipped to 1e-30 mW to avoid log(0).
    """
    return 10.0 * np.log10(np.maximum(np.asarray(mw, dtype=float), 1e-30))


def db_to_linear(db: ArrayLike) -> np.ndarray:
    """
    Convert decibels to linear scale.

    Args:
        db: Value(s) in dB

    Returns:
        Linear scale values
    """
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


def linear_to_db(lin: ArrayLike) -> np.ndarray:
    """
    Convert linear scale to decibels.

    Args:
        lin: Linear scale value(s)

    Returns:
        Values in dB

    Note:
        Values <= 0 are clipped to 1e-30 to avoid log(0).
    """
    return 10.0 * np.log10(np.maximum(np.asarray(lin, dtype=float), 1e-30))


def thermal_noise_dbm(bw_hz: float, temp_K: float = 290.0) -> float:
    """
    Calculate thermal noise power in dBm for a given bandwidth.

    Uses the formula: N = kTB, where k is Boltzmann's constant.
    Reference: -174 dBm/Hz at 290 K.

    Args:
        bw_hz: Bandwidth in Hz
        temp_K: Temperature in Kelvin (default: 290 K)

    Returns:
        Thermal noise power in dBm
    """
    # If temperature deviates from 290 K, adjust: -174 dBm/Hz + 10*log10(T/290)
    per_hz_dbm = -174.0 + 10.0 * np.log10(max(temp_K, 1e-9) / 290.0)
    return per_hz_dbm + 10.0 * np.log10(max(bw_hz, 1.0))


def blur1d(a: np.ndarray, k: int, axis: int = 0) -> np.ndarray:
    """
    Apply 1D box blur (moving average) along a specified axis.

    Uses cumsum-based algorithm for efficient computation. Edge values
    are extended using 'edge' padding mode.

    Args:
        a: Input array
        k: Blur radius (window size = 2k)
        axis: Axis along which to blur (default: 0)

    Returns:
        Blurred array with same shape as input
    """
    if k <= 0:
        return a
    a = np.asarray(a)
    if axis < 0:
        axis = a.ndim + axis

    # Pad array along the specified axis
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (k, k)
    padded = np.pad(a, pad_width, mode='edge').cumsum(axis=axis)

    # Extract window sums using cumsum differences
    slicer_hi = [slice(None)] * a.ndim
    slicer_lo = [slice(None)] * a.ndim
    slicer_hi[axis] = slice(2 * k, None)
    slicer_lo[axis] = slice(None, -2 * k)

    window_sum = padded[tuple(slicer_hi)] - padded[tuple(slicer_lo)]
    return window_sum / float(2 * k)


def resolve_prb_bandwidth_hz(scs_khz: float) -> float:
    """
    Calculate PRB bandwidth in Hz from subcarrier spacing.

    Each PRB contains 12 subcarriers.

    Args:
        scs_khz: Subcarrier spacing in kHz (e.g., 15, 30, 60)

    Returns:
        PRB bandwidth in Hz
    """
    return float(scs_khz) * 1e3 * 12.0
