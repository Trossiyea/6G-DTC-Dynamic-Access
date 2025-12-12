# -*- coding: utf-8 -*-
"""
BLER Curve Management for NR Link Adaptation.

Provides:
- AWGN-like sigmoid BLER curves with configurable slope/margin
- External BLER curve registration from JSON
- SINR threshold interpolation from CQI anchors
"""

from typing import Dict, Optional, Tuple
import json
import os
import numpy as np

from .cqi import get_nr_cqi_table
from .mcs import MCS, get_default_tables, _canonical_table_kind


# Registered external BLER curves: table_kind -> {mcs_idx: (sinr_db[], bler[])}
_BLER_CURVES: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

# Build AWGN 10% BLER thresholds by interpolating CQI anchors
_CQI_TABLE = get_nr_cqi_table("nr_64qam")
_CQI_SE = (_CQI_TABLE[:, 1] * _CQI_TABLE[:, 2]) / 1024.0
_CQI_THR_DB = _CQI_TABLE[:, 0]


def _sinr10_from_se(se: float) -> float:
    """Interpolate (or gently extrapolate) SINR for 10% BLER given spectral efficiency."""
    s = float(se)
    if s <= _CQI_SE[0]:
        slope = (_CQI_THR_DB[1] - _CQI_THR_DB[0]) / (_CQI_SE[1] - _CQI_SE[0])
        return float(_CQI_THR_DB[0] + slope * (s - _CQI_SE[0]))
    if s >= _CQI_SE[-1]:
        slope = (_CQI_THR_DB[-1] - _CQI_THR_DB[-2]) / (_CQI_SE[-1] - _CQI_SE[-2])
        return float(_CQI_THR_DB[-1] + slope * (s - _CQI_SE[-1]))
    return float(np.interp(s, _CQI_SE, _CQI_THR_DB))


def _shannon_required_sinr_db(se_bits_per_hz: float) -> float:
    """Invert Shannon to get the SNR needed for given SE."""
    se = max(1e-9, float(se_bits_per_hz))
    gamma = (2.0 ** se) - 1.0
    return 10.0 * np.log10(gamma)


# Pre-computed default SINR thresholds for 10% BLER per MCS table
_DEFAULT_SINR_THRESH_DB: Dict[str, Dict[int, float]] = {}
for name, tbl in get_default_tables().items():
    thr_map: Dict[int, float] = {}
    for m in tbl:
        thr_map[m.idx] = _sinr10_from_se(m.se)
    _DEFAULT_SINR_THRESH_DB[name] = thr_map


def register_bler_curves_from_file(path: str) -> None:
    """Register BLER vs SINR curves per MCS and table from external JSON.

    Expected schema:
    {
      "3gpp_table_2": [
        {"idx": 10, "sinr_db": [...], "bler": [...]},
        ...
      ],
      "table_1_64qam": [...]
    }

    BLER values clipped to [0, 1]. Malformed entries silently skipped.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"BLER curve file not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    for tbl, arr in data.items():
        try:
            cur: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for ent in arr:
                idx = int(ent.get("idx"))
                x = np.asarray(ent.get("sinr_db", []), dtype=float)
                y = np.asarray(ent.get("bler", []), dtype=float)
                if x.size < 2 or y.size != x.size:
                    continue
                y = np.clip(y, 0.0, 1.0)
                # Ensure increasing x for interpolation
                order = np.argsort(x)
                x = x[order]
                y = y[order]
                cur[idx] = (x, y)
            if cur:
                _BLER_CURVES[str(tbl).lower()] = cur
        except Exception:
            continue


def bler_awgn_sigmoid(
    sinr_eff_db: np.ndarray,
    mcs: MCS,
    slope_db: float = 1.0,
    margin_db: float = 1.5,
    table_kind: Optional[str] = None,
) -> np.ndarray:
    """Simple BLER curve using a logistic fit around the 10% BLER SINR point.

    When table_kind is provided, uses the interpolated 38.214 AWGN anchor
    for that MCS index. Otherwise falls back to Shannon+margin heuristic.

    Args:
        sinr_eff_db: Effective SINR in dB
        mcs: MCS entry
        slope_db: Steepness of the sigmoid (dB)
        margin_db: Margin above Shannon for fallback threshold
        table_kind: MCS table kind for threshold lookup

    Returns:
        BLER probability in [0, 1]
    """
    sinr_eff_db = np.asarray(sinr_eff_db, dtype=float)
    k = max(1e-6, float(slope_db))

    th_override = None
    if table_kind is not None:
        canon, _ = _canonical_table_kind(table_kind)
        thr_map = _DEFAULT_SINR_THRESH_DB.get(canon)
        if thr_map is not None:
            th_override = thr_map.get(int(mcs.idx))

    if th_override is None:
        se = mcs.se
        th_db = _shannon_required_sinr_db(se) + float(margin_db)
    else:
        th_db = float(th_override)

    mu = th_db - k * np.log(9.0)  # ensures BLER ~ 0.1 at SINR = th_db
    x = (sinr_eff_db - mu) / k
    p = 1.0 / (1.0 + np.exp(x))
    return np.clip(p, 0.0, 1.0)


def bler_from_registered_curves(
    sinr_eff_db: np.ndarray,
    mcs: MCS,
    table_kind: Optional[str],
) -> Optional[np.ndarray]:
    """Interpolate BLER from registered curves if available.

    Args:
        sinr_eff_db: Effective SINR in dB
        mcs: MCS entry
        table_kind: Table kind to look up curves for

    Returns:
        BLER probability array, or None if curves not available
    """
    if table_kind is None:
        return None
    tbl = _BLER_CURVES.get(str(table_kind).lower())
    if not tbl:
        return None
    xy = tbl.get(int(mcs.idx))
    if xy is None:
        return None
    x, y = xy
    xx = np.asarray(sinr_eff_db, dtype=float)
    # Extrapolate with edge values
    y_interp = np.interp(xx, x, y, left=y[0], right=y[-1])
    return np.clip(y_interp, 0.0, 1.0)


def get_sinr_threshold(mcs: MCS, table_kind: str = "table_1_64qam") -> float:
    """Get SINR threshold for 10% BLER for given MCS.

    Args:
        mcs: MCS entry
        table_kind: MCS table kind

    Returns:
        SINR threshold in dB
    """
    canon, _ = _canonical_table_kind(table_kind)
    thr_map = _DEFAULT_SINR_THRESH_DB.get(canon)
    if thr_map is not None and mcs.idx in thr_map:
        return thr_map[mcs.idx]
    # Fallback to Shannon + margin
    return _shannon_required_sinr_db(mcs.se) + 1.5
