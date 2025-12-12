# -*- coding: utf-8 -*-
"""
CQI Table Management for NR (TS 38.214).

Provides:
- CQI table entries (SINR threshold, Qm, R_x1024)
- SINR to CQI mapping
- CQI to spectral efficiency conversion
"""

from typing import Dict
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


def _resolve_cqi_table(table: str) -> np.ndarray:
    """Resolve table name to CQI table array."""
    key = str(table or "nr_64qam").lower()

    # Map 3GPP table names to CQI table equivalents
    if key == "3gpp_table_1":
        key = "nr_64qam"
    elif key == "3gpp_table_2":
        key = "nr_256qam"
    elif key == "3gpp_table_3":
        key = "nr_64qam"  # Table 3 uses similar modulation as Table 1
    elif key == "legacy":
        key = "nr_64qam"

    if key not in _NR_CQI_TABLE:
        raise ValueError(f"Unknown CQI table: {table}")
    return _NR_CQI_TABLE[key]


def get_nr_cqi_table(table: str = "nr_64qam") -> np.ndarray:
    """Return a copy of the CQI table entries (threshold_dB, Qm, R_x1024).

    Args:
        table: Table identifier ('nr_64qam', 'nr_256qam', '3gpp_table_1', etc.)

    Returns:
        Array of shape [15, 3] with (SINR_threshold_dB, Qm, R_x1024)
    """
    return _resolve_cqi_table(table).copy()


def sinr_to_cqi(sinr_db: np.ndarray, table: str = "nr_64qam") -> np.ndarray:
    """Map SINR (dB) to CQI index (0..15) using 38.214 thresholds.

    Args:
        sinr_db: SINR values in dB
        table: CQI table identifier

    Returns:
        CQI indices (0 means out-of-range, 1-15 are valid CQI)
    """
    sinr_db = np.asarray(sinr_db, dtype=float)
    tbl = _resolve_cqi_table(table)
    thr = tbl[:, 0]
    idx = np.searchsorted(thr, sinr_db, side="right")
    return np.clip(idx, 0, tbl.shape[0])


def cqi_to_se(cqi: np.ndarray, table: str = "nr_64qam") -> np.ndarray:
    """Convert CQI (0..15) into spectral efficiency (bits/s/Hz).

    Args:
        cqi: CQI indices (0 means no transmission, 1-15 valid)
        table: CQI table identifier

    Returns:
        Spectral efficiency values (0 for CQI=0)
    """
    cqi = np.asarray(cqi, dtype=int)
    tbl = _resolve_cqi_table(table)
    se_vals = (tbl[:, 1] * tbl[:, 2]) / 1024.0
    out = np.zeros_like(cqi, dtype=float)
    mask = cqi > 0
    idx = np.clip(cqi[mask] - 1, 0, se_vals.size - 1)
    out[mask] = se_vals[idx]
    return out


def get_cqi_thresholds(table: str = "nr_64qam") -> np.ndarray:
    """Get SINR thresholds for 10% BLER from CQI table.

    Args:
        table: CQI table identifier

    Returns:
        Array of SINR thresholds in dB
    """
    tbl = _resolve_cqi_table(table)
    return tbl[:, 0].copy()


def get_cqi_se_values(table: str = "nr_64qam") -> np.ndarray:
    """Get spectral efficiency values from CQI table.

    Args:
        table: CQI table identifier

    Returns:
        Array of SE values (bits/s/Hz)
    """
    tbl = _resolve_cqi_table(table)
    return (tbl[:, 1] * tbl[:, 2]) / 1024.0
