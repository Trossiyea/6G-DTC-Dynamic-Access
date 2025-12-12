# -*- coding: utf-8 -*-
"""
Link Adaptation for NR.

Provides unified MCS selection and SINR-to-SE mapping for scheduling.
"""

from typing import Dict, Optional
import numpy as np

from .mcs import MCS, get_mcs_table, _canonical_table_kind
from .cqi import sinr_to_cqi, cqi_to_se
from .bler import bler_awgn_sigmoid, bler_from_registered_curves


def choose_mcs_from_sinr(
    sinr_eff_db: float,
    table_kind: str,
    target_bler: float = 0.1,
    slope_db: float = 1.0,
    margin_db: float = 1.5,
) -> MCS:
    """Pick the highest MCS (by SE) with predicted BLER <= target.

    If no MCS satisfies the target, picks the most robust (lowest SE) one.

    Args:
        sinr_eff_db: Effective SINR in dB
        table_kind: MCS table identifier
        target_bler: Target BLER threshold (default 0.1 = 10%)
        slope_db: BLER curve slope parameter
        margin_db: BLER curve margin parameter

    Returns:
        Selected MCS entry
    """
    canon, override = _canonical_table_kind(table_kind)
    cands = get_mcs_table(table_kind)
    best = cands[0]

    for m in cands:
        # Try registered curves first
        p_curve = None
        for key in filter(None, [override, canon]):
            p_curve = bler_from_registered_curves(sinr_eff_db, m, key)
            if p_curve is not None:
                break

        if p_curve is None:
            p = float(
                bler_awgn_sigmoid(
                    sinr_eff_db,
                    m,
                    slope_db=slope_db,
                    margin_db=margin_db,
                    table_kind=canon,
                )
            )
        else:
            p = float(np.squeeze(p_curve))

        if p <= target_bler and m.se >= best.se:
            best = m

    # If all exceed target, pick the lowest SE
    if best is None:
        best = min(cands, key=lambda m: m.se)

    return best


def sinr_to_se_mcs(sinr_db: np.ndarray, table: str = "legacy") -> np.ndarray:
    """Map SINR to spectral efficiency via CQI tables.

    Args:
        sinr_db: SINR values in dB
        table: Table identifier ('legacy', 'nr_64qam', '3gpp_table_1', etc.)

    Returns:
        Spectral efficiency values (bits/s/Hz)
    """
    sinr_db = np.asarray(sinr_db, dtype=float)
    tbl_key = str(table or "nr_64qam").lower()

    # Map 3GPP table names to CQI table equivalents
    if tbl_key == "3gpp_table_1":
        tbl_key = "nr_64qam"
    elif tbl_key == "3gpp_table_2":
        tbl_key = "nr_256qam"
    elif tbl_key == "3gpp_table_3":
        tbl_key = "nr_64qam"  # Table 3 uses similar modulation as Table 1
    elif tbl_key not in ("legacy", "nr_64qam", "nr_256qam"):
        raise ValueError(f"Unknown MCS table: {table}")

    if tbl_key == "legacy":
        tbl_key = "nr_64qam"

    cqi = sinr_to_cqi(sinr_db, table=tbl_key)
    return cqi_to_se(cqi, table=tbl_key)


def snr_to_se_sched(
    snr_lin: np.ndarray,
    use_mcs: bool,
    mcs_params: Optional[Dict],
    enable_cqi_quant: bool = False,
    cqi_table: str = "nr_64qam"
) -> np.ndarray:
    """Map instantaneous SNR (linear) to SE for scheduling metric.

    Args:
        snr_lin: SNR in linear scale
        use_mcs: Whether MCS-based mapping is enabled (caller handles if True)
        mcs_params: MCS parameters (unused here, for interface compat)
        enable_cqi_quant: Quantize SINR to CQI and map to SE
        cqi_table: CQI table identifier

    Returns:
        Spectral efficiency values

    Note:
        If enable_cqi_quant is False and use_mcs is True, caller should
        use main.se_from_snr; this function only handles the CQI path.
    """
    s = np.asarray(snr_lin, dtype=float)
    if enable_cqi_quant:
        sinr_db = 10.0 * np.log10(np.maximum(s, 1e-12))
        cqi = sinr_to_cqi(sinr_db, table=cqi_table)
        return cqi_to_se(cqi, table=cqi_table)
    # Fallback: Shannon as neutral default
    return np.log2(1.0 + np.maximum(s, 0.0))
