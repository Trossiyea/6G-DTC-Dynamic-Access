# -*- coding: utf-8 -*-
"""
Core simulation utilities and capacity computation.
"""

from .units import (
    dbm_to_mw,
    mw_to_dbm,
    db_to_linear,
    linear_to_db,
    thermal_noise_dbm,
    blur1d,
)

from .capacity import (
    SEMapper,
    MCSParams,
    se_from_snr,
    se_from_snr_with_split,
    se_from_cap_shannon_with_split,
    block_se_from_snr_vec,
    se_metric_strategy,
    apply_ici_penalty,
    _apply_ici_penalty_lin,
    _block_se_from_snr_vec,
)

__all__ = [
    # units
    "dbm_to_mw",
    "mw_to_dbm",
    "db_to_linear",
    "linear_to_db",
    "thermal_noise_dbm",
    "blur1d",
    # capacity
    "SEMapper",
    "MCSParams",
    "se_from_snr",
    "se_from_snr_with_split",
    "se_from_cap_shannon_with_split",
    "block_se_from_snr_vec",
    "se_metric_strategy",
    "apply_ici_penalty",
    "_apply_ici_penalty_lin",
    "_block_se_from_snr_vec",
]
