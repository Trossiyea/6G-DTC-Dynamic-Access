# -*- coding: utf-8 -*-
"""
Link Layer Module for NR-NTN Simulation.

Unified link adaptation subsystem containing:
- MCS tables (TS 38.214)
- CQI tables and mapping
- BLER curves and models
- EESM/TBS calculation
- OLLA (Outer Loop Link Adaptation)
- HARQ management

Public API exports maintain backward compatibility with previous code structure.
"""

# MCS management
from .mcs import (
    MCS,
    get_mcs_table,
    register_mcs_tables_from_file,
)

# CQI management
from .cqi import (
    get_nr_cqi_table,
    sinr_to_cqi,
    cqi_to_se,
    get_cqi_thresholds,
    get_cqi_se_values,
)

# BLER curves
from .bler import (
    register_bler_curves_from_file,
    bler_awgn_sigmoid,
    bler_from_registered_curves,
    get_sinr_threshold,
)

# EESM
from .eesm import (
    effective_sinr_eesm,
    eff_sinr_eesm_db,
    combine_eff_sinr_db,
)

# TBS calculation
from .tbs import (
    n_sym_per_slot,
    n_re_per_prb,
    calc_tbs_bits,
    re_per_prb_from_config,
)

# OLLA
from .olla import (
    OLLA,
    create_olla_from_config,
)

# Link adaptation
from .adaptation import (
    choose_mcs_from_sinr,
    sinr_to_se_mcs,
    snr_to_se_sched,
)

# HARQ management
from .harq import (
    HarqManager,
    HarqManagerFull,
)

__all__ = [
    # MCS
    "MCS",
    "get_mcs_table",
    "register_mcs_tables_from_file",
    # CQI
    "get_nr_cqi_table",
    "sinr_to_cqi",
    "cqi_to_se",
    "get_cqi_thresholds",
    "get_cqi_se_values",
    # BLER
    "register_bler_curves_from_file",
    "bler_awgn_sigmoid",
    "bler_from_registered_curves",
    "get_sinr_threshold",
    # EESM
    "effective_sinr_eesm",
    "eff_sinr_eesm_db",
    "combine_eff_sinr_db",
    # TBS
    "n_sym_per_slot",
    "n_re_per_prb",
    "calc_tbs_bits",
    "re_per_prb_from_config",
    # OLLA
    "OLLA",
    "create_olla_from_config",
    # Adaptation
    "choose_mcs_from_sinr",
    "sinr_to_se_mcs",
    "snr_to_se_sched",
    # HARQ
    "HarqManager",
    "HarqManagerFull",
]
