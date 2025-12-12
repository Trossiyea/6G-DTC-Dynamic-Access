"""
Link adaptation utilities for NR (TS 38.214 inspired).

DEPRECATED: This module has been refactored into code/link/.
This file provides backward compatibility by re-exporting from the new location.

New code should import from 'link' directly:
    from link import MCS, choose_mcs_from_sinr, calc_tbs_bits, ...
"""

# Re-export all public APIs from link module for backward compatibility
from link.mcs import (
    MCS,
    get_mcs_table,
    register_mcs_tables_from_file,
)
from link.bler import (
    register_bler_curves_from_file,
    bler_awgn_sigmoid,
    bler_from_registered_curves,
)
from link.eesm import (
    eff_sinr_eesm_db,
    combine_eff_sinr_db,
)
from link.tbs import (
    n_sym_per_slot,
    n_re_per_prb,
    calc_tbs_bits,
    re_per_prb_from_config,
)
from link.olla import OLLA
from link.adaptation import choose_mcs_from_sinr

__all__ = [
    "MCS",
    "get_mcs_table",
    "register_mcs_tables_from_file",
    "register_bler_curves_from_file",
    "bler_awgn_sigmoid",
    "bler_from_registered_curves",
    "eff_sinr_eesm_db",
    "combine_eff_sinr_db",
    "n_sym_per_slot",
    "n_re_per_prb",
    "calc_tbs_bits",
    "re_per_prb_from_config",
    "OLLA",
    "choose_mcs_from_sinr",
]
