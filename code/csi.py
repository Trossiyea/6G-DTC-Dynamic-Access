"""
CSI utilities for NR-NTN simulation.

DEPRECATED: This module has been refactored into code/link/.
This file provides backward compatibility by re-exporting from the new location.

New code should import from 'link' directly:
    from link import get_nr_cqi_table, sinr_to_cqi, cqi_to_se, ...
"""

# Re-export all public APIs from link module for backward compatibility
from link.cqi import (
    get_nr_cqi_table,
    sinr_to_cqi,
    cqi_to_se,
)
from link.adaptation import sinr_to_se_mcs
from link.eesm import effective_sinr_eesm
from link.olla import OLLA

__all__ = [
    "get_nr_cqi_table",
    "sinr_to_cqi",
    "cqi_to_se",
    "sinr_to_se_mcs",
    "OLLA",
    "effective_sinr_eesm",
]
