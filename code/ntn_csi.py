"""
CSI/CQI handling for scheduler metrics.

DEPRECATED: This module has been refactored into code/link/.
This file provides backward compatibility by re-exporting from the new location.

New code should import from 'link' directly:
    from link import snr_to_se_sched
"""

# Re-export from link module for backward compatibility
from link.adaptation import snr_to_se_sched

__all__ = [
    "snr_to_se_sched",
]
