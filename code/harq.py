"""
HARQ ACK deferral manager for NR-NTN.

DEPRECATED: This module has been refactored into code/link/.
This file provides backward compatibility by re-exporting from the new location.

New code should import from 'link' directly:
    from link import HarqManager, HarqManagerFull
"""

# Re-export all public APIs from link module for backward compatibility
from link.harq import HarqManager, HarqManagerFull

__all__ = [
    "HarqManager",
    "HarqManagerFull",
]
