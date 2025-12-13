"""
Beam and coverage abstractions for NR-NTN simulation.

DEPRECATED: This module has been refactored into code/ntn/.
This file provides backward compatibility by re-exporting from the new location.

New code should import from 'ntn' directly:
    from ntn import BeamManager, simple_beam_pattern_db
"""

# Re-export all public APIs from ntn module for backward compatibility
from ntn.beams import (
    BeamManager,
    simple_beam_pattern_db,
)


__all__ = [
    "BeamManager",
    "simple_beam_pattern_db",
]
