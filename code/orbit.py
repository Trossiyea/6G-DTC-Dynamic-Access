"""
Orbit and beam geometry utilities for NR-NTN simulation.

DEPRECATED: This module has been refactored into code/ntn/.
This file provides backward compatibility by re-exporting from the new location.

New code should import from 'ntn' directly:
    from ntn import OrbitModel, compute_geometry_and_beam
    from ntn import fspl_db, beam_gain_db
"""

# Re-export all public APIs from ntn module for backward compatibility
from ntn.orbit import (
    OrbitModel,
    compute_geometry_and_beam,
    _parse_orbit_start_utc,
)

from ntn.geometry import (
    fspl_db,
    beam_gain_db as simple_beam_gain_db,
)


__all__ = [
    "OrbitModel",
    "compute_geometry_and_beam",
    "_parse_orbit_start_utc",
    "fspl_db",
    "simple_beam_gain_db",
]
