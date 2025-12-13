"""
Constellation orbit utilities for multi-satellite NR-NTN simulation.

DEPRECATED: This module has been refactored into code/ntn/.
This file provides backward compatibility by re-exporting from the new location.

New code should import from 'ntn' directly:
    from ntn import ConstellationOrbit, TLESatellite, parse_tle_catalog
"""

# Re-export all public APIs from ntn module for backward compatibility
from ntn.constellation import (
    TLESatellite,
    ConstellationOrbit,
    parse_tle_catalog,
)

# Also re-export haversine for any code that might use it
from ntn.geometry import haversine_km as _haversine_km


__all__ = [
    "TLESatellite",
    "ConstellationOrbit",
    "parse_tle_catalog",
    "_haversine_km",
]
