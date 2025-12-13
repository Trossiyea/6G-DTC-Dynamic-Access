"""
NTN (Non-Terrestrial Network) module for NR-NTN simulation.

This package provides unified orbit, constellation, channel, and beam
utilities for NR-NTN downlink simulation.

Phase 6 Modularization:
    - geometry.py: Unified geometry utilities (FSPL, beam gain, coordinate conversion)
    - orbit.py: Single-satellite TLE-driven orbit model
    - constellation.py: Multi-satellite constellation management
    - channel.py: 3GPP TR 38.811/38.821 channel models
    - beams.py: Beam pattern and management

Usage:
    from ntn import OrbitModel, ConstellationOrbit, sample_3gpp_ntn_fading
    from ntn import beam_gain_db, fspl_db, map_xy_to_latlon

Public API (25 exports):
    Geometry:
        - fspl_db
        - beam_gain_db
        - map_xy_to_latlon
        - latlon_to_map_xy
        - haversine_km
        - compute_slant_range_km
        - compute_offaxis_deg
        - compute_elevation_deg
        - EARTH_RADIUS_KM
        - KM_PER_DEG_LAT

    Orbit:
        - OrbitModel
        - compute_geometry_and_beam

    Constellation:
        - TLESatellite
        - ConstellationOrbit
        - parse_tle_catalog

    Channel:
        - NTNChannelProfile
        - NTN_CHANNEL_PROFILES
        - sample_3gpp_ntn_fading
        - describe_profile

    Beams:
        - BeamManager
        - simple_beam_pattern_db
"""

# Geometry utilities (unified, eliminates duplication)
from .geometry import (
    EARTH_RADIUS_KM,
    KM_PER_DEG_LAT,
    fspl_db,
    beam_gain_db,
    map_xy_to_latlon,
    latlon_to_map_xy,
    haversine_km,
    compute_slant_range_km,
    compute_offaxis_deg,
    compute_elevation_deg,
)

# Orbit model (single satellite)
from .orbit import (
    OrbitModel,
    compute_geometry_and_beam,
    _parse_orbit_start_utc,
)

# Constellation management (multi-satellite)
from .constellation import (
    TLESatellite,
    ConstellationOrbit,
    parse_tle_catalog,
)

# Channel models (3GPP TR 38.811/38.821)
from .channel import (
    NTNChannelProfile,
    NTN_CHANNEL_PROFILES,
    sample_3gpp_ntn_fading,
    describe_profile,
)

# Beam management
from .beams import (
    BeamManager,
    simple_beam_pattern_db,
)


__all__ = [
    # Geometry
    "EARTH_RADIUS_KM",
    "KM_PER_DEG_LAT",
    "fspl_db",
    "beam_gain_db",
    "map_xy_to_latlon",
    "latlon_to_map_xy",
    "haversine_km",
    "compute_slant_range_km",
    "compute_offaxis_deg",
    "compute_elevation_deg",
    # Orbit
    "OrbitModel",
    "compute_geometry_and_beam",
    "_parse_orbit_start_utc",
    # Constellation
    "TLESatellite",
    "ConstellationOrbit",
    "parse_tle_catalog",
    # Channel
    "NTNChannelProfile",
    "NTN_CHANNEL_PROFILES",
    "sample_3gpp_ntn_fading",
    "describe_profile",
    # Beams
    "BeamManager",
    "simple_beam_pattern_db",
]


__version__ = "1.0.0"
__phase__ = 6
