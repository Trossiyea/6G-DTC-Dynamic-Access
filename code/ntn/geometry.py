"""
Unified geometry utilities for NTN/satellite simulations.

This module consolidates common geometry functions used across orbit,
constellation, and beam calculations, eliminating code duplication.

Functions:
    - fspl_db: Free space path loss calculation
    - beam_gain_db: Cos^m beam pattern gain
    - map_xy_to_latlon: Convert grid coordinates to lat/lon
    - latlon_to_map_xy: Convert lat/lon to grid coordinates
    - haversine_km: Great circle distance calculation
"""

from __future__ import annotations

from typing import Tuple
import math
import numpy as np


# Earth's mean radius in km
EARTH_RADIUS_KM = 6371.0

# Approximate km per degree latitude
KM_PER_DEG_LAT = 111.0


def fspl_db(distance_km: np.ndarray, freq_GHz: float) -> np.ndarray:
    """Compute Free Space Path Loss (FSPL) in dB.

    Uses the standard formula: FSPL = 32.45 + 20*log10(d_km) + 20*log10(f_MHz)

    Args:
        distance_km: Slant range distance(s) in kilometers.
        freq_GHz: Carrier frequency in GHz.

    Returns:
        FSPL value(s) in dB.
    """
    d = np.maximum(np.asarray(distance_km, dtype=float), 1e-6)
    f_MHz = max(freq_GHz, 1e-9) * 1e3
    return 32.45 + 20.0 * np.log10(d) + 20.0 * np.log10(f_MHz)


def beam_gain_db(
    offaxis_deg: np.ndarray,
    boresight_gain_db: float,
    half_bw_deg: float,
    edge_drop_db: float = 3.0,
) -> np.ndarray:
    """Compute antenna beam gain using a Cos^m pattern model.

    The exponent m is computed so that the gain drops by edge_drop_db
    at half_bw_deg from boresight.

    Args:
        offaxis_deg: Off-axis angle(s) in degrees.
        boresight_gain_db: Peak gain at boresight in dBi.
        half_bw_deg: Half-power (3dB) beamwidth in degrees.
        edge_drop_db: Gain reduction at half_bw_deg (default 3.0 dB).

    Returns:
        Beam gain value(s) in dBi.
    """
    theta = np.asarray(offaxis_deg, dtype=float)
    theta = np.abs(theta)
    theta = np.minimum(theta, 89.9)  # Avoid numerical issues near 90 degrees

    # Calculate exponent m from edge drop requirement
    hb = max(half_bw_deg, 1e-3) * math.pi / 180.0
    target = 10.0 ** (-edge_drop_db / 10.0)
    c = max(math.cos(hb), 1e-6)
    m = math.log(max(target, 1e-9)) / math.log(c)

    # Apply pattern
    th_rad = theta * math.pi / 180.0
    patt = np.power(np.maximum(np.cos(th_rad), 1e-6), m)
    gain = boresight_gain_db + 10.0 * np.log10(np.maximum(patt, 1e-9))

    return gain


def map_xy_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    ref_lat_deg: float,
    ref_lon_deg: float,
    cell_km: float,
    grid_X: int,
    grid_Y: int,
    rotation_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert grid (x, y) coordinates to geographic (lat, lon).

    Grid coordinates are relative to the center of the map. The conversion
    uses a small-angle approximation suitable for local areas.

    Args:
        x: Grid x coordinate(s) (column index).
        y: Grid y coordinate(s) (row index).
        ref_lat_deg: Reference latitude at grid center (degrees).
        ref_lon_deg: Reference longitude at grid center (degrees).
        cell_km: Grid cell size in kilometers.
        grid_X: Total grid width (number of columns).
        grid_Y: Total grid height (number of rows).
        rotation_deg: Map rotation angle (degrees). 0 = +x is east, +y is north.
            Positive rotation turns +x towards north.

    Returns:
        Tuple of (lat_deg, lon_deg) arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Convert grid indices to km offsets from center
    x_off_km = (x - grid_X / 2.0) * cell_km
    y_off_km = (y - grid_Y / 2.0) * cell_km

    # Apply rotation to get ENU (East-North-Up) offsets
    phi = math.radians(rotation_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    east_km = x_off_km * cos_p - y_off_km * sin_p
    north_km = x_off_km * sin_p + y_off_km * cos_p

    # Convert ENU to lat/lon using small-angle approximation
    # 1 deg latitude ~ 111 km
    # 1 deg longitude ~ 111 km * cos(lat)
    deg_per_km_lat = 1.0 / KM_PER_DEG_LAT
    deg_per_km_lon = 1.0 / (KM_PER_DEG_LAT * max(1e-6, math.cos(math.radians(ref_lat_deg))))

    lat_deg = ref_lat_deg + north_km * deg_per_km_lat
    lon_deg = ref_lon_deg + east_km * deg_per_km_lon

    return lat_deg, lon_deg


def latlon_to_map_xy(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    ref_lat_deg: float,
    ref_lon_deg: float,
    cell_km: float,
    grid_X: int,
    grid_Y: int,
    rotation_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert geographic (lat, lon) to grid (x, y) coordinates.

    Inverse of map_xy_to_latlon.

    Args:
        lat_deg: Latitude(s) in degrees.
        lon_deg: Longitude(s) in degrees.
        ref_lat_deg: Reference latitude at grid center (degrees).
        ref_lon_deg: Reference longitude at grid center (degrees).
        cell_km: Grid cell size in kilometers.
        grid_X: Total grid width (number of columns).
        grid_Y: Total grid height (number of rows).
        rotation_deg: Map rotation angle (degrees).

    Returns:
        Tuple of (x, y) grid coordinate arrays.
    """
    lat_deg = np.asarray(lat_deg, dtype=float)
    lon_deg = np.asarray(lon_deg, dtype=float)

    # Convert lat/lon offsets to ENU km
    km_per_deg_lat = KM_PER_DEG_LAT
    km_per_deg_lon = KM_PER_DEG_LAT * max(1e-6, math.cos(math.radians(ref_lat_deg)))

    north_km = (lat_deg - ref_lat_deg) * km_per_deg_lat
    east_km = (lon_deg - ref_lon_deg) * km_per_deg_lon

    # Apply inverse rotation
    phi = math.radians(rotation_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    # Inverse rotation: (cos, -sin; sin, cos)^T = (cos, sin; -sin, cos)
    x_off_km = east_km * cos_p + north_km * sin_p
    y_off_km = -east_km * sin_p + north_km * cos_p

    # Convert km offsets to grid indices
    x = x_off_km / cell_km + grid_X / 2.0
    y = y_off_km / cell_km + grid_Y / 2.0

    return x, y


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great circle distance between two points on Earth.

    Uses the Haversine formula for numerical stability.

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        Distance in kilometers.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-15, 1.0 - a)))

    return EARTH_RADIUS_KM * c


def compute_slant_range_km(ground_distance_km: np.ndarray, altitude_km: float) -> np.ndarray:
    """Compute slant range from ground distance and satellite altitude.

    Uses simple Pythagorean approximation (valid for small coverage areas).

    Args:
        ground_distance_km: Horizontal distance(s) from subsatellite point.
        altitude_km: Satellite altitude above ground.

    Returns:
        Slant range(s) in kilometers.
    """
    r_ground = np.asarray(ground_distance_km, dtype=float)
    return np.sqrt(r_ground * r_ground + altitude_km * altitude_km)


def compute_offaxis_deg(ground_distance_km: np.ndarray, altitude_km: float) -> np.ndarray:
    """Compute off-axis angle at satellite from ground distance.

    Args:
        ground_distance_km: Horizontal distance(s) from subsatellite point.
        altitude_km: Satellite altitude above ground.

    Returns:
        Off-axis angle(s) in degrees.
    """
    r_ground = np.asarray(ground_distance_km, dtype=float)
    return np.rad2deg(np.arctan2(r_ground, altitude_km))


def compute_elevation_deg(ground_distance_km: np.ndarray, altitude_km: float) -> np.ndarray:
    """Compute elevation angle at UE from ground distance.

    Args:
        ground_distance_km: Horizontal distance(s) from subsatellite point.
        altitude_km: Satellite altitude above ground.

    Returns:
        Elevation angle(s) in degrees (from horizon).
    """
    r_ground = np.maximum(np.asarray(ground_distance_km, dtype=float), 1e-6)
    return np.rad2deg(np.arctan2(altitude_km, r_ground))


__all__ = [
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
]
