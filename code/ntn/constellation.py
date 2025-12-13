"""
Constellation orbit utilities for multi-satellite NR-NTN simulation.

This module provides TLE catalog parsing and multi-satellite geometry
computation using Skyfield.

Refactored from the original code/constellation.py to use unified
geometry utilities from ntn/geometry.py.

Classes:
    TLESatellite: Container for TLE satellite data.
    ConstellationOrbit: Multi-satellite constellation manager.

Functions:
    parse_tle_catalog: Parse TLE catalog file into satellite list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from .geometry import (
    fspl_db,
    beam_gain_db,
    map_xy_to_latlon,
    haversine_km,
)


try:
    from skyfield.api import EarthSatellite, load, wgs84  # type: ignore
    _SKYFIELD_OK = True
except Exception:  # pragma: no cover
    EarthSatellite = None  # type: ignore
    load = None  # type: ignore
    wgs84 = None  # type: ignore
    _SKYFIELD_OK = False


@dataclass
class TLESatellite:
    """Container for TLE satellite data.

    Attributes:
        name: Satellite name (from TLE or auto-generated).
        l1: TLE line 1.
        l2: TLE line 2.
        obj: Skyfield EarthSatellite object (when available).
    """
    name: str
    l1: str
    l2: str
    obj: Optional[object] = None


def parse_tle_catalog(path: str) -> List[TLESatellite]:
    """Parse a TLE catalog file into a list of TLESatellite objects.

    Supports both 2-line (unnamed) and 3-line (name + TLE) formats.

    Args:
        path: Path to the TLE catalog file.

    Returns:
        List of TLESatellite objects.

    Raises:
        FileNotFoundError: If the catalog file doesn't exist.
        ValueError: If no valid TLE entries are found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"TLE catalog not found: {path}")

    sats: List[TLESatellite] = []
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    i = 0
    unnamed_idx = 0

    while i < len(lines):
        line = lines[i]

        # Three-line group: name + L1 + L2
        if (i + 2 < len(lines)) and lines[i + 1].startswith('1 ') and lines[i + 2].startswith('2 '):
            name = line
            l1 = lines[i + 1]
            l2 = lines[i + 2]
            sats.append(TLESatellite(name=name, l1=l1, l2=l2))
            i += 3
            continue

        # Two-line group: L1 + L2
        if line.startswith('1 ') and (i + 1 < len(lines)) and lines[i + 1].startswith('2 '):
            name = f"SAT-{unnamed_idx:05d}"
            unnamed_idx += 1
            l1 = line
            l2 = lines[i + 1]
            sats.append(TLESatellite(name=name, l1=l1, l2=l2))
            i += 2
            continue

        # Otherwise skip stray line
        i += 1

    if not sats:
        raise ValueError(f"No valid TLE entries parsed from: {path}")

    return sats


class ConstellationOrbit:
    """Multi-satellite constellation manager using TLE catalog.

    Provides per-TTI satellite visibility filtering and geometry computation
    for all UEs relative to any visible satellite.
    """

    def __init__(self, config: Dict, X: int, Y: int):
        """Initialize constellation from TLE catalog.

        Args:
            config: Configuration dictionary with tle_catalog_path and reference point.
            X, Y: Grid dimensions.

        Raises:
            RuntimeError: If Skyfield is not available or no valid satellites found.
        """
        if not _SKYFIELD_OK:
            raise RuntimeError(
                "Skyfield/sgp4 not available. Please `pip install skyfield sgp4` to use constellation mode."
            )

        self.config = dict(config)
        self.X = int(X)
        self.Y = int(Y)
        self.cell_km = float(config.get("cell_size_km", 1.0))
        self.map_rot_deg = float(config.get("map_rotation_deg", 0.0))
        self.min_elev_deg = float(config.get("min_elev_deg", 5.0))
        self.max_radius_km = float(config.get("constellation_max_ground_radius_km", 2500.0))
        self.max_sats_tti = int(config.get("constellation_max_sats_per_tti", 24))

        self.ref_lat_deg = float(config.get("ref_lat_deg", 0.0))
        self.ref_lon_deg = float(config.get("ref_lon_deg", 0.0))

        # Parse start time
        t0_str = config.get("orbit_start_datetime", None)
        if t0_str:
            try:
                self.t0 = datetime.fromisoformat(
                    str(t0_str).replace('Z', '+00:00')
                ).astimezone(timezone.utc)
            except Exception:
                self.t0 = datetime.now(tz=timezone.utc)
        else:
            self.t0 = datetime.now(tz=timezone.utc)

        self.tti_s = float(config.get("tti_ms", 1.0)) * 1e-3
        self.ts = load.timescale()

        # Load TLE catalog
        catalog = config.get("tle_catalog_path")
        sats = parse_tle_catalog(catalog)

        # Build Skyfield objects
        self.sats: List[TLESatellite] = []
        for s in sats:
            try:
                s.obj = EarthSatellite(s.l1, s.l2, s.name)
                self.sats.append(s)
            except Exception:
                # Skip malformed entries
                pass

        if not self.sats:
            raise RuntimeError("No valid satellites constructed from TLE catalog.")

    def _ts_of(self, t_idx: int):
        """Get Skyfield time object for a given TTI index."""
        dt = self.t0 + timedelta(seconds=self.tti_s * t_idx)
        return self.ts.utc(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6
        )

    def _map_xy_to_latlon(self, ue_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert grid positions to lat/lon using unified geometry function."""
        return map_xy_to_latlon(
            ue_pos[:, 0], ue_pos[:, 1],
            self.ref_lat_deg, self.ref_lon_deg,
            self.cell_km, self.X, self.Y,
            self.map_rot_deg
        )

    def candidate_indices_at(self, t_idx: int) -> List[int]:
        """Get indices of visible satellites at a given TTI.

        Returns satellites whose subsatellite point is within max_radius_km
        of the reference point, sorted by distance (closest first).

        Args:
            t_idx: TTI index.

        Returns:
            List of satellite indices, limited to max_sats_tti entries.
        """
        ts_t = self._ts_of(t_idx)
        lat0, lon0 = self.ref_lat_deg, self.ref_lon_deg
        dists: List[Tuple[float, int]] = []

        for i, s in enumerate(self.sats):
            try:
                sp = s.obj.at(ts_t)
                sub = wgs84.subpoint_of(sp)
                lat = float(sub.latitude.degrees)
                lon = float(sub.longitude.degrees)
                # Normalize longitude to [-180, 180]
                lon = ((lon + 180.0) % 360.0) - 180.0
                d = haversine_km(lat, lon, lat0, lon0)
                if d <= self.max_radius_km:
                    dists.append((d, i))
            except Exception:
                continue

        if not dists:
            return []

        dists.sort(key=lambda x: x[0])
        return [idx for (_, idx) in dists[:self.max_sats_tti]]

    def geometry_for_sat(
        self,
        ue_pos: np.ndarray,
        sat_idx: int,
        t_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute geometry for all UEs relative to a specific satellite.

        Args:
            ue_pos: UE positions as (N, 2) array of (x, y) grid coordinates.
            sat_idx: Index of satellite in self.sats.
            t_idx: TTI index.

        Returns:
            Tuple of (L_fs, G_rx, tau_s, f_d_hz, elev_deg) arrays.
        """
        s = self.sats[sat_idx]
        ts_t = self._ts_of(t_idx)
        sat_itrs = s.obj.at(ts_t)

        # Sub-satellite ground point at t (for beam boresight)
        sub = wgs84.subpoint_of(sat_itrs)
        g_sub = wgs84.latlon(sub.latitude.degrees, sub.longitude.degrees, elevation_m=0.0)
        g_sub_itrs = g_sub.at(ts_t)
        sat_pos_km = sat_itrs.position.km
        bore_vec = g_sub_itrs.position.km - sat_pos_km
        bore_dir = bore_vec / max(1e-9, np.linalg.norm(bore_vec))

        # Grid -> lat/lon using unified function
        lat_deg, lon_deg = self._map_xy_to_latlon(ue_pos)

        # Per-UE geometry
        c_kmps = 299792.458
        f_c_hz = float(self.config.get("carrier_freq_GHz", 2.0)) * 1e9

        num_ue = ue_pos.shape[0]
        L_fs = np.zeros(num_ue, dtype=float)
        G_rx = np.zeros_like(L_fs)
        tau_s = np.zeros_like(L_fs)
        f_d_hz = np.zeros_like(L_fs)
        elev_deg_arr = np.zeros_like(L_fs)

        # Beam model params
        boresight_gain_db = float(self.config.get("G_rx_db", 32.0))
        half_bw_deg = float(self.config.get("beam_half_bw_deg", 4.0))
        edge_drop_db = float(self.config.get("beam_edge_drop_db", 3.0))

        for i in range(num_ue):
            g = wgs84.latlon(float(lat_deg[i]), float(lon_deg[i]), elevation_m=0.0)
            g_itrs = g.at(ts_t)
            rel = sat_itrs - g_itrs  # vector from ground to sat
            pos_km = rel.position.km
            vel_kmps = rel.velocity.km_per_s
            rng_km = max(1e-9, np.linalg.norm(pos_km))

            # FSPL using unified function
            L_fs[i] = fspl_db(rng_km, self.config.get("carrier_freq_GHz", 2.0))

            # Off-axis angle
            dir_ue = -pos_km / rng_km
            cosang = np.clip(np.dot(bore_dir, dir_ue), -1.0, 1.0)
            offaxis = math.degrees(math.acos(cosang))

            # Beam gain using unified function
            G_rx[i] = beam_gain_db(
                np.array([offaxis]),
                boresight_gain_db, half_bw_deg, edge_drop_db
            )[0]

            # Elevation
            ground_pos = g_itrs.position.km
            up = ground_pos / max(1e-9, np.linalg.norm(ground_pos))
            horiz = pos_km - np.dot(pos_km, up) * up
            elev_rad = math.atan2(np.dot(pos_km, up), max(1e-9, np.linalg.norm(horiz)))
            elev_deg_arr[i] = math.degrees(elev_rad)

            # Delay
            tau_s[i] = rng_km / c_kmps

            # Doppler
            v_rel_kmps = vel_kmps
            v_rad = -np.dot(v_rel_kmps, dir_ue)
            f_d_hz[i] = (v_rad / c_kmps) * f_c_hz

        return L_fs, G_rx, tau_s, f_d_hz, elev_deg_arr


__all__ = [
    "TLESatellite",
    "ConstellationOrbit",
    "parse_tle_catalog",
]
