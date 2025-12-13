"""
Orbit and beam geometry utilities for NR-NTN simulation.

This module provides orbit modeling using TLE data via Skyfield.
Refactored from the original code/orbit.py to use the unified
geometry utilities in ntn/geometry.py.

Classes:
    OrbitModel: TLE-driven orbit/beam model for single-satellite scenarios.

Functions:
    compute_geometry_and_beam: Quick geometry computation without TLE.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union, Optional
import numpy as np
import math
from datetime import datetime, timedelta, timezone
import re

from .geometry import (
    fspl_db,
    beam_gain_db,
    map_xy_to_latlon,
    compute_slant_range_km,
    compute_offaxis_deg,
    compute_elevation_deg,
)


try:
    # Optional Skyfield import; used only if enabled via config
    from skyfield.api import EarthSatellite, load, wgs84  # type: ignore
    _SKYFIELD_OK = True
except Exception:  # pragma: no cover
    EarthSatellite = None  # type: ignore
    load = None  # type: ignore
    wgs84 = None  # type: ignore
    _SKYFIELD_OK = False


def _parse_orbit_start_utc(t0_val) -> datetime:
    """Parse various ISO8601-like inputs to a timezone-aware UTC datetime.

    Accepts strings like:
      - '2025-10-07T21:38:31.574982Z'
      - '2025-10-07T21:38:31.574982+00:00'
      - '2025-10-07 21:38:31'
      - accidental '...+00:00Z' (will be normalized)

    Falls back to current UTC time if parsing fails or t0_val is None.
    """
    # If already a datetime, normalize to UTC
    if isinstance(t0_val, datetime):
        try:
            if t0_val.tzinfo is None:
                return t0_val.replace(tzinfo=timezone.utc)
            return t0_val.astimezone(timezone.utc)
        except Exception:
            return datetime.now(tz=timezone.utc)

    if t0_val is None:
        return datetime.now(tz=timezone.utc)

    s = str(t0_val).strip()
    if not s:
        return datetime.now(tz=timezone.utc)

    # Remove any trailing 'Z' (UTC marker) — we'll normalize the tz offset below.
    s_noz = re.sub(r"[Zz]+$", "", s)

    # Ensure there is exactly one timezone offset at the end.
    # Accept forms like +HH:MM or -HH:MM (fromisoformat requirement).
    # If there is no offset, append '+00:00'. If there is an offset without colon, insert it.
    m_with_colon = re.search(r"([+-]\d{2}:\d{2})$", s_noz)
    m_without = re.search(r"([+-])(\d{2})(\d{2})$", s_noz)
    if m_with_colon:
        s_norm = s_noz  # already standard
    elif m_without:
        sign, hh, mm = m_without.groups()
        s_norm = re.sub(r"([+-]\d{2}\d{2})$", f"{sign}{hh}:{mm}", s_noz)
    else:
        # No explicit offset present; append UTC offset
        s_norm = s_noz + "+00:00"

    # Replace whitespace separator with 'T' to be safe (fromisoformat accepts both)
    s_norm = s_norm.replace(" ", "T")

    try:
        dt = datetime.fromisoformat(s_norm)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        # Final fallback: now in UTC
        return datetime.now(tz=timezone.utc)


def compute_geometry_and_beam(
    config: Dict,
    X: int,
    Y: int,
    ue_pos: np.ndarray,
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float], np.ndarray]:
    """Compute per-UE FSPL, beam gain and elevation angle without TLE dynamics.

    Uses simple geometric calculations based on satellite altitude and
    beam center position.

    Args:
        config: Configuration dictionary with sat_altitude_km, beam_center_xy, etc.
        X, Y: Grid dimensions.
        ue_pos: UE positions as (N, 2) array of (x, y) grid coordinates.

    Returns:
        Tuple of (L_fs_per_ue, G_rx_per_ue, elev_deg) arrays.
    """
    bc = config.get("beam_center_xy", None)
    if bc is None:
        cx, cy = (X // 2, Y // 2)
    else:
        cx, cy = bc

    cell_km = config.get("cell_size_km", 1.0)
    dx = (ue_pos[:, 0] - cx) * cell_km
    dy = (ue_pos[:, 1] - cy) * cell_km
    r_ground = np.sqrt(dx * dx + dy * dy)  # km

    alt_km = config.get("sat_altitude_km", 600.0)
    slant_km = compute_slant_range_km(r_ground, alt_km)

    L_fs_per_ue = fspl_db(slant_km, config.get("carrier_freq_GHz", 2.0))
    offaxis_deg = compute_offaxis_deg(r_ground, alt_km)
    elev_deg = compute_elevation_deg(r_ground, alt_km)

    G_rx_per_ue = beam_gain_db(
        offaxis_deg,
        boresight_gain_db=config.get("G_rx_db", 32.0),
        half_bw_deg=config.get("beam_half_bw_deg", 4.0),
        edge_drop_db=config.get("beam_edge_drop_db", 3.0)
    )

    return L_fs_per_ue, G_rx_per_ue, np.asarray(elev_deg, dtype=float)


class OrbitModel:
    """TLE-driven orbit/beam model using Skyfield.

    Provides per-UE geometry computations including FSPL, beam gain,
    propagation delay, Doppler shift, and elevation angle.

    Requires Skyfield and sgp4 packages to be installed.
    """

    def __init__(self, config: Dict, X: int, Y: int):
        """Initialize the orbit model.

        Args:
            config: Configuration dictionary with TLE and reference point info.
            X, Y: Grid dimensions.

        Raises:
            RuntimeError: If Skyfield is not available or TLE parsing fails.
        """
        if not _SKYFIELD_OK:
            raise RuntimeError(
                "Skyfield/sgp4 not available. Please install them to run with TLE dynamics."
            )

        self.config = dict(config)
        self.X = X
        self.Y = Y
        self.cell_km = float(config.get("cell_size_km", 1.0))
        self.alt_km = float(config.get("sat_altitude_km", 600.0))
        self.tti_s = float(config.get("tti_ms", 1.0)) * 1e-3

        # Skyfield state
        self.sf_sat = None
        self.sf_ts = None
        self.sf_t0 = None
        self.ref_lat_deg = None
        self.ref_lon_deg = None
        self.auto_ref_from_tle = bool(config.get("auto_ref_from_tle", False))
        self.map_rot_deg = float(config.get("map_rotation_deg", 0.0))

        try:
            tle_lines = config.get("tle_lines")
            tle_path = config.get("tle_path")
            if tle_lines is None and tle_path:
                with open(tle_path, 'r') as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                # Accept 2-line or name+2-line
                if len(lines) >= 2:
                    tle_lines = lines[-2:]

            if (isinstance(tle_lines, (list, tuple)) and len(tle_lines) >= 2
                    and isinstance(tle_lines[0], str) and isinstance(tle_lines[1], str)):
                name = config.get("tle_name", "SAT")
                self.sf_sat = EarthSatellite(tle_lines[0], tle_lines[1], name)
                self.sf_ts = load.timescale()

                # Start time (robust ISO8601 parsing to UTC)
                self.sf_t0 = _parse_orbit_start_utc(config.get("orbit_start_datetime", None))

                # Local map reference (lat/lon)
                self.ref_lat_deg = config.get("ref_lat_deg", None)
                self.ref_lon_deg = config.get("ref_lon_deg", None)

                # If requested, set reference to sub-satellite point at t0
                if self.auto_ref_from_tle:
                    ts0 = self.sf_ts.utc(
                        self.sf_t0.year, self.sf_t0.month, self.sf_t0.day,
                        self.sf_t0.hour, self.sf_t0.minute,
                        self.sf_t0.second + self.sf_t0.microsecond * 1e-6
                    )
                    itrs0 = self.sf_sat.at(ts0)
                    sub0 = wgs84.subpoint_of(itrs0)
                    self.ref_lat_deg = float(sub0.latitude.degrees)
                    self.ref_lon_deg = float(sub0.longitude.degrees)

                if self.ref_lat_deg is None or self.ref_lon_deg is None:
                    raise RuntimeError(
                        "ref_lat_deg/ref_lon_deg must be set or auto_ref_from_tle=True"
                    )
            else:
                raise ValueError("TLE lines or tle_path must be provided for OrbitModel")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize TLE orbit: {e}")

    def get_geometry(
        self,
        ue_pos: np.ndarray,
        t: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute geometry for all UEs at a given TTI.

        Args:
            ue_pos: UE positions as (N, 2) array of (x, y) grid coordinates.
            t: TTI index (time step).

        Returns:
            Tuple of (L_fs, G_rx, tau_s, f_d_hz, elev_deg) arrays.
        """
        # Build Skyfield time object for this TTI
        dt = self.sf_t0 + timedelta(seconds=self.tti_s * t)
        ts_t = self.sf_ts.utc(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6
        )
        sat_itrs = self.sf_sat.at(ts_t)

        # Sub-satellite ground point at t (for beam boresight)
        sub = wgs84.subpoint_of(sat_itrs)
        g_sub = wgs84.latlon(sub.latitude.degrees, sub.longitude.degrees, elevation_m=0.0)
        g_sub_itrs = g_sub.at(ts_t)
        sat_pos_km = sat_itrs.position.km
        bore_vec = g_sub_itrs.position.km - sat_pos_km
        bore_dir = bore_vec / max(1e-9, np.linalg.norm(bore_vec))

        # Map pixel -> lat/lon using unified geometry function
        lat0 = float(self.ref_lat_deg) if self.ref_lat_deg is not None else 0.0
        lon0 = float(self.ref_lon_deg) if self.ref_lon_deg is not None else 0.0

        lat_deg, lon_deg = map_xy_to_latlon(
            ue_pos[:, 0], ue_pos[:, 1],
            lat0, lon0,
            self.cell_km, self.X, self.Y,
            self.map_rot_deg
        )

        # For each UE, compute topocentric geometry
        c_kmps = 299792.458
        f_c_hz = float(self.config.get("carrier_freq_GHz", 2.0)) * 1e9

        num_ue = ue_pos.shape[0]
        L_fs = np.zeros(num_ue, dtype=float)
        G_rx = np.zeros_like(L_fs)
        tau_s = np.zeros_like(L_fs)
        f_d_hz = np.zeros_like(L_fs)
        elev_deg_arr = np.zeros_like(L_fs)

        for i in range(num_ue):
            g = wgs84.latlon(float(lat_deg[i]), float(lon_deg[i]), elevation_m=0.0)
            g_itrs = g.at(ts_t)
            rel = sat_itrs - g_itrs  # vector from ground to sat
            pos_km = rel.position.km
            vel_kmps = rel.velocity.km_per_s
            rng_km = max(1e-9, np.linalg.norm(pos_km))

            # FSPL
            L_fs[i] = fspl_db(rng_km, self.config.get("carrier_freq_GHz", 2.0))

            # Off-axis angle at satellite: between boresight and sat->UE vector
            dir_ue = -pos_km / rng_km  # sat -> UE is negative of (UE->sat)
            cosang = np.clip(np.dot(bore_dir, dir_ue), -1.0, 1.0)
            offaxis = math.degrees(math.acos(cosang))

            G_rx[i] = beam_gain_db(
                np.array([offaxis]),
                boresight_gain_db=self.config.get("G_rx_db", 32.0),
                half_bw_deg=self.config.get("beam_half_bw_deg", 4.0),
                edge_drop_db=self.config.get("beam_edge_drop_db", 3.0)
            )[0]

            # Elevation angle
            ground_pos = g_itrs.position.km
            up = ground_pos / max(1e-9, np.linalg.norm(ground_pos))
            horiz = pos_km - np.dot(pos_km, up) * up
            elev_rad = math.atan2(np.dot(pos_km, up), max(1e-9, np.linalg.norm(horiz)))
            elev_deg_arr[i] = math.degrees(elev_rad)

            # Delay
            tau_s[i] = rng_km / c_kmps

            # Doppler: radial rate along LoS (sat->UE)
            v_rel_kmps = vel_kmps  # sat relative to ground
            v_rad = -np.dot(v_rel_kmps, dir_ue)  # positive if moving towards UE
            f_d_hz[i] = (v_rad / c_kmps) * f_c_hz

        return L_fs, G_rx, tau_s, f_d_hz, elev_deg_arr


__all__ = [
    "OrbitModel",
    "compute_geometry_and_beam",
    "_parse_orbit_start_utc",
]
