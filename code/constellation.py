"""
Constellation orbit utilities: load multiple TLEs and compute per-satellite
geometry for all UEs at a given time index t.

Design goals
- Robust TLE parsing from a catalog text file (name + 2 lines or bare 2 lines).
- Efficient candidate filtering by sub-satellite proximity to the mapped area.
- Vectorized geometry based on Skyfield (if available).

This module is intentionally independent of code/main.py scheduling logic to
avoid circular imports. It only computes geometry and returns arrays ready for
compute_caps() in main.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

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
    name: str
    l1: str
    l2: str
    obj: Optional[object] = None  # EarthSatellite when available


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-15, 1.0 - a)))
    return R * c


def parse_tle_catalog(path: str) -> List[TLESatellite]:
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
    """
    Manage multiple TLE-driven satellites and provide geometry per TTI.
    """

    def __init__(self, config: Dict, X: int, Y: int):
        if not _SKYFIELD_OK:
            raise RuntimeError("Skyfield/sgp4 not available. Please `pip install skyfield sgp4` to use constellation mode.")
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
        t0_str = config.get("orbit_start_datetime", None)
        if t0_str:
            try:
                self.t0 = datetime.fromisoformat(str(t0_str).replace('Z', '+00:00')).astimezone(timezone.utc)
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
        dt = self.t0 + timedelta(seconds=self.tti_s * t_idx)
        return self.ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)

    def _map_xy_to_latlon(self, ue_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Map grid offsets (x,y) around center to ENU offsets and then to lat/lon
        lat0, lon0 = self.ref_lat_deg, self.ref_lon_deg
        phi = math.radians(self.map_rot_deg)
        cos_p, sin_p = math.cos(phi), math.sin(phi)
        x_off_km = (ue_pos[:, 0].astype(float) - (self.X / 2.0)) * self.cell_km
        y_off_km = (ue_pos[:, 1].astype(float) - (self.Y / 2.0)) * self.cell_km
        east_km = x_off_km * cos_p - y_off_km * sin_p
        north_km = x_off_km * sin_p + y_off_km * cos_p
        deg_per_km_lat = 1.0 / 111.0
        deg_per_km_lon = 1.0 / (111.0 * max(1e-6, math.cos(math.radians(lat0))))
        lat_deg = lat0 + north_km * deg_per_km_lat
        lon_deg = lon0 + east_km * deg_per_km_lon
        return lat_deg, lon_deg

    def candidate_indices_at(self, t_idx: int) -> List[int]:
        """Return indices of satellites whose subsatellite point is near the map ref within radius.
        Limited to at most `max_sats_tti` entries (closest first).
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
                lon = ((lon + 180.0) % 360.0) - 180.0
                d = _haversine_km(lat, lon, lat0, lon0)
                if d <= self.max_radius_km:
                    dists.append((d, i))
            except Exception:
                continue
        if not dists:
            return []
        dists.sort(key=lambda x: x[0])
        return [idx for (_, idx) in dists[: self.max_sats_tti]]

    def geometry_for_sat(self, ue_pos: np.ndarray, sat_idx: int, t_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute (L_fs, G_rx, tau_s, f_d_hz, elev_deg) for all UEs w.r.t a satellite.
        Matches the Skyfield branch semantics in code/orbit.py.
        """
        s = self.sats[sat_idx]
        ts_t = self._ts_of(t_idx)
        sat_itrs = s.obj.at(ts_t)
        # Sub-satellite ground point at t (for beam boresight)
        sub = wgs84.subpoint_of(sat_itrs)
        g_sub = wgs84.latlon(sub.latitude.degrees, sub.longitude.degrees, elevation_m=0.0)
        g_sub_itrs = g_sub.at(ts_t)
        sat_pos_km, sat_vel_kmps = sat_itrs.position.km, sat_itrs.velocity.km_per_s
        bore_vec = g_sub_itrs.position.km - sat_pos_km
        bore_dir = bore_vec / max(1e-9, np.linalg.norm(bore_vec))

        # Grid -> lat/lon
        lat_deg, lon_deg = self._map_xy_to_latlon(ue_pos)

        # Per-UE geometry
        c_kmps = 299792.458
        f_c_hz = float(self.config.get("carrier_freq_GHz", 2.0)) * 1e9
        L_fs = np.zeros(ue_pos.shape[0], dtype=float)
        G_rx = np.zeros_like(L_fs)
        tau_s = np.zeros_like(L_fs)
        f_d_hz = np.zeros_like(L_fs)
        elev_deg = np.zeros_like(L_fs)

        # Beam model params
        boresight_gain_db = float(self.config.get("G_rx_db", 32.0))
        half_bw_deg = float(self.config.get("beam_half_bw_deg", 4.0))
        edge_drop_db = float(self.config.get("beam_edge_drop_db", 3.0))

        def simple_beam_gain_db(offaxis_deg: np.ndarray,
                                boresight_gain_db: float,
                                half_bw_deg: float,
                                edge_drop_db: float = 3.0) -> np.ndarray:
            theta = np.asarray(offaxis_deg, dtype=float)
            theta = np.abs(theta)
            theta = np.minimum(theta, 89.9)
            hb = max(half_bw_deg, 1e-3) * math.pi / 180.0
            target = 10.0 ** (-edge_drop_db / 10.0)
            c = max(math.cos(hb), 1e-6)
            m = math.log(max(target, 1e-9)) / math.log(c)
            th_rad = theta * math.pi / 180.0
            patt = np.power(np.maximum(np.cos(th_rad), 1e-6), m)
            gain = boresight_gain_db + 10.0 * np.log10(np.maximum(patt, 1e-9))
            return gain

        for i in range(ue_pos.shape[0]):
            g = wgs84.latlon(float(lat_deg[i]), float(lon_deg[i]), elevation_m=0.0)
            g_itrs = g.at(ts_t)
            rel = sat_itrs - g_itrs  # vector from ground to sat
            pos_km = rel.position.km
            vel_kmps = rel.velocity.km_per_s
            rng_km = max(1e-9, np.linalg.norm(pos_km))
            # FSPL
            f_MHz = float(self.config.get("carrier_freq_GHz", 2.0)) * 1e3
            L_fs[i] = 32.45 + 20.0 * np.log10(rng_km) + 20.0 * np.log10(f_MHz)
            # Off-axis
            dir_ue = -pos_km / rng_km
            cosang = np.clip(np.dot(bore_dir, dir_ue), -1.0, 1.0)
            offaxis = math.degrees(math.acos(cosang))
            G_rx[i] = simple_beam_gain_db(np.array([offaxis]), boresight_gain_db, half_bw_deg, edge_drop_db)[0]
            # Elevation
            ground_pos = g_itrs.position.km
            up = ground_pos / max(1e-9, np.linalg.norm(ground_pos))
            horiz = pos_km - np.dot(pos_km, up) * up
            elev_rad = math.atan2(np.dot(pos_km, up), max(1e-9, np.linalg.norm(horiz)))
            elev_deg[i] = math.degrees(elev_rad)
            # Delay
            tau_s[i] = rng_km / c_kmps
            # Doppler
            v_rel_kmps = vel_kmps
            v_rad = -np.dot(v_rel_kmps, dir_ue)
            f_d_hz[i] = (v_rad / c_kmps) * f_c_hz

        return L_fs, G_rx, tau_s, f_d_hz, elev_deg

