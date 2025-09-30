"""
Orbit and beam geometry utilities for NR-NTN simulation.

Initial step: extract the static geometry + beam gain from main.py
to keep behavior unchanged, while preparing for time-varying orbits.
"""

from typing import Dict, Tuple, Union, Optional
import numpy as np
import math
from datetime import datetime, timedelta

try:
    # Optional Skyfield import; used only if enabled via config
    from skyfield.api import EarthSatellite, load, wgs84  # type: ignore
    _SKYFIELD_OK = True
except Exception:  # pragma: no cover
    EarthSatellite = None  # type: ignore
    load = None  # type: ignore
    wgs84 = None  # type: ignore
    _SKYFIELD_OK = False


def fspl_db(distance_km: np.ndarray, freq_GHz: float) -> np.ndarray:
    d = np.maximum(np.asarray(distance_km, dtype=float), 1e-6)
    f_MHz = max(freq_GHz, 1e-9) * 1e3
    return 32.45 + 20.0 * np.log10(d) + 20.0 * np.log10(f_MHz)


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
    m = np.log(max(target, 1e-9)) / np.log(c)
    th_rad = theta * math.pi / 180.0
    patt = np.power(np.maximum(np.cos(th_rad), 1e-6), m)
    gain = boresight_gain_db + 10.0 * np.log10(np.maximum(patt, 1e-9))
    return gain


def compute_geometry_and_beam(
    config: Dict,
    X: int,
    Y: int,
    ue_pos: np.ndarray,
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float], np.ndarray]:
    """Resolve per-UE FSPL, beam gain and elevation angle (always enabled)."""
    bc = config.get("beam_center_xy", None)
    if bc is None:
        cx, cy = (X // 2, Y // 2)
    else:
        cx, cy = bc
    dx = (ue_pos[:, 0] - cx) * config.get("cell_size_km", 1.0)
    dy = (ue_pos[:, 1] - cy) * config.get("cell_size_km", 1.0)
    r_ground = np.sqrt(dx * dx + dy * dy)  # km
    alt_km = config.get("sat_altitude_km", 600.0)
    slant_km = np.sqrt(r_ground * r_ground + alt_km * alt_km)
    L_fs_per_ue = fspl_db(slant_km, config.get("carrier_freq_GHz", 2.0))
    offaxis_deg = np.rad2deg(np.arctan2(r_ground, alt_km))
    elev_deg = np.rad2deg(np.arctan2(alt_km, np.maximum(r_ground, 1e-6)))
    G_rx_per_ue = simple_beam_gain_db(
        offaxis_deg,
        boresight_gain_db=config.get("G_rx_db", 32.0),
        half_bw_deg=config.get("beam_half_bw_deg", 4.0),
        edge_drop_db=config.get("beam_edge_drop_db", 3.0)
    )
    return L_fs_per_ue, G_rx_per_ue, np.asarray(elev_deg, dtype=float)


class OrbitModel:
    """
    TLE-driven orbit/beam model (Skyfield required). No simple fallback.
    """

    def __init__(self, config: Dict, X: int, Y: int):
        if not _SKYFIELD_OK:
            raise RuntimeError("Skyfield/sgp4 not available. Please install them to run with TLE dynamics.")
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
                # Start time
                t0_str = config.get("orbit_start_datetime", None)
                if t0_str:
                    try:
                        self.sf_t0 = datetime.fromisoformat(str(t0_str).replace('Z', '+00:00'))
                    except Exception:
                        from datetime import timezone
                        self.sf_t0 = datetime.now(tz=timezone.utc)
                else:
                    from datetime import timezone
                    self.sf_t0 = datetime.now(tz=timezone.utc)
                # Local map reference (lat/lon)
                self.ref_lat_deg = config.get("ref_lat_deg", None)
                self.ref_lon_deg = config.get("ref_lon_deg", None)
                # If requested, set reference to sub-satellite point at t0
                if self.auto_ref_from_tle:
                    ts0 = self.sf_ts.utc(self.sf_t0.year, self.sf_t0.month, self.sf_t0.day,
                                         self.sf_t0.hour, self.sf_t0.minute, self.sf_t0.second + self.sf_t0.microsecond * 1e-6)
                    itrs0 = self.sf_sat.at(ts0)
                    sub0 = wgs84.subpoint_of(itrs0)
                    self.ref_lat_deg = float(sub0.latitude.degrees)
                    self.ref_lon_deg = float(sub0.longitude.degrees)
                if self.ref_lat_deg is None or self.ref_lon_deg is None:
                    raise RuntimeError("ref_lat_deg/ref_lon_deg must be set or auto_ref_from_tle=True")
            else:
                raise ValueError("TLE lines or tle_path must be provided for OrbitModel")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TLE orbit: {e}")

    def get_geometry(
        self,
        ue_pos: np.ndarray,
        t: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return FSPL, beam gain, delay, Doppler and elevation per UE."""
        # Skyfield path
        # Build Skyfield time object for this TTI
        dt = self.sf_t0 + timedelta(seconds=self.tti_s * t)
        ts_t = self.sf_ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
        sat_itrs = self.sf_sat.at(ts_t)

        # Sub-satellite ground point at t (for beam boresight)
        sub = wgs84.subpoint_of(sat_itrs)
        g_sub = wgs84.latlon(sub.latitude.degrees, sub.longitude.degrees, elevation_m=0.0)
        g_sub_itrs = g_sub.at(ts_t)
        sat_pos_km, sat_vel_kmps = sat_itrs.position.km, sat_itrs.velocity.km_per_s
        bore_vec = g_sub_itrs.position.km - sat_pos_km
        bore_dir = bore_vec / max(1e-9, np.linalg.norm(bore_vec))

        # Map pixel -> local ENU offsets (east,north) at reference lat/lon
        # Resolve reference lat/lon (must exist in skyfield mode)
        if self.ref_lat_deg is None or self.ref_lon_deg is None:
            # Fallback: use sub-satellite point at t=0 if available
            try:
                ts0 = self.sf_ts.utc(self.sf_t0.year, self.sf_t0.month, self.sf_t0.day,
                                     self.sf_t0.hour, self.sf_t0.minute, self.sf_t0.second + self.sf_t0.microsecond * 1e-6)
                itrs0 = self.sf_sat.at(ts0)
                sub0 = wgs84.subpoint_of(itrs0)
                self.ref_lat_deg = float(sub0.latitude.degrees)
                self.ref_lon_deg = float(sub0.longitude.degrees)
            except Exception:
                self.ref_lat_deg = 0.0
                self.ref_lon_deg = 0.0
        lat0 = float(self.ref_lat_deg)
        lon0 = float(self.ref_lon_deg)
        # Optional map rotation (deg): 0 means +x=east, +y=north; positive rotates x towards north.
        phi = math.radians(self.map_rot_deg)
        cos_p, sin_p = math.cos(phi), math.sin(phi)
        x_off_km = (ue_pos[:, 0].astype(float) - (self.X / 2.0)) * self.cell_km
        y_off_km = (ue_pos[:, 1].astype(float) - (self.Y / 2.0)) * self.cell_km
        # Rotate to ENU
        east_km = x_off_km * cos_p - y_off_km * sin_p
        north_km = x_off_km * sin_p + y_off_km * cos_p

        # Convert ENU offsets to lat/lon via small-angle approximation
        # 1 deg lat ~ 111 km; 1 deg lon ~ 111 km * cos(lat)
        deg_per_km_lat = 1.0 / 111.0
        deg_per_km_lon = 1.0 / (111.0 * max(1e-6, math.cos(math.radians(lat0))))
        lat_deg = lat0 + north_km * deg_per_km_lat
        lon_deg = lon0 + east_km * deg_per_km_lon

        # For each UE, compute topocentric geometry
        c_kmps = 299792.458
        f_c_hz = float(self.config.get("carrier_freq_GHz", 2.0)) * 1e9
        L_fs = np.zeros(ue_pos.shape[0], dtype=float)
        G_rx = np.zeros_like(L_fs)
        tau_s = np.zeros_like(L_fs)
        f_d_hz = np.zeros_like(L_fs)
        elev_deg = np.zeros_like(L_fs)

        for i in range(ue_pos.shape[0]):
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
            G_rx[i] = simple_beam_gain_db(np.array([offaxis]),
                                          boresight_gain_db=self.config.get("G_rx_db", 32.0),
                                          half_bw_deg=self.config.get("beam_half_bw_deg", 4.0),
                                          edge_drop_db=self.config.get("beam_edge_drop_db", 3.0))[0]
            ground_pos = g_itrs.position.km
            up = ground_pos / max(1e-9, np.linalg.norm(ground_pos))
            horiz = pos_km - np.dot(pos_km, up) * up
            elev_rad = math.atan2(np.dot(pos_km, up), max(1e-9, np.linalg.norm(horiz)))
            elev_deg[i] = math.degrees(elev_rad)
            # Delay
            tau_s[i] = rng_km / c_kmps
            # Doppler: radial rate along LoS (sat->UE)
            v_rel_kmps = vel_kmps  # sat relative to ground
            v_rad = -np.dot(v_rel_kmps, dir_ue)  # positive if moving towards UE
            f_d_hz[i] = (v_rad / c_kmps) * f_c_hz

        return L_fs, G_rx, tau_s, f_d_hz, elev_deg


## Minimal DL-only: Multi-beam wrapper removed to reduce complexity
