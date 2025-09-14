"""
Skyfield-based orbit model (Stage-3, practical mapping to simulation grid).

Uses TLE to compute satellite subpoint (lat, lon, alt) each TTI via Skyfield,
then maps the subpoint to the simulation grid using an equirectangular
approximation around a configured map center. The model returns per-UE
free-space loss, beam gain, propagation delay, and an approximate Doppler
by projecting the ground-track velocity onto the LoS ground unit vector.

Notes:
- Requires 'skyfield' to be installed. Falls back to OrbitModel if unavailable
  or TLE not provided.
- This is a transitional model: it preserves the grid-based geometry pipeline
  while using Skyfield for subpoint and altitude, keeping API compatibility.
"""

from typing import Dict, Tuple
import numpy as np
import math

try:
    from skyfield.api import EarthSatellite, wgs84, load
    HAS_SKYFIELD = True
except Exception:
    HAS_SKYFIELD = False

from orbit import OrbitModel, fspl_db, simple_beam_gain_db


class OrbitSkyfield(OrbitModel):
    def __init__(self, config: Dict, X: int, Y: int):
        super().__init__(config, X, Y)
        self.enabled = HAS_SKYFIELD and bool(config.get("tle_line1")) and bool(config.get("tle_line2"))
        if not self.enabled:
            return
        self.tle1 = str(config.get("tle_line1"))
        self.tle2 = str(config.get("tle_line2"))
        self.sat = EarthSatellite(self.tle1, self.tle2)
        # Time origin
        ts = load.timescale()
        self.ts = ts
        self.use_now = bool(config.get("tle_ref_use_now", False))
        self.ref_year = int(config.get("tle_ref_year", 2024))
        self.ref_month = int(config.get("tle_ref_month", 1))
        self.ref_day = int(config.get("tle_ref_day", 1))
        self.ref_hour = int(config.get("tle_ref_hour", 0))
        self.ref_min = int(config.get("tle_ref_min", 0))
        self.ref_sec = float(config.get("tle_ref_sec", 0.0))
        self._t0 = self.ts.now() if self.use_now else None
        # Map center for equirectangular projection
        self.lat0_deg = float(config.get("grid_center_lat_deg", 0.0))
        self.lon0_deg = float(config.get("grid_center_lon_deg", 0.0))
        # Precompute degrees per km at reference latitude
        self.R_earth_km = 6371.0
        self.deg_per_km_lat = 1.0 / (2.0 * math.pi * self.R_earth_km / 360.0)
        self.deg_per_km_lon = self.deg_per_km_lat / max(math.cos(math.radians(self.lat0_deg)), 1e-6)
        # Optionally auto-center grid to initial subpoint
        if bool(config.get("auto_grid_center_from_tle", False)):
            t0 = self._time_at(0)
            sp = wgs84.subpoint(self.sat.at(t0))
            self.lat0_deg = float(sp.latitude.degrees)
            self.lon0_deg = float(sp.longitude.degrees)
            self.deg_per_km_lon = self.deg_per_km_lat / max(math.cos(math.radians(self.lat0_deg)), 1e-6)

    def _time_at(self, t_idx: int):
        if self.use_now and self._t0 is not None:
            # Offset from now by t_idx * TTI
            return self.ts.tt_jd(self._t0.tt + (t_idx * self.tti_s) / 86400.0)
        return self.ts.utc(self.ref_year, self.ref_month, self.ref_day,
                           self.ref_hour, self.ref_min, self.ref_sec + t_idx * self.tti_s)

    def _subpoint_px(self, t_idx: int) -> Tuple[float, float, float]:
        """Return (cx, cy, alt_km) where (cx,cy) are beam-center pixel coords."""
        if not self.enabled:
            return super().beam_center_at(t_idx)[0], super().beam_center_at(t_idx)[1], self.alt_km
        t = self._time_at(t_idx)
        g = self.sat.at(t)
        sp = wgs84.subpoint(g)
        lat = float(sp.latitude.degrees)
        lon = float(sp.longitude.degrees)
        alt_km = float(sp.elevation.km)
        dlat = lat - self.lat0_deg
        dlon = lon - self.lon0_deg
        # Convert deg to km (local equirectangular), then to pixels
        dy_km = dlat / max(self.deg_per_km_lat, 1e-9)
        dx_km = dlon / max(self.deg_per_km_lon, 1e-9)
        cx = self.cx0 + dx_km / self.cell_km
        cy = self.cy0 + dy_km / self.cell_km
        # wrap to [0,X), [0,Y)
        cx = cx % self.X
        cy = cy % self.Y
        return float(cx), float(cy), float(alt_km)

    def beam_center_at(self, t: int) -> Tuple[float, float]:
        cx, cy, _ = self._subpoint_px(t)
        return cx, cy

    @staticmethod
    def _ecef_from_latlon(lat_deg: float, lon_deg: float, alt_km: float, R_earth_km: float = 6371.0) -> np.ndarray:
        lat = math.radians(float(lat_deg))
        lon = math.radians(float(lon_deg))
        r = float(R_earth_km) + float(alt_km)
        x = r * math.cos(lat) * math.cos(lon)
        y = r * math.cos(lat) * math.sin(lon)
        z = r * math.sin(lat)
        return np.array([x, y, z], dtype=float)

    def _latlon_from_px(self, px_x: float, px_y: float) -> Tuple[float, float]:
        dx_km = (float(px_x) - float(self.cx0)) * self.cell_km
        dy_km = (float(px_y) - float(self.cy0)) * self.cell_km
        lat = self.lat0_deg + dy_km * self.deg_per_km_lat
        lon = self.lon0_deg + dx_km * self.deg_per_km_lon
        return float(lat), float(lon)

    def get_slant_and_offaxis(self, ue_pos: np.ndarray, t: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return super().get_slant_and_offaxis(ue_pos, t)
        cx, cy, alt_km = self._subpoint_px(t)
        # Keep off-axis consistent with beam model (ground distance vs altitude)
        dxg = (ue_pos[:, 0].astype(float) - cx) * self.cell_km
        dyg = (ue_pos[:, 1].astype(float) - cy) * self.cell_km
        r_ground = np.sqrt(dxg * dxg + dyg * dyg)
        offaxis_deg = np.rad2deg(np.arctan2(r_ground, alt_km))
        # Slant using ECEF positions (spherical Earth approximation)
        lat_sub, lon_sub = self._latlon_from_px(cx, cy)
        sat_ecef = self._ecef_from_latlon(lat_sub, lon_sub, alt_km, self.R_earth_km)
        ue_ecef = []
        for i in range(ue_pos.shape[0]):
            lat_i, lon_i = self._latlon_from_px(ue_pos[i,0], ue_pos[i,1])
            ue_ecef.append(self._ecef_from_latlon(lat_i, lon_i, 0.0, self.R_earth_km))
        ue_ecef = np.vstack(ue_ecef)
        slant_km = np.linalg.norm(sat_ecef.reshape(1,3) - ue_ecef, axis=1)
        return slant_km, offaxis_deg

    def get_geometry(self, ue_pos: np.ndarray, t: int = 0):
        if not self.enabled:
            return super().get_geometry(ue_pos, t)
        # Satellite ECEF and velocity from Skyfield (GCRS), with UE positions mapped to the same frame
        cx, cy, alt_km = self._subpoint_px(t)
        tsky = self._time_at(t)
        # Satellite geocentric vectors (GCRS) from Skyfield
        geoc = self.sat.at(tsky)
        sat_pos = geoc.position.km  # (3,)
        sat_vel = geoc.velocity.km_per_s  # (3,)

        # UE geocentric positions using Skyfield wgs84 at the same time
        ue_pos_gcrs = []
        for i in range(ue_pos.shape[0]):
            plat, plon = self._latlon_from_px(ue_pos[i,0], ue_pos[i,1])
            gp = wgs84.latlon(plat, plon, elevation_m=0.0)
            ue_geo = gp.at(tsky)
            ue_pos_gcrs.append(ue_geo.position.km)
        ue_pos_gcrs = np.vstack(ue_pos_gcrs)  # [UE,3]

        r_vec = sat_pos.reshape(1,3) - ue_pos_gcrs
        slant_km = np.linalg.norm(r_vec, axis=1)
        r_hat = r_vec / np.maximum(slant_km.reshape(-1,1), 1e-12)
        # Off-axis: prefer ECEF/GCRS angle between boresight (to subpoint) and LoS if enabled
        if bool(self.config.get("offaxis_ecef", True)):
            lat0, lon0 = self._latlon_from_px(cx, cy)  # subpoint
            sub_ecef = self._ecef_from_latlon(lat0, lon0, 0.0, self.R_earth_km)
            b_hat = (sub_ecef - sat_pos) / np.maximum(np.linalg.norm(sub_ecef - sat_pos), 1e-12)
            cos_th = np.clip(np.dot(r_hat, b_hat), -1.0, 1.0)
            offaxis_deg = np.rad2deg(np.arccos(cos_th))
        else:
            dxg = (ue_pos[:, 0].astype(float) - cx) * self.cell_km
            dyg = (ue_pos[:, 1].astype(float) - cy) * self.cell_km
            r_ground = np.sqrt(dxg * dxg + dyg * dyg)
            offaxis_deg = np.rad2deg(np.arctan2(r_ground, alt_km))

        L_fs = fspl_db(slant_km, self.config.get("carrier_freq_GHz", 2.0))
        G_rx = simple_beam_gain_db(
            offaxis_deg,
            boresight_gain_db=self.config.get("G_rx_db", 32.0),
            half_bw_deg=self.config.get("beam_half_bw_deg", 4.0),
            edge_drop_db=self.config.get("beam_edge_drop_db", 3.0),
        )
        c_kmps = 299792.458
        tau_s = slant_km / c_kmps
        # Doppler using geocentric radial component of satellite velocity (UE static)
        v_r_kmps = np.dot(r_hat, sat_vel)
        f_c_hz = float(self.config.get("carrier_freq_GHz", 2.0)) * 1e9
        f_d_hz = (v_r_kmps / c_kmps) * f_c_hz
        return L_fs, G_rx, tau_s, f_d_hz
