"""
Orbit and beam geometry utilities for NR-NTN simulation.

Initial step: extract the static geometry + beam gain from main.py
to keep behavior unchanged, while preparing for time-varying orbits.
"""

from typing import Dict, Tuple, Union, Optional
import numpy as np
import math


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


def compute_geometry_and_beam(config: Dict,
                              X: int,
                              Y: int,
                              ue_pos: np.ndarray) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Drop-in replacement for main.compute_geometry_and_beam with identical behavior.
    """
    if config.get("enable_geometry", False):
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
        G_rx_per_ue = simple_beam_gain_db(
            offaxis_deg,
            boresight_gain_db=config.get("G_rx_db", 32.0),
            half_bw_deg=config.get("beam_half_bw_deg", 4.0),
            edge_drop_db=config.get("beam_edge_drop_db", 3.0)
        )
    else:
        L_fs_val = config.get("L_fs_db")
        G_rx_val = config.get("G_rx_db")
        L_fs_per_ue = float(L_fs_val) if isinstance(L_fs_val, (int, float)) else L_fs_val
        G_rx_per_ue = float(G_rx_val) if isinstance(G_rx_val, (int, float)) else G_rx_val
    return L_fs_per_ue, G_rx_per_ue


class OrbitModel:
    """
    Minimal orbit/beam placeholder for future time-varying geometry.
    For now, returns fixed altitude and boresight centered at a point.
    """

    def __init__(self, config: Dict, X: int, Y: int):
        self.config = dict(config)
        self.X = X
        self.Y = Y
        self.cell_km = float(config.get("cell_size_km", 1.0))
        bc = config.get("beam_center_xy")
        self.cx0 = float(X // 2) if bc is None else float(bc[0])
        self.cy0 = float(Y // 2) if bc is None else float(bc[1])
        self.alt_km = float(config.get("sat_altitude_km", 600.0))
        # Ground-track dynamics
        self.tti_s = float(config.get("tti_ms", 1.0)) * 1e-3
        v_kmps = float(config.get("sat_ground_speed_kms", 7.5))
        head_deg = float(config.get("sat_heading_deg", 0.0))
        head_rad = math.radians(head_deg)
        self.vx_kmps = v_kmps * math.cos(head_rad)
        self.vy_kmps = v_kmps * math.sin(head_rad)

    def beam_center_at(self, t: int) -> Tuple[float, float]:
        # Advance beam center with wrap-around on the tile
        step_km = self.tti_s
        cx = self.cx0 + (self.vx_kmps * step_km / self.cell_km) * t
        cy = self.cy0 + (self.vy_kmps * step_km / self.cell_km) * t
        # wrap around to keep within [0, X), [0, Y)
        cx = cx % self.X
        cy = cy % self.Y
        return cx, cy

    def get_slant_and_offaxis(self, ue_pos: np.ndarray, t: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        cx, cy = self.beam_center_at(t)
        dx = (ue_pos[:, 0].astype(float) - cx) * self.cell_km
        dy = (ue_pos[:, 1].astype(float) - cy) * self.cell_km
        r_ground = np.sqrt(dx * dx + dy * dy)
        slant_km = np.sqrt(r_ground * r_ground + self.alt_km * self.alt_km)
        offaxis_deg = np.rad2deg(np.arctan2(r_ground, self.alt_km))
        return slant_km, offaxis_deg

    def get_geometry(self, ue_pos: np.ndarray, t: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns per-UE at time t:
        - L_fs_db [UE]
        - G_rx_db [UE]
        - tau_s [UE]
        - f_d_hz [UE]
        """
        cx, cy = self.beam_center_at(t)
        x_ue_km = ue_pos[:, 0].astype(float) * self.cell_km
        y_ue_km = ue_pos[:, 1].astype(float) * self.cell_km
        x_sat_km = cx * self.cell_km
        y_sat_km = cy * self.cell_km
        dx = x_sat_km - x_ue_km
        dy = y_sat_km - y_ue_km
        r_ground = np.sqrt(dx * dx + dy * dy)
        slant_km = np.sqrt(r_ground * r_ground + self.alt_km * self.alt_km)
        offaxis_deg = np.rad2deg(np.arctan2(r_ground, self.alt_km))
        L_fs = fspl_db(slant_km, self.config.get("carrier_freq_GHz", 2.0))
        G_rx = simple_beam_gain_db(
            offaxis_deg,
            boresight_gain_db=self.config.get("G_rx_db", 32.0),
            half_bw_deg=self.config.get("beam_half_bw_deg", 4.0),
            edge_drop_db=self.config.get("beam_edge_drop_db", 3.0),
        )
        # Propagation delay (s)
        c_kmps = 299792.458
        tau_s = slant_km / c_kmps
        # Doppler: project sat ground velocity onto LoS unit vector
        # LoS unit vector from UE to Sat: u = [dx, dy, alt] / slant
        u_x = dx / slant_km
        u_y = dy / slant_km
        v_r_kmps = self.vx_kmps * u_x + self.vy_kmps * u_y
        f_c_hz = float(self.config.get("carrier_freq_GHz", 2.0)) * 1e9
        f_d_hz = (v_r_kmps / c_kmps) * f_c_hz
        return L_fs, G_rx, tau_s, f_d_hz


class OrbitModelMultiBeam(OrbitModel):
    """
    Multi-beam wrapper around OrbitModel that quantizes the instantaneous
    boresight center to a discrete beam grid or hops beams with a fixed period.

    Config keys (optional):
    - n_beams_x, n_beams_y: beam grid dimensions (defaults derived from map size)
    - beam_grid_spacing_px: spacing in pixels between adjacent beams (defaults ~ 2*half_bw footprint)
    - beam_hop_period_ttis: if >0, hop through beams in a raster order every P TTIs
    """

    def __init__(self, config: Dict, X: int, Y: int):
        super().__init__(config, X, Y)
        self.nx = int(config.get("n_beams_x", 0) or 0)
        self.ny = int(config.get("n_beams_y", 0) or 0)
        # Build beam centers
        spacing_px = config.get("beam_grid_spacing_px", None)
        if spacing_px is None:
            # Default spacing: approximate 2*half_bw footprint in pixels
            # half_bw_deg -> ground km at altitude: tan(hb) * alt; divide by cell size to get pixels
            hb_deg = float(config.get("beam_half_bw_deg", 4.0))
            r_km = math.tan(math.radians(hb_deg)) * self.alt_km * 2.0
            spacing_px = max(1, int(round(r_km / self.cell_km)))
        self.spacing_px = int(spacing_px)
        if self.nx <= 0 or self.ny <= 0:
            # Derive a moderate grid from map dimensions
            self.nx = max(2, self.X // max(1, self.spacing_px))
            self.ny = max(2, self.Y // max(1, self.spacing_px))
        xs = np.linspace(self.spacing_px//2, self.X - self.spacing_px//2, self.nx)
        ys = np.linspace(self.spacing_px//2, self.Y - self.spacing_px//2, self.ny)
        self.grid = np.array([(x, y) for x in xs for y in ys], dtype=float)
        self.hop_period = int(config.get("beam_hop_period_ttis", 0) or 0)

    def _nearest_beam_center(self, cx: float, cy: float) -> tuple:
        """Snap arbitrary center to nearest beam grid location."""
        if self.grid.size == 0:
            return cx, cy
        d2 = (self.grid[:,0] - cx)**2 + (self.grid[:,1] - cy)**2
        i = int(np.argmin(d2))
        return float(self.grid[i,0]), float(self.grid[i,1])

    def beam_center_at(self, t: int) -> Tuple[float, float]:
        # Base center from parent motion
        cx, cy = super().beam_center_at(t)
        if self.hop_period and self.hop_period > 0:
            # Raster hop through beams with period
            idx = (t // self.hop_period) % self.grid.shape[0]
            gx, gy = self.grid[idx]
            return float(gx), float(gy)
        # Else snap to nearest grid
        gx, gy = self._nearest_beam_center(cx, cy)
        return gx, gy
