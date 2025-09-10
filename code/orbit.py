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
        self.cx, self.cy = (X // 2, Y // 2) if config.get("beam_center_xy") is None else config["beam_center_xy"]
        self.alt_km = float(config.get("sat_altitude_km", 600.0))

    def get_slant_and_offaxis(self, ue_pos: np.ndarray, t: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        dx = (ue_pos[:, 0] - self.cx) * self.config.get("cell_size_km", 1.0)
        dy = (ue_pos[:, 1] - self.cy) * self.config.get("cell_size_km", 1.0)
        r_ground = np.sqrt(dx * dx + dy * dy)
        slant_km = np.sqrt(r_ground * r_ground + self.alt_km * self.alt_km)
        offaxis_deg = np.rad2deg(np.arctan2(r_ground, self.alt_km))
        return slant_km, offaxis_deg

