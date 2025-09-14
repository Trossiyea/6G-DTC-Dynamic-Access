"""
Beam and coverage abstractions for NR-NTN simulation.

Stage-0 scaffold: keep a simple API that can later grow into
multi-beam, beam hopping, and elevation masks without changing callers.

Current role:
- Provide helpers to compute approximate receive gain from off-axis angle.
- Define a BeamManager placeholder; by default, OrbitModel continues to
  compute G_rx and callers do not need this module.
"""

from typing import Optional
import numpy as np
import math


def simple_beam_pattern_db(offaxis_deg: np.ndarray,
                           boresight_gain_db: float,
                           half_bw_deg: float,
                           edge_drop_db: float = 3.0) -> np.ndarray:
    """Cos^m pattern matching orbit.simple_beam_gain_db behavior."""
    theta = np.asarray(offaxis_deg, dtype=float)
    theta = np.abs(theta)
    theta = np.minimum(theta, 89.9)
    hb = max(half_bw_deg, 1e-3) * math.pi / 180.0
    target = 10.0 ** (-edge_drop_db / 10.0)
    c = max(math.cos(hb), 1e-6)
    m = np.log(max(target, 1e-9)) / np.log(c)
    th_rad = theta * math.pi / 180.0
    patt = np.power(np.maximum(np.cos(th_rad), 1e-6), m)
    gain = float(boresight_gain_db) + 10.0 * np.log10(np.maximum(patt, 1e-9))
    return gain


class BeamManager:
    """
    Placeholder for multi-beam and beam-hopping controller.

    For Stage-0/1 it exposes a compute interface that mirrors the
    simple pattern; in later stages it will manage multiple beams,
    neighbor lists, hopping cadence, and elevation/visibility masks.
    """

    def __init__(self,
                 boresight_gain_db: float,
                 half_bw_deg: float,
                 edge_drop_db: float = 3.0,
                 elevation_min_deg: Optional[float] = None):
        self.G0 = float(boresight_gain_db)
        self.half_bw_deg = float(half_bw_deg)
        self.edge_drop_db = float(edge_drop_db)
        self.el_min = None if elevation_min_deg is None else float(elevation_min_deg)

    def gain_from_offaxis(self, offaxis_deg: np.ndarray) -> np.ndarray:
        return simple_beam_pattern_db(offaxis_deg, self.G0, self.half_bw_deg, self.edge_drop_db)

    def is_visible(self, elevation_deg: np.ndarray) -> np.ndarray:
        if self.el_min is None:
            return np.ones_like(np.asarray(elevation_deg), dtype=bool)
        return np.asarray(elevation_deg, dtype=float) >= self.el_min

