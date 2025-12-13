"""
Beam and coverage abstractions for NR-NTN simulation.

This module provides beam pattern computation and beam management utilities.
Uses the unified geometry functions from ntn/geometry.py.

Classes:
    BeamManager: Placeholder for multi-beam and beam-hopping controller.

Functions:
    simple_beam_pattern_db: Compute Cos^m beam pattern gain.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from .geometry import beam_gain_db


def simple_beam_pattern_db(
    offaxis_deg: np.ndarray,
    boresight_gain_db: float,
    half_bw_deg: float,
    edge_drop_db: float = 3.0,
) -> np.ndarray:
    """Compute Cos^m beam pattern gain.

    This is a wrapper around geometry.beam_gain_db for backward compatibility.

    Args:
        offaxis_deg: Off-axis angle(s) in degrees.
        boresight_gain_db: Peak gain at boresight in dBi.
        half_bw_deg: Half-power (3dB) beamwidth in degrees.
        edge_drop_db: Gain reduction at half_bw_deg (default 3.0 dB).

    Returns:
        Beam gain value(s) in dBi.
    """
    return beam_gain_db(offaxis_deg, boresight_gain_db, half_bw_deg, edge_drop_db)


class BeamManager:
    """Placeholder for multi-beam and beam-hopping controller.

    For Stage-0/1 it exposes a compute interface that mirrors the
    simple pattern; in later stages it will manage multiple beams,
    neighbor lists, hopping cadence, and elevation/visibility masks.

    Attributes:
        G0: Boresight gain in dBi.
        half_bw_deg: Half-power beamwidth in degrees.
        edge_drop_db: Edge gain drop in dB.
        el_min: Minimum elevation angle for visibility (optional).
    """

    def __init__(
        self,
        boresight_gain_db: float,
        half_bw_deg: float,
        edge_drop_db: float = 3.0,
        elevation_min_deg: Optional[float] = None,
    ):
        """Initialize beam manager.

        Args:
            boresight_gain_db: Peak antenna gain at boresight (dBi).
            half_bw_deg: Half-power beamwidth (degrees).
            edge_drop_db: Gain reduction at half_bw_deg (dB).
            elevation_min_deg: Minimum elevation for visibility (optional).
        """
        self.G0 = float(boresight_gain_db)
        self.half_bw_deg = float(half_bw_deg)
        self.edge_drop_db = float(edge_drop_db)
        self.el_min = None if elevation_min_deg is None else float(elevation_min_deg)

    def gain_from_offaxis(self, offaxis_deg: np.ndarray) -> np.ndarray:
        """Compute beam gain for given off-axis angles.

        Args:
            offaxis_deg: Off-axis angle(s) in degrees.

        Returns:
            Beam gain(s) in dBi.
        """
        return beam_gain_db(
            offaxis_deg,
            self.G0,
            self.half_bw_deg,
            self.edge_drop_db
        )

    def is_visible(self, elevation_deg: np.ndarray) -> np.ndarray:
        """Check visibility based on elevation angle.

        Args:
            elevation_deg: Elevation angle(s) in degrees.

        Returns:
            Boolean array indicating visibility.
        """
        if self.el_min is None:
            return np.ones_like(np.asarray(elevation_deg), dtype=bool)
        return np.asarray(elevation_deg, dtype=float) >= self.el_min


__all__ = [
    "BeamManager",
    "simple_beam_pattern_db",
]
