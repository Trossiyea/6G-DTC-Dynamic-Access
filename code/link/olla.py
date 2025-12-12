# -*- coding: utf-8 -*-
"""
OLLA (Outer Loop Link Adaptation) for NR.

Provides adaptive SINR offset based on ACK/NACK feedback
to achieve target BLER.
"""

from typing import Optional
import numpy as np


class OLLA:
    """Outer Loop Link Adaptation with separate up/down step sizes.

    Adjusts SINR offset based on ACK/NACK feedback to track target BLER.
    - ACK: increase offset (more aggressive MCS selection)
    - NACK: decrease offset (more conservative MCS selection)
    """

    def __init__(
        self,
        step_up_db: float = 0.1,
        step_down_db: float = 0.1,
        init_offset_db: float = 0.0,
        p_target: float = 0.1,
        min_offset_db: float = -6.0,
        max_offset_db: float = 6.0,
    ):
        """Initialize OLLA.

        Args:
            step_up_db: Offset increase on ACK (dB)
            step_down_db: Offset decrease on NACK (dB)
            init_offset_db: Initial offset value (dB)
            p_target: Target BLER (used for documentation, actual tracking via updates)
            min_offset_db: Minimum allowed offset (dB)
            max_offset_db: Maximum allowed offset (dB)
        """
        self.step_up_db = float(step_up_db)
        self.step_down_db = float(step_down_db)
        self.offset_db = float(init_offset_db)
        self.p_target = float(p_target)
        self.min_offset_db = float(min_offset_db)
        self.max_offset_db = float(max_offset_db)

    def apply(self, sinr_db: np.ndarray) -> np.ndarray:
        """Apply OLLA offset to SINR values.

        Args:
            sinr_db: Input SINR in dB

        Returns:
            Adjusted SINR in dB
        """
        return np.asarray(sinr_db, dtype=float) + self.offset_db

    def update(self, is_ack: bool) -> None:
        """Update offset based on ACK/NACK feedback.

        Args:
            is_ack: True for ACK, False for NACK
        """
        if is_ack:
            self.offset_db += self.step_up_db
        else:
            self.offset_db -= self.step_down_db

        # Clamp to avoid runaway
        self.offset_db = max(self.min_offset_db, min(self.max_offset_db, self.offset_db))

    def reset(self, init_offset_db: Optional[float] = None) -> None:
        """Reset OLLA offset.

        Args:
            init_offset_db: New initial offset (uses constructor value if None)
        """
        if init_offset_db is not None:
            self.offset_db = float(init_offset_db)
        else:
            self.offset_db = 0.0

    @property
    def current_offset(self) -> float:
        """Get current offset value."""
        return self.offset_db


def create_olla_from_config(config: dict, idx: int = 0) -> OLLA:
    """Create OLLA instance from configuration dict.

    Args:
        config: Configuration dict with OLLA settings
        idx: UE index (for potential per-UE initialization)

    Returns:
        Configured OLLA instance
    """
    return OLLA(
        step_up_db=float(config.get("olla_step_up_db", 0.1)),
        step_down_db=float(config.get("olla_step_down_db", 0.1)),
        init_offset_db=float(config.get("olla_init_offset_db", config.get("csi_olla_offset_db", 0.0))),
        p_target=float(config.get("harq_target_bler", 0.1)),
        min_offset_db=float(config.get("olla_min_db", -6.0)),
        max_offset_db=float(config.get("olla_max_db", 6.0)),
    )
