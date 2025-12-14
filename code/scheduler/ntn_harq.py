# -*- coding: utf-8 -*-
"""
NTN-aware HARQ adapter for satellite scenarios.

Extends the link layer HARQ manager with NTN-specific timing:
- Dynamic K1 per UE based on propagation delay
- TA-aware HARQ RTT calculation
- Extended HARQ process pool for high-RTT scenarios

Usage:
    from scheduler.ntn_harq import NTNHarqAdapter

    adapter = NTNHarqAdapter(n_ue=100, config=config)
    adapter.update_from_geometry(tau_s_per_ue)

    # Use with scheduler
    harq_mgr = adapter.get_harq_manager()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import link layer HARQ
try:
    from link import HarqManager, HarqManagerFull
    HARQ_AVAILABLE = True
except ImportError:
    HARQ_AVAILABLE = False
    HarqManager = None
    HarqManagerFull = None

# Import MAC timing
try:
    from mac import (
        SchedulingTimingManager,
        HARQTimingAdapter,
        NTNTimingConfig,
        NTNScenario,
    )
    MAC_TIMING_AVAILABLE = True
except ImportError:
    MAC_TIMING_AVAILABLE = False


@dataclass
class NTNHarqConfig:
    """NTN HARQ configuration."""

    # Base HARQ parameters
    max_processes: int = 16
    max_retx: int = 4
    ack_delay_base_ttis: int = 4  # Base K1 for terrestrial

    # NTN extensions
    enable_ntn_timing: bool = True
    auto_scale_processes: bool = True  # Auto-increase processes for high RTT
    max_ntn_processes: int = 32  # Maximum processes in NTN mode

    # Target BLER
    target_bler: float = 0.1

    # HARQ mode
    use_full_harq: bool = True

    @classmethod
    def from_config_dict(cls, config: Dict) -> "NTNHarqConfig":
        """Create from configuration dictionary."""
        return cls(
            max_processes=config.get("harq_max_procs", 16),
            max_retx=config.get("harq_max_retx", 4),
            ack_delay_base_ttis=config.get("harq_ack_delay_ttis", 4),
            enable_ntn_timing=config.get("ntn_timing_adaptation", True),
            auto_scale_processes=config.get("harq_auto_scale_processes", True),
            max_ntn_processes=config.get("harq_max_ntn_processes", 32),
            target_bler=config.get("harq_target_bler", 0.1),
            use_full_harq=config.get("enable_harq_full", True),
        )


class NTNHarqAdapter:
    """NTN-aware HARQ adapter.

    Wraps the link layer HARQ manager with NTN timing integration:
    - Updates HARQ ACK delay per UE based on K1
    - Scales HARQ process count for high-RTT UEs
    - Provides timing-aware scheduling constraints
    """

    def __init__(
        self,
        n_ue: int,
        config: Optional[Dict] = None,
    ):
        """Initialize NTN HARQ adapter.

        Args:
            n_ue: Number of UEs
            config: Configuration dictionary
        """
        self.n_ue = n_ue
        self.config = config or {}
        self.harq_config = NTNHarqConfig.from_config_dict(self.config)

        # NTN Timing adapter (from MAC)
        self.timing_adapter: Optional[HARQTimingAdapter] = None
        if MAC_TIMING_AVAILABLE and self.harq_config.enable_ntn_timing:
            self.timing_adapter = HARQTimingAdapter(
                n_ue=n_ue,
                config=self.config,
            )

        # Per-UE K1 values
        self._ue_k1_slots = np.full(n_ue, self.harq_config.ack_delay_base_ttis, dtype=int)

        # Per-UE HARQ process count (may vary for NTN)
        self._ue_harq_processes = np.full(n_ue, self.harq_config.max_processes, dtype=int)

        # Per-UE RTT slots
        self._ue_rtt_slots = np.zeros(n_ue, dtype=int)

        # Underlying HARQ manager (created on demand)
        self._harq_manager = None

        # Current TTI
        self._current_tti = 0

    def update_from_geometry(
        self,
        tau_s_per_ue: np.ndarray,
        slot_duration_ms: Optional[float] = None,
    ) -> None:
        """Update HARQ timing from propagation delays.

        Args:
            tau_s_per_ue: One-way propagation delay per UE [N_UE] in seconds
            slot_duration_ms: Slot duration in ms
        """
        slot_ms = slot_duration_ms or self.config.get("tti_ms", 1.0)

        # Update timing adapter
        if self.timing_adapter is not None:
            self.timing_adapter.update_from_geometry(tau_s_per_ue, slot_ms)

            # Get K1 and HARQ processes from adapter
            for ue in range(min(self.n_ue, len(tau_s_per_ue))):
                self._ue_k1_slots[ue] = self.timing_adapter.get_k1_for_ue(ue)
                self._ue_rtt_slots[ue] = self.timing_adapter.get_harq_rtt_slots(ue)

                if self.harq_config.auto_scale_processes:
                    self._ue_harq_processes[ue] = self.timing_adapter.get_effective_harq_processes(
                        ue, self.harq_config.max_processes
                    )
        else:
            # Manual calculation without MAC timing
            for ue in range(min(self.n_ue, len(tau_s_per_ue))):
                rtt_ms = 2.0 * tau_s_per_ue[ue] * 1000.0
                rtt_slots = int(np.ceil(rtt_ms / slot_ms))
                self._ue_rtt_slots[ue] = rtt_slots

                # K1 = base + RTT adjustment
                k1 = self.harq_config.ack_delay_base_ttis + rtt_slots
                self._ue_k1_slots[ue] = min(k1, 256)  # Cap at max K1

                # Scale processes if needed
                if self.harq_config.auto_scale_processes:
                    min_procs = max(self.harq_config.max_processes, rtt_slots // 2 + 1)
                    self._ue_harq_processes[ue] = min(min_procs, self.harq_config.max_ntn_processes)

    def get_k1_for_ue(self, ue_id: int) -> int:
        """Get K1 (PDSCH to ACK delay) for a UE.

        Args:
            ue_id: UE index

        Returns:
            K1 in slots
        """
        if 0 <= ue_id < self.n_ue:
            return int(self._ue_k1_slots[ue_id])
        return self.harq_config.ack_delay_base_ttis

    def get_all_k1(self) -> np.ndarray:
        """Get K1 for all UEs.

        Returns:
            K1 values [N_UE]
        """
        return self._ue_k1_slots.copy()

    def get_harq_processes_for_ue(self, ue_id: int) -> int:
        """Get number of HARQ processes for a UE.

        Args:
            ue_id: UE index

        Returns:
            Number of HARQ processes
        """
        if 0 <= ue_id < self.n_ue:
            return int(self._ue_harq_processes[ue_id])
        return self.harq_config.max_processes

    def get_harq_rtt_slots(self, ue_id: int) -> int:
        """Get HARQ RTT for a UE.

        Args:
            ue_id: UE index

        Returns:
            RTT in slots
        """
        if 0 <= ue_id < self.n_ue:
            return int(self._ue_rtt_slots[ue_id])
        return 0

    def get_ack_slot(self, ue_id: int, tx_slot: int) -> int:
        """Get expected ACK slot for a transmission.

        Args:
            ue_id: UE index
            tx_slot: Transmission slot

        Returns:
            Expected ACK slot
        """
        k1 = self.get_k1_for_ue(ue_id)
        return tx_slot + k1

    def can_schedule_new_tx(self, ue_id: int) -> bool:
        """Check if UE can be scheduled for new transmission.

        Args:
            ue_id: UE index

        Returns:
            True if UE has free HARQ process
        """
        if self._harq_manager is not None:
            return self._harq_manager.can_schedule(ue_id)
        # Without HARQ manager, always allow
        return True

    def get_pending_retx_ues(self) -> List[int]:
        """Get UEs with pending retransmissions.

        Returns:
            List of UE IDs needing retransmission
        """
        if self._harq_manager is not None and hasattr(self._harq_manager, 'get_retx_ues'):
            return list(self._harq_manager.get_retx_ues())
        return []

    # -------------------------------------------------------------------------
    # HARQ Manager Access
    # -------------------------------------------------------------------------

    def get_harq_manager(self) -> Optional[HarqManager]:
        """Get underlying HARQ manager.

        Creates on first access with NTN-aware parameters.

        Returns:
            HARQ manager instance
        """
        if not HARQ_AVAILABLE:
            return None

        if self._harq_manager is None:
            self._create_harq_manager()

        return self._harq_manager

    def _create_harq_manager(self) -> None:
        """Create HARQ manager with NTN-aware config."""
        if not HARQ_AVAILABLE:
            return

        # Use max K1 across all UEs for global ACK delay
        global_k1 = int(np.max(self._ue_k1_slots))

        # Use max processes across all UEs
        global_processes = int(np.max(self._ue_harq_processes))

        harq_config = {
            "harq_max_procs": global_processes,
            "harq_max_retx": self.harq_config.max_retx,
            "harq_ack_delay_ttis": global_k1,
            "harq_target_bler": self.harq_config.target_bler,
        }
        harq_config.update(self.config)

        if self.harq_config.use_full_harq and HarqManagerFull is not None:
            self._harq_manager = HarqManagerFull(
                n_ue=self.n_ue,
                config=harq_config,
            )
        elif HarqManager is not None:
            self._harq_manager = HarqManager(
                n_ue=self.n_ue,
                config=harq_config,
            )

    def update_harq_manager_timing(self) -> None:
        """Update HARQ manager with current timing parameters.

        Call after geometry update to sync timing.
        """
        if self._harq_manager is None:
            return

        # For full HARQ manager, we may need to update per-UE ACK delays
        # This depends on the HARQ manager implementation
        # For now, we use the global max K1 approach

        global_k1 = int(np.max(self._ue_k1_slots))
        if hasattr(self._harq_manager, 'ack_delay'):
            self._harq_manager.ack_delay = global_k1
        elif hasattr(self._harq_manager, 'config') and isinstance(self._harq_manager.config, dict):
            self._harq_manager.config['harq_ack_delay_ttis'] = global_k1

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Get NTN HARQ adapter statistics."""
        return {
            "n_ue": self.n_ue,
            "enable_ntn_timing": self.harq_config.enable_ntn_timing,
            "k1_min": int(np.min(self._ue_k1_slots)),
            "k1_max": int(np.max(self._ue_k1_slots)),
            "k1_mean": float(np.mean(self._ue_k1_slots)),
            "rtt_slots_max": int(np.max(self._ue_rtt_slots)),
            "processes_min": int(np.min(self._ue_harq_processes)),
            "processes_max": int(np.max(self._ue_harq_processes)),
            "harq_manager_active": self._harq_manager is not None,
        }

    def reset(self) -> None:
        """Reset adapter state."""
        self._ue_k1_slots = np.full(self.n_ue, self.harq_config.ack_delay_base_ttis, dtype=int)
        self._ue_harq_processes = np.full(self.n_ue, self.harq_config.max_processes, dtype=int)
        self._ue_rtt_slots = np.zeros(self.n_ue, dtype=int)
        self._harq_manager = None
        self._current_tti = 0


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ntn_harq_adapter(
    n_ue: int,
    config: Dict,
    tau_s_per_ue: Optional[np.ndarray] = None,
) -> NTNHarqAdapter:
    """Create and initialize NTN HARQ adapter.

    Args:
        n_ue: Number of UEs
        config: Configuration dictionary
        tau_s_per_ue: Initial propagation delays (optional)

    Returns:
        Initialized NTNHarqAdapter
    """
    adapter = NTNHarqAdapter(n_ue, config)

    if tau_s_per_ue is not None:
        adapter.update_from_geometry(tau_s_per_ue)

    return adapter


def compute_min_harq_processes(
    rtt_ms: float,
    slot_ms: float = 1.0,
    base_processes: int = 16,
) -> int:
    """Compute minimum HARQ processes needed for given RTT.

    Args:
        rtt_ms: Round-trip time in milliseconds
        slot_ms: Slot duration in milliseconds
        base_processes: Baseline number of processes

    Returns:
        Minimum number of HARQ processes
    """
    rtt_slots = int(np.ceil(rtt_ms / slot_ms))
    # Need enough processes to keep pipeline full
    min_procs = max(base_processes, rtt_slots // 2 + 1)
    return min(min_procs, 32)  # Cap at 32 (NR maximum)
