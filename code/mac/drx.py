"""
Discontinuous Reception (DRX) state machine per 3GPP TS 38.321.

Implements DRX for power-saving with NTN-specific extensions to handle
long propagation delays in satellite communication.

DRX States (TS 38.321 §5.7):
- Active Time: UE monitors PDCCH continuously
- On Duration: Start of DRX cycle, PDCCH monitoring enabled
- Inactivity: After PDCCH reception, waiting for more activity
- Short DRX Cycle: Power-saving with frequent wake-ups
- Long DRX Cycle: Deep power-saving with infrequent wake-ups

DRX Timers:
- drx-onDurationTimer: Duration of On Duration period
- drx-InactivityTimer: Time after PDCCH before entering DRX
- drx-HARQ-RTT-TimerDL: Wait time for HARQ feedback
- drx-RetransmissionTimerDL: Wait for DL retransmission
- drx-ShortCycleTimer: Cycles before transitioning to Long Cycle
- drx-LongCycleStartOffset: Offset for Long Cycle start

NTN Extensions:
- Extended timer values to account for propagation delay
- Per-UE timing adjustment based on satellite geometry
- HARQ RTT scaling for NTN scenarios
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np


# =============================================================================
# DRX State Definitions
# =============================================================================

class DRXState(Enum):
    """DRX state per 3GPP TS 38.321 §5.7."""

    ACTIVE = auto()          # Continuous PDCCH monitoring (DRX disabled or Active Time)
    ON_DURATION = auto()     # On Duration Timer running, PDCCH monitoring
    INACTIVITY = auto()      # Inactivity Timer running, PDCCH monitoring
    SHORT_CYCLE = auto()     # In DRX Short Cycle (sleeping between On Durations)
    LONG_CYCLE = auto()      # In DRX Long Cycle (deep sleep)

    def is_monitoring_pdcch(self) -> bool:
        """Check if UE is monitoring PDCCH in this state."""
        return self in (DRXState.ACTIVE, DRXState.ON_DURATION, DRXState.INACTIVITY)

    def is_sleeping(self) -> bool:
        """Check if UE is in sleep/power-saving mode."""
        return self in (DRXState.SHORT_CYCLE, DRXState.LONG_CYCLE)


class DRXEvent(Enum):
    """Events that trigger DRX state transitions."""

    DRX_CYCLE_START = auto()       # Start of DRX cycle (On Duration begins)
    ON_DURATION_EXPIRED = auto()   # On Duration Timer expired
    INACTIVITY_EXPIRED = auto()    # Inactivity Timer expired
    SHORT_CYCLE_EXPIRED = auto()   # Short Cycle Timer expired (transition to Long)
    PDCCH_RECEIVED = auto()        # PDCCH reception (new/retx grant)
    HARQ_ACK = auto()              # HARQ ACK received
    HARQ_NACK = auto()             # HARQ NACK received (retransmission expected)
    HARQ_RTT_EXPIRED = auto()      # HARQ RTT Timer expired
    RETX_TIMER_EXPIRED = auto()    # Retransmission Timer expired
    SR_SENT = auto()               # Scheduling Request sent (for completeness)


# =============================================================================
# DRX Configuration
# =============================================================================

@dataclass
class DRXConfig:
    """DRX configuration parameters per 3GPP TS 38.331.

    All timer values in TTIs (subframes/slots).
    """

    # Enable/Disable
    enabled: bool = False

    # On Duration Timer (ms -> TTIs)
    on_duration_ms: float = 10.0

    # Inactivity Timer (ms -> TTIs)
    inactivity_timer_ms: float = 100.0

    # HARQ RTT Timer DL (slots)
    harq_rtt_timer_dl_slots: int = 40

    # Retransmission Timer DL (slots)
    retx_timer_dl_slots: int = 33

    # Short DRX Cycle (ms -> TTIs)
    short_cycle_ms: float = 20.0

    # drx-ShortCycleTimer (number of short cycles before long)
    short_cycle_timer: int = 2

    # Long DRX Cycle (ms -> TTIs)
    long_cycle_ms: float = 320.0

    # Long Cycle Start Offset (slots within long cycle)
    long_cycle_start_offset: int = 0

    # Slot offset (for cycle alignment)
    slot_offset: int = 0

    # NTN Extensions
    ntn_harq_rtt_extension_ms: float = 0.0  # Additional RTT for NTN
    ntn_inactivity_extension_ms: float = 0.0  # Additional inactivity for NTN

    # TTI duration for conversion
    tti_ms: float = 1.0

    def __post_init__(self):
        """Convert ms values to TTIs."""
        self._on_duration_ttis = max(1, int(self.on_duration_ms / self.tti_ms))
        self._inactivity_ttis = max(1, int(self.inactivity_timer_ms / self.tti_ms))
        self._short_cycle_ttis = max(1, int(self.short_cycle_ms / self.tti_ms))
        self._long_cycle_ttis = max(1, int(self.long_cycle_ms / self.tti_ms))

    @property
    def on_duration_ttis(self) -> int:
        return self._on_duration_ttis

    @property
    def inactivity_ttis(self) -> int:
        return self._inactivity_ttis

    @property
    def short_cycle_ttis(self) -> int:
        return self._short_cycle_ttis

    @property
    def long_cycle_ttis(self) -> int:
        return self._long_cycle_ttis

    @classmethod
    def from_config_dict(cls, config: Dict) -> "DRXConfig":
        """Create DRXConfig from flat configuration dictionary."""
        return cls(
            enabled=config.get("enable_drx", False),
            on_duration_ms=config.get("drx_on_duration_ms", 10.0),
            inactivity_timer_ms=config.get("drx_inactivity_timer_ms", 100.0),
            harq_rtt_timer_dl_slots=config.get("drx_harq_rtt_timer_slots", 40),
            retx_timer_dl_slots=config.get("drx_retx_timer_slots", 33),
            short_cycle_ms=config.get("drx_short_cycle_ms", 20.0),
            short_cycle_timer=config.get("drx_short_cycle_timer", 2),
            long_cycle_ms=config.get("drx_long_cycle_ms", 320.0),
            long_cycle_start_offset=config.get("drx_long_cycle_start_offset", 0),
            slot_offset=config.get("drx_slot_offset", 0),
            ntn_harq_rtt_extension_ms=config.get("ntn_harq_rtt_extension_ms", 0.0),
            ntn_inactivity_extension_ms=config.get("ntn_inactivity_extension_ms", 0.0),
            tti_ms=config.get("tti_ms", 1.0),
        )


# =============================================================================
# Per-UE DRX State
# =============================================================================

@dataclass
class UEDRXState:
    """Per-UE DRX state tracking."""

    ue_id: int
    state: DRXState = DRXState.ACTIVE
    config: Optional[DRXConfig] = None

    # Timer values (remaining TTIs, 0 = expired/not running)
    on_duration_timer: int = 0
    inactivity_timer: int = 0
    harq_rtt_timer: int = 0
    retx_timer: int = 0
    short_cycle_count: int = 0  # Number of short cycles completed

    # Cycle tracking
    current_cycle_start_tti: int = 0
    in_active_time: bool = True  # True if in Active Time

    # Statistics
    total_active_ttis: int = 0
    total_sleep_ttis: int = 0
    state_transitions: int = 0

    # NTN-specific
    ntn_rtt_offset_ttis: int = 0  # Per-UE RTT offset for NTN

    def is_monitoring_pdcch(self) -> bool:
        """Check if UE should be monitoring PDCCH."""
        if self.config is None or not self.config.enabled:
            return True  # DRX disabled, always monitoring
        return self.state.is_monitoring_pdcch() or self.in_active_time

    def is_schedulable(self) -> bool:
        """Check if UE can be scheduled (alias for is_monitoring_pdcch)."""
        return self.is_monitoring_pdcch()

    def get_power_state(self) -> str:
        """Get human-readable power state."""
        if not self.config or not self.config.enabled:
            return "always_on"
        if self.state.is_sleeping():
            return "sleep"
        return "active"


# =============================================================================
# DRX Controller
# =============================================================================

class DRXController:
    """DRX state machine controller for all UEs.

    Manages DRX states, timers, and transitions for power-efficient
    PDCCH monitoring with support for NTN scenarios.
    """

    def __init__(
        self,
        n_ue: int,
        config: Optional[Dict] = None,
        drx_config: Optional[DRXConfig] = None,
    ):
        """Initialize DRX controller.

        Args:
            n_ue: Number of UEs
            config: Flat configuration dictionary
            drx_config: Pre-built DRXConfig (overrides config dict)
        """
        self.n_ue = n_ue

        # Build DRX config
        if drx_config is not None:
            self.drx_config = drx_config
        elif config is not None:
            self.drx_config = DRXConfig.from_config_dict(config)
        else:
            self.drx_config = DRXConfig()

        # Per-UE states
        self._ue_states: List[UEDRXState] = [
            UEDRXState(ue_id=i, config=self.drx_config)
            for i in range(n_ue)
        ]

        # Event callbacks
        self._on_state_change: Optional[Callable] = None

        # Statistics
        self._current_tti = 0

    # -------------------------------------------------------------------------
    # Timer Management
    # -------------------------------------------------------------------------

    def advance_time(self, tti: int) -> Dict[int, DRXState]:
        """Advance DRX state machine by one TTI.

        Decrements timers, handles expirations, and updates states.

        Args:
            tti: Current TTI index

        Returns:
            Dictionary of UEs that changed state: {ue_id: new_state}
        """
        self._current_tti = tti
        state_changes = {}

        if not self.drx_config.enabled:
            return state_changes

        for ue_state in self._ue_states:
            old_state = ue_state.state
            self._advance_ue_drx(ue_state, tti)

            if ue_state.state != old_state:
                state_changes[ue_state.ue_id] = ue_state.state
                ue_state.state_transitions += 1

            # Update statistics
            if ue_state.is_monitoring_pdcch():
                ue_state.total_active_ttis += 1
            else:
                ue_state.total_sleep_ttis += 1

        return state_changes

    def _advance_ue_drx(self, ue_state: UEDRXState, tti: int) -> None:
        """Advance DRX for a single UE."""
        cfg = self.drx_config

        # Check for DRX cycle start
        if self._is_drx_cycle_start(ue_state, tti):
            self._handle_event(ue_state, DRXEvent.DRX_CYCLE_START, tti)
            return

        # Decrement and check timers
        if ue_state.on_duration_timer > 0:
            ue_state.on_duration_timer -= 1
            if ue_state.on_duration_timer == 0:
                self._handle_event(ue_state, DRXEvent.ON_DURATION_EXPIRED, tti)

        if ue_state.inactivity_timer > 0:
            ue_state.inactivity_timer -= 1
            if ue_state.inactivity_timer == 0:
                self._handle_event(ue_state, DRXEvent.INACTIVITY_EXPIRED, tti)

        if ue_state.harq_rtt_timer > 0:
            ue_state.harq_rtt_timer -= 1
            if ue_state.harq_rtt_timer == 0:
                self._handle_event(ue_state, DRXEvent.HARQ_RTT_EXPIRED, tti)

        if ue_state.retx_timer > 0:
            ue_state.retx_timer -= 1
            if ue_state.retx_timer == 0:
                self._handle_event(ue_state, DRXEvent.RETX_TIMER_EXPIRED, tti)

    def _is_drx_cycle_start(self, ue_state: UEDRXState, tti: int) -> bool:
        """Check if current TTI is start of DRX cycle."""
        cfg = self.drx_config

        if ue_state.state == DRXState.SHORT_CYCLE:
            cycle_len = cfg.short_cycle_ttis
        else:
            cycle_len = cfg.long_cycle_ttis

        # Check if TTI aligns with cycle start
        offset = cfg.slot_offset + ue_state.ue_id % cycle_len  # UE-specific offset
        return (tti - offset) % cycle_len == 0

    def _handle_event(self, ue_state: UEDRXState, event: DRXEvent, tti: int) -> None:
        """Handle DRX event and perform state transition."""
        cfg = self.drx_config
        old_state = ue_state.state

        if event == DRXEvent.DRX_CYCLE_START:
            # Start On Duration
            ue_state.state = DRXState.ON_DURATION
            ue_state.on_duration_timer = cfg.on_duration_ttis
            ue_state.current_cycle_start_tti = tti
            ue_state.in_active_time = True

        elif event == DRXEvent.ON_DURATION_EXPIRED:
            # On Duration ended without PDCCH
            if ue_state.inactivity_timer > 0 or ue_state.harq_rtt_timer > 0:
                # Still in Active Time due to other timers
                pass
            else:
                # Transition to sleep
                ue_state.in_active_time = False
                if ue_state.short_cycle_count < cfg.short_cycle_timer:
                    ue_state.state = DRXState.SHORT_CYCLE
                    ue_state.short_cycle_count += 1
                else:
                    ue_state.state = DRXState.LONG_CYCLE
                    ue_state.short_cycle_count = 0

        elif event == DRXEvent.INACTIVITY_EXPIRED:
            # Inactivity ended
            if ue_state.on_duration_timer == 0 and ue_state.harq_rtt_timer == 0:
                ue_state.in_active_time = False
                if ue_state.short_cycle_count < cfg.short_cycle_timer:
                    ue_state.state = DRXState.SHORT_CYCLE
                    ue_state.short_cycle_count += 1
                else:
                    ue_state.state = DRXState.LONG_CYCLE
                    ue_state.short_cycle_count = 0

        elif event == DRXEvent.PDCCH_RECEIVED:
            # PDCCH received - restart inactivity timer
            ue_state.state = DRXState.INACTIVITY
            ue_state.inactivity_timer = cfg.inactivity_ttis + ue_state.ntn_rtt_offset_ttis
            ue_state.in_active_time = True
            ue_state.short_cycle_count = 0  # Reset short cycle count

        elif event == DRXEvent.HARQ_NACK:
            # HARQ NACK - start retransmission timer
            ue_state.retx_timer = cfg.retx_timer_dl_slots + ue_state.ntn_rtt_offset_ttis
            ue_state.in_active_time = True

        elif event == DRXEvent.HARQ_ACK:
            # HARQ ACK - may affect Active Time
            # If no other timers running, may exit Active Time
            pass

        # Notify callback if state changed
        if ue_state.state != old_state and self._on_state_change:
            self._on_state_change(ue_state.ue_id, old_state, ue_state.state, tti)

    # -------------------------------------------------------------------------
    # External Event Interface
    # -------------------------------------------------------------------------

    def on_pdcch_received(self, ue_id: int, tti: int) -> None:
        """Notify DRX of PDCCH reception (scheduling grant).

        Args:
            ue_id: UE index
            tti: Current TTI
        """
        if not self.drx_config.enabled:
            return
        if 0 <= ue_id < self.n_ue:
            self._handle_event(self._ue_states[ue_id], DRXEvent.PDCCH_RECEIVED, tti)

    def on_harq_ack(self, ue_id: int, tti: int) -> None:
        """Notify DRX of HARQ ACK.

        Args:
            ue_id: UE index
            tti: Current TTI
        """
        if not self.drx_config.enabled:
            return
        if 0 <= ue_id < self.n_ue:
            self._handle_event(self._ue_states[ue_id], DRXEvent.HARQ_ACK, tti)

    def on_harq_nack(self, ue_id: int, tti: int) -> None:
        """Notify DRX of HARQ NACK.

        Args:
            ue_id: UE index
            tti: Current TTI
        """
        if not self.drx_config.enabled:
            return
        if 0 <= ue_id < self.n_ue:
            self._handle_event(self._ue_states[ue_id], DRXEvent.HARQ_NACK, tti)

    # -------------------------------------------------------------------------
    # Query Interface for Scheduler
    # -------------------------------------------------------------------------

    def is_ue_active(self, ue_id: int) -> bool:
        """Check if UE is in Active Time (schedulable).

        Args:
            ue_id: UE index

        Returns:
            True if UE is monitoring PDCCH and can be scheduled
        """
        if not self.drx_config.enabled:
            return True
        if 0 <= ue_id < self.n_ue:
            return self._ue_states[ue_id].is_schedulable()
        return False

    def get_active_ues(self) -> List[int]:
        """Get list of UEs currently in Active Time.

        Returns:
            List of UE IDs that can be scheduled
        """
        if not self.drx_config.enabled:
            return list(range(self.n_ue))
        return [ue for ue in range(self.n_ue) if self._ue_states[ue].is_schedulable()]

    def get_sleeping_ues(self) -> List[int]:
        """Get list of UEs currently in DRX sleep.

        Returns:
            List of UE IDs in sleep mode
        """
        if not self.drx_config.enabled:
            return []
        return [ue for ue in range(self.n_ue) if not self._ue_states[ue].is_schedulable()]

    def get_ue_state(self, ue_id: int) -> Optional[DRXState]:
        """Get current DRX state for a UE.

        Args:
            ue_id: UE index

        Returns:
            DRXState or None if invalid UE
        """
        if 0 <= ue_id < self.n_ue:
            return self._ue_states[ue_id].state
        return None

    def get_all_states(self) -> np.ndarray:
        """Get DRX states for all UEs as array.

        Returns:
            Array of state values [N_UE] (0=ACTIVE, 1=ON_DURATION, etc.)
        """
        return np.array([s.state.value for s in self._ue_states])

    # -------------------------------------------------------------------------
    # NTN Extensions
    # -------------------------------------------------------------------------

    def set_ntn_rtt_offset(self, ue_id: int, rtt_ttis: int) -> None:
        """Set NTN RTT offset for a UE.

        In NTN scenarios, timers need to account for propagation delay.
        This adds per-UE RTT offset to relevant timers.

        Args:
            ue_id: UE index
            rtt_ttis: Round-trip time in TTIs
        """
        if 0 <= ue_id < self.n_ue:
            self._ue_states[ue_id].ntn_rtt_offset_ttis = rtt_ttis

    def update_ntn_offsets_from_geometry(
        self,
        tau_s_per_ue: np.ndarray,
        tti_ms: float = 1.0
    ) -> None:
        """Update NTN RTT offsets from propagation delay array.

        Args:
            tau_s_per_ue: One-way propagation delay per UE [N_UE] in seconds
            tti_ms: TTI duration in milliseconds
        """
        for ue in range(min(self.n_ue, len(tau_s_per_ue))):
            rtt_ms = 2.0 * tau_s_per_ue[ue] * 1000.0  # Round-trip in ms
            rtt_ttis = int(np.ceil(rtt_ms / tti_ms))
            self._ue_states[ue].ntn_rtt_offset_ttis = rtt_ttis

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_state_change(self, callback: Callable) -> None:
        """Register callback for state changes.

        Args:
            callback: Function(ue_id, old_state, new_state, tti)
        """
        self._on_state_change = callback

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Get DRX statistics.

        Returns:
            Dictionary with DRX statistics
        """
        if not self.drx_config.enabled:
            return {
                "enabled": False,
                "active_ues": self.n_ue,
                "sleeping_ues": 0,
            }

        active_ues = len(self.get_active_ues())
        sleeping_ues = len(self.get_sleeping_ues())

        total_active = sum(s.total_active_ttis for s in self._ue_states)
        total_sleep = sum(s.total_sleep_ttis for s in self._ue_states)
        total_transitions = sum(s.state_transitions for s in self._ue_states)

        # State distribution
        state_counts = {}
        for state in DRXState:
            state_counts[state.name] = sum(
                1 for s in self._ue_states if s.state == state
            )

        return {
            "enabled": True,
            "active_ues": active_ues,
            "sleeping_ues": sleeping_ues,
            "total_active_ttis": total_active,
            "total_sleep_ttis": total_sleep,
            "total_transitions": total_transitions,
            "duty_cycle": total_active / max(1, total_active + total_sleep),
            "state_distribution": state_counts,
            "config": {
                "on_duration_ttis": self.drx_config.on_duration_ttis,
                "inactivity_ttis": self.drx_config.inactivity_ttis,
                "short_cycle_ttis": self.drx_config.short_cycle_ttis,
                "long_cycle_ttis": self.drx_config.long_cycle_ttis,
            },
        }

    def get_power_saving_ratio(self) -> float:
        """Get average power saving ratio across all UEs.

        Returns:
            Ratio of sleep time to total time (0.0 to 1.0)
        """
        total_active = sum(s.total_active_ttis for s in self._ue_states)
        total_sleep = sum(s.total_sleep_ttis for s in self._ue_states)
        total = total_active + total_sleep
        return total_sleep / total if total > 0 else 0.0

    # -------------------------------------------------------------------------
    # Control
    # -------------------------------------------------------------------------

    def enable(self) -> None:
        """Enable DRX for all UEs."""
        self.drx_config.enabled = True

    def disable(self) -> None:
        """Disable DRX for all UEs."""
        self.drx_config.enabled = False
        # Reset all UEs to ACTIVE state
        for ue_state in self._ue_states:
            ue_state.state = DRXState.ACTIVE
            ue_state.in_active_time = True

    def reset(self) -> None:
        """Reset all DRX states."""
        self._ue_states = [
            UEDRXState(ue_id=i, config=self.drx_config)
            for i in range(self.n_ue)
        ]
        self._current_tti = 0


# =============================================================================
# NTN-Aware DRX Controller
# =============================================================================

class NTNDRXController(DRXController):
    """DRX Controller with NTN-specific enhancements.

    Extends DRXController with:
    - Automatic RTT offset computation from satellite geometry
    - Extended timer values for NTN scenarios
    - Discontinuous coverage support (future)
    """

    def __init__(
        self,
        n_ue: int,
        config: Optional[Dict] = None,
        tau_s_per_ue: Optional[np.ndarray] = None,
    ):
        """Initialize NTN DRX controller.

        Args:
            n_ue: Number of UEs
            config: Configuration dictionary
            tau_s_per_ue: Initial propagation delays per UE [N_UE] in seconds
        """
        super().__init__(n_ue, config)

        self.tti_ms = config.get("tti_ms", 1.0) if config else 1.0

        # Apply initial NTN offsets
        if tau_s_per_ue is not None:
            self.update_ntn_offsets_from_geometry(tau_s_per_ue, self.tti_ms)

    def update_geometry(self, tau_s_per_ue: np.ndarray) -> None:
        """Update NTN geometry (call when satellite moves).

        Args:
            tau_s_per_ue: Updated propagation delays per UE [N_UE] in seconds
        """
        self.update_ntn_offsets_from_geometry(tau_s_per_ue, self.tti_ms)

    def get_effective_inactivity_timer(self, ue_id: int) -> int:
        """Get effective inactivity timer for a UE including NTN offset.

        Args:
            ue_id: UE index

        Returns:
            Effective timer value in TTIs
        """
        base = self.drx_config.inactivity_ttis
        if 0 <= ue_id < self.n_ue:
            return base + self._ue_states[ue_id].ntn_rtt_offset_ttis
        return base
