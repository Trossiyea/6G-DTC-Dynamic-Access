"""
NTN-specific timing control per 3GPP TS 38.213/38.214.

Implements timing parameters and Timing Advance (TA) management for
Non-Terrestrial Networks with long propagation delays.

Key Timing Parameters (TS 38.214):
- K0: Slot offset from DCI to PDSCH (0-32 slots)
- K1: Slot offset from PDSCH to HARQ-ACK (0-15 slots, extended for NTN)
- K2: Slot offset from DCI to PUSCH (0-32 slots)

NTN Extensions (TR 38.821):
- Extended K1 values to accommodate satellite RTT
- Common TA from satellite ephemeris
- UE-specific TA adjustment
- Autonomous TA update based on GNSS

Timing Advance:
- Compensates for propagation delay
- Common TA: Derived from satellite ephemeris data
- UE-specific TA: Fine adjustment based on UE location
- Total TA = Common TA + UE-specific TA
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np


# =============================================================================
# Constants
# =============================================================================

# Speed of light (m/s)
SPEED_OF_LIGHT_M_S = 299792458.0

# Maximum timing advance in NR (microseconds)
MAX_TA_US = 2005.0  # TS 38.213 (for terrestrial)

# NTN maximum TA (much larger due to satellite distances)
# LEO: ~2-10 ms one-way, GEO: ~120 ms one-way
MAX_NTN_TA_MS = 300.0  # Allow up to 300 ms for GEO scenarios


# =============================================================================
# Timing Parameter Definitions
# =============================================================================

class NTNScenario(Enum):
    """NTN scenario types affecting timing parameters."""

    TERRESTRIAL = "terrestrial"    # No NTN extensions
    LEO_600KM = "leo_600km"        # LEO at ~600 km altitude
    LEO_1200KM = "leo_1200km"      # LEO at ~1200 km altitude
    MEO = "meo"                    # MEO constellation
    GEO = "geo"                    # Geostationary orbit

    @property
    def typical_delay_ms(self) -> float:
        """Get typical one-way propagation delay for this scenario."""
        delays = {
            NTNScenario.TERRESTRIAL: 0.0,
            NTNScenario.LEO_600KM: 4.0,    # ~4 ms one-way
            NTNScenario.LEO_1200KM: 8.0,   # ~8 ms one-way
            NTNScenario.MEO: 40.0,         # ~40 ms one-way
            NTNScenario.GEO: 120.0,        # ~120 ms one-way
        }
        return delays.get(self, 0.0)

    @property
    def typical_rtt_ms(self) -> float:
        """Get typical round-trip time for this scenario."""
        return 2.0 * self.typical_delay_ms


@dataclass
class K1Table:
    """K1 (PDSCH-to-HARQ-ACK) timing table.

    Per 3GPP TS 38.213 Table 9.2.3-1, extended for NTN.
    """

    # Standard K1 values (slots)
    standard_values: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8]
    )

    # Extended K1 values for NTN (slots)
    ntn_extended_values: List[int] = field(
        default_factory=lambda: [8, 10, 12, 16, 20, 24, 28, 32]
    )

    # Maximum K1 for different scenarios
    max_k1_by_scenario: Dict[NTNScenario, int] = field(default_factory=lambda: {
        NTNScenario.TERRESTRIAL: 8,
        NTNScenario.LEO_600KM: 16,
        NTNScenario.LEO_1200KM: 24,
        NTNScenario.MEO: 64,
        NTNScenario.GEO: 256,
    })

    def get_k1_values(self, scenario: NTNScenario) -> List[int]:
        """Get available K1 values for a scenario."""
        max_k1 = self.max_k1_by_scenario.get(scenario, 8)
        if scenario == NTNScenario.TERRESTRIAL:
            return [k for k in self.standard_values if k <= max_k1]
        else:
            all_values = self.standard_values + self.ntn_extended_values
            return sorted(set(k for k in all_values if k <= max_k1))


# =============================================================================
# NTN Timing Configuration
# =============================================================================

@dataclass
class NTNTimingConfig:
    """NTN timing configuration parameters."""

    # Scenario
    scenario: NTNScenario = NTNScenario.LEO_600KM

    # Slot duration
    slot_duration_ms: float = 1.0  # 1 ms for 15 kHz SCS

    # K0: DCI to PDSCH offset (slots)
    k0_slots: int = 0

    # K1: PDSCH to HARQ-ACK offset (slots)
    k1_slots_base: int = 4
    k1_ntn_extension_slots: int = 0  # Additional slots for NTN

    # K2: DCI to PUSCH offset (slots)
    k2_slots: int = 4

    # Timing Advance
    enable_ta_control: bool = True
    ta_common_ms: float = 0.0       # Common TA from ephemeris
    ta_granularity_us: float = 0.52  # TA command granularity (TS 38.213)

    # NTN-specific
    enable_ntn_timing: bool = True
    enable_ue_autonomous_ta: bool = True  # UE adjusts TA based on GNSS
    ta_update_period_ms: float = 100.0    # How often to update TA

    # HARQ timing
    harq_rtt_scaling: bool = True  # Scale HARQ RTT with propagation delay

    @property
    def k1_effective_slots(self) -> int:
        """Get effective K1 including NTN extension."""
        return self.k1_slots_base + self.k1_ntn_extension_slots

    @classmethod
    def from_config_dict(cls, config: Dict) -> "NTNTimingConfig":
        """Create from flat configuration dictionary."""
        scenario_str = config.get("ntn_scenario", "leo_600km")
        try:
            scenario = NTNScenario(scenario_str)
        except ValueError:
            scenario = NTNScenario.LEO_600KM

        return cls(
            scenario=scenario,
            slot_duration_ms=config.get("tti_ms", 1.0),
            k0_slots=config.get("k0_slots", 0),
            k1_slots_base=config.get("k1_slots_base", 4),
            k1_ntn_extension_slots=config.get("k1_ntn_extension_slots", 0),
            k2_slots=config.get("k2_slots", 4),
            enable_ta_control=config.get("enable_ta_control", True),
            ta_common_ms=config.get("ta_common_ms", 0.0),
            ta_granularity_us=config.get("ta_granularity_us", 0.52),
            enable_ntn_timing=config.get("ntn_timing_adaptation", True),
            enable_ue_autonomous_ta=config.get("enable_ue_autonomous_ta", True),
            ta_update_period_ms=config.get("ta_update_period_ms", 100.0),
            harq_rtt_scaling=config.get("harq_rtt_scaling", True),
        )

    @classmethod
    def for_scenario(cls, scenario: NTNScenario, slot_duration_ms: float = 1.0) -> "NTNTimingConfig":
        """Create configuration preset for a specific NTN scenario."""
        config = cls(scenario=scenario, slot_duration_ms=slot_duration_ms)

        # Set K1 extension based on scenario
        rtt_ms = scenario.typical_rtt_ms
        rtt_slots = int(np.ceil(rtt_ms / slot_duration_ms))
        config.k1_ntn_extension_slots = max(0, rtt_slots - config.k1_slots_base)

        return config


# =============================================================================
# Per-UE Timing State
# =============================================================================

@dataclass
class UETimingState:
    """Per-UE timing state."""

    ue_id: int

    # Propagation delay (one-way, seconds)
    propagation_delay_s: float = 0.0

    # Timing Advance
    ta_common_us: float = 0.0      # Common TA (from ephemeris)
    ta_ue_specific_us: float = 0.0  # UE-specific TA adjustment
    ta_total_us: float = 0.0       # Total TA = common + UE-specific

    # TA command history
    last_ta_command_tti: int = -1
    ta_adjustment_us: float = 0.0  # Pending TA adjustment

    # Statistics
    ta_updates: int = 0
    max_delay_experienced_ms: float = 0.0

    @property
    def rtt_s(self) -> float:
        """Get round-trip time in seconds."""
        return 2.0 * self.propagation_delay_s

    @property
    def rtt_ms(self) -> float:
        """Get round-trip time in milliseconds."""
        return self.rtt_s * 1000.0

    def update_from_geometry(self, slant_range_m: float) -> None:
        """Update timing from slant range.

        Args:
            slant_range_m: Slant range to satellite in meters
        """
        self.propagation_delay_s = slant_range_m / SPEED_OF_LIGHT_M_S
        self.max_delay_experienced_ms = max(
            self.max_delay_experienced_ms,
            self.propagation_delay_s * 1000.0
        )


# =============================================================================
# Timing Advance Controller
# =============================================================================

class TimingAdvanceController:
    """Manages Timing Advance for all UEs.

    Handles both common TA (from satellite ephemeris) and
    UE-specific TA adjustments.
    """

    def __init__(
        self,
        n_ue: int,
        config: Optional[Dict] = None,
        timing_config: Optional[NTNTimingConfig] = None,
    ):
        """Initialize TA controller.

        Args:
            n_ue: Number of UEs
            config: Flat configuration dictionary
            timing_config: Pre-built timing configuration
        """
        self.n_ue = n_ue

        # Build timing config
        if timing_config is not None:
            self.timing_config = timing_config
        elif config is not None:
            self.timing_config = NTNTimingConfig.from_config_dict(config)
        else:
            self.timing_config = NTNTimingConfig()

        # Per-UE timing states
        self._ue_states: List[UETimingState] = [
            UETimingState(ue_id=i) for i in range(n_ue)
        ]

        # Common TA (applies to all UEs in cell)
        self._common_ta_us = self.timing_config.ta_common_ms * 1000.0

        # Current TTI
        self._current_tti = 0

        # Statistics
        self._ta_commands_sent = 0

    # -------------------------------------------------------------------------
    # TA Computation
    # -------------------------------------------------------------------------

    def compute_ta_from_delay(self, propagation_delay_s: float) -> float:
        """Compute TA from propagation delay.

        TA compensates for round-trip time.

        Args:
            propagation_delay_s: One-way propagation delay in seconds

        Returns:
            TA value in microseconds
        """
        rtt_us = 2.0 * propagation_delay_s * 1e6
        return rtt_us

    def update_common_ta(self, ta_ms: float) -> None:
        """Update common TA for the cell.

        Args:
            ta_ms: Common TA in milliseconds
        """
        self._common_ta_us = ta_ms * 1000.0

        # Update all UE states
        for state in self._ue_states:
            state.ta_common_us = self._common_ta_us
            state.ta_total_us = state.ta_common_us + state.ta_ue_specific_us

    def update_ue_ta_from_geometry(
        self,
        tau_s_per_ue: np.ndarray,
        reference_delay_s: Optional[float] = None,
    ) -> None:
        """Update UE-specific TA from propagation delays.

        Args:
            tau_s_per_ue: One-way propagation delay per UE [N_UE] in seconds
            reference_delay_s: Reference delay for common TA (e.g., cell center)
        """
        if reference_delay_s is not None:
            # Update common TA based on reference point
            self._common_ta_us = self.compute_ta_from_delay(reference_delay_s)

        for ue in range(min(self.n_ue, len(tau_s_per_ue))):
            state = self._ue_states[ue]

            # Update propagation delay
            state.propagation_delay_s = float(tau_s_per_ue[ue])

            # Compute TA
            total_ta = self.compute_ta_from_delay(state.propagation_delay_s)

            if reference_delay_s is not None:
                # UE-specific = total - common
                state.ta_common_us = self._common_ta_us
                state.ta_ue_specific_us = total_ta - self._common_ta_us
            else:
                # No common TA reference, all in UE-specific
                state.ta_common_us = 0.0
                state.ta_ue_specific_us = total_ta

            state.ta_total_us = total_ta
            state.ta_updates += 1

    def get_ta_for_ue(self, ue_id: int) -> float:
        """Get total TA for a UE.

        Args:
            ue_id: UE index

        Returns:
            Total TA in microseconds
        """
        if 0 <= ue_id < self.n_ue:
            return self._ue_states[ue_id].ta_total_us
        return 0.0

    def get_ta_for_ue_ms(self, ue_id: int) -> float:
        """Get total TA for a UE in milliseconds.

        Args:
            ue_id: UE index

        Returns:
            Total TA in milliseconds
        """
        return self.get_ta_for_ue(ue_id) / 1000.0

    def get_all_ta_ms(self) -> np.ndarray:
        """Get TA for all UEs.

        Returns:
            Array of TA values in milliseconds [N_UE]
        """
        return np.array([
            state.ta_total_us / 1000.0 for state in self._ue_states
        ])

    # -------------------------------------------------------------------------
    # TA Commands
    # -------------------------------------------------------------------------

    def generate_ta_command(self, ue_id: int, target_ta_us: float) -> int:
        """Generate TA command for a UE.

        Args:
            ue_id: UE index
            target_ta_us: Target TA in microseconds

        Returns:
            TA command value (quantized)
        """
        if not (0 <= ue_id < self.n_ue):
            return 0

        state = self._ue_states[ue_id]
        delta_ta = target_ta_us - state.ta_total_us

        # Quantize to TA granularity
        granularity = self.timing_config.ta_granularity_us
        ta_command = int(round(delta_ta / granularity))

        # Clamp to valid range (-31 to 31 for NR)
        ta_command = max(-31, min(31, ta_command))

        if ta_command != 0:
            self._ta_commands_sent += 1
            state.last_ta_command_tti = self._current_tti
            state.ta_adjustment_us = ta_command * granularity

        return ta_command

    def apply_ta_command(self, ue_id: int, ta_command: int) -> None:
        """Apply TA command to UE state.

        Args:
            ue_id: UE index
            ta_command: TA command value
        """
        if not (0 <= ue_id < self.n_ue):
            return

        state = self._ue_states[ue_id]
        adjustment = ta_command * self.timing_config.ta_granularity_us
        state.ta_ue_specific_us += adjustment
        state.ta_total_us = state.ta_common_us + state.ta_ue_specific_us

    # -------------------------------------------------------------------------
    # Query Interface
    # -------------------------------------------------------------------------

    def get_propagation_delay_ms(self, ue_id: int) -> float:
        """Get propagation delay for a UE.

        Args:
            ue_id: UE index

        Returns:
            One-way propagation delay in milliseconds
        """
        if 0 <= ue_id < self.n_ue:
            return self._ue_states[ue_id].propagation_delay_s * 1000.0
        return 0.0

    def get_rtt_ms(self, ue_id: int) -> float:
        """Get round-trip time for a UE.

        Args:
            ue_id: UE index

        Returns:
            RTT in milliseconds
        """
        if 0 <= ue_id < self.n_ue:
            return self._ue_states[ue_id].rtt_ms
        return 0.0

    def get_all_rtt_ms(self) -> np.ndarray:
        """Get RTT for all UEs.

        Returns:
            Array of RTT values in milliseconds [N_UE]
        """
        return np.array([state.rtt_ms for state in self._ue_states])

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Get TA controller statistics."""
        delays = [s.propagation_delay_s * 1000.0 for s in self._ue_states]
        tas = [s.ta_total_us / 1000.0 for s in self._ue_states]

        return {
            "common_ta_ms": self._common_ta_us / 1000.0,
            "min_delay_ms": min(delays) if delays else 0.0,
            "max_delay_ms": max(delays) if delays else 0.0,
            "mean_delay_ms": np.mean(delays) if delays else 0.0,
            "min_ta_ms": min(tas) if tas else 0.0,
            "max_ta_ms": max(tas) if tas else 0.0,
            "mean_ta_ms": np.mean(tas) if tas else 0.0,
            "ta_commands_sent": self._ta_commands_sent,
            "scenario": self.timing_config.scenario.value,
        }

    def reset(self) -> None:
        """Reset TA controller state."""
        self._ue_states = [UETimingState(ue_id=i) for i in range(self.n_ue)]
        self._ta_commands_sent = 0
        self._current_tti = 0


# =============================================================================
# HARQ Timing Adapter
# =============================================================================

class HARQTimingAdapter:
    """Adapts HARQ timing parameters for NTN scenarios.

    Adjusts K1 (PDSCH-to-HARQ-ACK) and HARQ process timing based on
    satellite propagation delay.
    """

    def __init__(
        self,
        n_ue: int,
        config: Optional[Dict] = None,
        timing_config: Optional[NTNTimingConfig] = None,
    ):
        """Initialize HARQ timing adapter.

        Args:
            n_ue: Number of UEs
            config: Flat configuration dictionary
            timing_config: Pre-built timing configuration
        """
        self.n_ue = n_ue

        if timing_config is not None:
            self.timing_config = timing_config
        elif config is not None:
            self.timing_config = NTNTimingConfig.from_config_dict(config)
        else:
            self.timing_config = NTNTimingConfig()

        # K1 table
        self.k1_table = K1Table()

        # Per-UE K1 values (may vary by UE location)
        self._ue_k1_slots: np.ndarray = np.full(
            n_ue, self.timing_config.k1_effective_slots, dtype=int
        )

        # Per-UE HARQ RTT (slots)
        self._ue_harq_rtt_slots: np.ndarray = np.zeros(n_ue, dtype=int)

    def update_from_geometry(
        self,
        tau_s_per_ue: np.ndarray,
        slot_duration_ms: Optional[float] = None,
    ) -> None:
        """Update HARQ timing from propagation delays.

        Args:
            tau_s_per_ue: One-way propagation delay per UE [N_UE] in seconds
            slot_duration_ms: Slot duration in ms (default from config)
        """
        slot_ms = slot_duration_ms or self.timing_config.slot_duration_ms

        for ue in range(min(self.n_ue, len(tau_s_per_ue))):
            rtt_ms = 2.0 * tau_s_per_ue[ue] * 1000.0
            rtt_slots = int(np.ceil(rtt_ms / slot_ms))

            # K1 must cover at least RTT + processing
            base_k1 = self.timing_config.k1_slots_base
            min_k1 = rtt_slots + base_k1
            max_k1 = self.k1_table.max_k1_by_scenario.get(
                self.timing_config.scenario, 256
            )

            self._ue_k1_slots[ue] = min(max(min_k1, base_k1), max_k1)
            self._ue_harq_rtt_slots[ue] = rtt_slots

    def get_k1_for_ue(self, ue_id: int) -> int:
        """Get K1 value for a UE.

        Args:
            ue_id: UE index

        Returns:
            K1 in slots
        """
        if 0 <= ue_id < self.n_ue:
            return int(self._ue_k1_slots[ue_id])
        return self.timing_config.k1_effective_slots

    def get_harq_rtt_slots(self, ue_id: int) -> int:
        """Get HARQ RTT for a UE.

        Args:
            ue_id: UE index

        Returns:
            HARQ RTT in slots
        """
        if 0 <= ue_id < self.n_ue:
            return int(self._ue_harq_rtt_slots[ue_id])
        return 0

    def get_effective_harq_processes(
        self,
        ue_id: int,
        base_processes: int = 16
    ) -> int:
        """Get effective number of HARQ processes needed for a UE.

        In NTN, more HARQ processes may be needed due to long RTT.

        Args:
            ue_id: UE index
            base_processes: Base number of processes

        Returns:
            Effective number of HARQ processes
        """
        rtt_slots = self.get_harq_rtt_slots(ue_id)
        # Need enough processes to cover RTT
        min_processes = max(base_processes, rtt_slots // 2 + 1)
        return min(min_processes, 32)  # Max 32 processes in NR

    def get_pdsch_to_ack_delay_ms(self, ue_id: int) -> float:
        """Get PDSCH to ACK delay for a UE.

        Args:
            ue_id: UE index

        Returns:
            Delay in milliseconds
        """
        k1 = self.get_k1_for_ue(ue_id)
        return k1 * self.timing_config.slot_duration_ms


# =============================================================================
# Scheduling Timing Manager
# =============================================================================

class SchedulingTimingManager:
    """Manages scheduling timing parameters (K0/K1/K2) for NTN.

    Coordinates timing between DCI, PDSCH, PUSCH, and HARQ feedback.
    """

    def __init__(
        self,
        n_ue: int,
        config: Optional[Dict] = None,
    ):
        """Initialize scheduling timing manager.

        Args:
            n_ue: Number of UEs
            config: Flat configuration dictionary
        """
        self.n_ue = n_ue
        self.config = config or {}

        # Timing config
        self.timing_config = NTNTimingConfig.from_config_dict(self.config)

        # Sub-controllers
        self.ta_controller = TimingAdvanceController(
            n_ue, config, self.timing_config
        )
        self.harq_adapter = HARQTimingAdapter(
            n_ue, config, self.timing_config
        )

        # Per-UE timing parameters
        self._ue_k0: np.ndarray = np.full(n_ue, self.timing_config.k0_slots, dtype=int)
        self._ue_k2: np.ndarray = np.full(n_ue, self.timing_config.k2_slots, dtype=int)

    def update_from_geometry(
        self,
        tau_s_per_ue: np.ndarray,
        reference_delay_s: Optional[float] = None,
    ) -> None:
        """Update all timing from geometry.

        Args:
            tau_s_per_ue: One-way propagation delay per UE [N_UE] in seconds
            reference_delay_s: Reference delay for common TA
        """
        # Update TA
        self.ta_controller.update_ue_ta_from_geometry(tau_s_per_ue, reference_delay_s)

        # Update HARQ timing
        self.harq_adapter.update_from_geometry(tau_s_per_ue)

    # -------------------------------------------------------------------------
    # K-value Query Interface
    # -------------------------------------------------------------------------

    def get_k0(self, ue_id: int) -> int:
        """Get K0 (DCI to PDSCH offset) for a UE."""
        if 0 <= ue_id < self.n_ue:
            return int(self._ue_k0[ue_id])
        return self.timing_config.k0_slots

    def get_k1(self, ue_id: int) -> int:
        """Get K1 (PDSCH to HARQ-ACK offset) for a UE."""
        return self.harq_adapter.get_k1_for_ue(ue_id)

    def get_k2(self, ue_id: int) -> int:
        """Get K2 (DCI to PUSCH offset) for a UE."""
        if 0 <= ue_id < self.n_ue:
            return int(self._ue_k2[ue_id])
        return self.timing_config.k2_slots

    def get_all_k1(self) -> np.ndarray:
        """Get K1 for all UEs."""
        return np.array([self.get_k1(ue) for ue in range(self.n_ue)])

    # -------------------------------------------------------------------------
    # Scheduling Constraints
    # -------------------------------------------------------------------------

    def get_earliest_pdsch_slot(self, ue_id: int, dci_slot: int) -> int:
        """Get earliest PDSCH slot after DCI.

        Args:
            ue_id: UE index
            dci_slot: Slot containing DCI

        Returns:
            Earliest PDSCH slot
        """
        return dci_slot + self.get_k0(ue_id)

    def get_harq_ack_slot(self, ue_id: int, pdsch_slot: int) -> int:
        """Get expected HARQ ACK slot after PDSCH.

        Args:
            ue_id: UE index
            pdsch_slot: Slot containing PDSCH

        Returns:
            Expected HARQ ACK slot
        """
        return pdsch_slot + self.get_k1(ue_id)

    def get_earliest_retx_slot(self, ue_id: int, nack_slot: int) -> int:
        """Get earliest retransmission slot after NACK.

        Args:
            ue_id: UE index
            nack_slot: Slot containing NACK

        Returns:
            Earliest retransmission slot
        """
        # After NACK, need K0 for new DCI
        return nack_slot + self.get_k0(ue_id)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Get timing manager statistics."""
        ta_stats = self.ta_controller.get_statistics()
        k1_values = self.get_all_k1()

        return {
            "scenario": self.timing_config.scenario.value,
            "k0_slots": int(self.timing_config.k0_slots),
            "k1_base_slots": int(self.timing_config.k1_slots_base),
            "k2_slots": int(self.timing_config.k2_slots),
            "k1_min": int(np.min(k1_values)),
            "k1_max": int(np.max(k1_values)),
            "k1_mean": float(np.mean(k1_values)),
            "ta_stats": ta_stats,
        }

    def reset(self) -> None:
        """Reset timing manager."""
        self.ta_controller.reset()
        self._ue_k0 = np.full(self.n_ue, self.timing_config.k0_slots, dtype=int)
        self._ue_k2 = np.full(self.n_ue, self.timing_config.k2_slots, dtype=int)
