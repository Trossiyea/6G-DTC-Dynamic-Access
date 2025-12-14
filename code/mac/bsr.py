"""
Buffer Status Report (BSR) management per 3GPP TS 38.321.

Implements BSR tables, Logical Channel Group (LCG) mapping, and BSR
report generation for MAC-layer buffer status reporting to scheduler.

3GPP TS 38.321 defines:
- Table 6.1.3.1-1: Buffer size levels for 5-bit BSR (Short BSR)
- Table 6.1.3.1-2: Buffer size levels for 8-bit BSR (Long BSR)

BSR Types:
- Short BSR: Single LCG with data, 5-bit buffer size field
- Short Truncated BSR: Multiple LCGs, reports highest priority LCG
- Long BSR: All LCGs with data, 8-bit buffer size fields
- Long Truncated BSR: Truncated when MAC PDU size limited

Trigger Conditions (TS 38.321 §5.4.5):
- Regular BSR: New data arrives for LCG with higher priority than current
- Periodic BSR: Configured periodic timer expires
- Padding BSR: Padding available in MAC PDU
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from traffic import BufferManager


# =============================================================================
# 3GPP TS 38.321 BSR Tables
# =============================================================================

# Table 6.1.3.1-1: Buffer size levels (in bytes) for 5-bit Buffer Size field
# Index 0 = 0 bytes, Index 31 = >150,000 bytes
BSR_TABLE_5BIT: List[int] = [
    0, 10, 14, 20, 28, 38, 53, 74,
    102, 142, 198, 276, 384, 535, 745, 1038,
    1446, 2014, 2806, 3909, 5446, 7587, 10570, 14726,
    20516, 28581, 39818, 55474, 77284, 107669, 150000, 150001,  # 31 = >150000
]

# Table 6.1.3.1-2: Buffer size levels (in bytes) for 8-bit Buffer Size field
# Index 0 = 0 bytes, Index 255 = >81,338,368 bytes
# Simplified version with key breakpoints (full table has 256 entries)
def _generate_8bit_table() -> List[int]:
    """Generate full 8-bit BSR table per 3GPP TS 38.321 Table 6.1.3.1-2."""
    table = [0]

    # Ranges with different growth rates (approximated from spec)
    # 0-63: linear growth to ~1500 bytes
    for i in range(1, 64):
        table.append(int(10 + i * 24))

    # 64-127: exponential growth to ~30,000 bytes
    for i in range(64, 128):
        table.append(int(1500 * (1.05 ** (i - 64))))

    # 128-191: faster exponential growth to ~1,000,000 bytes
    for i in range(128, 192):
        table.append(int(30000 * (1.08 ** (i - 128))))

    # 192-254: rapid growth to 81,338,368 bytes
    for i in range(192, 255):
        table.append(int(1000000 * (1.12 ** (i - 192))))

    # Index 255 = more than max
    table.append(81338369)

    return table


BSR_TABLE_8BIT: List[int] = _generate_8bit_table()


class BSRTable:
    """BSR lookup table for buffer size encoding/decoding.

    Provides methods to convert between actual buffer sizes and BSR indices.
    """

    def __init__(self, bits: int = 8):
        """Initialize BSR table.

        Args:
            bits: Table size (5 or 8 bits)
        """
        if bits == 5:
            self.table = BSR_TABLE_5BIT
            self.max_index = 31
        elif bits == 8:
            self.table = BSR_TABLE_8BIT
            self.max_index = 255
        else:
            raise ValueError(f"BSR table must be 5 or 8 bits, got {bits}")

        self.bits = bits

    def bytes_to_index(self, buffer_bytes: int) -> int:
        """Convert buffer size to BSR index.

        Uses binary search to find the appropriate index.

        Args:
            buffer_bytes: Buffer size in bytes

        Returns:
            BSR index (0 to max_index)
        """
        if buffer_bytes <= 0:
            return 0

        # Binary search for appropriate index
        low, high = 0, self.max_index
        while low < high:
            mid = (low + high + 1) // 2
            if self.table[mid] <= buffer_bytes:
                low = mid
            else:
                high = mid - 1

        return low

    def index_to_bytes(self, index: int) -> int:
        """Convert BSR index to buffer size.

        Args:
            index: BSR index

        Returns:
            Buffer size in bytes (upper bound)
        """
        index = max(0, min(index, self.max_index))
        return self.table[index]

    def index_to_bytes_range(self, index: int) -> Tuple[int, int]:
        """Get buffer size range for a BSR index.

        Args:
            index: BSR index

        Returns:
            (min_bytes, max_bytes) tuple
        """
        index = max(0, min(index, self.max_index))
        min_bytes = self.table[index]

        if index < self.max_index:
            max_bytes = self.table[index + 1] - 1
        else:
            max_bytes = float('inf')

        return (min_bytes, max_bytes)


# =============================================================================
# Logical Channel Group (LCG) Management
# =============================================================================

def qci_to_lcg(qci: int) -> int:
    """Map QCI/5QI to Logical Channel Group.

    Default mapping based on QoS characteristics:
    - LCG 0: Signaling (QCI 5)
    - LCG 1: Voice/VoIP (QCI 1, 65, 66, 67)
    - LCG 2: Video/Real-time (QCI 2, 3, 4, 7)
    - LCG 3: URLLC (QCI 80-85)
    - LCG 4-6: Reserved for future use
    - LCG 7: Best effort (QCI 6, 8, 9, others)

    Args:
        qci: 5QI/QCI value

    Returns:
        LCG ID (0-7)
    """
    if qci == 5:  # IMS Signaling
        return 0
    elif qci in (1, 65, 66, 67):  # Voice
        return 1
    elif qci in (2, 3, 4, 7):  # Video/Real-time
        return 2
    elif 80 <= qci <= 85:  # URLLC
        return 3
    else:  # Best effort (6, 8, 9, etc.)
        return 7


class LCGManager:
    """Manages Logical Channel Group configuration and mapping.

    LCGs aggregate logical channels for BSR reporting, allowing the UE
    to report buffer status per group rather than per channel.
    """

    # Default LCG priority (lower = higher priority)
    DEFAULT_LCG_PRIORITY: Dict[int, int] = {
        0: 1,   # Signaling - highest priority
        1: 2,   # Voice
        2: 3,   # Video
        3: 0,   # URLLC - highest among data
        4: 4,   # Reserved
        5: 5,   # Reserved
        6: 6,   # Reserved
        7: 7,   # Best effort - lowest priority
    }

    def __init__(self, custom_mapping: Optional[Dict[int, int]] = None):
        """Initialize LCG manager.

        Args:
            custom_mapping: Optional QCI-to-LCG mapping override
        """
        self._qci_to_lcg = custom_mapping or {}
        self._lcg_priority = dict(self.DEFAULT_LCG_PRIORITY)

    def get_lcg(self, qci: int) -> int:
        """Get LCG for a QCI.

        Args:
            qci: 5QI/QCI value

        Returns:
            LCG ID (0-7)
        """
        if qci in self._qci_to_lcg:
            return self._qci_to_lcg[qci]
        return qci_to_lcg(qci)

    def get_priority(self, lcg: int) -> int:
        """Get priority for an LCG.

        Args:
            lcg: LCG ID

        Returns:
            Priority value (lower = higher priority)
        """
        return self._lcg_priority.get(lcg, 99)

    def get_highest_priority_lcg(self, lcgs: List[int]) -> int:
        """Get highest priority LCG from a list.

        Args:
            lcgs: List of LCG IDs

        Returns:
            LCG ID with highest priority
        """
        if not lcgs:
            return 7
        return min(lcgs, key=lambda x: self.get_priority(x))

    def set_qci_mapping(self, qci: int, lcg: int) -> None:
        """Set custom QCI-to-LCG mapping.

        Args:
            qci: 5QI/QCI value
            lcg: LCG ID (0-7)
        """
        if not 0 <= lcg <= 7:
            raise ValueError(f"LCG must be 0-7, got {lcg}")
        self._qci_to_lcg[qci] = lcg

    def set_lcg_priority(self, lcg: int, priority: int) -> None:
        """Set priority for an LCG.

        Args:
            lcg: LCG ID (0-7)
            priority: Priority value (lower = higher priority)
        """
        self._lcg_priority[lcg] = priority


# =============================================================================
# BSR Report Data Structures
# =============================================================================

class BSRTrigger(Enum):
    """BSR trigger conditions per 3GPP TS 38.321 §5.4.5."""

    REGULAR = "regular"          # New high-priority data arrival
    PERIODIC = "periodic"        # Periodic BSR timer expired
    PADDING = "padding"          # Padding available in MAC PDU
    RETX_PERIODIC = "retx_periodic"  # Retransmission BSR timer


@dataclass
class BSRReport:
    """Single BSR report for one LCG.

    Represents the buffer status reported by a UE for a specific
    Logical Channel Group at a specific TTI.
    """

    ue_id: int
    lcg_id: int
    buffer_size_bytes: int
    bsr_index: int              # 5-bit or 8-bit index
    timestamp_tti: int
    trigger: BSRTrigger = BSRTrigger.REGULAR
    is_truncated: bool = False  # True if truncated BSR

    @property
    def is_empty(self) -> bool:
        """Check if this LCG has no data."""
        return self.buffer_size_bytes == 0

    def __repr__(self) -> str:
        return (f"BSRReport(ue={self.ue_id}, lcg={self.lcg_id}, "
                f"bytes={self.buffer_size_bytes}, idx={self.bsr_index}, "
                f"trigger={self.trigger.value})")


@dataclass
class UEBSRState:
    """Per-UE BSR state tracking.

    Maintains the latest BSR reports per LCG and related timers.
    """

    ue_id: int
    lcg_buffers: Dict[int, int] = field(default_factory=dict)  # LCG -> bytes
    last_report_tti: int = -1
    periodic_timer_remaining: int = 0
    retx_timer_remaining: int = 0

    # Pending BSR flags
    regular_bsr_pending: bool = False
    periodic_bsr_pending: bool = False

    def get_total_buffer_bytes(self) -> int:
        """Get total buffer across all LCGs."""
        return sum(self.lcg_buffers.values())

    def get_non_empty_lcgs(self) -> List[int]:
        """Get list of LCGs with data."""
        return [lcg for lcg, size in self.lcg_buffers.items() if size > 0]

    def clear_buffers(self) -> None:
        """Clear all LCG buffers (e.g., after full BSR sent)."""
        self.lcg_buffers.clear()
        self.regular_bsr_pending = False
        self.periodic_bsr_pending = False


# =============================================================================
# BSR Manager
# =============================================================================

class BSRManager:
    """Manages Buffer Status Reports for all UEs.

    Provides interface between traffic layer buffers and scheduler,
    generating BSR reports based on buffer states and trigger conditions.
    """

    def __init__(
        self,
        n_ue: int,
        config: Optional[Dict] = None,
        lcg_manager: Optional[LCGManager] = None,
    ):
        """Initialize BSR manager.

        Args:
            n_ue: Number of UEs
            config: Configuration dictionary
            lcg_manager: Optional custom LCG manager
        """
        self.n_ue = n_ue
        self.config = config or {}
        self.lcg_manager = lcg_manager or LCGManager()

        # Configuration
        self.bsr_bits = self.config.get("bsr_table_bits", 8)
        self.periodic_timer_ttis = int(
            self.config.get("bsr_periodic_timer_ms", 20.0) /
            self.config.get("tti_ms", 1.0)
        )
        self.retx_timer_ttis = int(
            self.config.get("bsr_retx_timer_ms", 10.0) /
            self.config.get("tti_ms", 1.0)
        )
        self.enable_bsr = self.config.get("enable_bsr", True)

        # BSR table
        self.bsr_table = BSRTable(bits=self.bsr_bits)

        # Per-UE BSR state
        self._ue_states: List[UEBSRState] = [
            UEBSRState(ue_id=i) for i in range(n_ue)
        ]

        # Latest BSR reports for scheduler
        self._latest_reports: Dict[int, List[BSRReport]] = {}

    def update_from_buffer_manager(
        self,
        buffer_mgr: "BufferManager",
        current_tti: int,
    ) -> List[BSRReport]:
        """Update BSR state from traffic layer buffer manager.

        Synchronizes BSR state with actual buffer occupancy and
        generates BSR reports based on trigger conditions.

        Args:
            buffer_mgr: Traffic layer BufferManager instance
            current_tti: Current TTI index

        Returns:
            List of BSR reports generated this TTI
        """
        if not self.enable_bsr:
            return []

        all_reports = []

        for ue_id in range(min(self.n_ue, buffer_mgr.n_ue)):
            ue_state = self._ue_states[ue_id]
            ue_buffer = buffer_mgr.buffers[ue_id]

            # Build LCG -> bytes mapping from buffer state
            new_lcg_buffers: Dict[int, int] = {}

            for bearer_id, queue in ue_buffer.queues.items():
                if not queue:
                    continue

                # Get representative QCI from first packet
                first_pkt = queue[0]
                lcg = self.lcg_manager.get_lcg(first_pkt.qci)

                # Sum buffer size for this bearer
                bearer_bytes = sum(pkt.remaining_bits for pkt in queue) // 8
                new_lcg_buffers[lcg] = new_lcg_buffers.get(lcg, 0) + bearer_bytes

            # Detect trigger conditions
            trigger = self._check_bsr_trigger(ue_state, new_lcg_buffers, current_tti)

            # Update state
            old_lcg_buffers = ue_state.lcg_buffers.copy()
            ue_state.lcg_buffers = new_lcg_buffers

            # Generate BSR reports if triggered
            if trigger is not None:
                reports = self._generate_bsr_reports(
                    ue_id, new_lcg_buffers, current_tti, trigger
                )
                all_reports.extend(reports)
                ue_state.last_report_tti = current_tti

                # Cache latest reports
                self._latest_reports[ue_id] = reports

            # Update timers
            self._update_timers(ue_state, current_tti)

        return all_reports

    def _check_bsr_trigger(
        self,
        ue_state: UEBSRState,
        new_lcg_buffers: Dict[int, int],
        current_tti: int,
    ) -> Optional[BSRTrigger]:
        """Check if BSR should be triggered.

        Args:
            ue_state: UE's BSR state
            new_lcg_buffers: New buffer state per LCG
            current_tti: Current TTI

        Returns:
            BSRTrigger if triggered, None otherwise
        """
        # Check periodic timer
        if ue_state.periodic_timer_remaining <= 0:
            if ue_state.get_total_buffer_bytes() > 0 or sum(new_lcg_buffers.values()) > 0:
                return BSRTrigger.PERIODIC

        # Check for new high-priority data (Regular BSR trigger)
        old_lcgs = set(ue_state.lcg_buffers.keys())
        new_lcgs_with_data = {lcg for lcg, size in new_lcg_buffers.items() if size > 0}

        # New LCG with data
        new_lcgs = new_lcgs_with_data - old_lcgs
        if new_lcgs:
            # Check if new LCG has higher priority
            if old_lcgs:
                old_highest = self.lcg_manager.get_highest_priority_lcg(list(old_lcgs))
                new_highest = self.lcg_manager.get_highest_priority_lcg(list(new_lcgs))
                if self.lcg_manager.get_priority(new_highest) < self.lcg_manager.get_priority(old_highest):
                    return BSRTrigger.REGULAR
            else:
                return BSRTrigger.REGULAR

        # Significant buffer increase (configurable threshold)
        for lcg, new_size in new_lcg_buffers.items():
            old_size = ue_state.lcg_buffers.get(lcg, 0)
            if new_size > old_size * 1.5 and new_size - old_size > 1000:  # 50% increase and 1KB
                return BSRTrigger.REGULAR

        return None

    def _generate_bsr_reports(
        self,
        ue_id: int,
        lcg_buffers: Dict[int, int],
        current_tti: int,
        trigger: BSRTrigger,
    ) -> List[BSRReport]:
        """Generate BSR reports for a UE.

        Args:
            ue_id: UE index
            lcg_buffers: Buffer size per LCG
            current_tti: Current TTI
            trigger: Trigger condition

        Returns:
            List of BSRReport objects
        """
        reports = []

        for lcg_id, buffer_bytes in sorted(lcg_buffers.items()):
            bsr_index = self.bsr_table.bytes_to_index(buffer_bytes)

            report = BSRReport(
                ue_id=ue_id,
                lcg_id=lcg_id,
                buffer_size_bytes=buffer_bytes,
                bsr_index=bsr_index,
                timestamp_tti=current_tti,
                trigger=trigger,
            )
            reports.append(report)

        # If no data in any LCG, generate single empty report
        if not reports:
            reports.append(BSRReport(
                ue_id=ue_id,
                lcg_id=7,  # Default best-effort
                buffer_size_bytes=0,
                bsr_index=0,
                timestamp_tti=current_tti,
                trigger=trigger,
            ))

        return reports

    def _update_timers(self, ue_state: UEBSRState, current_tti: int) -> None:
        """Update BSR timers.

        Args:
            ue_state: UE's BSR state
            current_tti: Current TTI
        """
        # Decrement periodic timer
        if ue_state.periodic_timer_remaining > 0:
            ue_state.periodic_timer_remaining -= 1
        else:
            # Reset periodic timer
            ue_state.periodic_timer_remaining = self.periodic_timer_ttis

        # Decrement retx timer
        if ue_state.retx_timer_remaining > 0:
            ue_state.retx_timer_remaining -= 1

    # -------------------------------------------------------------------------
    # Public API for Scheduler Integration
    # -------------------------------------------------------------------------

    def get_pending_data_bytes(self, ue_id: int) -> int:
        """Get total pending data for a UE across all LCGs.

        Args:
            ue_id: UE index

        Returns:
            Total buffer size in bytes
        """
        if not (0 <= ue_id < self.n_ue):
            return 0
        return self._ue_states[ue_id].get_total_buffer_bytes()

    def get_pending_data_per_lcg(self, ue_id: int) -> Dict[int, int]:
        """Get pending data per LCG for a UE.

        Args:
            ue_id: UE index

        Returns:
            Dictionary mapping LCG ID to buffer size in bytes
        """
        if not (0 <= ue_id < self.n_ue):
            return {}
        return dict(self._ue_states[ue_id].lcg_buffers)

    def get_all_pending_data(self) -> np.ndarray:
        """Get pending data for all UEs.

        Returns:
            Array of buffer sizes in bytes [N_UE]
        """
        return np.array([
            self._ue_states[ue].get_total_buffer_bytes()
            for ue in range(self.n_ue)
        ])

    def get_latest_reports(self, ue_id: int) -> List[BSRReport]:
        """Get latest BSR reports for a UE.

        Args:
            ue_id: UE index

        Returns:
            List of most recent BSRReport objects
        """
        return self._latest_reports.get(ue_id, [])

    def get_non_empty_ues(self) -> List[int]:
        """Get list of UEs with non-empty buffers.

        Returns:
            List of UE IDs with pending data
        """
        return [
            ue for ue in range(self.n_ue)
            if self._ue_states[ue].get_total_buffer_bytes() > 0
        ]

    def get_ues_with_high_priority_data(self, priority_threshold: int = 3) -> List[int]:
        """Get UEs with high-priority LCG data.

        Args:
            priority_threshold: Include LCGs with priority <= this value

        Returns:
            List of UE IDs with high-priority data
        """
        ues = []
        for ue in range(self.n_ue):
            for lcg in self._ue_states[ue].get_non_empty_lcgs():
                if self.lcg_manager.get_priority(lcg) <= priority_threshold:
                    ues.append(ue)
                    break
        return ues

    def dequeue_bytes(self, ue_id: int, bytes_transmitted: int) -> None:
        """Notify BSR manager of transmitted bytes.

        Called by scheduler/HARQ when bytes are successfully delivered.
        Updates internal BSR state to reflect reduced buffer.

        Args:
            ue_id: UE index
            bytes_transmitted: Number of bytes successfully transmitted
        """
        if not (0 <= ue_id < self.n_ue):
            return

        ue_state = self._ue_states[ue_id]
        remaining = bytes_transmitted

        # Deduct from LCGs in priority order
        for lcg in sorted(ue_state.lcg_buffers.keys(),
                         key=lambda x: self.lcg_manager.get_priority(x)):
            if remaining <= 0:
                break

            lcg_bytes = ue_state.lcg_buffers[lcg]
            deduct = min(remaining, lcg_bytes)
            ue_state.lcg_buffers[lcg] = lcg_bytes - deduct
            remaining -= deduct

            # Remove empty LCGs
            if ue_state.lcg_buffers[lcg] <= 0:
                del ue_state.lcg_buffers[lcg]

    def reset(self) -> None:
        """Reset all BSR state."""
        self._ue_states = [UEBSRState(ue_id=i) for i in range(self.n_ue)]
        self._latest_reports.clear()

    # -------------------------------------------------------------------------
    # Statistics and Debugging
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Get BSR manager statistics.

        Returns:
            Dictionary with BSR statistics
        """
        total_buffer = sum(
            state.get_total_buffer_bytes() for state in self._ue_states
        )
        non_empty_ues = len(self.get_non_empty_ues())

        lcg_totals = {}
        for state in self._ue_states:
            for lcg, size in state.lcg_buffers.items():
                lcg_totals[lcg] = lcg_totals.get(lcg, 0) + size

        return {
            "total_buffer_bytes": total_buffer,
            "non_empty_ues": non_empty_ues,
            "total_ues": self.n_ue,
            "buffer_per_lcg": lcg_totals,
            "bsr_table_bits": self.bsr_bits,
            "periodic_timer_ttis": self.periodic_timer_ttis,
        }
