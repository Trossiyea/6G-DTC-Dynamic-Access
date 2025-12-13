"""
Buffer and queue management for per-UE packet queuing.

Implements multi-bearer queues per UE with QoS-aware packet management,
expiry detection, and buffer state reporting for schedulers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from .packets import Packet


@dataclass
class UEBuffer:
    """Per-UE buffer with multiple bearer queues.

    Each UE can have multiple logical bearers (e.g., separate queues for
    URLLC, eMBB, voice), each with its own QoS characteristics.

    Attributes:
        ue_id: UE index
        queues: Dictionary mapping bearer_id to FIFO packet queue
        total_bits: Total buffered bits across all bearers
        total_packets: Total number of packets in buffer
        urllc_bits: Bits in URLLC-class queues (QCI <= 5 or >= 80)
        embb_bits: Bits in eMBB-class queues
    """

    ue_id: int
    queues: Dict[int, Deque[Packet]] = field(default_factory=dict)
    total_bits: int = 0
    total_packets: int = 0
    urllc_bits: int = 0
    embb_bits: int = 0

    def enqueue(self, packet: Packet) -> None:
        """Add packet to appropriate bearer queue.

        Args:
            packet: Packet to enqueue
        """
        bid = packet.bearer_id
        if bid not in self.queues:
            self.queues[bid] = deque()

        self.queues[bid].append(packet)
        self.total_bits += packet.size_bits
        self.total_packets += 1

        # Track by QoS type (URLLC vs eMBB)
        if packet.qci <= 5 or packet.qci >= 80:
            self.urllc_bits += packet.size_bits
        else:
            self.embb_bits += packet.size_bits

    def get_head_of_line_packet(self, bearer_id: Optional[int] = None) -> Optional[Packet]:
        """Get head-of-line packet for a bearer or highest priority bearer.

        Args:
            bearer_id: Specific bearer to query (None = highest priority)

        Returns:
            HOL packet or None if buffer empty
        """
        if bearer_id is not None:
            queue = self.queues.get(bearer_id)
            return queue[0] if queue else None

        # Find highest priority non-empty queue (lowest bearer_id by convention)
        for bid in sorted(self.queues.keys()):
            if self.queues[bid]:
                return self.queues[bid][0]
        return None

    def dequeue_bits(
        self,
        bits: int,
        current_tti: int,
        bearer_id: Optional[int] = None
    ) -> List[Packet]:
        """Dequeue bits from buffer, returning completed packets.

        Bits are deducted from packets in FIFO order within the specified bearer.
        Packets are considered complete when remaining_bits <= 0.

        Args:
            bits: Number of bits successfully transmitted
            current_tti: Current TTI (for completion timestamp)
            bearer_id: Bearer to dequeue from (None = all bearers in priority order)

        Returns:
            List of completed packets
        """
        completed_packets = []
        remaining_bits = bits

        # Determine bearers to process
        if bearer_id is not None:
            bearer_list = [bearer_id] if bearer_id in self.queues else []
        else:
            bearer_list = sorted(self.queues.keys())

        for bid in bearer_list:
            queue = self.queues[bid]
            if not queue or remaining_bits <= 0:
                continue

            # Process packets in FIFO order
            while queue and remaining_bits > 0:
                pkt = queue[0]

                # Deduct bits from packet
                bits_to_deduct = min(remaining_bits, pkt.remaining_bits)
                pkt.deduct_bits(bits_to_deduct)
                remaining_bits -= bits_to_deduct

                # Remove if complete
                if pkt.is_complete():
                    pkt.completed_tti = current_tti
                    completed_packets.append(pkt)
                    queue.popleft()

                    # Update counters
                    self.total_bits -= pkt.size_bits
                    self.total_packets -= 1
                    if pkt.qci <= 5 or pkt.qci >= 80:
                        self.urllc_bits -= pkt.size_bits
                    else:
                        self.embb_bits -= pkt.size_bits

        return completed_packets

    def drop_expired(self, current_tti: int) -> List[Packet]:
        """Remove and return all expired packets from all bearers.

        Args:
            current_tti: Current TTI index

        Returns:
            List of dropped packets
        """
        dropped = []

        for bid in list(self.queues.keys()):
            queue = self.queues[bid]
            expired_count = 0

            # Count expired packets at head of queue
            for pkt in queue:
                if pkt.is_expired(current_tti):
                    expired_count += 1
                else:
                    break  # FIFO: stop at first non-expired

            # Remove expired packets
            for _ in range(expired_count):
                pkt = queue.popleft()
                dropped.append(pkt)

                # Update counters
                self.total_bits -= pkt.size_bits
                self.total_packets -= 1
                if pkt.qci <= 5 or pkt.qci >= 80:
                    self.urllc_bits -= pkt.size_bits
                else:
                    self.embb_bits -= pkt.size_bits

            # Remove empty queues
            if not queue:
                del self.queues[bid]

        return dropped

    def head_of_line_delay(self, current_tti: int, bearer_id: Optional[int] = None) -> int:
        """Get maximum head-of-line delay across bearers.

        Args:
            current_tti: Current TTI index
            bearer_id: Specific bearer to query (None = max across all)

        Returns:
            Maximum HOL delay in TTIs (0 if buffer empty)
        """
        if bearer_id is not None:
            pkt = self.get_head_of_line_packet(bearer_id)
            return pkt.head_of_line_delay(current_tti) if pkt else 0

        # Max HOL delay across all bearers
        max_delay = 0
        for queue in self.queues.values():
            if queue:
                delay = queue[0].head_of_line_delay(current_tti)
                max_delay = max(max_delay, delay)
        return max_delay

    def is_empty(self) -> bool:
        """Check if buffer is completely empty."""
        return self.total_packets == 0

    def get_buffer_occupancy_bits(self) -> int:
        """Get total buffer occupancy in bits."""
        return self.total_bits


class BufferManager:
    """Manages buffers for all UEs in the simulation.

    Provides centralized buffer state queries for scheduler integration.
    """

    def __init__(self, n_ue: int, config: Optional[Dict] = None):
        """Initialize buffer manager.

        Args:
            n_ue: Number of UEs
            config: Optional configuration dictionary
        """
        self.n_ue = n_ue
        self.buffers: List[UEBuffer] = [UEBuffer(ue_id=i) for i in range(n_ue)]
        self.config = config or {}

    def enqueue(self, packet: Packet) -> None:
        """Enqueue packet to appropriate UE buffer.

        Args:
            packet: Packet to enqueue
        """
        if 0 <= packet.ue_id < self.n_ue:
            self.buffers[packet.ue_id].enqueue(packet)

    def dequeue_bits(
        self,
        ue: int,
        bits: int,
        current_tti: int,
        bearer_id: Optional[int] = None
    ) -> List[Packet]:
        """Dequeue bits from a UE's buffer.

        Args:
            ue: UE index
            bits: Number of bits successfully transmitted
            current_tti: Current TTI
            bearer_id: Optional bearer specifier

        Returns:
            List of completed packets
        """
        if 0 <= ue < self.n_ue:
            return self.buffers[ue].dequeue_bits(bits, current_tti, bearer_id)
        return []

    def drop_all_expired(self, current_tti: int) -> List[Packet]:
        """Drop expired packets from all UE buffers.

        Args:
            current_tti: Current TTI index

        Returns:
            List of all dropped packets
        """
        dropped = []
        for buf in self.buffers:
            dropped.extend(buf.drop_expired(current_tti))
        return dropped

    def get_buffer_state(self, ue: int) -> Dict:
        """Get buffer state for scheduler decision-making.

        Args:
            ue: UE index

        Returns:
            Dictionary with buffer metrics:
                - total_bits: Total buffered bits
                - total_packets: Number of packets
                - urllc_bits: URLLC-class bits
                - embb_bits: eMBB-class bits
                - hol_delay: Maximum HOL delay in TTIs
                - has_urllc: Boolean indicating URLLC data presence
                - is_empty: Boolean indicating empty buffer
        """
        if not (0 <= ue < self.n_ue):
            return {
                "total_bits": 0,
                "total_packets": 0,
                "urllc_bits": 0,
                "embb_bits": 0,
                "hol_delay": 0,
                "has_urllc": False,
                "is_empty": True,
            }

        buf = self.buffers[ue]
        # Use dummy current_tti=0 if not tracking delays
        return {
            "total_bits": buf.total_bits,
            "total_packets": buf.total_packets,
            "urllc_bits": buf.urllc_bits,
            "embb_bits": buf.embb_bits,
            "hol_delay": 0,  # Will be set by caller with current_tti
            "has_urllc": buf.urllc_bits > 0,
            "is_empty": buf.is_empty(),
        }

    def get_all_buffer_states(self, current_tti: int) -> List[Dict]:
        """Get buffer states for all UEs.

        Args:
            current_tti: Current TTI for HOL delay calculation

        Returns:
            List of buffer state dictionaries
        """
        states = []
        for ue in range(self.n_ue):
            state = self.get_buffer_state(ue)
            state["hol_delay"] = self.buffers[ue].head_of_line_delay(current_tti)
            states.append(state)
        return states

    def get_total_buffer_occupancy(self) -> int:
        """Get total buffered bits across all UEs."""
        return sum(buf.total_bits for buf in self.buffers)

    def get_total_packet_count(self) -> int:
        """Get total number of packets across all UEs."""
        return sum(buf.total_packets for buf in self.buffers)
