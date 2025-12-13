"""
Packet and queue data structures for traffic modeling.

This module defines the core packet representation and queue management
for simulating traffic flows with QoS differentiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Packet:
    """Represents a single downlink packet in the simulation.

    A packet tracks its lifecycle from arrival at the gNB buffer through
    transmission attempts to final delivery or timeout.

    Attributes:
        packet_id: Unique identifier within the UE's packet stream
        ue_id: Index of the owning UE (0 to N_UE-1)
        bearer_id: Logical bearer identifier (maps to QCI/5QI)
        size_bits: Total packet size in bits
        arrival_tti: TTI when packet arrived at gNB buffer
        deadline_tti: Absolute deadline TTI (arrival + PDB), None if no deadline
        priority: Scheduling priority (lower value = higher priority)
        qci: QCI/5QI class identifier
        remaining_bits: Bits not yet successfully transmitted (decremented on TX)
        first_tx_tti: TTI of first transmission attempt (None if not started)
        completed_tti: TTI when fully delivered (None if incomplete)
        harq_attempts: Number of HARQ transmission attempts
    """

    packet_id: int
    ue_id: int
    bearer_id: int
    size_bits: int
    arrival_tti: int
    deadline_tti: Optional[int]
    priority: int
    qci: int

    # Transmission state (initialized in __post_init__)
    remaining_bits: int = field(init=False, default=0)
    first_tx_tti: Optional[int] = field(default=None)
    completed_tti: Optional[int] = field(default=None)
    harq_attempts: int = field(default=0)

    def __post_init__(self):
        """Initialize transmission state."""
        self.remaining_bits = self.size_bits

    def is_expired(self, current_tti: int) -> bool:
        """Check if packet has exceeded its delay budget.

        Args:
            current_tti: Current TTI index

        Returns:
            True if packet has a deadline and has expired, False otherwise
        """
        if self.deadline_tti is None:
            return False
        return current_tti > self.deadline_tti

    def head_of_line_delay(self, current_tti: int) -> int:
        """Compute head-of-line delay in TTIs.

        Args:
            current_tti: Current TTI index

        Returns:
            Number of TTIs since packet arrival
        """
        return current_tti - self.arrival_tti

    def latency_ms(self, delivery_tti: int, tti_duration_ms: float) -> float:
        """Compute end-to-end latency in milliseconds.

        Args:
            delivery_tti: TTI when packet was fully delivered
            tti_duration_ms: TTI duration in milliseconds

        Returns:
            Latency in milliseconds
        """
        return (delivery_tti - self.arrival_tti) * tti_duration_ms

    def is_complete(self) -> bool:
        """Check if packet has been fully transmitted."""
        return self.remaining_bits <= 0

    def deduct_bits(self, bits: int) -> None:
        """Deduct successfully transmitted bits from remaining count.

        Args:
            bits: Number of bits successfully transmitted
        """
        self.remaining_bits = max(0, self.remaining_bits - bits)

    def __repr__(self) -> str:
        status = "complete" if self.is_complete() else f"{self.remaining_bits}b remain"
        return (f"Packet(id={self.packet_id}, ue={self.ue_id}, "
                f"qci={self.qci}, size={self.size_bits}b, {status})")
