"""
QoS class definitions and management for 5G NR.

Implements 5QI (5G QoS Identifier) classes per 3GPP TS 23.501 Table 5.7.4-1,
with NTN-specific delay budget extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class QoSClass:
    """5QI/QCI class definition per 3GPP TS 23.501.

    Attributes:
        qci: 5QI value (1-79 standardized, 80-85 for URLLC)
        name: Human-readable service description
        resource_type: "GBR", "Non-GBR", or "Delay-Critical GBR"
        priority_level: Scheduling priority (1-127, lower = higher priority)
        packet_delay_budget_ms: Maximum acceptable end-to-end delay
        packet_error_rate: Target packet error rate
        default_averaging_window_ms: Window for rate averaging (typically 2s)
        max_data_burst_volume: Maximum burst size for URLLC (bytes)
    """

    qci: int
    name: str
    resource_type: str
    priority_level: int
    packet_delay_budget_ms: float
    packet_error_rate: float
    default_averaging_window_ms: float = 2000.0
    max_data_burst_volume: Optional[int] = None

    def get_deadline_ttis(self, tti_ms: float) -> int:
        """Convert PDB to number of TTIs.

        Args:
            tti_ms: TTI duration in milliseconds

        Returns:
            Number of TTIs corresponding to PDB
        """
        return int(np.ceil(self.packet_delay_budget_ms / tti_ms))


# Standard 5QI definitions per 3GPP TS 23.501 Table 5.7.4-1
STANDARD_5QI: Dict[int, QoSClass] = {
    # GBR flows (conversational)
    1: QoSClass(
        qci=1, name="Conversational Voice", resource_type="GBR",
        priority_level=20, packet_delay_budget_ms=100.0, packet_error_rate=1e-2
    ),
    2: QoSClass(
        qci=2, name="Conversational Video (Live Streaming)", resource_type="GBR",
        priority_level=40, packet_delay_budget_ms=150.0, packet_error_rate=1e-3
    ),
    3: QoSClass(
        qci=3, name="Real-time Gaming", resource_type="GBR",
        priority_level=30, packet_delay_budget_ms=50.0, packet_error_rate=1e-3
    ),
    4: QoSClass(
        qci=4, name="Non-Conversational Video (Buffered Streaming)", resource_type="GBR",
        priority_level=50, packet_delay_budget_ms=300.0, packet_error_rate=1e-6
    ),
    # Non-GBR flows
    5: QoSClass(
        qci=5, name="IMS Signaling", resource_type="Non-GBR",
        priority_level=10, packet_delay_budget_ms=100.0, packet_error_rate=1e-6
    ),
    6: QoSClass(
        qci=6, name="Video (Buffered Streaming) TCP-based", resource_type="Non-GBR",
        priority_level=60, packet_delay_budget_ms=300.0, packet_error_rate=1e-6
    ),
    7: QoSClass(
        qci=7, name="Voice, Video (Live Streaming), Interactive Gaming",
        resource_type="Non-GBR",
        priority_level=70, packet_delay_budget_ms=100.0, packet_error_rate=1e-3
    ),
    8: QoSClass(
        qci=8, name="Video (Buffered Streaming) TCP-based", resource_type="Non-GBR",
        priority_level=80, packet_delay_budget_ms=300.0, packet_error_rate=1e-6
    ),
    9: QoSClass(
        qci=9, name="Video (Buffered Streaming) TCP-based / Default Bearer",
        resource_type="Non-GBR",
        priority_level=90, packet_delay_budget_ms=300.0, packet_error_rate=1e-6
    ),
    # URLLC Delay-Critical GBR (5QI 80-85)
    80: QoSClass(
        qci=80, name="Low Latency eMBB", resource_type="Non-GBR",
        priority_level=68, packet_delay_budget_ms=10.0, packet_error_rate=1e-6
    ),
    82: QoSClass(
        qci=82, name="Discrete Automation", resource_type="Delay-Critical GBR",
        priority_level=19, packet_delay_budget_ms=10.0, packet_error_rate=1e-4,
        max_data_burst_volume=255
    ),
    83: QoSClass(
        qci=83, name="Discrete Automation (Larger Burst)", resource_type="Delay-Critical GBR",
        priority_level=22, packet_delay_budget_ms=10.0, packet_error_rate=1e-4,
        max_data_burst_volume=1354
    ),
    84: QoSClass(
        qci=84, name="Intelligent Transport Systems", resource_type="Delay-Critical GBR",
        priority_level=24, packet_delay_budget_ms=30.0, packet_error_rate=1e-5,
        max_data_burst_volume=1354
    ),
    85: QoSClass(
        qci=85, name="Electricity Distribution - High Voltage", resource_type="Delay-Critical GBR",
        priority_level=21, packet_delay_budget_ms=5.0, packet_error_rate=1e-5,
        max_data_burst_volume=255
    ),
}


class QoSManager:
    """Manages QoS classes and bearer-to-QoS mapping.

    Provides deadline calculation and priority lookup for packet scheduling.
    """

    def __init__(self, config: Dict):
        """Initialize QoS manager.

        Args:
            config: Configuration dictionary containing TTI duration
        """
        self.classes = dict(STANDARD_5QI)
        self.tti_ms = config.get("tti_ms", 1.0)
        self.default_qci = 9  # Default best-effort

    def get_qos_class(self, qci: int) -> QoSClass:
        """Get QoS class definition.

        Args:
            qci: 5QI identifier

        Returns:
            QoSClass object (falls back to default if not found)
        """
        return self.classes.get(qci, self.classes[self.default_qci])

    def get_deadline_ttis(self, qci: int) -> int:
        """Get deadline in TTIs for a QoS class.

        Args:
            qci: 5QI identifier

        Returns:
            Number of TTIs corresponding to PDB
        """
        qos = self.get_qos_class(qci)
        return qos.get_deadline_ttis(self.tti_ms)

    def get_priority(self, qci: int) -> int:
        """Get scheduling priority for a QoS class.

        Args:
            qci: 5QI identifier

        Returns:
            Priority level (lower = higher priority)
        """
        qos = self.get_qos_class(qci)
        return qos.priority_level

    def is_urllc(self, qci: int) -> bool:
        """Check if QoS class is URLLC (low latency).

        Args:
            qci: 5QI identifier

        Returns:
            True if URLLC class (QCI <= 5 or QCI >= 80)
        """
        return qci <= 5 or qci >= 80


class NTNQoSManager(QoSManager):
    """QoS manager with NTN-specific delay compensation.

    Accounts for satellite propagation delay when computing effective PDB.
    """

    def __init__(self, config: Dict, tau_s_per_ue: Optional[np.ndarray] = None):
        """Initialize NTN QoS manager.

        Args:
            config: Configuration dictionary
            tau_s_per_ue: One-way propagation delay per UE in seconds [N_UE]
        """
        super().__init__(config)
        self.tau_s = tau_s_per_ue  # [N_UE] or None
        self.compensate_rtt = config.get("compensate_rtt_in_pdb", True)
        self.extension_factor = config.get("ntn_pdb_extension_factor", 1.0)

    def get_effective_pdb_ttis(self, ue: int, qci: int) -> int:
        """Get effective PDB accounting for NTN propagation delay.

        Args:
            ue: UE index
            qci: 5QI identifier

        Returns:
            Effective PDB in TTIs after RTT compensation
        """
        base_pdb_ms = self.get_qos_class(qci).packet_delay_budget_ms

        if self.compensate_rtt and self.tau_s is not None and ue < len(self.tau_s):
            # Subtract estimated RTT from available budget
            rtt_ms = 2.0 * float(self.tau_s[ue]) * 1000.0  # Round-trip time
            effective_pdb_ms = max(base_pdb_ms * self.extension_factor - rtt_ms, 1.0)
        else:
            # No compensation, just apply extension factor
            effective_pdb_ms = base_pdb_ms * self.extension_factor

        return int(np.ceil(effective_pdb_ms / self.tti_ms))

    def get_deadline_ttis(self, qci: int, ue: Optional[int] = None) -> int:
        """Get deadline with optional UE-specific NTN compensation.

        Args:
            qci: 5QI identifier
            ue: UE index (if None, uses base PDB without compensation)

        Returns:
            Deadline in TTIs
        """
        if ue is not None:
            return self.get_effective_pdb_ttis(ue, qci)
        else:
            return super().get_deadline_ttis(qci)
