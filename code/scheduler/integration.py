# -*- coding: utf-8 -*-
"""
MAC-Scheduler integration layer for NTN-aware resource allocation.

This module bridges the MAC layer (BSR, DRX, NTN Timing) with the scheduler,
enabling realistic MAC-aware scheduling decisions.

Integration Points:
- DRX → UE mask: Filter sleeping UEs from scheduling
- BSR → Buffer priority: Use buffer status for resource allocation
- QoS → Scheduling metric: M-LWDF, EXP-PF algorithms
- NTN Timing → HARQ: K1-aware HARQ timing

Usage:
    from scheduler.integration import MACSchedulerBridge, QoSScheduler

    # Create bridge
    bridge = MACSchedulerBridge(n_ue=100, config=config)

    # Update from geometry
    bridge.update_from_geometry(tau_s_per_ue)

    # Get scheduling inputs
    ue_mask = bridge.get_ue_mask(tti)
    qos_metric = bridge.compute_qos_metric(se_metric, head_of_line_delay)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Union

import numpy as np

# Import MAC modules
try:
    from mac import (
        BSRManager,
        DRXController,
        NTNDRXController,
        SchedulingTimingManager,
        TimingAdvanceController,
        NTNScenario,
    )
    MAC_AVAILABLE = True
except ImportError:
    MAC_AVAILABLE = False

# Import Traffic modules
try:
    from traffic import BufferManager, QoSManager, NTNQoSManager
    TRAFFIC_AVAILABLE = True
except ImportError:
    TRAFFIC_AVAILABLE = False


# =============================================================================
# QoS Scheduling Algorithms
# =============================================================================

class QoSSchedulerType(Enum):
    """QoS-aware scheduler algorithm types."""

    PF = "pf"                # Proportional Fair (baseline)
    M_LWDF = "m-lwdf"        # Modified Largest Weighted Delay First
    EXP_PF = "exp-pf"        # Exponential PF
    EDF = "edf"              # Earliest Deadline First


@dataclass
class QoSSchedulerConfig:
    """Configuration for QoS-aware scheduling."""

    # Algorithm selection
    algorithm: QoSSchedulerType = QoSSchedulerType.PF

    # PF parameters
    pf_beta: float = 0.1  # Averaging factor

    # M-LWDF parameters (per 3GPP)
    mlwdf_delta: float = 0.01  # Target packet loss rate
    mlwdf_tau: float = 100.0   # Delay bound (ms)

    # EXP-PF parameters
    exppf_beta: float = 1.0    # Delay sensitivity
    exppf_c: float = 1.0       # Normalization constant

    # NTN-specific
    compensate_rtt_in_delay: bool = True  # Subtract RTT from HoL delay

    @classmethod
    def from_config_dict(cls, config: Dict) -> "QoSSchedulerConfig":
        """Create from flat configuration dictionary."""
        algo_str = config.get("scheduler_algorithm", "pf").lower()
        algo_map = {
            "pf": QoSSchedulerType.PF,
            "m-lwdf": QoSSchedulerType.M_LWDF,
            "mlwdf": QoSSchedulerType.M_LWDF,
            "exp-pf": QoSSchedulerType.EXP_PF,
            "exppf": QoSSchedulerType.EXP_PF,
            "edf": QoSSchedulerType.EDF,
        }
        algorithm = algo_map.get(algo_str, QoSSchedulerType.PF)

        return cls(
            algorithm=algorithm,
            pf_beta=config.get("pf_beta", 0.1),
            mlwdf_delta=config.get("mlwdf_delta", 0.01),
            mlwdf_tau=config.get("mlwdf_tau", 100.0),
            exppf_beta=config.get("exppf_beta", 1.0),
            exppf_c=config.get("exppf_c", 1.0),
            compensate_rtt_in_delay=config.get("compensate_rtt_in_delay", True),
        )


# =============================================================================
# QoS Metric Computation
# =============================================================================

def compute_pf_metric(
    se_metric: np.ndarray,
    avg_throughput: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute Proportional Fair metric.

    PF metric = R_i(t) / R_bar_i

    Args:
        se_metric: Instantaneous achievable rate [N_UE] or [N_UE, Z]
        avg_throughput: Average throughput [N_UE]
        epsilon: Small value to avoid division by zero

    Returns:
        PF metric [N_UE] or [N_UE, Z]
    """
    Rbar = np.maximum(avg_throughput, epsilon)

    if se_metric.ndim == 1:
        return se_metric / Rbar
    else:
        # Per-PRB metric
        return se_metric / Rbar[:, np.newaxis]


def compute_mlwdf_metric(
    se_metric: np.ndarray,
    avg_throughput: np.ndarray,
    hol_delay_ms: np.ndarray,
    qos_params: np.ndarray,
    delta: float = 0.01,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute Modified Largest Weighted Delay First (M-LWDF) metric.

    M-LWDF metric = a_i * W_i(t) * R_i(t) / R_bar_i

    where:
        a_i = -log(delta_i) / T_i  (QoS weight)
        W_i(t) = D_HoL_i(t)        (Head-of-line delay)

    Reference: 3GPP TS 23.501, Andrews et al. "Providing QoS over CDMA systems"

    Args:
        se_metric: Instantaneous achievable rate [N_UE] or [N_UE, Z]
        avg_throughput: Average throughput [N_UE]
        hol_delay_ms: Head-of-line packet delay per UE [N_UE] (ms)
        qos_params: QoS parameters [N_UE, 2] where [:, 0] = delta, [:, 1] = tau_ms
        delta: Default target packet loss rate
        epsilon: Small value to avoid division by zero

    Returns:
        M-LWDF metric [N_UE] or [N_UE, Z]
    """
    n_ue = se_metric.shape[0]

    # Extract QoS parameters
    if qos_params is not None and qos_params.shape[0] == n_ue:
        delta_i = qos_params[:, 0]
        tau_i = qos_params[:, 1]
    else:
        delta_i = np.full(n_ue, delta)
        tau_i = np.full(n_ue, 100.0)  # Default 100 ms

    # Compute QoS weight: a_i = -log(delta_i) / tau_i
    a_i = -np.log(np.maximum(delta_i, 1e-10)) / np.maximum(tau_i, epsilon)

    # Delay weight: W_i = D_HoL
    W_i = np.maximum(hol_delay_ms, 0.0)

    # PF part
    Rbar = np.maximum(avg_throughput, epsilon)

    if se_metric.ndim == 1:
        pf_metric = se_metric / Rbar
        return a_i * W_i * pf_metric
    else:
        pf_metric = se_metric / Rbar[:, np.newaxis]
        return (a_i * W_i)[:, np.newaxis] * pf_metric


def compute_exppf_metric(
    se_metric: np.ndarray,
    avg_throughput: np.ndarray,
    hol_delay_ms: np.ndarray,
    qos_params: np.ndarray,
    beta: float = 1.0,
    c: float = 1.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute Exponential Proportional Fair (EXP-PF) metric.

    EXP-PF metric = exp(a_i * W_i(t) - c * W_avg) * R_i(t) / R_bar_i

    where:
        a_i = -log(delta_i) / T_i
        W_i(t) = D_HoL_i(t)
        W_avg = (1/N) * sum(a_j * W_j)

    Reference: Rhee et al. "Exponential Rule for QoS provisioning"

    Args:
        se_metric: Instantaneous achievable rate [N_UE] or [N_UE, Z]
        avg_throughput: Average throughput [N_UE]
        hol_delay_ms: Head-of-line packet delay per UE [N_UE] (ms)
        qos_params: QoS parameters [N_UE, 2] where [:, 0] = delta, [:, 1] = tau_ms
        beta: Delay sensitivity parameter
        c: Normalization constant
        epsilon: Small value to avoid division by zero

    Returns:
        EXP-PF metric [N_UE] or [N_UE, Z]
    """
    n_ue = se_metric.shape[0]

    # Extract QoS parameters
    if qos_params is not None and qos_params.shape[0] == n_ue:
        delta_i = qos_params[:, 0]
        tau_i = qos_params[:, 1]
    else:
        delta_i = np.full(n_ue, 0.01)
        tau_i = np.full(n_ue, 100.0)

    # QoS weight
    a_i = -np.log(np.maximum(delta_i, 1e-10)) / np.maximum(tau_i, epsilon)

    # Delay weight
    W_i = np.maximum(hol_delay_ms, 0.0)

    # Average weighted delay
    aW = a_i * W_i
    W_avg = np.mean(aW) if n_ue > 0 else 0.0

    # Exponential weight (clipped to avoid overflow)
    exp_weight = np.exp(np.clip(beta * (aW - c * W_avg), -50, 50))

    # PF part
    Rbar = np.maximum(avg_throughput, epsilon)

    if se_metric.ndim == 1:
        pf_metric = se_metric / Rbar
        return exp_weight * pf_metric
    else:
        pf_metric = se_metric / Rbar[:, np.newaxis]
        return exp_weight[:, np.newaxis] * pf_metric


def compute_edf_metric(
    hol_delay_ms: np.ndarray,
    deadline_ms: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute Earliest Deadline First (EDF) metric.

    EDF metric = 1 / (deadline - current_delay)

    Higher metric = closer to deadline = higher priority.

    Args:
        hol_delay_ms: Head-of-line packet delay per UE [N_UE] (ms)
        deadline_ms: Packet deadline per UE [N_UE] (ms)
        epsilon: Small value to avoid division by zero

    Returns:
        EDF metric [N_UE]
    """
    slack = np.maximum(deadline_ms - hol_delay_ms, epsilon)
    return 1.0 / slack


# =============================================================================
# MAC-Scheduler Bridge
# =============================================================================

class MACSchedulerBridge:
    """Bridge between MAC layer and scheduler.

    Integrates:
    - DRX: Provides UE activity mask for scheduler
    - BSR: Provides buffer status for scheduling priority
    - NTN Timing: Provides K values and TA for HARQ timing
    - QoS: Computes QoS-aware scheduling metrics
    """

    def __init__(
        self,
        n_ue: int,
        config: Optional[Dict] = None,
    ):
        """Initialize MAC-Scheduler bridge.

        Args:
            n_ue: Number of UEs
            config: Configuration dictionary
        """
        self.n_ue = n_ue
        self.config = config or {}

        # Initialize MAC components if available
        self._init_mac_components()

        # Initialize QoS scheduler config
        self.qos_config = QoSSchedulerConfig.from_config_dict(self.config)

        # Per-UE state
        self._avg_throughput = np.full(n_ue, 1e-3)
        self._hol_delay_ms = np.zeros(n_ue)
        self._qos_params = np.zeros((n_ue, 2))  # [delta, tau]
        self._qos_params[:, 0] = 0.01  # Default delta
        self._qos_params[:, 1] = 100.0  # Default tau (ms)

        # Current TTI
        self._current_tti = 0

    def _init_mac_components(self) -> None:
        """Initialize MAC layer components."""
        # DRX Controller
        self.drx_controller: Optional[DRXController] = None
        if MAC_AVAILABLE and self.config.get("enable_drx", False):
            use_ntn_drx = self.config.get("ntn_scenario", "") not in ("", "terrestrial")
            if use_ntn_drx:
                self.drx_controller = NTNDRXController(
                    n_ue=self.n_ue,
                    config=self.config,
                )
            else:
                self.drx_controller = DRXController(
                    n_ue=self.n_ue,
                    config=self.config,
                )

        # BSR Manager
        self.bsr_manager: Optional[BSRManager] = None
        if MAC_AVAILABLE and self.config.get("enable_bsr", False):
            self.bsr_manager = BSRManager(
                n_ue=self.n_ue,
                config=self.config,
            )

        # NTN Timing Manager
        self.timing_manager: Optional[SchedulingTimingManager] = None
        if MAC_AVAILABLE and self.config.get("enable_ntn_timing", True):
            self.timing_manager = SchedulingTimingManager(
                n_ue=self.n_ue,
                config=self.config,
            )

        # Buffer Manager (from traffic module)
        self.buffer_manager: Optional[BufferManager] = None
        if TRAFFIC_AVAILABLE and self.config.get("enable_traffic", False):
            self.buffer_manager = BufferManager(
                n_ue=self.n_ue,
                config=self.config,
            )

        # QoS Manager
        self.qos_manager: Optional[QoSManager] = None
        if TRAFFIC_AVAILABLE and self.config.get("enable_qos", False):
            use_ntn_qos = self.config.get("ntn_scenario", "") not in ("", "terrestrial")
            if use_ntn_qos:
                self.qos_manager = NTNQoSManager(self.config)
            else:
                self.qos_manager = QoSManager(self.config)

    # -------------------------------------------------------------------------
    # Geometry Update
    # -------------------------------------------------------------------------

    def update_from_geometry(
        self,
        tau_s_per_ue: np.ndarray,
        reference_delay_s: Optional[float] = None,
    ) -> None:
        """Update all components from satellite geometry.

        Args:
            tau_s_per_ue: One-way propagation delay per UE [N_UE] in seconds
            reference_delay_s: Reference delay for common TA
        """
        # Update NTN Timing
        if self.timing_manager is not None:
            self.timing_manager.update_from_geometry(tau_s_per_ue, reference_delay_s)

        # Update DRX RTT offsets
        if self.drx_controller is not None:
            tti_ms = self.config.get("tti_ms", 1.0)
            self.drx_controller.update_ntn_offsets_from_geometry(tau_s_per_ue, tti_ms)

        # Update QoS delay compensation
        if self.qos_manager is not None and hasattr(self.qos_manager, 'update_rtt'):
            rtt_ms = 2.0 * tau_s_per_ue * 1000.0
            for ue in range(min(self.n_ue, len(rtt_ms))):
                self.qos_manager.update_rtt(ue, rtt_ms[ue])

    # -------------------------------------------------------------------------
    # Time Advancement
    # -------------------------------------------------------------------------

    def advance_time(self, tti: int) -> Dict[int, any]:
        """Advance time for all MAC components.

        Args:
            tti: Current TTI index

        Returns:
            Dictionary of state changes (if any)
        """
        self._current_tti = tti
        changes = {}

        # Advance DRX
        if self.drx_controller is not None:
            drx_changes = self.drx_controller.advance_time(tti)
            if drx_changes:
                changes["drx"] = drx_changes

        # Advance BSR timers
        if self.bsr_manager is not None:
            self.bsr_manager.advance_time(tti)

        return changes

    # -------------------------------------------------------------------------
    # UE Mask Generation (DRX Integration)
    # -------------------------------------------------------------------------

    def get_ue_mask(self, tti: Optional[int] = None) -> np.ndarray:
        """Get UE scheduling mask for current TTI.

        Combines DRX state with other constraints.

        Args:
            tti: TTI index (optional, uses current if not provided)

        Returns:
            Boolean mask [N_UE] where True = schedulable
        """
        mask = np.ones(self.n_ue, dtype=bool)

        # Apply DRX mask
        if self.drx_controller is not None:
            active_ues = self.drx_controller.get_active_ues()
            mask = np.zeros(self.n_ue, dtype=bool)
            mask[active_ues] = True

        # Apply buffer state mask (no data = don't schedule)
        if self.buffer_manager is not None:
            for ue in range(self.n_ue):
                if mask[ue] and self.buffer_manager.get_total_buffer_bytes(ue) == 0:
                    # UE has no data to send - could still schedule for polling
                    pass  # Keep schedulable for now

        return mask

    def build_ue_mask_time(self, T: int) -> np.ndarray:
        """Build UE mask for multiple TTIs.

        Args:
            T: Number of TTIs

        Returns:
            Boolean mask [T, N_UE]
        """
        mask_time = np.ones((T, self.n_ue), dtype=bool)

        if self.drx_controller is None:
            return mask_time

        # Save current state
        saved_tti = self._current_tti

        for t in range(T):
            self.advance_time(saved_tti + t)
            mask_time[t] = self.get_ue_mask()

        # Restore (note: DRX state has advanced, may need reset for actual sim)
        self._current_tti = saved_tti

        return mask_time

    # -------------------------------------------------------------------------
    # Buffer Status (BSR Integration)
    # -------------------------------------------------------------------------

    def get_buffer_bytes(self, ue_id: int) -> int:
        """Get total buffer size for a UE.

        Args:
            ue_id: UE index

        Returns:
            Total buffer size in bytes
        """
        if self.buffer_manager is not None:
            return self.buffer_manager.get_total_buffer_bytes(ue_id)
        if self.bsr_manager is not None:
            return self.bsr_manager.get_total_buffer_bytes(ue_id)
        return 0

    def get_all_buffer_bytes(self) -> np.ndarray:
        """Get buffer sizes for all UEs.

        Returns:
            Buffer sizes [N_UE] in bytes
        """
        if self.buffer_manager is not None:
            return np.array([
                self.buffer_manager.get_total_buffer_bytes(ue)
                for ue in range(self.n_ue)
            ])
        if self.bsr_manager is not None:
            return np.array([
                self.bsr_manager.get_total_buffer_bytes(ue)
                for ue in range(self.n_ue)
            ])
        return np.zeros(self.n_ue)

    def get_buffer_priority(self) -> np.ndarray:
        """Get buffer-based priority for scheduling.

        Higher priority for UEs with more data (simplified).

        Returns:
            Priority scores [N_UE]
        """
        buffer_bytes = self.get_all_buffer_bytes()
        # Normalize to [0, 1] range
        max_buf = np.max(buffer_bytes)
        if max_buf > 0:
            return buffer_bytes / max_buf
        return np.zeros(self.n_ue)

    # -------------------------------------------------------------------------
    # QoS Metric Computation
    # -------------------------------------------------------------------------

    def update_hol_delay(self, hol_delay_ms: np.ndarray) -> None:
        """Update head-of-line delay for all UEs.

        Args:
            hol_delay_ms: HoL delay per UE [N_UE] in milliseconds
        """
        self._hol_delay_ms = np.array(hol_delay_ms)

        # Compensate for RTT if configured
        if (self.qos_config.compensate_rtt_in_delay and
            self.timing_manager is not None):
            rtt_ms = self.timing_manager.ta_controller.get_all_ta_ms()
            self._hol_delay_ms = np.maximum(0, self._hol_delay_ms - rtt_ms)

    def update_qos_params(
        self,
        ue_id: int,
        delta: float,
        tau_ms: float,
    ) -> None:
        """Update QoS parameters for a UE.

        Args:
            ue_id: UE index
            delta: Target packet loss rate
            tau_ms: Delay bound in milliseconds
        """
        if 0 <= ue_id < self.n_ue:
            self._qos_params[ue_id, 0] = delta
            self._qos_params[ue_id, 1] = tau_ms

    def update_avg_throughput(
        self,
        throughput: np.ndarray,
        beta: Optional[float] = None,
    ) -> None:
        """Update average throughput (exponential moving average).

        Args:
            throughput: Current throughput per UE [N_UE]
            beta: Averaging factor (default from config)
        """
        if beta is None:
            beta = self.qos_config.pf_beta
        self._avg_throughput = (1 - beta) * self._avg_throughput + beta * throughput

    def compute_qos_metric(
        self,
        se_metric: np.ndarray,
        hol_delay_ms: Optional[np.ndarray] = None,
        algorithm: Optional[QoSSchedulerType] = None,
    ) -> np.ndarray:
        """Compute QoS-aware scheduling metric.

        Args:
            se_metric: Instantaneous achievable rate [N_UE] or [N_UE, Z]
            hol_delay_ms: Head-of-line delay [N_UE] (uses stored if None)
            algorithm: Algorithm to use (default from config)

        Returns:
            QoS-aware metric [N_UE] or [N_UE, Z]
        """
        if hol_delay_ms is not None:
            self.update_hol_delay(hol_delay_ms)

        algo = algorithm or self.qos_config.algorithm

        if algo == QoSSchedulerType.PF:
            return compute_pf_metric(
                se_metric,
                self._avg_throughput,
            )

        elif algo == QoSSchedulerType.M_LWDF:
            return compute_mlwdf_metric(
                se_metric,
                self._avg_throughput,
                self._hol_delay_ms,
                self._qos_params,
                delta=self.qos_config.mlwdf_delta,
            )

        elif algo == QoSSchedulerType.EXP_PF:
            return compute_exppf_metric(
                se_metric,
                self._avg_throughput,
                self._hol_delay_ms,
                self._qos_params,
                beta=self.qos_config.exppf_beta,
                c=self.qos_config.exppf_c,
            )

        elif algo == QoSSchedulerType.EDF:
            deadline_ms = self._qos_params[:, 1]
            return compute_edf_metric(
                self._hol_delay_ms,
                deadline_ms,
            )

        else:
            # Fallback to PF
            return compute_pf_metric(se_metric, self._avg_throughput)

    # -------------------------------------------------------------------------
    # NTN Timing Integration
    # -------------------------------------------------------------------------

    def get_k1(self, ue_id: int) -> int:
        """Get K1 (PDSCH to HARQ-ACK) for a UE.

        Args:
            ue_id: UE index

        Returns:
            K1 in slots
        """
        if self.timing_manager is not None:
            return self.timing_manager.get_k1(ue_id)
        return self.config.get("k1_slots_base", 4)

    def get_all_k1(self) -> np.ndarray:
        """Get K1 for all UEs.

        Returns:
            K1 values [N_UE] in slots
        """
        if self.timing_manager is not None:
            return self.timing_manager.get_all_k1()
        k1_base = self.config.get("k1_slots_base", 4)
        return np.full(self.n_ue, k1_base, dtype=int)

    def get_ta_ms(self, ue_id: int) -> float:
        """Get Timing Advance for a UE.

        Args:
            ue_id: UE index

        Returns:
            TA in milliseconds
        """
        if self.timing_manager is not None:
            return self.timing_manager.ta_controller.get_ta_for_ue_ms(ue_id)
        return 0.0

    def get_harq_ack_slot(self, ue_id: int, pdsch_slot: int) -> int:
        """Get expected HARQ ACK slot.

        Args:
            ue_id: UE index
            pdsch_slot: PDSCH transmission slot

        Returns:
            Expected ACK slot
        """
        if self.timing_manager is not None:
            return self.timing_manager.get_harq_ack_slot(ue_id, pdsch_slot)
        k1 = self.config.get("k1_slots_base", 4)
        return pdsch_slot + k1

    # -------------------------------------------------------------------------
    # Event Notifications
    # -------------------------------------------------------------------------

    def on_pdcch_scheduled(self, ue_id: int, tti: int) -> None:
        """Notify that UE was scheduled (PDCCH sent).

        Args:
            ue_id: UE index
            tti: TTI of scheduling
        """
        if self.drx_controller is not None:
            self.drx_controller.on_pdcch_received(ue_id, tti)

    def on_harq_ack(self, ue_id: int, tti: int) -> None:
        """Notify HARQ ACK received.

        Args:
            ue_id: UE index
            tti: TTI of ACK
        """
        if self.drx_controller is not None:
            self.drx_controller.on_harq_ack(ue_id, tti)

    def on_harq_nack(self, ue_id: int, tti: int) -> None:
        """Notify HARQ NACK received.

        Args:
            ue_id: UE index
            tti: TTI of NACK
        """
        if self.drx_controller is not None:
            self.drx_controller.on_harq_nack(ue_id, tti)

    def on_data_delivered(
        self,
        ue_id: int,
        bytes_delivered: int,
        tti: int,
    ) -> None:
        """Notify data delivery to UE.

        Args:
            ue_id: UE index
            bytes_delivered: Bytes delivered
            tti: TTI of delivery
        """
        if self.buffer_manager is not None:
            self.buffer_manager.dequeue_bytes(ue_id, bytes_delivered)
        if self.bsr_manager is not None:
            # Update BSR buffer tracking
            pass  # BSR manager syncs from buffer manager

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Get integration layer statistics."""
        stats = {
            "n_ue": self.n_ue,
            "qos_algorithm": self.qos_config.algorithm.value,
            "components_enabled": {
                "drx": self.drx_controller is not None,
                "bsr": self.bsr_manager is not None,
                "timing": self.timing_manager is not None,
                "buffer": self.buffer_manager is not None,
                "qos": self.qos_manager is not None,
            },
        }

        # DRX stats
        if self.drx_controller is not None:
            stats["drx"] = self.drx_controller.get_statistics()

        # Timing stats
        if self.timing_manager is not None:
            stats["timing"] = self.timing_manager.get_statistics()

        return stats

    def reset(self) -> None:
        """Reset all components."""
        self._avg_throughput = np.full(self.n_ue, 1e-3)
        self._hol_delay_ms = np.zeros(self.n_ue)
        self._current_tti = 0

        if self.drx_controller is not None:
            self.drx_controller.reset()
        if self.bsr_manager is not None:
            self.bsr_manager.reset()
        if self.timing_manager is not None:
            self.timing_manager.reset()
        if self.buffer_manager is not None:
            self.buffer_manager.reset()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_mac_scheduler_bridge(
    n_ue: int,
    config: Dict,
    tau_s_per_ue: Optional[np.ndarray] = None,
) -> MACSchedulerBridge:
    """Create and initialize MAC-Scheduler bridge.

    Args:
        n_ue: Number of UEs
        config: Configuration dictionary
        tau_s_per_ue: Initial propagation delays (optional)

    Returns:
        Initialized MACSchedulerBridge
    """
    bridge = MACSchedulerBridge(n_ue, config)

    if tau_s_per_ue is not None:
        bridge.update_from_geometry(tau_s_per_ue)

    return bridge


def build_qos_params_from_qci(
    qci_per_ue: np.ndarray,
    qos_table: Optional[Dict] = None,
) -> np.ndarray:
    """Build QoS parameters from QCI values.

    Args:
        qci_per_ue: QCI per UE [N_UE]
        qos_table: Optional custom QoS table {qci: (delta, tau_ms)}

    Returns:
        QoS parameters [N_UE, 2]
    """
    # Default 5QI/QCI table (simplified)
    default_table = {
        1: (0.01, 100),    # Conversational Voice
        2: (0.001, 150),   # Conversational Video
        3: (0.001, 50),    # Real-time Gaming
        4: (0.001, 300),   # Non-conv Video
        5: (0.0001, 100),  # IMS Signaling
        6: (0.000001, 300), # Video (TCP)
        7: (0.0001, 100),  # Video (Live)
        8: (0.000001, 300), # TCP-based
        9: (0.000001, 300), # TCP-based (lower priority)
        65: (0.01, 75),    # MCPTT
        66: (0.01, 100),   # MCPTT Non-critical
        67: (0.001, 100),  # Mission Critical Data
    }

    table = qos_table or default_table
    n_ue = len(qci_per_ue)
    params = np.zeros((n_ue, 2))

    for ue in range(n_ue):
        qci = int(qci_per_ue[ue])
        if qci in table:
            params[ue, 0], params[ue, 1] = table[qci]
        else:
            # Default: QCI 9
            params[ue, 0], params[ue, 1] = 0.000001, 300

    return params
