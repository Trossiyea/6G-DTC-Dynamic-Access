"""
Traffic layer simulator using post-processing approach.

This module simulates traffic dynamics (arrivals, queuing, latency, drops)
on top of scheduler output without modifying the core scheduling algorithm.

The scheduler runs with full-buffer assumption, then this module:
1. Generates packet arrivals according to traffic model
2. Queues packets in per-UE buffers
3. Uses scheduler's per-TTI throughput output to dequeue packets
4. Tracks latency, drops, and computes KPIs

This approach guarantees:
- Zero changes to scheduler code
- traffic_model="full_buffer" produces identical results
- Complete backward compatibility
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from traffic import (
    BufferManager,
    Packet,
    QoSManager,
    NTNQoSManager,
    TrafficStatisticsTracker,
    create_traffic_generator,
)


class TrafficSimulator:
    """Post-processing traffic simulator.

    Takes scheduler output (per-TTI per-UE throughput) and simulates
    traffic layer dynamics to compute latency/goodput KPIs.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize traffic simulator.

        Args:
            config: Configuration dictionary (flat format)
        """
        self.config = config
        self.traffic_model = config.get("traffic_model", "full_buffer")
        self.n_ue = config.get("N_UE", 100)
        self.T = config.get("T", 2000)
        self.tti_ms = config.get("tti_ms", 1.0)
        self.seed = config.get("seed", 42)

        # Will be initialized in simulate()
        self._buffer_mgr: Optional[BufferManager] = None
        self._traffic_gen = None
        self._qos_mgr: Optional[QoSManager] = None
        self._stats: Optional[TrafficStatisticsTracker] = None
        self._rng: Optional[np.random.Generator] = None

    def simulate(
        self,
        scheduler_result: Dict[str, Any],
        tau_s_per_ue: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run traffic simulation on scheduler output.

        Args:
            scheduler_result: Dictionary from scheduler containing:
                - ue_thr_time_rm or ue_thr_time_base: [T, N_UE] throughput in SE units
                - Or per_ue_throughput_*_bps for simplified mode
            tau_s_per_ue: Optional propagation delay per UE [N_UE] for NTN

        Returns:
            Extended result dictionary with traffic_kpi added
        """
        # Full buffer mode: no traffic simulation needed
        if self.traffic_model == "full_buffer":
            return scheduler_result

        # Extract per-TTI per-UE throughput from scheduler output
        ue_thr_time = self._extract_ue_throughput(scheduler_result)

        if ue_thr_time is None:
            # Fall back to average-based simulation
            return self._simulate_average_mode(scheduler_result)

        # Initialize components
        self._init_components(tau_s_per_ue)

        # Get RE per PRB for bit conversion
        re_per_prb = self._get_re_per_prb()

        # Run TTI-by-TTI simulation
        for t_idx in range(self.T):
            self._simulate_tti(t_idx, ue_thr_time[t_idx], re_per_prb)

        # Build result
        traffic_kpi = self._stats.get_summary()

        # Add traffic-specific metrics
        traffic_kpi["simulation_mode"] = "post_processing"
        traffic_kpi["traffic_model"] = self.traffic_model

        return {
            **scheduler_result,
            "traffic_kpi": traffic_kpi,
        }

    def _extract_ue_throughput(
        self, scheduler_result: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract per-TTI per-UE throughput from scheduler result.

        Returns:
            Array of shape [T, N_UE] with SE values per TTI, or None if not available
        """
        # Try RadioMap scheduler output first
        if "ue_thr_time_rm" in scheduler_result:
            thr_list = scheduler_result["ue_thr_time_rm"]
            if thr_list and len(thr_list) > 0:
                return np.array(thr_list)

        # Try baseline scheduler output
        if "ue_thr_time_base" in scheduler_result:
            thr_list = scheduler_result["ue_thr_time_base"]
            if thr_list and len(thr_list) > 0:
                return np.array(thr_list)

        return None

    def _simulate_average_mode(
        self, scheduler_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback: simulate using average throughput (less accurate).

        When per-TTI throughput is not available, use average rates.
        """
        # Get average per-UE throughput
        avg_thr_bps = scheduler_result.get("per_ue_throughput_radiomap_bps")
        if avg_thr_bps is None:
            avg_thr_bps = scheduler_result.get("per_ue_throughput_baseline_bps")
        if avg_thr_bps is None:
            # Cannot simulate without throughput data
            return {
                **scheduler_result,
                "traffic_kpi": {
                    "error": "No per-UE throughput data available",
                    "hint": "Set record_ue_thr=True in config",
                },
            }

        avg_thr_bps = np.array(avg_thr_bps)

        # Initialize components
        self._init_components(None)

        # Simulate with constant throughput per UE
        bits_per_tti = avg_thr_bps * (self.tti_ms / 1000.0)

        for t_idx in range(self.T):
            self._simulate_tti_with_bits(t_idx, bits_per_tti)

        traffic_kpi = self._stats.get_summary()
        traffic_kpi["simulation_mode"] = "average_throughput"
        traffic_kpi["traffic_model"] = self.traffic_model
        traffic_kpi["warning"] = "Using average throughput; set record_ue_thr=True for accurate simulation"

        return {
            **scheduler_result,
            "traffic_kpi": traffic_kpi,
        }

    def _init_components(self, tau_s_per_ue: Optional[np.ndarray]) -> None:
        """Initialize traffic simulation components."""
        self._rng = np.random.default_rng(self.seed + 1000)  # Offset seed

        # Buffer manager
        self._buffer_mgr = BufferManager(self.n_ue, self.config)

        # Traffic generator
        self._traffic_gen = create_traffic_generator(self.config)

        # QoS manager (NTN-aware if propagation delay available)
        if tau_s_per_ue is not None:
            self._qos_mgr = NTNQoSManager(self.config, tau_s_per_ue)
        else:
            self._qos_mgr = QoSManager(self.config)

        # Statistics tracker
        self._stats = TrafficStatisticsTracker(
            self.config, self.n_ue, self.tti_ms
        )

    def _get_re_per_prb(self) -> float:
        """Get resource elements per PRB from config."""
        scs_khz = self.config.get("scs_khz", 30)
        cp_type = self.config.get("cp_type", "normal")

        # Symbols per slot
        if cp_type == "extended":
            symbols_per_slot = 12
        else:
            symbols_per_slot = 14

        # Subcarriers per PRB
        sc_per_prb = 12

        # DMRS overhead
        dmrs_sym = self.config.get("pdsch_dmrs_sym_per_slot", 1)
        dmrs_re_per_sym = self.config.get("dmrs_re_per_sym_per_prb", 6)
        dmrs_re = dmrs_sym * dmrs_re_per_sym

        # Other overhead
        oh_prb = self.config.get("oh_prb", 0)

        # Available REs per PRB per slot
        total_re = symbols_per_slot * sc_per_prb
        available_re = total_re - dmrs_re - oh_prb

        return float(available_re)

    def _simulate_tti(
        self,
        t_idx: int,
        ue_thr_se: np.ndarray,
        re_per_prb: float,
    ) -> None:
        """Simulate one TTI.

        Args:
            t_idx: TTI index
            ue_thr_se: Per-UE throughput in SE units (bits per RE) [N_UE]
            re_per_prb: REs per PRB
        """
        # Convert SE to bits (SE is typically bits per PRB after overhead)
        # ue_thr_se[ue] = sum of (SE * num_prbs) for assigned PRBs
        # Already in "total bits per UE this TTI" after accounting for PRBs
        bits_per_ue = ue_thr_se * re_per_prb

        self._simulate_tti_with_bits(t_idx, bits_per_ue)

    def _simulate_tti_with_bits(
        self,
        t_idx: int,
        bits_per_ue: np.ndarray,
    ) -> None:
        """Simulate one TTI with given bits per UE.

        Args:
            t_idx: TTI index
            bits_per_ue: Bits transmitted per UE this TTI [N_UE]
        """
        # 1. Generate arrivals
        for ue in range(self.n_ue):
            arrivals = self._traffic_gen.generate_arrivals(t_idx, ue, self._rng)
            for pkt in arrivals:
                # Set deadline based on QoS
                pkt.deadline_tti = t_idx + self._qos_mgr.get_deadline_ttis(pkt.qci)
                pkt.priority = self._qos_mgr.get_priority(pkt.qci)

                self._buffer_mgr.enqueue(pkt)
                self._stats.on_packet_arrived(pkt)

        # 2. Drop expired packets
        dropped = self._buffer_mgr.drop_all_expired(t_idx)
        for pkt in dropped:
            self._stats.on_packet_dropped(pkt, t_idx, "timeout")

        # 3. Dequeue scheduled bits
        for ue in range(self.n_ue):
            scheduled_bits = bits_per_ue[ue]
            if scheduled_bits <= 0:
                continue

            # Get available bits in buffer
            buf_state = self._buffer_mgr.get_buffer_state(ue)
            available_bits = buf_state["total_bits"]

            if available_bits <= 0:
                # Scheduler allocated PRBs but no data to send
                # This is "wasted" capacity in non-full-buffer mode
                continue

            # Dequeue actual bits (min of scheduled and available)
            actual_bits = min(scheduled_bits, available_bits)
            completed = self._buffer_mgr.dequeue_bits(ue, int(actual_bits), t_idx)

            for pkt in completed:
                self._stats.on_packet_delivered(pkt, t_idx)


def simulate_traffic_layer(
    config: Dict[str, Any],
    scheduler_result: Dict[str, Any],
    tau_s_per_ue: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Convenience function to run traffic simulation.

    Args:
        config: Configuration dictionary
        scheduler_result: Scheduler output dictionary
        tau_s_per_ue: Optional propagation delay per UE

    Returns:
        Extended result with traffic_kpi
    """
    simulator = TrafficSimulator(config)
    return simulator.simulate(scheduler_result, tau_s_per_ue)
