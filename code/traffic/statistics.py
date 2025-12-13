"""
Traffic statistics tracking and KPI calculation.

Collects per-packet latency, packet loss events, and computes system-level
metrics including latency CDF, goodput, and per-QoS statistics.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .packets import Packet


class TrafficStatisticsTracker:
    """Tracks all traffic-related KPIs during simulation.

    Maintains counters for arrivals, deliveries, drops, and optional
    per-packet latency samples for CDF computation.
    """

    def __init__(self, config: Dict, n_ue: int, tti_ms: float):
        """Initialize statistics tracker.

        Args:
            config: Configuration dictionary
            n_ue: Number of UEs
            tti_ms: TTI duration in milliseconds
        """
        self.n_ue = n_ue
        self.tti_ms = tti_ms
        self.record_per_packet = config.get("record_packet_latency", True)
        self.max_samples = config.get("max_stored_packets", 1_000_000)
        self.record_per_qos = config.get("record_per_qos_stats", True)

        # Global counters
        self.arrived = 0
        self.delivered = 0
        self.dropped_timeout = 0
        self.dropped_overflow = 0

        # Bit counters
        self.total_bits_arrived = 0
        self.total_bits_delivered = 0
        self.total_bits_dropped = 0

        # Latency tracking
        self.latency_samples: List[float] = []  # in ms
        self.latency_per_qos: Dict[int, List[float]] = defaultdict(list)

        # Per-UE tracking
        self.per_ue_delivered_bits = np.zeros(n_ue, dtype=np.int64)
        self.per_ue_arrived_bits = np.zeros(n_ue, dtype=np.int64)
        self.per_ue_delivered_packets = np.zeros(n_ue, dtype=np.int64)
        self.per_ue_arrived_packets = np.zeros(n_ue, dtype=np.int64)

        # Optional time series tracking
        self.buffer_occupancy_time: List[np.ndarray] = []

    def on_packet_arrived(self, packet: Packet) -> None:
        """Record packet arrival.

        Args:
            packet: Newly arrived packet
        """
        self.arrived += 1
        self.total_bits_arrived += packet.size_bits
        if packet.ue_id < self.n_ue:
            self.per_ue_arrived_bits[packet.ue_id] += packet.size_bits
            self.per_ue_arrived_packets[packet.ue_id] += 1

    def on_packet_delivered(self, packet: Packet, delivery_tti: int) -> None:
        """Record successful packet delivery.

        Args:
            packet: Completed packet
            delivery_tti: TTI when packet was fully delivered
        """
        self.delivered += 1
        self.total_bits_delivered += packet.size_bits

        if packet.ue_id < self.n_ue:
            self.per_ue_delivered_bits[packet.ue_id] += packet.size_bits
            self.per_ue_delivered_packets[packet.ue_id] += 1

        # Record latency
        latency_ttis = delivery_tti - packet.arrival_tti
        latency_ms = latency_ttis * self.tti_ms

        if self.record_per_packet and len(self.latency_samples) < self.max_samples:
            self.latency_samples.append(latency_ms)

            if self.record_per_qos:
                self.latency_per_qos[packet.qci].append(latency_ms)

    def on_packet_dropped(self, packet: Packet, drop_tti: int, reason: str = "timeout") -> None:
        """Record packet drop.

        Args:
            packet: Dropped packet
            drop_tti: TTI when packet was dropped
            reason: Drop reason ("timeout" or "overflow")
        """
        self.total_bits_dropped += packet.size_bits

        if reason == "timeout":
            self.dropped_timeout += 1
        else:
            self.dropped_overflow += 1

    def record_buffer_snapshot(self, buffer_occupancy: np.ndarray) -> None:
        """Record buffer occupancy snapshot for time series analysis.

        Args:
            buffer_occupancy: Buffer bits per UE [N_UE]
        """
        if len(self.buffer_occupancy_time) < 10000:  # Limit memory
            self.buffer_occupancy_time.append(np.copy(buffer_occupancy))

    def compute_latency_cdf(self, percentiles: Optional[List[float]] = None) -> Dict:
        """Compute latency CDF and percentiles.

        Args:
            percentiles: List of percentiles to compute (default: [50, 90, 95, 99, 99.9])

        Returns:
            Dictionary with latency statistics
        """
        if not self.latency_samples:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "percentiles": {},
            }

        arr = np.array(self.latency_samples)
        percentiles = percentiles or [50.0, 90.0, 95.0, 99.0, 99.9]

        return {
            "count": len(arr),
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "percentiles": {
                f"p{p}": float(np.percentile(arr, p)) for p in percentiles
            },
        }

    def compute_per_qos_latency(self) -> Dict[int, Dict]:
        """Compute latency statistics per QoS class.

        Returns:
            Dictionary mapping QCI to latency stats
        """
        result = {}
        for qci, samples in self.latency_per_qos.items():
            if samples:
                arr = np.array(samples)
                result[qci] = {
                    "count": len(arr),
                    "mean_ms": float(np.mean(arr)),
                    "p50_ms": float(np.percentile(arr, 50)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "p99_ms": float(np.percentile(arr, 99)),
                }
        return result

    def compute_per_ue_goodput_fairness(self) -> Dict:
        """Compute fairness index on per-UE goodput.

        Returns:
            Dictionary with fairness metrics
        """
        # Jain's fairness index: (sum x)^2 / (n * sum x^2)
        goodput = self.per_ue_delivered_bits
        s = np.sum(goodput)
        s2 = np.sum(goodput * goodput)
        n = self.n_ue

        if s2 > 0:
            fairness = float((s * s) / (n * s2))
        else:
            fairness = 0.0

        return {
            "jain_index": fairness,
            "min_goodput_bits": int(np.min(goodput)),
            "max_goodput_bits": int(np.max(goodput)),
            "mean_goodput_bits": float(np.mean(goodput)),
            "std_goodput_bits": float(np.std(goodput)),
        }

    def get_summary(self) -> Dict:
        """Get final statistics summary.

        Returns:
            Dictionary with all KPIs
        """
        total = self.arrived
        packet_loss_rate = (self.dropped_timeout + self.dropped_overflow) / max(total, 1)
        goodput_ratio = self.delivered / max(total, 1)

        # Throughput vs goodput (bits)
        throughput_goodput_ratio = self.total_bits_delivered / max(self.total_bits_arrived, 1)

        summary = {
            # Packet counters
            "packets_arrived": self.arrived,
            "packets_delivered": self.delivered,
            "packets_dropped_timeout": self.dropped_timeout,
            "packets_dropped_overflow": self.dropped_overflow,
            "packet_loss_rate": packet_loss_rate,
            "packet_delivery_rate": goodput_ratio,

            # Bit counters
            "total_bits_arrived": self.total_bits_arrived,
            "total_bits_delivered": self.total_bits_delivered,
            "total_bits_dropped": self.total_bits_dropped,
            "goodput_ratio": throughput_goodput_ratio,

            # Latency
            "latency_cdf": self.compute_latency_cdf(),
            "latency_per_qos": self.compute_per_qos_latency(),

            # Per-UE
            "per_ue_goodput_bits": self.per_ue_delivered_bits.tolist(),
            "per_ue_goodput_fairness": self.compute_per_ue_goodput_fairness(),
        }

        return summary

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("Traffic Statistics Summary")
        print("=" * 60)

        print(f"\nPackets:")
        print(f"  Arrived:   {summary['packets_arrived']}")
        print(f"  Delivered: {summary['packets_delivered']}")
        print(f"  Dropped:   {summary['packets_dropped_timeout'] + summary['packets_dropped_overflow']}")
        print(f"    - Timeout:  {summary['packets_dropped_timeout']}")
        print(f"    - Overflow: {summary['packets_dropped_overflow']}")
        print(f"  Loss Rate: {summary['packet_loss_rate']:.4f}")

        print(f"\nGoodput:")
        print(f"  Ratio:     {summary['goodput_ratio']:.4f}")
        print(f"  Delivered: {summary['total_bits_delivered'] / 1e6:.2f} Mbits")

        lat = summary['latency_cdf']
        if lat['count'] > 0:
            print(f"\nLatency (ms):")
            print(f"  Mean:  {lat['mean_ms']:.2f}")
            print(f"  P50:   {lat['percentiles'].get('p50.0', 0):.2f}")
            print(f"  P90:   {lat['percentiles'].get('p90.0', 0):.2f}")
            print(f"  P95:   {lat['percentiles'].get('p95.0', 0):.2f}")
            print(f"  P99:   {lat['percentiles'].get('p99.0', 0):.2f}")

        fairness = summary['per_ue_goodput_fairness']
        print(f"\nFairness:")
        print(f"  Jain Index: {fairness['jain_index']:.4f}")

        print("=" * 60 + "\n")
