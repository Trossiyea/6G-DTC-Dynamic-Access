"""
Unit tests for traffic module.

Tests cover:
- Packet data structure
- Buffer management
- Traffic generators
- QoS management
- Statistics tracking
- Traffic simulator integration
"""

import sys
import os

# Add the DL directory to path for direct submodule imports
_DL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.join(_DL_ROOT, "code")
sys.path.insert(0, _DL_ROOT)
sys.path.insert(0, _CODE_ROOT)  # For imports like "from traffic import ..."

# Import traffic submodules directly to avoid code/__init__.py issues
import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Pre-load traffic submodules to avoid code/__init__.py
_traffic_path = os.path.join(_DL_ROOT, "code", "traffic")
_simulation_path = os.path.join(_DL_ROOT, "code", "simulation")
_packets = _import_module("code.traffic.packets", os.path.join(_traffic_path, "packets.py"))
_qos = _import_module("code.traffic.qos", os.path.join(_traffic_path, "qos.py"))
_buffer = _import_module("code.traffic.buffer", os.path.join(_traffic_path, "buffer.py"))
_models = _import_module("code.traffic.models", os.path.join(_traffic_path, "models.py"))
_statistics = _import_module("code.traffic.statistics", os.path.join(_traffic_path, "statistics.py"))
_traffic_simulator = _import_module("code.simulation.traffic_simulator", os.path.join(_simulation_path, "traffic_simulator.py"))

import unittest
import numpy as np
from typing import List

# Import from preloaded modules
Packet = _packets.Packet
UEBuffer = _buffer.UEBuffer
BufferManager = _buffer.BufferManager
QoSClass = _qos.QoSClass
QoSManager = _qos.QoSManager
NTNQoSManager = _qos.NTNQoSManager
STANDARD_5QI = _qos.STANDARD_5QI
FullBufferGenerator = _models.FullBufferGenerator
PoissonGenerator = _models.PoissonGenerator
FTPModel3Generator = _models.FTPModel3Generator
VideoStreamingGenerator = _models.VideoStreamingGenerator
VoIPGenerator = _models.VoIPGenerator
create_traffic_generator = _models.create_traffic_generator
TrafficStatisticsTracker = _statistics.TrafficStatisticsTracker
TrafficSimulator = _traffic_simulator.TrafficSimulator


class TestPacket(unittest.TestCase):
    """Test Packet data structure."""

    def test_packet_creation(self):
        """Test basic packet creation."""
        pkt = Packet(
            packet_id=1,
            ue_id=0,
            bearer_id=0,
            size_bits=12000,
            arrival_tti=10,
            deadline_tti=100,
            priority=0,
            qci=9,
        )
        self.assertEqual(pkt.packet_id, 1)
        self.assertEqual(pkt.size_bits, 12000)
        self.assertEqual(pkt.remaining_bits, 12000)
        self.assertFalse(pkt.is_complete())

    def test_packet_expiry(self):
        """Test packet expiry detection."""
        pkt = Packet(0, 0, 0, 1000, arrival_tti=10, deadline_tti=50, priority=0, qci=9)

        self.assertFalse(pkt.is_expired(30))
        self.assertFalse(pkt.is_expired(50))
        self.assertTrue(pkt.is_expired(51))

    def test_packet_no_deadline(self):
        """Test packet without deadline (None)."""
        pkt = Packet(0, 0, 0, 1000, arrival_tti=10, deadline_tti=None, priority=0, qci=9)

        self.assertFalse(pkt.is_expired(100))
        self.assertFalse(pkt.is_expired(10000))

    def test_packet_deduct_bits(self):
        """Test bit deduction."""
        pkt = Packet(0, 0, 0, 1000, arrival_tti=0, deadline_tti=100, priority=0, qci=9)

        pkt.deduct_bits(300)
        self.assertEqual(pkt.remaining_bits, 700)

        pkt.deduct_bits(800)  # More than remaining
        self.assertEqual(pkt.remaining_bits, 0)
        self.assertTrue(pkt.is_complete())

    def test_hol_delay(self):
        """Test head-of-line delay calculation."""
        pkt = Packet(0, 0, 0, 1000, arrival_tti=10, deadline_tti=100, priority=0, qci=9)

        self.assertEqual(pkt.head_of_line_delay(10), 0)
        self.assertEqual(pkt.head_of_line_delay(25), 15)


class TestUEBuffer(unittest.TestCase):
    """Test UEBuffer class."""

    def test_enqueue(self):
        """Test packet enqueueing."""
        buf = UEBuffer(ue_id=0)

        pkt1 = Packet(0, 0, 0, 1000, 0, 100, 0, 9)
        pkt2 = Packet(1, 0, 0, 2000, 1, 100, 0, 9)

        buf.enqueue(pkt1)
        self.assertEqual(buf.total_bits, 1000)
        self.assertEqual(buf.total_packets, 1)

        buf.enqueue(pkt2)
        self.assertEqual(buf.total_bits, 3000)
        self.assertEqual(buf.total_packets, 2)

    def test_urllc_embb_tracking(self):
        """Test URLLC vs eMBB bit tracking."""
        buf = UEBuffer(ue_id=0)

        # URLLC packet (QCI=1)
        pkt_urllc = Packet(0, 0, 0, 1000, 0, 100, 0, 1)
        buf.enqueue(pkt_urllc)
        self.assertEqual(buf.urllc_bits, 1000)
        self.assertEqual(buf.embb_bits, 0)

        # eMBB packet (QCI=9)
        pkt_embb = Packet(1, 0, 0, 2000, 0, 100, 0, 9)
        buf.enqueue(pkt_embb)
        self.assertEqual(buf.urllc_bits, 1000)
        self.assertEqual(buf.embb_bits, 2000)

    def test_dequeue_bits(self):
        """Test bit dequeuing."""
        buf = UEBuffer(ue_id=0)

        pkt1 = Packet(0, 0, 0, 1000, 0, 100, 0, 9)
        pkt2 = Packet(1, 0, 0, 2000, 1, 100, 0, 9)
        buf.enqueue(pkt1)
        buf.enqueue(pkt2)

        # Dequeue part of first packet
        completed = buf.dequeue_bits(500, current_tti=5)
        self.assertEqual(len(completed), 0)  # Not complete yet
        self.assertEqual(buf.total_bits, 3000)  # Still tracking original

        # Dequeue rest of first packet
        completed = buf.dequeue_bits(600, current_tti=6)
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].packet_id, 0)
        self.assertEqual(buf.total_bits, 2000)

    def test_drop_expired(self):
        """Test expired packet dropping."""
        buf = UEBuffer(ue_id=0)

        pkt1 = Packet(0, 0, 0, 1000, 0, 50, 0, 9)  # Expires at TTI 50
        pkt2 = Packet(1, 0, 0, 2000, 0, 100, 0, 9)  # Expires at TTI 100
        buf.enqueue(pkt1)
        buf.enqueue(pkt2)

        dropped = buf.drop_expired(current_tti=60)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].packet_id, 0)
        self.assertEqual(buf.total_bits, 2000)

    def test_hol_delay(self):
        """Test HOL delay calculation."""
        buf = UEBuffer(ue_id=0)

        pkt = Packet(0, 0, 0, 1000, arrival_tti=10, deadline_tti=100, priority=0, qci=9)
        buf.enqueue(pkt)

        self.assertEqual(buf.head_of_line_delay(current_tti=25), 15)


class TestBufferManager(unittest.TestCase):
    """Test BufferManager class."""

    def test_creation(self):
        """Test buffer manager creation."""
        mgr = BufferManager(n_ue=10, config={})
        self.assertEqual(len(mgr.buffers), 10)

    def test_enqueue_dequeue(self):
        """Test enqueue and dequeue through manager."""
        mgr = BufferManager(n_ue=5, config={})

        pkt = Packet(0, 2, 0, 1000, 0, 100, 0, 9)
        mgr.enqueue(pkt)

        state = mgr.get_buffer_state(2)
        self.assertEqual(state["total_bits"], 1000)

        completed = mgr.dequeue_bits(2, 1000, current_tti=5)
        self.assertEqual(len(completed), 1)

    def test_drop_all_expired(self):
        """Test dropping expired packets across all UEs."""
        mgr = BufferManager(n_ue=3, config={})

        # Add packets to different UEs
        mgr.enqueue(Packet(0, 0, 0, 1000, 0, 50, 0, 9))
        mgr.enqueue(Packet(1, 1, 0, 1000, 0, 50, 0, 9))
        mgr.enqueue(Packet(2, 2, 0, 1000, 0, 100, 0, 9))

        dropped = mgr.drop_all_expired(current_tti=60)
        self.assertEqual(len(dropped), 2)  # UE 0 and UE 1 packets dropped


class TestQoSManager(unittest.TestCase):
    """Test QoS management."""

    def test_standard_5qi(self):
        """Test standard 5QI definitions."""
        self.assertIn(1, STANDARD_5QI)
        self.assertIn(9, STANDARD_5QI)
        self.assertIn(82, STANDARD_5QI)

        # Check conversational voice
        qos_voice = STANDARD_5QI[1]
        self.assertEqual(qos_voice.packet_delay_budget_ms, 100.0)
        self.assertEqual(qos_voice.resource_type, "GBR")

    def test_deadline_calculation(self):
        """Test deadline TTI calculation."""
        qos_mgr = QoSManager({"tti_ms": 1.0})

        # QCI 9: 300ms PDB
        deadline = qos_mgr.get_deadline_ttis(9)
        self.assertEqual(deadline, 300)

        # QCI 1: 100ms PDB
        deadline = qos_mgr.get_deadline_ttis(1)
        self.assertEqual(deadline, 100)

    def test_priority(self):
        """Test priority lookup."""
        qos_mgr = QoSManager({"tti_ms": 1.0})

        # Lower priority number = higher priority
        self.assertLess(qos_mgr.get_priority(1), qos_mgr.get_priority(9))

    def test_ntn_qos_manager(self):
        """Test NTN QoS manager with RTT compensation."""
        tau_s = np.array([0.010, 0.020])  # 10ms, 20ms one-way delay

        ntn_qos = NTNQoSManager(
            {"tti_ms": 1.0, "compensate_rtt_in_pdb": True},
            tau_s_per_ue=tau_s
        )

        # QCI 9: 300ms PDB, UE 0: 20ms RTT
        # Effective: 300 - 20 = 280 TTIs
        eff_pdb = ntn_qos.get_effective_pdb_ttis(ue=0, qci=9)
        self.assertEqual(eff_pdb, 280)

        # UE 1: 40ms RTT
        # Effective: 300 - 40 = 260 TTIs
        eff_pdb = ntn_qos.get_effective_pdb_ttis(ue=1, qci=9)
        self.assertEqual(eff_pdb, 260)


class TestTrafficGenerators(unittest.TestCase):
    """Test traffic generator implementations."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_full_buffer(self):
        """Test full buffer generator."""
        gen = FullBufferGenerator()

        arrivals = gen.generate_arrivals(0, 0, self.rng)
        self.assertEqual(len(arrivals), 0)  # No new arrivals
        self.assertTrue(gen.is_full_buffer())

    def test_poisson_generator(self):
        """Test Poisson traffic generator."""
        gen = PoissonGenerator(
            arrival_rate_hz=1000.0,  # 1000 packets/s
            packet_size_bytes=100,
            tti_duration_ms=1.0,
            qci=9,
        )

        # Generate arrivals over many TTIs
        total_arrivals = 0
        for tti in range(1000):
            arrivals = gen.generate_arrivals(tti, 0, self.rng)
            total_arrivals += len(arrivals)
            for pkt in arrivals:
                self.assertEqual(pkt.size_bits, 800)  # 100 bytes * 8
                self.assertEqual(pkt.qci, 9)

        # Should be approximately 1000 arrivals (Poisson)
        self.assertGreater(total_arrivals, 800)
        self.assertLess(total_arrivals, 1200)

    def test_ftp3_generator(self):
        """Test FTP Model 3 generator."""
        gen = FTPModel3Generator(
            file_size_bytes=1000,
            reading_time_ms=100.0,
            tti_duration_ms=1.0,
        )

        # First TTI should generate a file
        arrivals = gen.generate_arrivals(0, 0, self.rng)
        self.assertEqual(len(arrivals), 1)
        self.assertEqual(arrivals[0].size_bits, 8000)

        # Next few TTIs should not generate (reading time)
        for tti in range(1, 50):
            arrivals = gen.generate_arrivals(tti, 0, self.rng)
            # May or may not have arrivals depending on exponential sample

    def test_video_streaming(self):
        """Test video streaming generator."""
        gen = VideoStreamingGenerator(
            frame_rate_fps=30.0,
            i_frame_size_bytes=5000,
            p_frame_size_bytes=1000,
            gop_size=15,
            tti_duration_ms=1.0,
        )

        # First frame should be I-frame
        arrivals = gen.generate_arrivals(0, 0, self.rng)
        self.assertEqual(len(arrivals), 1)
        self.assertEqual(arrivals[0].size_bits, 5000 * 8)

        # Generate frames over 1 second (30 frames)
        i_frames = 0
        p_frames = 0
        for tti in range(1000):  # 1 second
            arrivals = gen.generate_arrivals(tti, 0, self.rng)
            for pkt in arrivals:
                if pkt.size_bits == 5000 * 8:
                    i_frames += 1
                else:
                    p_frames += 1

        # Should have ~2 I-frames per second (every 15 frames)
        self.assertGreaterEqual(i_frames, 1)
        self.assertLessEqual(i_frames, 4)

    def test_voip_generator(self):
        """Test VoIP generator."""
        gen = VoIPGenerator(
            codec="AMR-WB",
            activity_factor=1.0,  # Always active for testing
            packet_interval_ms=20.0,
            tti_duration_ms=1.0,
        )

        # Generate over 100ms (should be ~5 packets at 20ms interval)
        total_packets = 0
        for tti in range(100):
            arrivals = gen.generate_arrivals(tti, 0, self.rng)
            total_packets += len(arrivals)

        # Should be approximately 5 packets
        self.assertGreater(total_packets, 3)
        self.assertLess(total_packets, 8)

    def test_create_traffic_generator(self):
        """Test factory function."""
        config_fb = {"traffic_model": "full_buffer"}
        gen = create_traffic_generator(config_fb)
        self.assertIsInstance(gen, FullBufferGenerator)

        config_poisson = {
            "traffic_model": "poisson",
            "poisson_arrival_rate_hz": 100.0,
            "poisson_packet_size_bytes": 1500,
            "tti_ms": 1.0,
        }
        gen = create_traffic_generator(config_poisson)
        self.assertIsInstance(gen, PoissonGenerator)


class TestStatisticsTracker(unittest.TestCase):
    """Test statistics tracking."""

    def test_basic_tracking(self):
        """Test basic arrival/delivery tracking."""
        config = {"record_packet_latency": True, "max_stored_packets": 1000}
        stats = TrafficStatisticsTracker(config, n_ue=5, tti_ms=1.0)

        # Simulate arrivals
        pkt1 = Packet(0, 0, 0, 1000, arrival_tti=0, deadline_tti=100, priority=0, qci=9)
        pkt2 = Packet(1, 0, 0, 2000, arrival_tti=5, deadline_tti=100, priority=0, qci=9)

        stats.on_packet_arrived(pkt1)
        stats.on_packet_arrived(pkt2)

        self.assertEqual(stats.arrived, 2)
        self.assertEqual(stats.total_bits_arrived, 3000)

        # Deliver first packet
        stats.on_packet_delivered(pkt1, delivery_tti=10)
        self.assertEqual(stats.delivered, 1)

        # Drop second packet
        stats.on_packet_dropped(pkt2, drop_tti=101, reason="timeout")
        self.assertEqual(stats.dropped_timeout, 1)

    def test_latency_cdf(self):
        """Test latency CDF computation."""
        config = {"record_packet_latency": True, "max_stored_packets": 1000}
        stats = TrafficStatisticsTracker(config, n_ue=1, tti_ms=1.0)

        # Simulate 100 packets with varying latency
        for i in range(100):
            pkt = Packet(i, 0, 0, 1000, arrival_tti=i*10, deadline_tti=1000, priority=0, qci=9)
            stats.on_packet_arrived(pkt)
            # Latency = 5 + (i % 10) TTIs = 5-14ms
            stats.on_packet_delivered(pkt, delivery_tti=i*10 + 5 + (i % 10))

        cdf = stats.compute_latency_cdf([50, 90, 99])
        self.assertEqual(cdf["count"], 100)
        self.assertGreater(cdf["mean_ms"], 0)
        self.assertIn("p50", cdf["percentiles"])

    def test_per_qos_stats(self):
        """Test per-QoS statistics."""
        config = {"record_packet_latency": True, "record_per_qos_stats": True}
        stats = TrafficStatisticsTracker(config, n_ue=1, tti_ms=1.0)

        # URLLC packet
        pkt_urllc = Packet(0, 0, 0, 1000, 0, 100, 0, 1)
        stats.on_packet_arrived(pkt_urllc)
        stats.on_packet_delivered(pkt_urllc, 5)

        # eMBB packet
        pkt_embb = Packet(1, 0, 0, 1000, 0, 100, 0, 9)
        stats.on_packet_arrived(pkt_embb)
        stats.on_packet_delivered(pkt_embb, 20)

        per_qos = stats.compute_per_qos_latency()
        self.assertIn(1, per_qos)
        self.assertIn(9, per_qos)
        self.assertEqual(per_qos[1]["mean_ms"], 5.0)
        self.assertEqual(per_qos[9]["mean_ms"], 20.0)

    def test_summary(self):
        """Test summary generation."""
        config = {"record_packet_latency": True}
        stats = TrafficStatisticsTracker(config, n_ue=2, tti_ms=1.0)

        for i in range(10):
            pkt = Packet(i, i % 2, 0, 1000, i, 100, 0, 9)
            stats.on_packet_arrived(pkt)
            if i < 8:
                stats.on_packet_delivered(pkt, i + 5)
            else:
                stats.on_packet_dropped(pkt, 101, "timeout")

        summary = stats.get_summary()
        self.assertEqual(summary["packets_arrived"], 10)
        self.assertEqual(summary["packets_delivered"], 8)
        self.assertEqual(summary["packets_dropped_timeout"], 2)
        self.assertAlmostEqual(summary["packet_loss_rate"], 0.2)


class TestTrafficSimulator(unittest.TestCase):
    """Test traffic simulator integration."""

    def test_full_buffer_passthrough(self):
        """Test that full_buffer mode passes through unchanged."""
        config = {
            "traffic_model": "full_buffer",
            "N_UE": 10,
            "T": 100,
        }

        scheduler_result = {
            "avg_se_radiomap": 2.5,
            "some_data": [1, 2, 3],
        }

        simulator = TrafficSimulator(config)
        result = simulator.simulate(scheduler_result)

        # Should return unchanged
        self.assertEqual(result["avg_se_radiomap"], 2.5)
        self.assertNotIn("traffic_kpi", result)

    def test_simulation_with_throughput_data(self):
        """Test simulation with per-TTI throughput data."""
        config = {
            "traffic_model": "poisson",
            "N_UE": 5,
            "T": 100,
            "tti_ms": 1.0,
            "seed": 42,
            "poisson_arrival_rate_hz": 50.0,
            "poisson_packet_size_bytes": 1500,
        }

        # Simulate scheduler output: each UE gets ~10 bits/RE per TTI
        ue_thr_time = np.random.rand(100, 5) * 20  # [T, N_UE]

        scheduler_result = {
            "avg_se_radiomap": 2.5,
            "ue_thr_time_rm": ue_thr_time.tolist(),
        }

        simulator = TrafficSimulator(config)
        result = simulator.simulate(scheduler_result)

        self.assertIn("traffic_kpi", result)
        kpi = result["traffic_kpi"]
        self.assertIn("packets_arrived", kpi)
        self.assertIn("packets_delivered", kpi)
        self.assertIn("latency_cdf", kpi)
        self.assertEqual(kpi["simulation_mode"], "post_processing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
