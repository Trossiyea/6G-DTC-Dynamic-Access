"""
Unit tests for Scheduler Integration module (Phase 10).

Tests cover:
- QoS metric computation (PF, M-LWDF, EXP-PF, EDF)
- MACSchedulerBridge functionality
- NTN HARQ adapter
- Integration with MAC modules
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

import unittest
import numpy as np


class TestQoSMetrics(unittest.TestCase):
    """Test QoS metric computation functions."""

    def test_import(self):
        """Test module can be imported."""
        from scheduler.integration import (
            compute_pf_metric,
            compute_mlwdf_metric,
            compute_exppf_metric,
            compute_edf_metric,
        )
        self.assertIsNotNone(compute_pf_metric)
        self.assertIsNotNone(compute_mlwdf_metric)
        self.assertIsNotNone(compute_exppf_metric)
        self.assertIsNotNone(compute_edf_metric)

    def test_pf_metric_1d(self):
        """Test PF metric with 1D input."""
        from scheduler.integration import compute_pf_metric

        se_metric = np.array([10.0, 20.0, 30.0])
        avg_throughput = np.array([5.0, 10.0, 15.0])

        metric = compute_pf_metric(se_metric, avg_throughput)

        # PF = se / avg_thr
        expected = se_metric / avg_throughput
        np.testing.assert_array_almost_equal(metric, expected)

    def test_pf_metric_2d(self):
        """Test PF metric with 2D input (per-PRB)."""
        from scheduler.integration import compute_pf_metric

        se_metric = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])  # [2 UE, 3 PRB]
        avg_throughput = np.array([1.0, 2.0])

        metric = compute_pf_metric(se_metric, avg_throughput)

        self.assertEqual(metric.shape, (2, 3))
        # UE 0: se / 1.0 = se
        np.testing.assert_array_almost_equal(metric[0], [1.0, 2.0, 3.0])
        # UE 1: se / 2.0 = se / 2
        np.testing.assert_array_almost_equal(metric[1], [2.0, 2.5, 3.0])

    def test_mlwdf_metric(self):
        """Test M-LWDF metric computation."""
        from scheduler.integration import compute_mlwdf_metric

        se_metric = np.array([10.0, 20.0])
        avg_throughput = np.array([5.0, 10.0])
        hol_delay_ms = np.array([50.0, 100.0])
        qos_params = np.array([
            [0.01, 100.0],  # delta=0.01, tau=100ms
            [0.001, 50.0],  # delta=0.001, tau=50ms
        ])

        metric = compute_mlwdf_metric(
            se_metric, avg_throughput, hol_delay_ms, qos_params
        )

        # Higher delay should give higher metric
        self.assertEqual(len(metric), 2)
        # UE 1 has higher HoL delay and stricter QoS, should have higher priority component
        # (ignoring SE/Rbar part)

    def test_mlwdf_delay_sensitivity(self):
        """Test M-LWDF is sensitive to delay."""
        from scheduler.integration import compute_mlwdf_metric

        se_metric = np.array([10.0, 10.0])  # Same SE
        avg_throughput = np.array([5.0, 5.0])  # Same avg
        qos_params = np.array([
            [0.01, 100.0],
            [0.01, 100.0],
        ])  # Same QoS

        # Different delays
        hol_delay_1 = np.array([10.0, 10.0])
        hol_delay_2 = np.array([10.0, 50.0])  # UE 1 has higher delay

        metric_1 = compute_mlwdf_metric(
            se_metric, avg_throughput, hol_delay_1, qos_params
        )
        metric_2 = compute_mlwdf_metric(
            se_metric, avg_throughput, hol_delay_2, qos_params
        )

        # UE 1 should have higher metric with higher delay
        self.assertAlmostEqual(metric_1[0], metric_1[1])  # Same with same delay
        self.assertGreater(metric_2[1], metric_2[0])  # Higher delay = higher metric

    def test_exppf_metric(self):
        """Test EXP-PF metric computation."""
        from scheduler.integration import compute_exppf_metric

        se_metric = np.array([10.0, 20.0, 15.0])
        avg_throughput = np.array([5.0, 10.0, 7.5])
        hol_delay_ms = np.array([50.0, 100.0, 75.0])
        qos_params = np.array([
            [0.01, 100.0],
            [0.01, 100.0],
            [0.01, 100.0],
        ])

        metric = compute_exppf_metric(
            se_metric, avg_throughput, hol_delay_ms, qos_params
        )

        self.assertEqual(len(metric), 3)
        # All values should be positive
        self.assertTrue(np.all(metric > 0))

    def test_edf_metric(self):
        """Test EDF metric computation."""
        from scheduler.integration import compute_edf_metric

        hol_delay_ms = np.array([10.0, 50.0, 90.0])
        deadline_ms = np.array([100.0, 100.0, 100.0])

        metric = compute_edf_metric(hol_delay_ms, deadline_ms)

        # EDF = 1 / (deadline - delay)
        # Lower slack = higher metric = higher priority
        self.assertGreater(metric[2], metric[1])  # 90ms delay > 50ms delay priority
        self.assertGreater(metric[1], metric[0])  # 50ms delay > 10ms delay priority


class TestQoSSchedulerConfig(unittest.TestCase):
    """Test QoS scheduler configuration."""

    def test_default_config(self):
        """Test default configuration."""
        from scheduler.integration import QoSSchedulerConfig, QoSSchedulerType

        cfg = QoSSchedulerConfig()
        self.assertEqual(cfg.algorithm, QoSSchedulerType.PF)
        self.assertEqual(cfg.pf_beta, 0.1)

    def test_from_config_dict(self):
        """Test configuration from dictionary."""
        from scheduler.integration import QoSSchedulerConfig, QoSSchedulerType

        config_dict = {
            "scheduler_algorithm": "m-lwdf",
            "pf_beta": 0.2,
            "mlwdf_delta": 0.001,
            "mlwdf_tau": 50.0,
        }

        cfg = QoSSchedulerConfig.from_config_dict(config_dict)
        self.assertEqual(cfg.algorithm, QoSSchedulerType.M_LWDF)
        self.assertEqual(cfg.pf_beta, 0.2)
        self.assertEqual(cfg.mlwdf_delta, 0.001)
        self.assertEqual(cfg.mlwdf_tau, 50.0)

    def test_algorithm_aliases(self):
        """Test algorithm name aliases."""
        from scheduler.integration import QoSSchedulerConfig, QoSSchedulerType

        # Test various aliases
        for algo_str, expected in [
            ("pf", QoSSchedulerType.PF),
            ("m-lwdf", QoSSchedulerType.M_LWDF),
            ("mlwdf", QoSSchedulerType.M_LWDF),
            ("exp-pf", QoSSchedulerType.EXP_PF),
            ("exppf", QoSSchedulerType.EXP_PF),
            ("edf", QoSSchedulerType.EDF),
        ]:
            cfg = QoSSchedulerConfig.from_config_dict({"scheduler_algorithm": algo_str})
            self.assertEqual(cfg.algorithm, expected, f"Failed for {algo_str}")


class TestMACSchedulerBridge(unittest.TestCase):
    """Test MAC-Scheduler bridge functionality."""

    def test_bridge_creation(self):
        """Test MACSchedulerBridge creation."""
        from scheduler.integration import MACSchedulerBridge

        bridge = MACSchedulerBridge(n_ue=10)
        self.assertEqual(bridge.n_ue, 10)

    def test_ue_mask_default(self):
        """Test UE mask returns all active by default."""
        from scheduler.integration import MACSchedulerBridge

        bridge = MACSchedulerBridge(n_ue=5)
        mask = bridge.get_ue_mask()

        # All UEs should be schedulable by default
        self.assertTrue(np.all(mask))
        self.assertEqual(len(mask), 5)

    def test_qos_metric_pf(self):
        """Test QoS metric computation (PF)."""
        from scheduler.integration import MACSchedulerBridge

        bridge = MACSchedulerBridge(n_ue=3, config={"scheduler_algorithm": "pf"})

        se_metric = np.array([10.0, 20.0, 30.0])
        bridge._avg_throughput = np.array([5.0, 10.0, 15.0])

        metric = bridge.compute_qos_metric(se_metric)

        # PF = se / avg
        expected = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(metric, expected)

    def test_qos_metric_mlwdf(self):
        """Test QoS metric computation (M-LWDF)."""
        from scheduler.integration import MACSchedulerBridge

        bridge = MACSchedulerBridge(n_ue=2, config={"scheduler_algorithm": "m-lwdf"})

        se_metric = np.array([10.0, 10.0])
        bridge._avg_throughput = np.array([5.0, 5.0])
        bridge._hol_delay_ms = np.array([10.0, 50.0])

        metric = bridge.compute_qos_metric(se_metric)

        # UE with higher delay should have higher metric
        self.assertGreater(metric[1], metric[0])

    def test_update_avg_throughput(self):
        """Test average throughput update."""
        from scheduler.integration import MACSchedulerBridge

        bridge = MACSchedulerBridge(n_ue=3)
        bridge._avg_throughput = np.array([1.0, 1.0, 1.0])

        new_throughput = np.array([10.0, 20.0, 30.0])
        bridge.update_avg_throughput(new_throughput, beta=0.5)

        # Avg = (1-0.5) * old + 0.5 * new
        expected = np.array([5.5, 10.5, 15.5])
        np.testing.assert_array_almost_equal(bridge._avg_throughput, expected)

    def test_update_hol_delay(self):
        """Test HoL delay update."""
        from scheduler.integration import MACSchedulerBridge

        bridge = MACSchedulerBridge(n_ue=3)
        hol_delay = np.array([10.0, 20.0, 30.0])
        bridge.update_hol_delay(hol_delay)

        np.testing.assert_array_almost_equal(bridge._hol_delay_ms, hol_delay)

    def test_update_qos_params(self):
        """Test QoS parameter update."""
        from scheduler.integration import MACSchedulerBridge

        bridge = MACSchedulerBridge(n_ue=3)
        bridge.update_qos_params(ue_id=1, delta=0.001, tau_ms=50.0)

        self.assertEqual(bridge._qos_params[1, 0], 0.001)
        self.assertEqual(bridge._qos_params[1, 1], 50.0)

    def test_factory_function(self):
        """Test create_mac_scheduler_bridge factory."""
        from scheduler.integration import create_mac_scheduler_bridge

        bridge = create_mac_scheduler_bridge(
            n_ue=10,
            config={"scheduler_algorithm": "exp-pf"},
        )

        self.assertEqual(bridge.n_ue, 10)
        from scheduler.integration import QoSSchedulerType
        self.assertEqual(bridge.qos_config.algorithm, QoSSchedulerType.EXP_PF)

    def test_statistics(self):
        """Test statistics retrieval."""
        from scheduler.integration import MACSchedulerBridge

        bridge = MACSchedulerBridge(n_ue=5, config={"scheduler_algorithm": "mlwdf"})
        stats = bridge.get_statistics()

        self.assertEqual(stats["n_ue"], 5)
        self.assertEqual(stats["qos_algorithm"], "m-lwdf")
        self.assertIn("components_enabled", stats)

    def test_reset(self):
        """Test bridge reset."""
        from scheduler.integration import MACSchedulerBridge

        bridge = MACSchedulerBridge(n_ue=3)
        bridge._avg_throughput = np.array([100.0, 200.0, 300.0])
        bridge._hol_delay_ms = np.array([10.0, 20.0, 30.0])

        bridge.reset()

        # Should be reset to defaults
        np.testing.assert_array_almost_equal(
            bridge._avg_throughput, np.array([1e-3, 1e-3, 1e-3])
        )
        np.testing.assert_array_almost_equal(
            bridge._hol_delay_ms, np.zeros(3)
        )


class TestBuildQoSParams(unittest.TestCase):
    """Test QoS parameter building from QCI."""

    def test_build_from_qci(self):
        """Test building QoS params from QCI values."""
        from scheduler.integration import build_qos_params_from_qci

        qci_per_ue = np.array([1, 5, 9])  # Voice, IMS, TCP
        params = build_qos_params_from_qci(qci_per_ue)

        self.assertEqual(params.shape, (3, 2))
        # QCI 1 (Voice): delta=0.01, tau=100
        self.assertEqual(params[0, 0], 0.01)
        self.assertEqual(params[0, 1], 100)

    def test_unknown_qci_fallback(self):
        """Test fallback for unknown QCI."""
        from scheduler.integration import build_qos_params_from_qci

        qci_per_ue = np.array([99, 100])  # Unknown QCIs
        params = build_qos_params_from_qci(qci_per_ue)

        # Should use default (QCI 9 equivalent)
        self.assertEqual(params.shape, (2, 2))


class TestNTNHarqAdapter(unittest.TestCase):
    """Test NTN HARQ adapter."""

    def test_adapter_creation(self):
        """Test NTNHarqAdapter creation."""
        from scheduler.ntn_harq import NTNHarqAdapter

        adapter = NTNHarqAdapter(n_ue=10)
        self.assertEqual(adapter.n_ue, 10)

    def test_default_k1(self):
        """Test default K1 values."""
        from scheduler.ntn_harq import NTNHarqAdapter

        adapter = NTNHarqAdapter(n_ue=5, config={"harq_ack_delay_ttis": 4})

        for ue in range(5):
            self.assertEqual(adapter.get_k1_for_ue(ue), 4)

    def test_update_from_geometry(self):
        """Test K1 update from geometry."""
        from scheduler.ntn_harq import NTNHarqAdapter

        # Use MEO scenario which has higher max_k1 (64) to allow K1 differentiation
        adapter = NTNHarqAdapter(n_ue=3, config={"tti_ms": 1.0, "ntn_scenario": "meo"})

        # Different one-way delays to produce different K1 values
        # 5ms one-way = 10ms RTT = 10 slots → K1 ~14
        # 10ms one-way = 20ms RTT = 20 slots → K1 ~24
        # 15ms one-way = 30ms RTT = 30 slots → K1 ~34
        tau_s = np.array([0.005, 0.010, 0.015])
        adapter.update_from_geometry(tau_s)

        # K1 should increase with delay
        k1_0 = adapter.get_k1_for_ue(0)
        k1_1 = adapter.get_k1_for_ue(1)
        k1_2 = adapter.get_k1_for_ue(2)

        self.assertLess(k1_0, k1_1)
        self.assertLess(k1_1, k1_2)

    def test_get_all_k1(self):
        """Test getting K1 for all UEs."""
        from scheduler.ntn_harq import NTNHarqAdapter

        adapter = NTNHarqAdapter(n_ue=5)
        all_k1 = adapter.get_all_k1()

        self.assertEqual(len(all_k1), 5)

    def test_harq_processes_scaling(self):
        """Test HARQ process scaling for high RTT."""
        from scheduler.ntn_harq import NTNHarqAdapter

        adapter = NTNHarqAdapter(n_ue=2, config={
            "harq_max_procs": 16,
            "harq_auto_scale_processes": True,
            "tti_ms": 1.0,
        })

        # Low delay - should use base processes
        tau_low = np.array([0.001, 0.001])  # 1ms
        adapter.update_from_geometry(tau_low)
        procs_low = adapter.get_harq_processes_for_ue(0)

        # Reset and use high delay
        adapter.reset()
        tau_high = np.array([0.050, 0.050])  # 50ms = 100ms RTT
        adapter.update_from_geometry(tau_high)
        procs_high = adapter.get_harq_processes_for_ue(0)

        # High RTT should have more processes
        self.assertGreaterEqual(procs_high, procs_low)

    def test_get_ack_slot(self):
        """Test ACK slot calculation."""
        from scheduler.ntn_harq import NTNHarqAdapter

        adapter = NTNHarqAdapter(n_ue=1, config={"harq_ack_delay_ttis": 4})

        ack_slot = adapter.get_ack_slot(ue_id=0, tx_slot=100)
        self.assertEqual(ack_slot, 104)

    def test_factory_function(self):
        """Test create_ntn_harq_adapter factory."""
        from scheduler.ntn_harq import create_ntn_harq_adapter

        tau_s = np.array([0.010] * 5)
        adapter = create_ntn_harq_adapter(
            n_ue=5,
            config={"tti_ms": 1.0},
            tau_s_per_ue=tau_s,
        )

        self.assertEqual(adapter.n_ue, 5)
        # Should have updated K1 from geometry
        k1 = adapter.get_k1_for_ue(0)
        self.assertGreater(k1, 0)

    def test_compute_min_harq_processes(self):
        """Test minimum HARQ process computation."""
        from scheduler.ntn_harq import compute_min_harq_processes

        # Low RTT
        procs_low = compute_min_harq_processes(rtt_ms=10.0, slot_ms=1.0)
        self.assertEqual(procs_low, 16)  # Base processes

        # High RTT (100ms = 100 slots, need ~51 processes, capped at 32)
        procs_high = compute_min_harq_processes(rtt_ms=100.0, slot_ms=1.0)
        self.assertEqual(procs_high, 32)  # Capped at maximum

    def test_statistics(self):
        """Test statistics retrieval."""
        from scheduler.ntn_harq import NTNHarqAdapter

        adapter = NTNHarqAdapter(n_ue=5)
        tau_s = np.array([0.005, 0.010, 0.015, 0.020, 0.025])
        adapter.update_from_geometry(tau_s)

        stats = adapter.get_statistics()

        self.assertEqual(stats["n_ue"], 5)
        self.assertIn("k1_min", stats)
        self.assertIn("k1_max", stats)
        self.assertIn("k1_mean", stats)
        self.assertGreater(stats["k1_max"], stats["k1_min"])

    def test_reset(self):
        """Test adapter reset."""
        from scheduler.ntn_harq import NTNHarqAdapter

        adapter = NTNHarqAdapter(n_ue=3, config={"harq_ack_delay_ttis": 4})
        tau_s = np.array([0.020] * 3)
        adapter.update_from_geometry(tau_s)

        # K1 should be > base after update
        k1_before = adapter.get_k1_for_ue(0)
        self.assertGreater(k1_before, 4)

        adapter.reset()

        # K1 should be back to base
        k1_after = adapter.get_k1_for_ue(0)
        self.assertEqual(k1_after, 4)


class TestSchedulerModuleExports(unittest.TestCase):
    """Test scheduler module exports."""

    def test_integration_exports(self):
        """Test integration classes are exported from scheduler."""
        from scheduler import (
            MACSchedulerBridge,
            QoSSchedulerType,
            QoSSchedulerConfig,
            create_mac_scheduler_bridge,
            compute_pf_metric,
            compute_mlwdf_metric,
            compute_exppf_metric,
            compute_edf_metric,
            build_qos_params_from_qci,
        )

        self.assertIsNotNone(MACSchedulerBridge)
        self.assertIsNotNone(QoSSchedulerType)
        self.assertIsNotNone(compute_mlwdf_metric)

    def test_ntn_harq_exports(self):
        """Test NTN HARQ classes are exported from scheduler."""
        from scheduler import (
            NTNHarqAdapter,
            NTNHarqConfig,
            create_ntn_harq_adapter,
            compute_min_harq_processes,
        )

        self.assertIsNotNone(NTNHarqAdapter)
        self.assertIsNotNone(NTNHarqConfig)


class TestIntegrationWithMACModules(unittest.TestCase):
    """Test integration with MAC modules (if available)."""

    def test_bridge_with_drx_config(self):
        """Test bridge with DRX configuration."""
        from scheduler.integration import MACSchedulerBridge

        bridge = MACSchedulerBridge(n_ue=5, config={
            "enable_drx": True,
            "drx_on_duration_ms": 10.0,
            "drx_inactivity_timer_ms": 100.0,
        })

        # Should have DRX controller if MAC module available
        stats = bridge.get_statistics()
        self.assertIn("components_enabled", stats)

    def test_bridge_geometry_update(self):
        """Test bridge geometry update."""
        from scheduler.integration import MACSchedulerBridge

        bridge = MACSchedulerBridge(n_ue=3, config={
            "enable_ntn_timing": True,
            "ntn_scenario": "leo_600km",
        })

        tau_s = np.array([0.010, 0.012, 0.015])
        bridge.update_from_geometry(tau_s)

        # Should be able to get K1 values
        k1 = bridge.get_k1(0)
        self.assertGreater(k1, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
