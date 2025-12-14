# -*- coding: utf-8 -*-
"""Unit tests for CSI delay benchmark module."""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))


class TestBenchmarkConfig(unittest.TestCase):
    """Test benchmark configuration."""

    def test_import(self):
        """Test module can be imported."""
        from benchmark import CSIDelayBenchmarkConfig, BenchmarkResult
        self.assertTrue(True)

    def test_default_config(self):
        """Test default configuration values."""
        from benchmark import CSIDelayBenchmarkConfig

        cfg = CSIDelayBenchmarkConfig()
        self.assertEqual(cfg.n_ue, 50)
        self.assertEqual(cfg.n_tti, 500)
        self.assertEqual(cfg.ntn_scenario, "leo_600km")
        self.assertTrue(cfg.enable_time_varying)
        self.assertEqual(cfg.rm_csi_delay_ttis, 0)

    def test_custom_config(self):
        """Test custom configuration."""
        from benchmark import CSIDelayBenchmarkConfig

        cfg = CSIDelayBenchmarkConfig(
            csi_delays_ttis=[0, 8, 16],
            n_ue=100,
            n_tti=200,
        )
        self.assertEqual(cfg.csi_delays_ttis, [0, 8, 16])
        self.assertEqual(cfg.n_ue, 100)
        self.assertEqual(cfg.n_tti, 200)


class TestBenchmarkResult(unittest.TestCase):
    """Test benchmark result dataclass."""

    def test_improvement_calculation(self):
        """Test improvement percentage calculation."""
        from benchmark import BenchmarkResult

        result = BenchmarkResult(
            csi_delay_ms=8.0,
            baseline_se=2.0,
            radiomap_se=2.5,
        )
        self.assertAlmostEqual(result.improvement_pct, 25.0)

    def test_zero_baseline(self):
        """Test improvement with zero baseline."""
        from benchmark import BenchmarkResult

        result = BenchmarkResult(
            csi_delay_ms=0.0,
            baseline_se=0.0,
            radiomap_se=1.0,
        )
        self.assertEqual(result.improvement_pct, 0.0)


class TestBaseConfig(unittest.TestCase):
    """Test base config generation."""

    def test_create_base_config(self):
        """Test base config creation."""
        from benchmark.csi_delay import create_base_config, CSIDelayBenchmarkConfig

        bench_cfg = CSIDelayBenchmarkConfig(n_ue=30, n_tti=100)
        config = create_base_config(bench_cfg)

        self.assertEqual(config["N_UE"], 30)
        self.assertEqual(config["T"], 100)
        self.assertEqual(config["sat_alt_km"], 600)
        self.assertTrue(config["enable_time_varying"])
        self.assertEqual(config["rm_csi_delay_ttis"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
