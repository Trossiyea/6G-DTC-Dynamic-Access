"""
Unit tests for MAC layer NTN Timing module (Phase 9.3).

Tests cover:
- NTN scenario definitions
- Timing configuration
- K1 table and extensions
- Timing Advance computation
- HARQ timing adaptation
- Scheduling timing management
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

import unittest
import numpy as np


class TestNTNScenario(unittest.TestCase):
    """Test NTN scenario definitions."""

    def test_import(self):
        """Test module can be imported."""
        from mac import NTNScenario, NTNTimingConfig, TimingAdvanceController
        self.assertIsNotNone(NTNScenario)
        self.assertIsNotNone(NTNTimingConfig)
        self.assertIsNotNone(TimingAdvanceController)

    def test_scenario_values(self):
        """Test NTN scenario enumeration."""
        from mac import NTNScenario

        # All scenarios should exist
        self.assertIsNotNone(NTNScenario.TERRESTRIAL)
        self.assertIsNotNone(NTNScenario.LEO_600KM)
        self.assertIsNotNone(NTNScenario.LEO_1200KM)
        self.assertIsNotNone(NTNScenario.MEO)
        self.assertIsNotNone(NTNScenario.GEO)

    def test_typical_delays(self):
        """Test typical delay values for scenarios."""
        from mac import NTNScenario

        # Terrestrial has no delay
        self.assertEqual(NTNScenario.TERRESTRIAL.typical_delay_ms, 0.0)

        # LEO has few ms delay
        self.assertGreater(NTNScenario.LEO_600KM.typical_delay_ms, 0)
        self.assertLess(NTNScenario.LEO_600KM.typical_delay_ms, 20)

        # GEO has ~120 ms delay
        self.assertGreater(NTNScenario.GEO.typical_delay_ms, 100)

    def test_typical_rtt(self):
        """Test RTT is twice the delay."""
        from mac import NTNScenario

        for scenario in NTNScenario:
            self.assertEqual(
                scenario.typical_rtt_ms,
                2.0 * scenario.typical_delay_ms
            )


class TestNTNTimingConfig(unittest.TestCase):
    """Test NTN timing configuration."""

    def test_default_config(self):
        """Test default timing configuration."""
        from mac import NTNTimingConfig, NTNScenario

        cfg = NTNTimingConfig()
        self.assertEqual(cfg.scenario, NTNScenario.LEO_600KM)
        self.assertEqual(cfg.k0_slots, 0)
        self.assertEqual(cfg.k1_slots_base, 4)
        self.assertEqual(cfg.k2_slots, 4)

    def test_from_config_dict(self):
        """Test configuration from flat dictionary."""
        from mac import NTNTimingConfig, NTNScenario

        config_dict = {
            "ntn_scenario": "geo",
            "k0_slots": 2,
            "k1_slots_base": 8,
            "k2_slots": 6,
            "tti_ms": 0.5,
        }

        cfg = NTNTimingConfig.from_config_dict(config_dict)
        self.assertEqual(cfg.scenario, NTNScenario.GEO)
        self.assertEqual(cfg.k0_slots, 2)
        self.assertEqual(cfg.k1_slots_base, 8)
        self.assertEqual(cfg.k2_slots, 6)
        self.assertEqual(cfg.slot_duration_ms, 0.5)

    def test_for_scenario_preset(self):
        """Test scenario preset configuration."""
        from mac import NTNTimingConfig, NTNScenario

        # LEO preset
        cfg_leo = NTNTimingConfig.for_scenario(NTNScenario.LEO_600KM)
        self.assertEqual(cfg_leo.scenario, NTNScenario.LEO_600KM)

        # GEO preset should have larger K1 extension
        cfg_geo = NTNTimingConfig.for_scenario(NTNScenario.GEO)
        self.assertGreater(cfg_geo.k1_ntn_extension_slots, cfg_leo.k1_ntn_extension_slots)

    def test_k1_effective_slots(self):
        """Test effective K1 calculation."""
        from mac import NTNTimingConfig

        cfg = NTNTimingConfig(
            k1_slots_base=4,
            k1_ntn_extension_slots=10
        )
        self.assertEqual(cfg.k1_effective_slots, 14)


class TestK1Table(unittest.TestCase):
    """Test K1 timing table."""

    def test_standard_values(self):
        """Test standard K1 values."""
        from mac.timing import K1Table

        table = K1Table()
        self.assertEqual(table.standard_values, [1, 2, 3, 4, 5, 6, 7, 8])

    def test_ntn_extended_values(self):
        """Test NTN extended K1 values."""
        from mac.timing import K1Table

        table = K1Table()
        self.assertIn(16, table.ntn_extended_values)
        self.assertIn(32, table.ntn_extended_values)

    def test_get_k1_values_terrestrial(self):
        """Test K1 values for terrestrial scenario."""
        from mac.timing import K1Table
        from mac import NTNScenario

        table = K1Table()
        values = table.get_k1_values(NTNScenario.TERRESTRIAL)

        # Should only include standard values up to max
        self.assertTrue(all(v <= 8 for v in values))

    def test_get_k1_values_ntn(self):
        """Test K1 values for NTN scenario."""
        from mac.timing import K1Table
        from mac import NTNScenario

        table = K1Table()
        values = table.get_k1_values(NTNScenario.LEO_600KM)

        # Should include extended values
        self.assertGreater(max(values), 8)


class TestTimingAdvanceController(unittest.TestCase):
    """Test Timing Advance controller."""

    def test_controller_creation(self):
        """Test TA controller creation."""
        from mac import TimingAdvanceController

        ctrl = TimingAdvanceController(n_ue=10)
        self.assertEqual(ctrl.n_ue, 10)

    def test_compute_ta_from_delay(self):
        """Test TA computation from propagation delay."""
        from mac import TimingAdvanceController

        ctrl = TimingAdvanceController(n_ue=1)

        # 10 ms one-way delay = 20 ms RTT = 20000 us TA
        ta = ctrl.compute_ta_from_delay(0.010)
        self.assertAlmostEqual(ta, 20000.0, places=1)

    def test_update_common_ta(self):
        """Test common TA update."""
        from mac import TimingAdvanceController

        ctrl = TimingAdvanceController(n_ue=3)
        ctrl.update_common_ta(ta_ms=10.0)

        # All UEs should have common TA updated
        for ue in range(3):
            self.assertEqual(ctrl._ue_states[ue].ta_common_us, 10000.0)

    def test_update_from_geometry(self):
        """Test TA update from propagation delays."""
        from mac import TimingAdvanceController

        ctrl = TimingAdvanceController(n_ue=3)

        # Propagation delays in seconds
        tau_s = np.array([0.005, 0.010, 0.015])  # 5, 10, 15 ms
        ctrl.update_ue_ta_from_geometry(tau_s)

        # Check TA values (RTT = 2 * delay)
        self.assertAlmostEqual(ctrl.get_ta_for_ue_ms(0), 10.0, places=1)
        self.assertAlmostEqual(ctrl.get_ta_for_ue_ms(1), 20.0, places=1)
        self.assertAlmostEqual(ctrl.get_ta_for_ue_ms(2), 30.0, places=1)

    def test_get_all_ta(self):
        """Test getting TA for all UEs."""
        from mac import TimingAdvanceController

        ctrl = TimingAdvanceController(n_ue=5)
        tau_s = np.array([0.010] * 5)  # 10 ms for all
        ctrl.update_ue_ta_from_geometry(tau_s)

        all_ta = ctrl.get_all_ta_ms()
        self.assertEqual(len(all_ta), 5)
        self.assertTrue(np.allclose(all_ta, 20.0))  # RTT = 20 ms

    def test_ta_with_reference(self):
        """Test TA computation with reference delay."""
        from mac import TimingAdvanceController

        ctrl = TimingAdvanceController(n_ue=3)

        # Reference delay (e.g., cell center)
        ref_delay = 0.010  # 10 ms

        # UE delays
        tau_s = np.array([0.008, 0.010, 0.012])  # 8, 10, 12 ms

        ctrl.update_ue_ta_from_geometry(tau_s, reference_delay_s=ref_delay)

        # Common TA should be based on reference
        self.assertAlmostEqual(ctrl._common_ta_us, 20000.0, places=1)

        # UE-specific should be delta from common
        # UE 0: 8ms -> 16ms RTT, common = 20ms, UE-specific = -4ms
        self.assertAlmostEqual(ctrl._ue_states[0].ta_ue_specific_us, -4000.0, places=1)

    def test_get_rtt(self):
        """Test RTT query."""
        from mac import TimingAdvanceController

        ctrl = TimingAdvanceController(n_ue=2)
        tau_s = np.array([0.010, 0.020])  # 10, 20 ms one-way
        ctrl.update_ue_ta_from_geometry(tau_s)

        self.assertAlmostEqual(ctrl.get_rtt_ms(0), 20.0, places=1)
        self.assertAlmostEqual(ctrl.get_rtt_ms(1), 40.0, places=1)

    def test_statistics(self):
        """Test TA statistics."""
        from mac import TimingAdvanceController

        ctrl = TimingAdvanceController(n_ue=5)
        tau_s = np.array([0.005, 0.010, 0.015, 0.020, 0.025])
        ctrl.update_ue_ta_from_geometry(tau_s)

        stats = ctrl.get_statistics()
        self.assertIn("min_delay_ms", stats)
        self.assertIn("max_delay_ms", stats)
        self.assertIn("mean_ta_ms", stats)


class TestHARQTimingAdapter(unittest.TestCase):
    """Test HARQ timing adapter."""

    def test_adapter_creation(self):
        """Test HARQ timing adapter creation."""
        from mac import HARQTimingAdapter

        adapter = HARQTimingAdapter(n_ue=10)
        self.assertEqual(adapter.n_ue, 10)

    def test_update_from_geometry(self):
        """Test K1 update from geometry."""
        from mac import HARQTimingAdapter

        adapter = HARQTimingAdapter(n_ue=3, config={"tti_ms": 1.0})

        # Propagation delays
        tau_s = np.array([0.005, 0.010, 0.020])  # 5, 10, 20 ms
        adapter.update_from_geometry(tau_s)

        # K1 should increase with delay
        k1_0 = adapter.get_k1_for_ue(0)
        k1_1 = adapter.get_k1_for_ue(1)
        k1_2 = adapter.get_k1_for_ue(2)

        self.assertLessEqual(k1_0, k1_1)
        self.assertLessEqual(k1_1, k1_2)

    def test_harq_rtt_slots(self):
        """Test HARQ RTT in slots."""
        from mac import HARQTimingAdapter

        adapter = HARQTimingAdapter(n_ue=1, config={"tti_ms": 1.0})
        tau_s = np.array([0.010])  # 10 ms one-way = 20 ms RTT
        adapter.update_from_geometry(tau_s)

        rtt_slots = adapter.get_harq_rtt_slots(0)
        self.assertEqual(rtt_slots, 20)  # 20 ms / 1 ms = 20 slots

    def test_effective_harq_processes(self):
        """Test effective HARQ process calculation."""
        from mac import HARQTimingAdapter

        adapter = HARQTimingAdapter(n_ue=1, config={"tti_ms": 1.0})

        # Large delay requires more processes
        tau_s = np.array([0.050])  # 50 ms one-way = 100 ms RTT
        adapter.update_from_geometry(tau_s)

        processes = adapter.get_effective_harq_processes(0, base_processes=16)
        self.assertGreaterEqual(processes, 16)
        self.assertLessEqual(processes, 32)

    def test_pdsch_to_ack_delay(self):
        """Test PDSCH to ACK delay calculation."""
        from mac import HARQTimingAdapter

        adapter = HARQTimingAdapter(n_ue=1, config={"tti_ms": 1.0})
        tau_s = np.array([0.010])
        adapter.update_from_geometry(tau_s)

        delay_ms = adapter.get_pdsch_to_ack_delay_ms(0)
        self.assertGreater(delay_ms, 0)


class TestSchedulingTimingManager(unittest.TestCase):
    """Test unified scheduling timing manager."""

    def test_manager_creation(self):
        """Test timing manager creation."""
        from mac import SchedulingTimingManager

        mgr = SchedulingTimingManager(n_ue=10)
        self.assertEqual(mgr.n_ue, 10)

    def test_update_from_geometry(self):
        """Test unified geometry update."""
        from mac import SchedulingTimingManager

        mgr = SchedulingTimingManager(n_ue=5, config={"tti_ms": 1.0})
        tau_s = np.array([0.010] * 5)
        mgr.update_from_geometry(tau_s)

        # Check K1 updated
        k1 = mgr.get_k1(0)
        self.assertGreater(k1, 0)

        # Check TA updated
        ta = mgr.ta_controller.get_ta_for_ue_ms(0)
        self.assertGreater(ta, 0)

    def test_get_k_values(self):
        """Test K value queries."""
        from mac import SchedulingTimingManager

        mgr = SchedulingTimingManager(n_ue=1, config={
            "k0_slots": 2,
            "k1_slots_base": 4,
            "k2_slots": 3,
        })

        self.assertEqual(mgr.get_k0(0), 2)
        self.assertEqual(mgr.get_k2(0), 3)
        # K1 depends on timing config

    def test_get_all_k1(self):
        """Test getting K1 for all UEs."""
        from mac import SchedulingTimingManager

        mgr = SchedulingTimingManager(n_ue=5)
        all_k1 = mgr.get_all_k1()

        self.assertEqual(len(all_k1), 5)
        self.assertTrue(all(k > 0 for k in all_k1))

    def test_scheduling_constraints(self):
        """Test scheduling constraint calculations."""
        from mac import SchedulingTimingManager

        mgr = SchedulingTimingManager(n_ue=1, config={"k0_slots": 2})

        # PDSCH slot after DCI
        pdsch_slot = mgr.get_earliest_pdsch_slot(0, dci_slot=10)
        self.assertEqual(pdsch_slot, 12)  # 10 + K0(2)

        # HARQ ACK slot after PDSCH
        ack_slot = mgr.get_harq_ack_slot(0, pdsch_slot=12)
        self.assertGreater(ack_slot, 12)

    def test_statistics(self):
        """Test timing statistics."""
        from mac import SchedulingTimingManager

        mgr = SchedulingTimingManager(n_ue=5)
        tau_s = np.array([0.005, 0.010, 0.015, 0.020, 0.025])
        mgr.update_from_geometry(tau_s)

        stats = mgr.get_statistics()
        self.assertIn("scenario", stats)
        self.assertIn("k1_min", stats)
        self.assertIn("k1_max", stats)
        self.assertIn("ta_stats", stats)

    def test_reset(self):
        """Test timing manager reset."""
        from mac import SchedulingTimingManager

        mgr = SchedulingTimingManager(n_ue=3)
        tau_s = np.array([0.010] * 3)
        mgr.update_from_geometry(tau_s)

        mgr.reset()

        # TA should be reset
        stats = mgr.ta_controller.get_statistics()
        self.assertEqual(stats["mean_ta_ms"], 0.0)


class TestUETimingState(unittest.TestCase):
    """Test per-UE timing state."""

    def test_state_creation(self):
        """Test UETimingState creation."""
        from mac.timing import UETimingState

        state = UETimingState(ue_id=0)
        self.assertEqual(state.ue_id, 0)
        self.assertEqual(state.propagation_delay_s, 0.0)

    def test_rtt_properties(self):
        """Test RTT properties."""
        from mac.timing import UETimingState

        state = UETimingState(ue_id=0, propagation_delay_s=0.010)
        self.assertAlmostEqual(state.rtt_s, 0.020, places=5)
        self.assertAlmostEqual(state.rtt_ms, 20.0, places=2)

    def test_update_from_geometry(self):
        """Test geometry update."""
        from mac.timing import UETimingState, SPEED_OF_LIGHT_M_S

        state = UETimingState(ue_id=0)

        # 600 km slant range
        slant_range_m = 600000.0
        state.update_from_geometry(slant_range_m)

        expected_delay = slant_range_m / SPEED_OF_LIGHT_M_S
        self.assertAlmostEqual(state.propagation_delay_s, expected_delay, places=6)


class TestConstants(unittest.TestCase):
    """Test timing constants."""

    def test_speed_of_light(self):
        """Test speed of light constant."""
        from mac import SPEED_OF_LIGHT_M_S

        # Should be approximately 3e8 m/s
        self.assertGreater(SPEED_OF_LIGHT_M_S, 299000000)
        self.assertLess(SPEED_OF_LIGHT_M_S, 300000000)

    def test_max_ntn_ta(self):
        """Test maximum NTN TA constant."""
        from mac import MAX_NTN_TA_MS

        # Should allow for GEO scenarios
        self.assertGreater(MAX_NTN_TA_MS, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
