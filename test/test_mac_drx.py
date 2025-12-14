"""
Unit tests for MAC layer DRX module (Phase 9.2).

Tests cover:
- DRX state enumeration and properties
- DRX configuration from dict
- DRX state machine transitions
- Timer management
- NTN extensions
- Scheduler integration (active UE filtering)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

import unittest
import numpy as np


class TestDRXState(unittest.TestCase):
    """Test DRX state enumeration."""

    def test_import(self):
        """Test module can be imported."""
        from mac import DRXState, DRXEvent, DRXConfig
        self.assertIsNotNone(DRXState)
        self.assertIsNotNone(DRXEvent)
        self.assertIsNotNone(DRXConfig)

    def test_state_values(self):
        """Test DRX state enumeration values."""
        from mac import DRXState

        # All states should exist
        self.assertIsNotNone(DRXState.ACTIVE)
        self.assertIsNotNone(DRXState.ON_DURATION)
        self.assertIsNotNone(DRXState.INACTIVITY)
        self.assertIsNotNone(DRXState.SHORT_CYCLE)
        self.assertIsNotNone(DRXState.LONG_CYCLE)

    def test_is_monitoring_pdcch(self):
        """Test PDCCH monitoring state check."""
        from mac import DRXState

        # Active states monitor PDCCH
        self.assertTrue(DRXState.ACTIVE.is_monitoring_pdcch())
        self.assertTrue(DRXState.ON_DURATION.is_monitoring_pdcch())
        self.assertTrue(DRXState.INACTIVITY.is_monitoring_pdcch())

        # Sleep states do not monitor
        self.assertFalse(DRXState.SHORT_CYCLE.is_monitoring_pdcch())
        self.assertFalse(DRXState.LONG_CYCLE.is_monitoring_pdcch())

    def test_is_sleeping(self):
        """Test sleep state check."""
        from mac import DRXState

        # Active states not sleeping
        self.assertFalse(DRXState.ACTIVE.is_sleeping())
        self.assertFalse(DRXState.ON_DURATION.is_sleeping())
        self.assertFalse(DRXState.INACTIVITY.is_sleeping())

        # Sleep states
        self.assertTrue(DRXState.SHORT_CYCLE.is_sleeping())
        self.assertTrue(DRXState.LONG_CYCLE.is_sleeping())


class TestDRXConfig(unittest.TestCase):
    """Test DRX configuration."""

    def test_default_config(self):
        """Test default DRX configuration."""
        from mac import DRXConfig

        cfg = DRXConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.on_duration_ms, 10.0)
        self.assertEqual(cfg.inactivity_timer_ms, 100.0)
        self.assertEqual(cfg.short_cycle_ms, 20.0)
        self.assertEqual(cfg.long_cycle_ms, 320.0)

    def test_tti_conversion(self):
        """Test ms to TTI conversion."""
        from mac import DRXConfig

        cfg = DRXConfig(
            on_duration_ms=10.0,
            inactivity_timer_ms=100.0,
            short_cycle_ms=20.0,
            long_cycle_ms=320.0,
            tti_ms=1.0
        )

        self.assertEqual(cfg.on_duration_ttis, 10)
        self.assertEqual(cfg.inactivity_ttis, 100)
        self.assertEqual(cfg.short_cycle_ttis, 20)
        self.assertEqual(cfg.long_cycle_ttis, 320)

    def test_from_config_dict(self):
        """Test configuration from flat dictionary."""
        from mac import DRXConfig

        config_dict = {
            "enable_drx": True,
            "drx_on_duration_ms": 5.0,
            "drx_inactivity_timer_ms": 50.0,
            "drx_short_cycle_ms": 10.0,
            "drx_long_cycle_ms": 160.0,
            "tti_ms": 0.5,
        }

        cfg = DRXConfig.from_config_dict(config_dict)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.on_duration_ms, 5.0)
        self.assertEqual(cfg.on_duration_ttis, 10)  # 5ms / 0.5ms = 10 TTIs


class TestDRXController(unittest.TestCase):
    """Test DRX controller functionality."""

    def test_controller_creation(self):
        """Test DRXController creation."""
        from mac import DRXController

        ctrl = DRXController(n_ue=10)
        self.assertEqual(ctrl.n_ue, 10)
        self.assertFalse(ctrl.drx_config.enabled)

    def test_drx_disabled_all_active(self):
        """Test all UEs active when DRX disabled."""
        from mac import DRXController

        ctrl = DRXController(n_ue=5, config={"enable_drx": False})

        # All UEs should be active
        self.assertEqual(len(ctrl.get_active_ues()), 5)
        self.assertEqual(len(ctrl.get_sleeping_ues()), 0)

        # Each UE should be schedulable
        for ue in range(5):
            self.assertTrue(ctrl.is_ue_active(ue))

    def test_drx_enabled_initial_state(self):
        """Test initial state when DRX enabled."""
        from mac import DRXController, DRXState

        ctrl = DRXController(n_ue=5, config={"enable_drx": True})

        # Initially all UEs in ACTIVE state
        for ue in range(5):
            self.assertEqual(ctrl.get_ue_state(ue), DRXState.ACTIVE)

    def test_advance_time(self):
        """Test time advancement."""
        from mac import DRXController

        ctrl = DRXController(n_ue=3, config={
            "enable_drx": True,
            "drx_on_duration_ms": 10.0,
            "drx_inactivity_timer_ms": 20.0,
            "drx_short_cycle_ms": 50.0,
            "tti_ms": 1.0,
        })

        # Advance time
        for tti in range(100):
            state_changes = ctrl.advance_time(tti)
            # May or may not have changes depending on cycle timing

    def test_pdcch_reception_restarts_inactivity(self):
        """Test PDCCH reception restarts inactivity timer."""
        from mac import DRXController, DRXState

        ctrl = DRXController(n_ue=1, config={
            "enable_drx": True,
            "drx_on_duration_ms": 5.0,
            "drx_inactivity_timer_ms": 10.0,
            "tti_ms": 1.0,
        })

        # Simulate PDCCH reception
        ctrl.on_pdcch_received(0, tti=5)

        # UE should be in INACTIVITY state
        self.assertEqual(ctrl.get_ue_state(0), DRXState.INACTIVITY)

        # UE should still be active (schedulable)
        self.assertTrue(ctrl.is_ue_active(0))

    def test_get_all_states(self):
        """Test getting all UE states."""
        from mac import DRXController

        ctrl = DRXController(n_ue=5)
        states = ctrl.get_all_states()

        self.assertEqual(len(states), 5)
        self.assertTrue(isinstance(states, np.ndarray))

    def test_enable_disable(self):
        """Test enabling and disabling DRX."""
        from mac import DRXController, DRXState

        ctrl = DRXController(n_ue=3, config={"enable_drx": False})

        # Initially disabled
        self.assertFalse(ctrl.drx_config.enabled)

        # Enable
        ctrl.enable()
        self.assertTrue(ctrl.drx_config.enabled)

        # Disable - all UEs should return to ACTIVE
        ctrl.disable()
        self.assertFalse(ctrl.drx_config.enabled)
        for ue in range(3):
            self.assertEqual(ctrl.get_ue_state(ue), DRXState.ACTIVE)

    def test_reset(self):
        """Test DRX state reset."""
        from mac import DRXController

        ctrl = DRXController(n_ue=3, config={"enable_drx": True})

        # Advance time
        for tti in range(50):
            ctrl.advance_time(tti)

        # Reset
        ctrl.reset()

        # All UEs back to initial state
        stats = ctrl.get_statistics()
        self.assertEqual(stats["total_transitions"], 0)


class TestDRXStatistics(unittest.TestCase):
    """Test DRX statistics collection."""

    def test_statistics_disabled(self):
        """Test statistics when DRX disabled."""
        from mac import DRXController

        ctrl = DRXController(n_ue=5, config={"enable_drx": False})
        stats = ctrl.get_statistics()

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["active_ues"], 5)
        self.assertEqual(stats["sleeping_ues"], 0)

    def test_statistics_enabled(self):
        """Test statistics when DRX enabled."""
        from mac import DRXController

        ctrl = DRXController(n_ue=5, config={
            "enable_drx": True,
            "drx_on_duration_ms": 10.0,
            "tti_ms": 1.0,
        })

        # Advance some time
        for tti in range(100):
            ctrl.advance_time(tti)

        stats = ctrl.get_statistics()
        self.assertTrue(stats["enabled"])
        self.assertIn("duty_cycle", stats)
        self.assertIn("state_distribution", stats)
        self.assertIn("config", stats)

    def test_power_saving_ratio(self):
        """Test power saving ratio calculation."""
        from mac import DRXController

        ctrl = DRXController(n_ue=1, config={
            "enable_drx": True,
            "drx_on_duration_ms": 10.0,
            "drx_inactivity_timer_ms": 10.0,
            "drx_short_cycle_ms": 100.0,
            "tti_ms": 1.0,
        })

        # Initially no time has passed
        self.assertEqual(ctrl.get_power_saving_ratio(), 0.0)

        # Advance time
        for tti in range(200):
            ctrl.advance_time(tti)

        # Should have some power saving (may be 0 if always active)
        ratio = ctrl.get_power_saving_ratio()
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)


class TestNTNDRXController(unittest.TestCase):
    """Test NTN-specific DRX extensions."""

    def test_ntn_controller_creation(self):
        """Test NTNDRXController creation."""
        from mac import NTNDRXController

        ctrl = NTNDRXController(n_ue=5, config={"enable_drx": True})
        self.assertIsNotNone(ctrl)

    def test_set_ntn_rtt_offset(self):
        """Test setting NTN RTT offset for a UE."""
        from mac import DRXController

        ctrl = DRXController(n_ue=3, config={"enable_drx": True})

        # Set RTT offset
        ctrl.set_ntn_rtt_offset(0, rtt_ttis=50)
        ctrl.set_ntn_rtt_offset(1, rtt_ttis=60)

        # Verify offsets stored (internal state)
        self.assertEqual(ctrl._ue_states[0].ntn_rtt_offset_ttis, 50)
        self.assertEqual(ctrl._ue_states[1].ntn_rtt_offset_ttis, 60)

    def test_update_from_geometry(self):
        """Test updating NTN offsets from propagation delay array."""
        from mac import NTNDRXController

        # Propagation delays in seconds (LEO ~10ms one-way)
        tau_s = np.array([0.010, 0.012, 0.015])  # 10ms, 12ms, 15ms

        ctrl = NTNDRXController(
            n_ue=3,
            config={"enable_drx": True, "tti_ms": 1.0},
            tau_s_per_ue=tau_s
        )

        # RTT = 2 * one-way delay
        # UE 0: 2 * 10ms = 20ms = 20 TTIs
        # UE 1: 2 * 12ms = 24ms = 24 TTIs
        # UE 2: 2 * 15ms = 30ms = 30 TTIs
        self.assertEqual(ctrl._ue_states[0].ntn_rtt_offset_ttis, 20)
        self.assertEqual(ctrl._ue_states[1].ntn_rtt_offset_ttis, 24)
        self.assertEqual(ctrl._ue_states[2].ntn_rtt_offset_ttis, 30)

    def test_effective_inactivity_timer(self):
        """Test effective inactivity timer with NTN offset."""
        from mac import NTNDRXController

        tau_s = np.array([0.020])  # 20ms one-way = 40ms RTT

        ctrl = NTNDRXController(
            n_ue=1,
            config={
                "enable_drx": True,
                "drx_inactivity_timer_ms": 100.0,
                "tti_ms": 1.0,
            },
            tau_s_per_ue=tau_s
        )

        # Effective = base (100) + RTT (40) = 140 TTIs
        effective = ctrl.get_effective_inactivity_timer(0)
        self.assertEqual(effective, 140)

    def test_geometry_update(self):
        """Test geometry update for moving satellite."""
        from mac import NTNDRXController

        ctrl = NTNDRXController(
            n_ue=2,
            config={"enable_drx": True, "tti_ms": 1.0},
        )

        # Initial geometry
        tau_s_1 = np.array([0.010, 0.015])
        ctrl.update_geometry(tau_s_1)

        self.assertEqual(ctrl._ue_states[0].ntn_rtt_offset_ttis, 20)
        self.assertEqual(ctrl._ue_states[1].ntn_rtt_offset_ttis, 30)

        # Updated geometry (satellite moved)
        tau_s_2 = np.array([0.012, 0.018])
        ctrl.update_geometry(tau_s_2)

        self.assertEqual(ctrl._ue_states[0].ntn_rtt_offset_ttis, 24)
        self.assertEqual(ctrl._ue_states[1].ntn_rtt_offset_ttis, 36)


class TestDRXSchedulerIntegration(unittest.TestCase):
    """Test DRX integration with scheduler."""

    def test_active_ue_filtering(self):
        """Test filtering of active UEs for scheduling."""
        from mac import DRXController

        ctrl = DRXController(n_ue=10, config={
            "enable_drx": True,
            "drx_on_duration_ms": 10.0,
            "drx_inactivity_timer_ms": 20.0,
            "tti_ms": 1.0,
        })

        # Get active UEs
        active = ctrl.get_active_ues()
        sleeping = ctrl.get_sleeping_ues()

        # Totals should match
        self.assertEqual(len(active) + len(sleeping), 10)

        # No overlap
        self.assertEqual(len(set(active) & set(sleeping)), 0)

    def test_callback_on_state_change(self):
        """Test state change callback."""
        from mac import DRXController, DRXState

        state_changes = []

        def on_change(ue_id, old_state, new_state, tti):
            state_changes.append((ue_id, old_state, new_state, tti))

        ctrl = DRXController(n_ue=1, config={
            "enable_drx": True,
            "drx_on_duration_ms": 5.0,
            "tti_ms": 1.0,
        })
        ctrl.on_state_change(on_change)

        # Trigger state change
        ctrl.on_pdcch_received(0, tti=0)

        # Should have recorded state change
        self.assertGreater(len(state_changes), 0)


class TestUEDRXState(unittest.TestCase):
    """Test per-UE DRX state structure."""

    def test_ue_state_creation(self):
        """Test UEDRXState creation."""
        from mac.drx import UEDRXState, DRXState

        state = UEDRXState(ue_id=0)
        self.assertEqual(state.ue_id, 0)
        self.assertEqual(state.state, DRXState.ACTIVE)
        self.assertTrue(state.in_active_time)

    def test_is_schedulable(self):
        """Test schedulability check."""
        from mac.drx import UEDRXState, DRXState, DRXConfig

        state = UEDRXState(ue_id=0, config=DRXConfig(enabled=True))

        # ACTIVE is schedulable
        state.state = DRXState.ACTIVE
        self.assertTrue(state.is_schedulable())

        # ON_DURATION is schedulable
        state.state = DRXState.ON_DURATION
        self.assertTrue(state.is_schedulable())

        # SHORT_CYCLE is not schedulable (unless in_active_time)
        state.state = DRXState.SHORT_CYCLE
        state.in_active_time = False
        self.assertFalse(state.is_schedulable())

    def test_power_state_string(self):
        """Test power state string representation."""
        from mac.drx import UEDRXState, DRXState, DRXConfig

        state = UEDRXState(ue_id=0, config=DRXConfig(enabled=True))

        state.state = DRXState.ACTIVE
        self.assertEqual(state.get_power_state(), "active")

        state.state = DRXState.SHORT_CYCLE
        self.assertEqual(state.get_power_state(), "sleep")

        # DRX disabled
        state.config = DRXConfig(enabled=False)
        self.assertEqual(state.get_power_state(), "always_on")


if __name__ == "__main__":
    unittest.main(verbosity=2)
