#!/usr/bin/env python3
"""
Unit tests for OALS (Orbit-Aware Lookahead Scheduling) modules.

Tests cover:
1. LookaheadFactor - Φ calculation and trend computation
2. OALSScheduler - metric correction and scheduling recommendations
3. PredictiveHandoverManager - handover prediction and scheduling phases
4. HARQLookaheadMCS - MCS adjustment strategies

Run: python -m pytest test/test_oals.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

import numpy as np

# Optional pytest import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Simple skip decorator when pytest not available
    class pytest:
        @staticmethod
        def skip(reason=""):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    print(f"  SKIP: {reason}")
                return wrapper
            return decorator


# ============== Test: Module Imports ==============

def test_import_lookahead():
    """Test OALS lookahead module imports."""
    from scheduler.lookahead import (
        OALSConfig,
        LookaheadFactor,
        LookaheadCache,
        OALSScheduler,
        SchedulingRecommendation,
        compute_correction_factor,
        compute_correction_factor_batch,
        create_oals_scheduler,
    )
    assert OALSConfig is not None
    assert LookaheadFactor is not None
    assert compute_correction_factor is not None


def test_import_predictive_ho():
    """Test predictive handover module imports."""
    from scheduler.predictive_ho import (
        HandoverReason,
        SchedulingPhase,
        HandoverPrediction,
        SchedulingAdjustment,
        PredictiveHOConfig,
        PredictiveHandoverManager,
        apply_handover_aware_correction,
        create_predictive_ho_manager,
    )
    assert HandoverReason is not None
    assert PredictiveHandoverManager is not None


def test_import_harq_lookahead():
    """Test HARQ lookahead module imports."""
    from scheduler.harq_lookahead import (
        MCSStrategy,
        MCSAdjustment,
        HARQLookaheadConfig,
        HARQLookaheadMCS,
        compute_harq_sinr_adjustments,
        create_harq_lookahead_mcs,
    )
    assert MCSStrategy is not None
    assert HARQLookaheadMCS is not None


def test_import_from_scheduler_package():
    """Test imports from scheduler package."""
    from scheduler import (
        OALSConfig,
        OALSScheduler,
        create_oals_scheduler,
        PredictiveHandoverManager,
        create_predictive_ho_manager,
        HARQLookaheadMCS,
        create_harq_lookahead_mcs,
        compute_correction_factor,
    )
    assert create_oals_scheduler is not None


# ============== Test: OALSConfig ==============

def test_oals_config_defaults():
    """Test OALSConfig default values."""
    from scheduler.lookahead import OALSConfig

    cfg = OALSConfig()

    assert cfg.lookahead_horizon_ttis == 200
    assert cfg.lookahead_sample_interval == 5
    assert cfg.alpha_urgent == 0.8
    assert cfg.beta_wait == 0.3
    assert cfg.theta_lookahead == 0.7
    assert cfg.gamma_decay == 2.0
    assert cfg.boost_factor == 2.0


def test_oals_config_from_dict():
    """Test OALSConfig creation from dictionary."""
    from scheduler.lookahead import OALSConfig

    cfg_dict = {
        'lookahead_horizon_ttis': 500,
        'alpha_urgent': 0.9,
        'gamma_decay': 3.0,
        'unknown_key': 'should_be_ignored',
    }

    cfg = OALSConfig.from_config_dict(cfg_dict)

    assert cfg.lookahead_horizon_ttis == 500
    assert cfg.alpha_urgent == 0.9
    assert cfg.gamma_decay == 3.0
    assert cfg.beta_wait == 0.3  # default unchanged


# ============== Test: Correction Factor ==============

def test_correction_factor_urgent():
    """Test correction factor in urgent region."""
    from scheduler.lookahead import OALSConfig, compute_correction_factor

    cfg = OALSConfig()

    # High urgency → boost
    f = compute_correction_factor(phi=0.5, urgency=0.95, trend=0.0, config=cfg)

    # f = 1 + boost * (urgency - alpha) / (1 - alpha)
    expected = 1.0 + 2.0 * (0.95 - 0.8) / (1.0 - 0.8)

    assert abs(f - expected) < 1e-6


def test_correction_factor_wait_region():
    """Test correction factor in wait region."""
    from scheduler.lookahead import OALSConfig, compute_correction_factor

    cfg = OALSConfig()

    # Low urgency + low phi → penalty
    f = compute_correction_factor(phi=0.3, urgency=0.1, trend=0.0, config=cfg)

    # f = phi^gamma = 0.3^2.0 = 0.09
    expected = 0.3 ** 2.0

    assert abs(f - expected) < 1e-6


def test_correction_factor_normal():
    """Test correction factor in normal region."""
    from scheduler.lookahead import OALSConfig, compute_correction_factor

    cfg = OALSConfig()

    # Normal case
    f = compute_correction_factor(phi=0.8, urgency=0.5, trend=0.0, config=cfg)

    # Should be 1.0 (normal region)
    assert abs(f - 1.0) < 1e-6


def test_correction_factor_trend_improving():
    """Test correction factor with improving trend."""
    from scheduler.lookahead import OALSConfig, compute_correction_factor

    cfg = OALSConfig()

    # Normal + improving trend → slightly lower
    f = compute_correction_factor(phi=0.8, urgency=0.5, trend=1.0, config=cfg)

    # f = 1.0 * (1 - epsilon) = 0.9
    expected = 1.0 * (1.0 - 0.1)

    assert abs(f - expected) < 1e-6


def test_correction_factor_trend_degrading():
    """Test correction factor with degrading trend."""
    from scheduler.lookahead import OALSConfig, compute_correction_factor

    cfg = OALSConfig()

    # Normal + degrading trend → slightly higher
    f = compute_correction_factor(phi=0.8, urgency=0.5, trend=-1.0, config=cfg)

    # f = 1.0 * (1 + epsilon) = 1.1
    expected = 1.0 * (1.0 + 0.1)

    assert abs(f - expected) < 1e-6


def test_correction_factor_batch():
    """Test batch correction factor computation."""
    from scheduler.lookahead import OALSConfig, compute_correction_factor_batch

    cfg = OALSConfig()
    n_ue = 5

    phi = np.array([0.2, 0.5, 0.8, 0.9, 0.3])
    urgency = np.array([0.1, 0.5, 0.85, 0.95, 0.2])
    trend = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    f = compute_correction_factor_batch(phi, urgency, trend, cfg)

    assert f.shape == (n_ue,)
    # Check wait region: UE 0, phi=0.2, urgency=0.1 < beta, phi < theta → phi^gamma
    assert abs(f[0] - 0.2**2.0) < 1e-6
    # Check normal region: UE 1
    assert abs(f[1] - 1.0) < 1e-6
    # Check urgent region: UE 3, urgency=0.95 > alpha
    expected_3 = 1.0 + 2.0 * (0.95 - 0.8) / (1.0 - 0.8)
    assert abs(f[3] - expected_3) < 1e-6


# ============== Test: Handover Types ==============

def test_handover_reason_enum():
    """Test HandoverReason enum values."""
    from scheduler.predictive_ho import HandoverReason

    assert HandoverReason.NONE.value == "none"
    assert HandoverReason.ELEVATION.value == "elevation"
    assert HandoverReason.SNR.value == "snr"


def test_scheduling_phase_enum():
    """Test SchedulingPhase enum values."""
    from scheduler.predictive_ho import SchedulingPhase

    assert SchedulingPhase.NORMAL.value == "normal"
    assert SchedulingPhase.PREPARATION.value == "preparation"
    assert SchedulingPhase.HANDOVER.value == "handover"
    assert SchedulingPhase.RECOVERY.value == "recovery"


def test_mcs_strategy_enum():
    """Test MCSStrategy enum values."""
    from scheduler.harq_lookahead import MCSStrategy

    assert MCSStrategy.STANDARD.value == "standard"
    assert MCSStrategy.AGGRESSIVE.value == "aggressive"
    assert MCSStrategy.CONSERVATIVE.value == "conservative"
    assert MCSStrategy.URGENT.value == "urgent"


# ============== Test: PredictiveHOConfig ==============

def test_predictive_ho_config():
    """Test PredictiveHOConfig default values."""
    from scheduler.predictive_ho import PredictiveHOConfig

    cfg = PredictiveHOConfig()

    assert cfg.min_elev_deg == 10.0
    assert cfg.ho_hyst_db == 3.0
    assert cfg.ho_ttt_ttis == 20
    assert cfg.lookahead_horizon_ttis == 200
    assert cfg.prep_window_ttis == 100


# ============== Test: HARQLookaheadConfig ==============

def test_harq_lookahead_config():
    """Test HARQLookaheadConfig default values."""
    from scheduler.harq_lookahead import HARQLookaheadConfig

    cfg = HARQLookaheadConfig()

    assert cfg.aggressive_sinr_boost_db == 1.5
    assert cfg.conservative_sinr_margin_db == 1.5
    assert cfg.delta_threshold_db == 3.0
    assert cfg.handover_prep_window_ttis == 100


# ============== Test: HandoverPrediction Dataclass ==============

def test_handover_prediction_dataclass():
    """Test HandoverPrediction dataclass."""
    from scheduler.predictive_ho import HandoverPrediction, HandoverReason

    pred = HandoverPrediction(
        ue_id=0,
        will_handover=True,
        predicted_time=100,
        target_satellite=2,
        source_satellite=1,
        reason=HandoverReason.SNR,
        time_to_handover=50,
        confidence=0.85,
        snr_improvement_db=3.5,
    )

    assert pred.will_handover is True
    assert pred.predicted_time == 100
    assert pred.target_satellite == 2
    assert pred.reason == HandoverReason.SNR
    assert pred.snr_improvement_db == 3.5


# ============== Test: MCSAdjustment Dataclass ==============

def test_mcs_adjustment_dataclass():
    """Test MCSAdjustment dataclass."""
    from scheduler.harq_lookahead import MCSAdjustment, MCSStrategy

    adj = MCSAdjustment(
        ue_id=0,
        sinr_adjustment_db=1.5,
        strategy=MCSStrategy.AGGRESSIVE,
        rationale="Future channel better by 5.0 dB",
        predicted_delta_db=5.0,
        k1_slots=4,
        confidence=0.7,
    )

    assert adj.sinr_adjustment_db == 1.5
    assert adj.strategy == MCSStrategy.AGGRESSIVE
    assert adj.predicted_delta_db == 5.0


# ============== Test: SchedulingAdjustment Dataclass ==============

def test_scheduling_adjustment_dataclass():
    """Test SchedulingAdjustment dataclass."""
    from scheduler.predictive_ho import SchedulingAdjustment, SchedulingPhase

    adj = SchedulingAdjustment(
        ue_id=0,
        phase=SchedulingPhase.PREPARATION,
        defer_non_realtime=True,
        accelerate_realtime=True,
        reduce_new_harq=True,
        target_satellite=2,
        priority_boost=1.5,
        reason="Handover prep: 50 TTIs to HO",
    )

    assert adj.phase == SchedulingPhase.PREPARATION
    assert adj.defer_non_realtime is True
    assert adj.priority_boost == 1.5


# ============== Test: Config Schema Integration ==============

def test_oals_config_in_schema():
    """Test OALSConfig is properly integrated in config schema."""
    try:
        from config.schema import OALSConfig, NTNSimConfig

        # Create full config
        cfg = NTNSimConfig()

        # Access OALS config
        assert hasattr(cfg, 'oals')
        assert cfg.oals.lookahead_horizon_ttis == 200
        assert cfg.oals.enable_oals is False  # default off

        # Test flat dict conversion
        flat = cfg.to_flat_dict()
        assert 'enable_oals' in flat
        assert 'lookahead_horizon_ttis' in flat
        assert 'alpha_urgent' in flat

    except ImportError:
        pytest.skip("Config schema not available")


# ============== Test: apply_handover_aware_correction ==============

def test_apply_handover_aware_correction():
    """Test handover-aware metric correction."""
    from scheduler.predictive_ho import (
        apply_handover_aware_correction,
        SchedulingAdjustment,
        SchedulingPhase,
    )

    n_ue = 3
    metric = np.ones(n_ue) * 10.0
    is_realtime = np.array([True, False, True])

    adjustments = [
        SchedulingAdjustment(
            ue_id=0,
            phase=SchedulingPhase.PREPARATION,
            defer_non_realtime=True,
            accelerate_realtime=True,
            reduce_new_harq=True,
            target_satellite=1,
            priority_boost=1.5,
            reason="Prep",
        ),
        SchedulingAdjustment(
            ue_id=1,
            phase=SchedulingPhase.PREPARATION,
            defer_non_realtime=True,
            accelerate_realtime=True,
            reduce_new_harq=True,
            target_satellite=1,
            priority_boost=1.5,
            reason="Prep",
        ),
        SchedulingAdjustment(
            ue_id=2,
            phase=SchedulingPhase.NORMAL,
            defer_non_realtime=False,
            accelerate_realtime=False,
            reduce_new_harq=False,
            target_satellite=None,
            priority_boost=1.0,
            reason="Normal",
        ),
    ]

    corrected = apply_handover_aware_correction(metric, adjustments, is_realtime)

    # UE 0: realtime + accelerate → boosted
    assert abs(corrected[0] - 15.0) < 1e-6  # 10 * 1.5
    # UE 1: non-realtime + defer → penalized
    assert abs(corrected[1] - 1.0) < 1e-6  # 10 * 0.1
    # UE 2: normal → unchanged
    assert abs(corrected[2] - 10.0) < 1e-6


# ============== Test: Orbit Batch Prediction Interface ==============

def test_orbit_model_has_batch_methods():
    """Test that OrbitModel has batch prediction methods."""
    from ntn.orbit import OrbitModel

    # Check method exists
    assert hasattr(OrbitModel, 'get_geometry_batch')
    assert hasattr(OrbitModel, 'get_g_sat_batch')


# ============== Main ==============

if __name__ == '__main__':
    # Run basic tests
    test_import_lookahead()
    test_import_predictive_ho()
    test_import_harq_lookahead()
    test_import_from_scheduler_package()

    test_oals_config_defaults()
    test_oals_config_from_dict()

    test_correction_factor_urgent()
    test_correction_factor_wait_region()
    test_correction_factor_normal()
    test_correction_factor_trend_improving()
    test_correction_factor_trend_degrading()
    test_correction_factor_batch()

    test_handover_reason_enum()
    test_scheduling_phase_enum()
    test_mcs_strategy_enum()

    test_predictive_ho_config()
    test_harq_lookahead_config()

    test_handover_prediction_dataclass()
    test_mcs_adjustment_dataclass()
    test_scheduling_adjustment_dataclass()

    test_apply_handover_aware_correction()
    test_orbit_model_has_batch_methods()

    print("\n" + "="*60)
    print("All OALS unit tests passed!")
    print("="*60)
