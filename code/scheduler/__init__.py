# -*- coding: utf-8 -*-
"""
Scheduler module for NR-NTN downlink resource allocation.

This module provides:
- Power allocation strategies (equal PRB, water-filling)
- Baseline PF schedulers (wideband, subband)
- RadioMap-aware contiguous block scheduler
- MAC-Scheduler integration (Phase 10)
- QoS-aware scheduling algorithms (M-LWDF, EXP-PF)
- NTN-aware HARQ timing
- OALS: Orbit-Aware Lookahead Scheduling (Patent Core Algorithm)

Public API (~35 exports):

Power Allocation:
    - waterfill_prb: Per-PRB water-filling
    - waterfill_groups: Group-level water-filling
    - apply_dl_power_allocation: DL power allocation wrapper

Schedulers:
    - pf_schedule_baseline: 3GPP-like wideband PF scheduler
    - pf_schedule_radiomap_blocks: RadioMap-aware contiguous block scheduler
    - pf_schedule_baseline_subband: Subband-level baseline scheduler

MAC-Scheduler Integration (Phase 10):
    - MACSchedulerBridge: Bridge MAC layer (BSR/DRX/Timing) to scheduler
    - QoSSchedulerType: QoS algorithm enumeration
    - QoSSchedulerConfig: QoS scheduler configuration
    - create_mac_scheduler_bridge: Factory function

QoS Metrics:
    - compute_pf_metric: Proportional Fair metric
    - compute_mlwdf_metric: M-LWDF metric
    - compute_exppf_metric: Exponential PF metric
    - compute_edf_metric: Earliest Deadline First metric
    - build_qos_params_from_qci: Build QoS params from QCI

NTN HARQ:
    - NTNHarqAdapter: NTN-aware HARQ timing adapter
    - NTNHarqConfig: NTN HARQ configuration
    - create_ntn_harq_adapter: Factory function

OALS - Orbit-Aware Lookahead Scheduling (Patent):
    - OALSConfig: OALS algorithm configuration
    - LookaheadFactor: Lookahead factor calculator (Φ computation)
    - OALSScheduler: Orbit-aware lookahead scheduler
    - compute_correction_factor: Metric correction function f(Φ, urgency)
    - create_oals_scheduler: Factory function

Predictive Handover:
    - PredictiveHOConfig: Predictive handover configuration
    - PredictiveHandoverManager: Predictive handover manager
    - HandoverPrediction: Handover prediction result
    - SchedulingAdjustment: Scheduling adjustment recommendation
    - create_predictive_ho_manager: Factory function

HARQ Lookahead MCS:
    - HARQLookaheadConfig: HARQ lookahead configuration
    - HARQLookaheadMCS: HARQ lookahead MCS selector
    - MCSAdjustment: MCS adjustment recommendation
    - MCSStrategy: MCS selection strategy enumeration
    - create_harq_lookahead_mcs: Factory function

Usage:
    # Basic scheduling
    from scheduler import pf_schedule_radiomap_blocks

    # MAC-aware scheduling
    from scheduler import MACSchedulerBridge, create_mac_scheduler_bridge
    bridge = create_mac_scheduler_bridge(n_ue=100, config=cfg, tau_s_per_ue=tau)
    ue_mask = bridge.get_ue_mask(tti)
    qos_metric = bridge.compute_qos_metric(se_metric, hol_delay)

    # NTN HARQ
    from scheduler import NTNHarqAdapter, create_ntn_harq_adapter
    harq_adapter = create_ntn_harq_adapter(n_ue=100, config=cfg, tau_s_per_ue=tau)
    k1 = harq_adapter.get_k1_for_ue(ue_id=0)

    # OALS (Orbit-Aware Lookahead Scheduling)
    from scheduler import OALSScheduler, create_oals_scheduler
    oals = create_oals_scheduler(n_ue=100, orbit_model=orbit)
    oals.update_lookahead(ue_pos, t)
    metric = oals.compute_oals_metric(pf_metric, qos_weight)

    # Predictive Handover
    from scheduler import PredictiveHandoverManager, create_predictive_ho_manager
    ho_mgr = create_predictive_ho_manager(n_ue=100, orbit=orbit)
    pred = ho_mgr.predict_handover(ue_id=0, ue_pos=pos, t=100)

    # HARQ Lookahead MCS
    from scheduler import HARQLookaheadMCS, create_harq_lookahead_mcs
    harq_mcs = create_harq_lookahead_mcs(lookahead_factor)
    adj = harq_mcs.get_mcs_adjustment(ue_id=0, sinr_now_db=10.0, k1_slots=4)
"""

from .power_alloc import (
    waterfill_prb,
    waterfill_groups,
    apply_dl_power_allocation,
)

from .baseline import (
    pf_schedule_baseline,
)

from .radiomap import (
    pf_schedule_radiomap_blocks,
)

from .subband import (
    pf_schedule_baseline_subband,
    group_ranges,
)

from .integration import (
    # Core classes
    MACSchedulerBridge,
    QoSSchedulerType,
    QoSSchedulerConfig,
    # Factory functions
    create_mac_scheduler_bridge,
    build_qos_params_from_qci,
    # QoS metric functions
    compute_pf_metric,
    compute_mlwdf_metric,
    compute_exppf_metric,
    compute_edf_metric,
)

from .ntn_harq import (
    NTNHarqAdapter,
    NTNHarqConfig,
    create_ntn_harq_adapter,
    compute_min_harq_processes,
)

# OALS - Orbit-Aware Lookahead Scheduling (Patent Core Algorithm)
from .lookahead import (
    OALSConfig,
    LookaheadFactor,
    LookaheadCache,
    OALSScheduler,
    SchedulingRecommendation,
    compute_correction_factor,
    compute_correction_factor_batch,
    create_oals_scheduler,
)

# Predictive Handover
from .predictive_ho import (
    HandoverReason,
    SchedulingPhase,
    HandoverPrediction,
    SchedulingAdjustment,
    PredictiveHOConfig,
    PredictiveHandoverManager,
    apply_handover_aware_correction,
    create_predictive_ho_manager,
)

# HARQ Lookahead MCS
from .harq_lookahead import (
    MCSStrategy,
    MCSAdjustment,
    HARQLookaheadConfig,
    HARQLookaheadMCS,
    compute_harq_sinr_adjustments,
    create_harq_lookahead_mcs,
)

__all__ = [
    # Power allocation
    "waterfill_prb",
    "waterfill_groups",
    "apply_dl_power_allocation",
    # Baseline scheduler
    "pf_schedule_baseline",
    # RadioMap scheduler
    "pf_schedule_radiomap_blocks",
    # Subband scheduler
    "pf_schedule_baseline_subband",
    "group_ranges",
    # MAC-Scheduler integration (Phase 10)
    "MACSchedulerBridge",
    "QoSSchedulerType",
    "QoSSchedulerConfig",
    "create_mac_scheduler_bridge",
    "build_qos_params_from_qci",
    # QoS metrics
    "compute_pf_metric",
    "compute_mlwdf_metric",
    "compute_exppf_metric",
    "compute_edf_metric",
    # NTN HARQ
    "NTNHarqAdapter",
    "NTNHarqConfig",
    "create_ntn_harq_adapter",
    "compute_min_harq_processes",
    # OALS - Orbit-Aware Lookahead Scheduling (Patent)
    "OALSConfig",
    "LookaheadFactor",
    "LookaheadCache",
    "OALSScheduler",
    "SchedulingRecommendation",
    "compute_correction_factor",
    "compute_correction_factor_batch",
    "create_oals_scheduler",
    # Predictive Handover
    "HandoverReason",
    "SchedulingPhase",
    "HandoverPrediction",
    "SchedulingAdjustment",
    "PredictiveHOConfig",
    "PredictiveHandoverManager",
    "apply_handover_aware_correction",
    "create_predictive_ho_manager",
    # HARQ Lookahead MCS
    "MCSStrategy",
    "MCSAdjustment",
    "HARQLookaheadConfig",
    "HARQLookaheadMCS",
    "compute_harq_sinr_adjustments",
    "create_harq_lookahead_mcs",
]

__version__ = "3.0.0"  # Phase 11: OALS - Orbit-Aware Lookahead Scheduling
