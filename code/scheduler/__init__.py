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

Public API (~20 exports):

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
]

__version__ = "2.0.0"  # Phase 10: MAC-Scheduler integration
