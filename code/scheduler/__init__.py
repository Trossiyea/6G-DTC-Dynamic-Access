# -*- coding: utf-8 -*-
"""
Scheduler module for NR-NTN downlink resource allocation.

This module provides:
- Power allocation strategies (equal PRB, water-filling)
- Baseline PF schedulers (wideband, subband)
- RadioMap-aware contiguous block scheduler
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
]
