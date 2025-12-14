"""
MAC layer subsystem for NTN RAN simulator (Phase 9).

This module provides MAC-layer protocol functionality including:
- Buffer Status Report (BSR) management
- Discontinuous Reception (DRX) state machine
- NTN-specific timing control (K0/K1/K2, Timing Advance)

Public API (~35 exports):

BSR Management:
    - BSRTable: 3GPP TS 38.321 BSR lookup tables (5-bit/8-bit)
    - BSRReport: Single BSR report data structure
    - BSRTrigger: BSR trigger condition enumeration
    - LCGManager: Logical Channel Group management
    - BSRManager: Per-UE BSR state management

DRX Management:
    - DRXState: DRX state enumeration (Active/OnDuration/Inactivity/ShortCycle/LongCycle)
    - DRXEvent: DRX event enumeration
    - DRXConfig: DRX parameter configuration
    - DRXController: DRX state machine controller
    - NTNDRXController: NTN-aware DRX with RTT compensation

NTN Timing:
    - NTNScenario: NTN scenario types (LEO/MEO/GEO)
    - NTNTimingConfig: NTN timing parameter configuration
    - K1Table: K1 timing table with NTN extensions
    - TimingAdvanceController: Timing Advance management
    - HARQTimingAdapter: HARQ timing for NTN
    - SchedulingTimingManager: Unified scheduling timing management

Usage:
    # BSR
    from mac import BSRManager
    bsr_mgr = BSRManager(n_ue=100, config=config)

    # DRX
    from mac import DRXController, NTNDRXController
    drx_ctrl = DRXController(n_ue=100, config=config)

    # NTN Timing
    from mac import SchedulingTimingManager, NTNScenario
    timing_mgr = SchedulingTimingManager(n_ue=100, config=config)
    timing_mgr.update_from_geometry(tau_s_per_ue)
    k1 = timing_mgr.get_k1(ue_id=0)
"""

from .bsr import (
    BSRTable,
    BSRReport,
    BSRTrigger,
    LCGManager,
    BSRManager,
    BSR_TABLE_5BIT,
    BSR_TABLE_8BIT,
    qci_to_lcg,
)

from .drx import (
    DRXState,
    DRXEvent,
    DRXConfig,
    UEDRXState,
    DRXController,
    NTNDRXController,
)

from .timing import (
    NTNScenario,
    NTNTimingConfig,
    K1Table,
    UETimingState,
    TimingAdvanceController,
    HARQTimingAdapter,
    SchedulingTimingManager,
    SPEED_OF_LIGHT_M_S,
    MAX_NTN_TA_MS,
)

__all__ = [
    # BSR classes
    "BSRTable",
    "BSRReport",
    "BSRTrigger",
    "LCGManager",
    "BSRManager",
    # BSR constants
    "BSR_TABLE_5BIT",
    "BSR_TABLE_8BIT",
    # BSR utility functions
    "qci_to_lcg",
    # DRX classes
    "DRXState",
    "DRXEvent",
    "DRXConfig",
    "UEDRXState",
    "DRXController",
    "NTNDRXController",
    # NTN Timing classes
    "NTNScenario",
    "NTNTimingConfig",
    "K1Table",
    "UETimingState",
    "TimingAdvanceController",
    "HARQTimingAdapter",
    "SchedulingTimingManager",
    # Timing constants
    "SPEED_OF_LIGHT_M_S",
    "MAX_NTN_TA_MS",
]

__version__ = "1.0.0"
