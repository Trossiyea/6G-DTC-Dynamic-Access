# -*- coding: utf-8 -*-
"""
TBS (Transport Block Size) Calculation for NR (TS 38.214 §5.1.3.2).

Provides:
- Symbol and RE accounting per slot
- TBS computation from PRBs and MCS
- Config-based RE resolution
"""

from typing import Dict
import numpy as np

from .mcs import MCS


def n_sym_per_slot(cp_type: str = "normal") -> int:
    """Number of OFDM symbols per slot.

    Args:
        cp_type: 'normal' (14 symbols) or 'extended' (12)

    Returns:
        Number of symbols
    """
    return 14 if (str(cp_type).lower() == "normal") else 12


def n_re_per_prb(
    cp_type: str = "normal",
    dmrs_sym_per_slot: int = 1,
    dmrs_re_per_sym_per_prb: int = 6,
    oh_prb: int = 0,
) -> int:
    """Effective RE per PRB per TTI (slot) after DMRS and overhead.

    Args:
        cp_type: 'normal' (14 symbols) or 'extended' (12)
        dmrs_sym_per_slot: Number of OFDM symbols with DMRS in slot
        dmrs_re_per_sym_per_prb: DMRS REs per DMRS symbol per PRB (typical 6)
        oh_prb: Additional overhead RE per PRB (0 if none)

    Returns:
        Effective RE count per PRB
    """
    n_sym = n_sym_per_slot(cp_type)
    total = 12 * n_sym
    dmrs = int(dmrs_sym_per_slot) * int(dmrs_re_per_sym_per_prb)
    return max(0, total - dmrs - int(oh_prb))


def calc_tbs_bits(
    n_prb: int,
    mcs: MCS,
    n_layers: int = 1,
    cp_type: str = "normal",
    dmrs_sym_per_slot: int = 1,
    dmrs_re_per_sym_per_prb: int = 6,
    oh_prb: int = 0,
) -> int:
    """Compute TBS bits for given PRBs and MCS.

    Follows the 5G NR TBS steps in a practical, widely-used form
    per TS 38.214 §5.1.3.2.

    Args:
        n_prb: Number of allocated PRBs
        mcs: MCS entry with Qm and R
        n_layers: Number of spatial layers
        cp_type: Cyclic prefix type
        dmrs_sym_per_slot: DMRS symbols per slot
        dmrs_re_per_sym_per_prb: DMRS REs per symbol per PRB
        oh_prb: Additional overhead REs per PRB

    Returns:
        TBS in bits
    """
    N_RE = n_prb * n_re_per_prb(cp_type, dmrs_sym_per_slot, dmrs_re_per_sym_per_prb, oh_prb)
    N_info = N_RE * mcs.Qm * mcs.R * max(1, int(n_layers))

    if N_info <= 0:
        return 0

    # NR practical TBS (see also 38.214/38.212)
    if N_info <= 3824:
        n = max(3, int(np.floor(np.log2(N_info))) - 6)
        TBS = max(24, int(2 ** n * np.floor(N_info / (2 ** n))))
    else:
        n = int(np.floor(np.log2(N_info - 24))) - 5
        Ninfo_prime = max(24, int(2 ** n * np.floor((N_info - 24) / (2 ** n))))
        C = int(np.ceil((Ninfo_prime + 24) / 8448))
        TBS = 8 * C * int(np.ceil((Ninfo_prime + 24) / (8 * C))) - 24

    return int(max(0, TBS))


def re_per_prb_from_config(cfg: Dict) -> int:
    """Resolve effective RE/PRB from config, preferring DL (PDSCH) keys.

    Backward-compatible: falls back to PUSCH keys if PDSCH ones are absent.

    Args:
        cfg: Configuration dict with numerology settings

    Returns:
        Effective RE count per PRB
    """
    cp = str(cfg.get("cp_type", "normal"))
    dmrs_sym = int(cfg.get("pdsch_dmrs_sym_per_slot", cfg.get("pusch_dmrs_sym_per_slot", 1)))
    dmrs_re = int(cfg.get("dmrs_re_per_sym_per_prb", 6))
    oh = int(cfg.get("oh_prb", 0))
    return n_re_per_prb(cp, dmrs_sym, dmrs_re, oh)
