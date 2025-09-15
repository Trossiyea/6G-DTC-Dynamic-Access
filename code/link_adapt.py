"""
Link adaptation utilities for NR (TS 38.214 inspired).

Provides:
- Configurable MCS candidate sets (Table-1/2/3 approximations)
- 38.214 §5.1.3.2-like TBS computation (practical implementation)
- Simple BLER curves (AWGN-like, configurable slope/margin) and OLLA

Notes
- Tables here provide a pragmatic baseline for simulations. Exact 3GPP
  table entries can be injected through overrides if needed.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ------------------------------
# Helpers: symbols/RE accounting
# ------------------------------
def n_sym_per_slot(cp_type: str = "normal") -> int:
    return 14 if (str(cp_type).lower() == "normal") else 12


def n_re_per_prb(
    cp_type: str = "normal",
    dmrs_sym_per_slot: int = 1,
    dmrs_re_per_sym_per_prb: int = 6,
    oh_prb: int = 0,
) -> int:
    """Effective RE per PRB per TTI (slot) after DMRS and overhead.

    - cp_type: 'normal' (14 symbols) or 'extended' (12)
    - dmrs_sym_per_slot: number of OFDM symbols with DMRS in slot
    - dmrs_re_per_sym_per_prb: DMRS REs per DMRS symbol per PRB (typical 6)
    - oh_prb: additional overhead RE per PRB (0 if none)
    """
    n_sym = n_sym_per_slot(cp_type)
    total = 12 * n_sym
    dmrs = int(dmrs_sym_per_slot) * int(dmrs_re_per_sym_per_prb)
    return max(0, total - dmrs - int(oh_prb))


# ------------------------------
# MCS candidates (approximate)
# ------------------------------
@dataclass(frozen=True)
class MCS:
    idx: int
    Qm: int
    R: float  # code rate in [0,1]

    @property
    def se(self) -> float:
        return self.Qm * self.R


def _table1_64qam() -> List[MCS]:
    """Approximate Table 5.1.3.1-1: 64QAM-capable set.

    We map a standard 15-step CQI SE vector into (Qm, R) pairs commonly used in
    practice. This yields a stable candidate set covering QPSK/16QAM/64QAM.
    """
    # SE values (bits/s/Hz) for CQI 1..15 near 10% BLER in AWGN (industry-common)
    se_vals = np.array([
        0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063,
        2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547
    ], dtype=float)
    # Assign Qm by typical CQI regions: 1..4->QPSK, 5..8->16QAM, 9..15->64QAM
    qm_by_idx = [2]*4 + [4]*4 + [6]*7
    out: List[MCS] = []
    j = 0
    for i, se in enumerate(se_vals, start=1):
        Qm = qm_by_idx[i-1]
        R = float(se) / float(Qm)
        out.append(MCS(idx=j, Qm=Qm, R=R))
        j += 1
    return out


def _table2_256qam() -> List[MCS]:
    """Approximate Table 5.1.3.1-2: 256QAM-capable set.

    Extend 64QAM set with a few 256QAM points for high-SE region.
    """
    base = _table1_64qam()
    # Add 256QAM candidates; choose code rates to span SE ~6.0..7.6
    extra = [
        MCS(idx=len(base)+0, Qm=8, R=0.75),   # se=6.00
        MCS(idx=len(base)+1, Qm=8, R=0.80),   # se=6.40
        MCS(idx=len(base)+2, Qm=8, R=0.85),   # se=6.80
        MCS(idx=len(base)+3, Qm=8, R=0.90),   # se=7.20
        MCS(idx=len(base)+4, Qm=8, R=0.95),   # se=7.60
    ]
    return base + extra


def _table3_low_se() -> List[MCS]:
    """Approximate Table 5.1.3.1-3: Low-SE set (coverage).

    Keep QPSK-only with smaller steps.
    """
    se_vals = [0.06, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    out: List[MCS] = []
    for i, se in enumerate(se_vals):
        out.append(MCS(idx=i, Qm=2, R=float(se)/2.0))
    return out


_MCS_TABLES_3GPP: Dict[str, List[MCS]] = {}
_BLER_CURVES: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}


def register_mcs_tables_from_file(path: str) -> None:
    """Register 3GPP MCS tables from an external JSON-like file.

    Expected schema (JSON):
    {
      "table_1": [ {"idx":0, "Qm":2, "R_x1024":120}, ... ],
      "table_2": [ {"idx":0, "Qm":2, "R_x1024":120}, ... ],
      "table_3": [ {"idx":0, "Qm":2, "R_x1024":30},  ... ]
    }
    - Indices should be contiguous from 0.
    - R is provided in 1/1024 units; converted to [0,1] here.
    """
    import json, os
    if not os.path.exists(path):
        raise FileNotFoundError(f"MCS table file not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    for key_in, key_out in [("table_1", "3gpp_table_1"), ("table_2", "3gpp_table_2"), ("table_3", "3gpp_table_3")]:
        if key_in not in data:
            continue
        arr = data[key_in]
        lst: List[MCS] = []
        for ent in arr:
            idx = int(ent.get("idx"))
            Qm = int(ent.get("Qm"))
            Rx = ent.get("R_x1024")
            # Skip reserved or invalid entries
            try:
                R = float(Rx) / 1024.0
            except Exception:
                continue
            lst.append(MCS(idx=idx, Qm=Qm, R=R))
        # sort by idx for safety
        lst.sort(key=lambda m: m.idx)
        _MCS_TABLES_3GPP[key_out] = lst


def register_bler_curves_from_file(path: str) -> None:
    """Register BLER vs SINR curves per MCS and table.

    JSON schema example:
    {
      "3gpp_table_2": [
        {"idx": 10, "sinr_db": [...], "bler": [...]},
        ...
      ],
      "3gpp_table_1": [...],
      "table_1_64qam": [...]
    }
    - `bler` values clipped to [0,1].
    - For unknown tables or malformed entries, silently skip.
    """
    import json, os
    if not os.path.exists(path):
        raise FileNotFoundError(f"BLER curve file not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    for tbl, arr in data.items():
        try:
            cur: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for ent in arr:
                idx = int(ent.get("idx"))
                x = np.asarray(ent.get("sinr_db", []), dtype=float)
                y = np.asarray(ent.get("bler", []), dtype=float)
                if x.size < 2 or y.size != x.size:
                    continue
                y = np.clip(y, 0.0, 1.0)
                # Ensure increasing x for interpolation
                order = np.argsort(x)
                x = x[order]
                y = y[order]
                cur[idx] = (x, y)
            if cur:
                _BLER_CURVES[str(tbl).lower()] = cur
        except Exception:
            continue


def get_mcs_table(kind: str) -> List[MCS]:
    k = str(kind).lower()
    # 3GPP tables (require registration from file)
    if k in ("3gpp_table_1", "3gpp_table_2", "3gpp_table_3"):
        if k not in _MCS_TABLES_3GPP:
            raise ValueError(f"3GPP MCS table '{k}' not registered. Provide CONFIG['mcs_3gpp_table_path'] to load.")
        return _MCS_TABLES_3GPP[k]
    if k in ("table_1", "table1", "nr_64qam", "table_1_64qam"):
        return _table1_64qam()
    if k in ("table_2", "table2", "nr_256qam", "table_2_256qam"):
        return _table2_256qam()
    if k in ("table_3", "table3", "low_se", "table_3_low_se"):
        return _table3_low_se()
    raise ValueError(f"Unknown MCS table kind: {kind}")


# ------------------------------
# TBS (38.214 §5.1.3.2 inspired)
# ------------------------------
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

    This follows the 5G NR TBS steps in a practical, widely-used form.
    """
    N_RE = n_prb * n_re_per_prb(cp_type, dmrs_sym_per_slot, dmrs_re_per_sym_per_prb, oh_prb)
    N_info = N_RE * mcs.Qm * mcs.R * max(1, int(n_layers))
    if N_info <= 0:
        return 0
    # NR practical TBS (see also 38.214/38.212); numeric-safe approximation
    if N_info <= 3824:
        n = max(3, int(np.floor(np.log2(N_info))) - 6)
        TBS = max(24, int(2 ** n * np.floor(N_info / (2 ** n))))
    else:
        n = int(np.floor(np.log2(N_info - 24))) - 5
        Ninfo_prime = max(24, int(2 ** n * np.floor((N_info - 24) / (2 ** n))))
        C = int(np.ceil((Ninfo_prime + 24) / 8448))
        TBS = 8 * C * int(np.ceil((Ninfo_prime + 24) / (8 * C))) - 24
    return int(max(0, TBS))


# ------------------------------
# BLER model and OLLA
# ------------------------------
def _shannon_required_sinr_db(se_bits_per_hz: float) -> float:
    """Invert Shannon to get the SNR needed for given SE; add guard for <=0."""
    se = max(1e-9, float(se_bits_per_hz))
    gamma = (2.0 ** se) - 1.0
    return 10.0 * np.log10(gamma)


def bler_awgn_sigmoid(
    sinr_eff_db: np.ndarray,
    mcs: MCS,
    slope_db: float = 1.0,
    margin_db: float = 1.5,
) -> np.ndarray:
    """Simple BLER curve: logistic vs (SINR - threshold).

    - threshold is derived from Shannon for the MCS SE, with a positive margin.
    - slope_db sets steepness (smaller -> sharper transition).
    Returns BLER in [0,1].
    """
    se = mcs.se
    th_db = _shannon_required_sinr_db(se) + float(margin_db)
    k = max(1e-6, float(slope_db))
    x = (np.asarray(sinr_eff_db, dtype=float) - th_db) / k
    # Logistic: p = 1/(1+exp(x)) gives p~0.5 at x=0; adjust to reach ~0.1 near +2.2k
    p = 1.0 / (1.0 + np.exp(x))
    return np.clip(p, 0.0, 1.0)


def bler_from_registered_curves(
    sinr_eff_db: np.ndarray,
    mcs: MCS,
    table_kind: Optional[str],
) -> Optional[np.ndarray]:
    """Interpolate BLER from registered curves if available for (table_kind, mcs.idx).
    Returns None if not available.
    """
    if table_kind is None:
        return None
    tbl = _BLER_CURVES.get(str(table_kind).lower())
    if not tbl:
        return None
    xy = tbl.get(int(mcs.idx))
    if xy is None:
        return None
    x, y = xy
    xx = np.asarray(sinr_eff_db, dtype=float)
    # Extrapolate with edge values
    y_interp = np.interp(xx, x, y, left=y[0], right=y[-1])
    return np.clip(y_interp, 0.0, 1.0)


class OLLA:
    def __init__(self, step_up_db: float = 0.1, step_down_db: float = 0.1, init_offset_db: float = 0.0, p_target: float = 0.1):
        self.step_up_db = float(step_up_db)
        self.step_down_db = float(step_down_db)
        self.offset_db = float(init_offset_db)
        self.p_target = float(p_target)

    def apply(self, sinr_db: np.ndarray) -> np.ndarray:
        return np.asarray(sinr_db, dtype=float) + self.offset_db

    def update(self, is_ack: bool) -> None:
        # NACK -> increase offset (more conservative); ACK -> decrease offset
        if is_ack:
            self.offset_db -= self.step_down_db
        else:
            self.offset_db += self.step_up_db


def choose_mcs_from_sinr(
    sinr_eff_db: float,
    table_kind: str,
    target_bler: float = 0.1,
    slope_db: float = 1.0,
    margin_db: float = 1.5,
) -> MCS:
    """Pick the highest MCS (by SE) with predicted BLER <= target.
    If none satisfies, pick the most robust one.
    """
    cands = get_mcs_table(table_kind)
    best = cands[0]
    for m in cands:
        p_curve = bler_from_registered_curves(sinr_eff_db, m, table_kind)
        if p_curve is None:
            p = float(bler_awgn_sigmoid(sinr_eff_db, m, slope_db=slope_db, margin_db=margin_db))
        else:
            p = float(np.squeeze(p_curve))
        if p <= target_bler and m.se >= best.se:
            best = m
    # If all exceed target, pick the lowest SE
    if best is None:
        best = min(cands, key=lambda m: m.se)
    return best


# ------------------------------
# EESM combine across transmissions
# ------------------------------
def combine_eff_sinr_db(prev_eff_db: Optional[float], curr_eff_db: float) -> float:
    """Approximate soft-combining by adding linear SNRs of effective values."""
    if prev_eff_db is None:
        return float(curr_eff_db)
    a = 10.0 ** (float(prev_eff_db) / 10.0)
    b = 10.0 ** (float(curr_eff_db) / 10.0)
    return 10.0 * np.log10(a + b)


def eff_sinr_eesm_db(sinr_db_vec: np.ndarray, beta_db: float = 1.0) -> float:
    """Convenience wrapper around per-PRB EESM to single scalar (dB)."""
    x = np.asarray(sinr_db_vec, dtype=float)
    beta = max(1e-6, float(beta_db))
    t = np.exp(-x / beta)
    m = np.maximum(np.mean(t, axis=-1), 1e-30)
    return float(-beta * np.log(m))


def re_per_prb_from_config(cfg: Dict) -> int:
    """Resolve effective RE/PRB from config, preferring DL (PDSCH) keys.

    Backward-compatible: falls back to PUSCH keys if PDSCH ones are absent.
    """
    cp = str(cfg.get("cp_type", "normal"))
    dmrs_sym = int(cfg.get("pdsch_dmrs_sym_per_slot", cfg.get("pusch_dmrs_sym_per_slot", 1)))
    dmrs_re = int(cfg.get("dmrs_re_per_sym_per_prb", 6))
    oh = int(cfg.get("oh_prb", 0))
    return n_re_per_prb(cp, dmrs_sym, dmrs_re, oh)
