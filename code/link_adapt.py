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

from csi import get_nr_cqi_table


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
# MCS tables (3GPP TS 38.214)
# ------------------------------
@dataclass(frozen=True)
class MCS:
    idx: int
    Qm: int
    R: float  # code rate in [0,1]

    @property
    def se(self) -> float:
        return self.Qm * self.R


_TABLE1_SPEC = [
    (0, 2, 120), (1, 2, 157), (2, 2, 193), (3, 2, 251), (4, 2, 308),
    (5, 2, 379), (6, 2, 449), (7, 2, 526), (8, 2, 602), (9, 2, 679),
    (10, 4, 340), (11, 4, 378), (12, 4, 434), (13, 4, 490), (14, 4, 553),
    (15, 4, 616), (16, 4, 658), (17, 6, 438), (18, 6, 466), (19, 6, 517),
    (20, 6, 567), (21, 6, 616), (22, 6, 666), (23, 6, 719), (24, 6, 772),
    (25, 6, 822), (26, 6, 873), (27, 6, 910), (28, 6, 948),
]

_TABLE2_SPEC = [
    (0, 2, 120), (1, 2, 193), (2, 2, 308), (3, 2, 449), (4, 2, 602),
    (5, 4, 378), (6, 4, 434), (7, 4, 490), (8, 4, 553), (9, 4, 616),
    (10, 4, 658), (11, 6, 466), (12, 6, 517), (13, 6, 567), (14, 6, 616),
    (15, 6, 666), (16, 6, 719), (17, 6, 772), (18, 6, 822), (19, 6, 873),
    (20, 8, 682.5), (21, 8, 711), (22, 8, 754), (23, 8, 797), (24, 8, 841),
    (25, 8, 885), (26, 8, 916.5), (27, 8, 948),
]

_TABLE3_SPEC = [
    (0, 2, 30), (1, 2, 40), (2, 2, 50), (3, 2, 64), (4, 2, 78),
    (5, 2, 99), (6, 2, 120), (7, 2, 157), (8, 2, 193), (9, 2, 251),
    (10, 2, 308), (11, 2, 379), (12, 2, 449), (13, 2, 526), (14, 2, 602),
    (15, 4, 340), (16, 4, 378), (17, 4, 434), (18, 4, 490), (19, 4, 553),
    (20, 4, 616), (21, 6, 438), (22, 6, 466), (23, 6, 517), (24, 6, 567),
    (25, 6, 616), (26, 6, 666), (27, 6, 719), (28, 6, 772),
]


def _build_table(entries: List[Tuple[int, int, Any]]) -> List[MCS]:
    table: List[MCS] = []
    for idx, qm, rx in entries:
        if isinstance(rx, str):
            continue
        R = float(rx) / 1024.0
        table.append(MCS(idx=idx, Qm=int(qm), R=R))
    table.sort(key=lambda m: m.idx)
    return table


_DEFAULT_MCS_TABLES: Dict[str, List[MCS]] = {
    "table_1_64qam": _build_table(_TABLE1_SPEC),
    "table_2_256qam": _build_table(_TABLE2_SPEC),
    "table_3_low_se": _build_table(_TABLE3_SPEC),
}


def _canonical_table_kind(kind: str) -> Tuple[str, Optional[str]]:
    """Return (canonical_name, override_key) for a requested table kind."""
    k = str(kind or "").lower()
    if k in ("legacy", "table_1", "table1", "table_1_64qam", "nr_64qam"):
        return "table_1_64qam", None
    if k in ("table_2", "table2", "table_2_256qam", "nr_256qam"):
        return "table_2_256qam", None
    if k in ("table_3", "table3", "table_3_low_se", "low_se"):
        return "table_3_low_se", None
    if k == "3gpp_table_1":
        return "table_1_64qam", "3gpp_table_1"
    if k == "3gpp_table_2":
        return "table_2_256qam", "3gpp_table_2"
    if k == "3gpp_table_3":
        return "table_3_low_se", "3gpp_table_3"
    return k, None


_MCS_TABLES_3GPP: Dict[str, List[MCS]] = {}
_BLER_CURVES: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

# Build AWGN 10% BLER thresholds by interpolating CQI anchors.
_CQI_TABLE = get_nr_cqi_table("nr_64qam")
_CQI_SE = (_CQI_TABLE[:, 1] * _CQI_TABLE[:, 2]) / 1024.0
_CQI_THR_DB = _CQI_TABLE[:, 0]


def _sinr10_from_se(se: float) -> float:
    """Interpolate (or gently extrapolate) SINR for 10% BLER given spectral efficiency."""
    s = float(se)
    if s <= _CQI_SE[0]:
        slope = (_CQI_THR_DB[1] - _CQI_THR_DB[0]) / (_CQI_SE[1] - _CQI_SE[0])
        return float(_CQI_THR_DB[0] + slope * (s - _CQI_SE[0]))
    if s >= _CQI_SE[-1]:
        slope = (_CQI_THR_DB[-1] - _CQI_THR_DB[-2]) / (_CQI_SE[-1] - _CQI_SE[-2])
        return float(_CQI_THR_DB[-1] + slope * (s - _CQI_SE[-1]))
    return float(np.interp(s, _CQI_SE, _CQI_THR_DB))


_DEFAULT_SINR_THRESH_DB: Dict[str, Dict[int, float]] = {}
for name, tbl in _DEFAULT_MCS_TABLES.items():
    thr_map: Dict[int, float] = {}
    for m in tbl:
        thr_map[m.idx] = _sinr10_from_se(m.se)
    _DEFAULT_SINR_THRESH_DB[name] = thr_map


def _table1_64qam() -> List[MCS]:
    return list(_DEFAULT_MCS_TABLES["table_1_64qam"])


def _table2_256qam() -> List[MCS]:
    return list(_DEFAULT_MCS_TABLES["table_2_256qam"])


def _table3_low_se() -> List[MCS]:
    return list(_DEFAULT_MCS_TABLES["table_3_low_se"])


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
    canon, override = _canonical_table_kind(kind)
    if override is not None:
        if override not in _MCS_TABLES_3GPP:
            raise ValueError(
                f"3GPP MCS table '{override}' not registered. Provide CONFIG['mcs_3gpp_table_path'] to load."
            )
        return list(_MCS_TABLES_3GPP[override])
    if canon in _DEFAULT_MCS_TABLES:
        return list(_DEFAULT_MCS_TABLES[canon])
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
    table_kind: Optional[str] = None,
) -> np.ndarray:
    """Simple BLER curve using a logistic fit around the 10% BLER SINR point.

    When ``table_kind`` is provided we use the interpolated 38.214 AWGN anchor
    for that MCS index. Otherwise we fall back to a Shannon+margin heuristic to
    preserve backward compatibility.
    """
    sinr_eff_db = np.asarray(sinr_eff_db, dtype=float)
    k = max(1e-6, float(slope_db))

    th_override = None
    if table_kind is not None:
        canon, _ = _canonical_table_kind(table_kind)
        thr_map = _DEFAULT_SINR_THRESH_DB.get(canon)
        if thr_map is not None:
            th_override = thr_map.get(int(mcs.idx))

    if th_override is None:
        se = mcs.se
        th_db = _shannon_required_sinr_db(se) + float(margin_db)
    else:
        th_db = float(th_override)

    mu = th_db - k * np.log(9.0)  # ensures BLER≈0.1 at SINR=th_db
    x = (sinr_eff_db - mu) / k
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
    canon, override = _canonical_table_kind(table_kind)
    cands = get_mcs_table(table_kind)
    best = cands[0]
    for m in cands:
        p_curve = None
        for key in filter(None, [override, canon]):
            p_curve = bler_from_registered_curves(sinr_eff_db, m, key)
            if p_curve is not None:
                break
        if p_curve is None:
            p = float(
                bler_awgn_sigmoid(
                    sinr_eff_db,
                    m,
                    slope_db=slope_db,
                    margin_db=margin_db,
                    table_kind=canon,
                )
            )
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
