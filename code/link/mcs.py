# -*- coding: utf-8 -*-
"""
MCS Table Management for NR (TS 38.214).

Provides:
- MCS dataclass with idx, Qm, R, and derived SE
- Built-in tables (Table 1/2/3) per 3GPP TS 38.214
- External 3GPP table registration from JSON
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
import os


@dataclass(frozen=True)
class MCS:
    """MCS entry with modulation order and code rate."""
    idx: int
    Qm: int       # Modulation order (2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM)
    R: float      # Code rate in [0, 1]

    @property
    def se(self) -> float:
        """Spectral efficiency (bits/s/Hz)."""
        return self.Qm * self.R


# ------------------------------
# Built-in MCS Tables (3GPP TS 38.214)
# ------------------------------
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
    """Build MCS table from spec entries."""
    table: List[MCS] = []
    for idx, qm, rx in entries:
        if isinstance(rx, str):
            continue  # Skip reserved entries
        R = float(rx) / 1024.0
        table.append(MCS(idx=idx, Qm=int(qm), R=R))
    table.sort(key=lambda m: m.idx)
    return table


_DEFAULT_MCS_TABLES: Dict[str, List[MCS]] = {
    "table_1_64qam": _build_table(_TABLE1_SPEC),
    "table_2_256qam": _build_table(_TABLE2_SPEC),
    "table_3_low_se": _build_table(_TABLE3_SPEC),
}

# External 3GPP tables loaded at runtime
_MCS_TABLES_3GPP: Dict[str, List[MCS]] = {}


def _canonical_table_kind(kind: str) -> Tuple[str, Optional[str]]:
    """Return (canonical_name, override_key) for a requested table kind.

    Returns:
        Tuple of (canonical table name for defaults, override key for 3GPP tables)
    """
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


def register_mcs_tables_from_file(path: str) -> None:
    """Register 3GPP MCS tables from an external JSON file.

    Expected schema:
    {
      "table_1": [ {"idx":0, "Qm":2, "R_x1024":120}, ... ],
      "table_2": [ {"idx":0, "Qm":2, "R_x1024":120}, ... ],
      "table_3": [ {"idx":0, "Qm":2, "R_x1024":30},  ... ]
    }

    R is provided in 1/1024 units; converted to [0,1] here.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MCS table file not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    for key_in, key_out in [
        ("table_1", "3gpp_table_1"),
        ("table_2", "3gpp_table_2"),
        ("table_3", "3gpp_table_3")
    ]:
        if key_in not in data:
            continue
        arr = data[key_in]
        lst: List[MCS] = []
        for ent in arr:
            idx = int(ent.get("idx"))
            Qm = int(ent.get("Qm"))
            Rx = ent.get("R_x1024")
            try:
                R = float(Rx) / 1024.0
            except Exception:
                continue  # Skip reserved or invalid entries
            lst.append(MCS(idx=idx, Qm=Qm, R=R))
        lst.sort(key=lambda m: m.idx)
        _MCS_TABLES_3GPP[key_out] = lst


def get_mcs_table(kind: str) -> List[MCS]:
    """Get MCS table by kind name.

    Args:
        kind: Table identifier (e.g., 'table_1_64qam', '3gpp_table_1')

    Returns:
        List of MCS entries (copy)

    Raises:
        ValueError: If table kind unknown or 3GPP table not registered
    """
    canon, override = _canonical_table_kind(kind)
    if override is not None:
        if override not in _MCS_TABLES_3GPP:
            raise ValueError(
                f"3GPP MCS table '{override}' not registered. "
                "Provide CONFIG['mcs_3gpp_table_path'] to load."
            )
        return list(_MCS_TABLES_3GPP[override])
    if canon in _DEFAULT_MCS_TABLES:
        return list(_DEFAULT_MCS_TABLES[canon])
    raise ValueError(f"Unknown MCS table kind: {kind}")


def get_default_tables() -> Dict[str, List[MCS]]:
    """Get reference to default MCS tables (for internal use)."""
    return _DEFAULT_MCS_TABLES
