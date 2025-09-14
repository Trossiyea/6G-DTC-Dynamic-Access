"""
Random Access (RA/RACH) manager (Stage-1 minimal).

Approximate initial access and post-HO re-access as a fixed-length
procedure that gates UE schedulability for a configured number of TTIs.

This scaffold does not simulate preamble power ramping, contention
resolution, or failures; these can be added later. It logs events and
produces a UE mask to combine with HO gating.
"""

from typing import Dict, List
import numpy as np


class RachManager:
    def __init__(self, config: Dict, ue_pos_xy: np.ndarray):
        self.cfg = dict(config)
        self.ue_pos = np.asarray(ue_pos_xy)
        self.enable = bool(self.cfg.get("enable_rach_gating", True))
        self.proc_len = max(1, int(self.cfg.get("rach_proc_ttis", 5)))
        self.on_ho = bool(self.cfg.get("rach_on_ho", True))
        # Events
        self.events: Dict[str, List[List[int]]] = {"rach_start": [], "rach_end": []}

    def build_mask(self, T: int, ho_events: List[List[int]]) -> np.ndarray:
        N_UE = int(self.ue_pos.shape[0])
        mask = np.ones((T, N_UE), dtype=bool)
        if not self.enable:
            self.events["rach_start"] = [[] for _ in range(N_UE)]
            self.events["rach_end"] = [[] for _ in range(N_UE)]
            return mask
        # Initial access at t=0
        rach_start_lists: List[List[int]] = [[] for _ in range(N_UE)]
        rach_end_lists: List[List[int]] = [[] for _ in range(N_UE)]
        for u in range(N_UE):
            # Initial access
            t0 = 0
            t1 = min(T, t0 + self.proc_len)
            if t0 < T:
                mask[t0:t1, u] = False
                rach_start_lists[u].append(t0)
                rach_end_lists[u].append(t1 - 1)
            # Post-HO access
            if self.on_ho and ho_events and len(ho_events[u]) > 0:
                for ts in ho_events[u]:
                    t0 = ts
                    t1 = min(T, t0 + self.proc_len)
                    mask[t0:t1, u] = False
                    rach_start_lists[u].append(t0)
                    rach_end_lists[u].append(t1 - 1)
        self.events["rach_start"] = rach_start_lists
        self.events["rach_end"] = rach_end_lists
        return mask

