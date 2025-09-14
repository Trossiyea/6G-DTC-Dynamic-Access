"""
HARQ ACK deferral manager (Stage-2 minimal).

Approximates 38.214 HARQ ACK timing by:
- Limiting the number of outstanding HARQ processes per UE (num_procs).
- Deferring ACK by a configurable number of TTIs (ack_delay_ttis).

Assumptions:
- Every scheduled UE transmission uses one HARQ process (per TTI granularity).
- All transmissions ACK after the delay (no NACK/retransmission modeling yet).
"""

from typing import Dict, List, Optional
import numpy as np


class HarqManager:
    def __init__(self, num_ue: int, num_procs: int, ack_delay_ttis: int):
        self.N = int(num_ue)
        self.P = max(1, int(num_procs))
        self.D = max(1, int(ack_delay_ttis))
        self._outstanding = np.zeros(self.N, dtype=int)
        # For each UE, a list of completion times (ttis) when ACK arrives
        self._completions: List[List[int]] = [[] for _ in range(self.N)]
        self.t = 0

    def advance_time(self, t: Optional[int] = None) -> None:
        """Advance internal time to t (or t+1) and apply completions."""
        if t is None:
            self.t += 1
        else:
            self.t = int(t)
        for u in range(self.N):
            # Pop any completions due at time t
            if not self._completions[u]:
                continue
            # The list is small; use while
            changed = True
            while changed and self._completions[u]:
                if self._completions[u][0] <= self.t:
                    self._completions[u].pop(0)
                    if self._outstanding[u] > 0:
                        self._outstanding[u] -= 1
                else:
                    changed = False

    def can_schedule(self, ue: int) -> bool:
        return self._outstanding[int(ue)] < self.P

    def on_scheduled(self, scheduled_ues: np.ndarray) -> None:
        """Mark scheduled UEs for this TTI (unique set), increasing outstanding and scheduling completion."""
        if scheduled_ues.size == 0:
            return
        t_ack = self.t + self.D
        for u in np.unique(scheduled_ues.astype(int)):
            self._outstanding[u] += 1
            # Append completion time in sorted order
            lst = self._completions[u]
            # Maintain monotonic order (append-only)
            lst.append(t_ack)

