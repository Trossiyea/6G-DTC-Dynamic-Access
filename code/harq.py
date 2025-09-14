"""
HARQ ACK deferral manager (Stage-2 minimal).

Approximates 38.214 HARQ ACK timing by:
- Limiting the number of outstanding HARQ processes per UE (num_procs).
- Deferring ACK by a configurable number of TTIs (ack_delay_ttis).

Assumptions:
- Every scheduled UE transmission uses one HARQ process (per TTI granularity).
- All transmissions ACK after the delay (no NACK/retransmission modeling yet).
"""

from typing import Dict, List, Optional, Tuple
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


class HarqManagerFull:
    """
    Full HARQ manager with ACK/NACK, RV cycling, soft combining and OLLA.

    Integration contract with schedulers (block-based preferred):
    - Call advance_time(t) at the start of each TTI to realize ACK/NACK and
      receive credited bits per UE (for goodput/SE accounting).
    - Use can_schedule(ue) for process gating.
    - After resource assignment, call on_scheduled_blocks(sched_info) once per TTI
      to register new transmissions or retransmissions.

    Notes
    - This abstraction treats one TB per UE per TTI (single codeword) which fits
      the current block scheduling path.
    - Retransmissions are scheduled only after NACK feedback (ack delay applied).
    """

    def __init__(
        self,
        num_ue: int,
        num_procs: int,
        ack_delay_ttis: int,
        config: Dict,
    ) -> None:
        from link_adapt import OLLA  # local import to avoid cycles

        self.N = int(num_ue)
        self.P = max(1, int(num_procs))
        self.D = max(1, int(ack_delay_ttis))
        self.t = 0
        # Outstanding HARQ processes per UE
        self._outstanding = np.zeros(self.N, dtype=int)
        # Per-UE OLLA
        self._olla = [
            OLLA(
                step_up_db=float(config.get("olla_step_up_db", 0.1)),
                step_down_db=float(config.get("olla_step_down_db", 0.1)),
                init_offset_db=float(config.get("olla_init_offset_db", config.get("csi_olla_offset_db", 0.0))),
                p_target=float(config.get("harq_target_bler", 0.1)),
            )
            for _ in range(self.N)
        ]
        # Pending ACK events per UE: list of (t_ack, tb_id, proc_idx)
        self._acks: List[List[Tuple[int, int, int]]] = [[] for _ in range(self.N)]
        # Per-UE processes: each proc stores TB state or None
        self._procs: List[List[Optional[Dict]]] = [[None for _ in range(self.P)] for _ in range(self.N)]
        # Retx pending per UE: (proc_idx, tb_id) to be scheduled ASAP after NACK
        self._retx: List[Optional[Tuple[int, int]]] = [None for _ in range(self.N)]
        # TB id counter per UE
        self._tb_next: List[int] = [0 for _ in range(self.N)]
        # Config for link adaptation
        self.cfg = dict(config)
        self.mode = "full"
        # Stats
        self._ack_count = 0
        self._nack_count = 0
        self._mcs_counts: Dict[int, int] = {}
        self._olla_hist_avg: List[float] = []

    # ------------------------
    # Public API used by schedulers
    # ------------------------
    def is_full(self) -> bool:
        return True

    def advance_time(self, t: Optional[int] = None) -> np.ndarray:
        """Advance time and materialize ACK/NACK. Returns acked bits per UE for this TTI."""
        if t is None:
            self.t += 1
        else:
            self.t = int(t)
        ack_bits = np.zeros(self.N, dtype=float)
        # Iterate per UE
        for u in range(self.N):
            if not self._acks[u]:
                continue
            # multiple events could fall on same t; process in order
            pending = self._acks[u]
            i = 0
            while i < len(pending) and pending[i][0] <= self.t:
                _, tb_id, pidx = pending[i]
                tb = self._procs[u][pidx]
                if tb is None or tb.get("tb_id") != tb_id:
                    # stale or already resolved
                    i += 1
                    continue
                # Evaluate BLER on current combined eff SINR
                is_ack = self._decide_ack(u, tb)
                # Update OLLA based on outcome
                self._olla[u].update(is_ack=is_ack)
                if is_ack:
                    self._ack_count += 1
                    ack_bits[u] += float(tb.get("tbs_bits", 0))
                    # Free the process
                    self._procs[u][pidx] = None
                    if self._outstanding[u] > 0:
                        self._outstanding[u] -= 1
                else:
                    self._nack_count += 1
                    # Mark for retransmission
                    self._retx[u] = (pidx, tb_id)
                i += 1
            # prune processed events
            self._acks[u] = [ev for ev in pending if ev[0] > self.t]
        # Record average OLLA offset snapshot after updates
        offsets = [o.offset_db for o in self._olla]
        self._olla_hist_avg.append(float(np.mean(offsets) if offsets else 0.0))
        return ack_bits

    def can_schedule(self, ue: int) -> bool:
        return self._outstanding[int(ue)] < self.P

    def get_retx_ues(self) -> np.ndarray:
        """Boolean mask of UEs that are pending retransmission."""
        mask = np.zeros(self.N, dtype=bool)
        for u in range(self.N):
            if self._retx[u] is not None:
                mask[u] = True
        return mask

    def on_scheduled_blocks(self, info: Dict[int, Dict]) -> None:
        """
        Register scheduled TBs for a set of UEs in this TTI.
        info[u] = {
          'sinr_vec_db': np.ndarray[PRB],
          'n_prb': int,
          'eesm_beta_db': float,
        }
        """
        from link_adapt import eff_sinr_eesm_db, combine_eff_sinr_db, choose_mcs_from_sinr, calc_tbs_bits

        if not info:
            return
        for u, d in info.items():
            u = int(u)
            if d is None:
                continue
            # Build or fetch TB state
            retx_entry = self._retx[u]
            if retx_entry is not None:
                pidx, tbid = retx_entry
                tb = self._procs[u][pidx]
                if tb is None or tb.get("tb_id") != tbid:
                    # Shouldn't happen; clear and skip
                    self._retx[u] = None
                    continue
                # Combine effective SINR with new attempt
                sinr_eff_now_db = eff_sinr_eesm_db(np.asarray(d['sinr_vec_db']), beta_db=float(d.get('eesm_beta_db', 1.0)))
                tb['sinr_eff_db'] = combine_eff_sinr_db(tb.get('sinr_eff_db'), sinr_eff_now_db)
                tb['rv_idx'] = (tb.get('rv_idx', 0) + 1) % 4
                tb['n_retx'] = int(tb.get('n_retx', 0)) + 1
                # Schedule feedback
                self._acks[u].append((self.t + self.D, tb['tb_id'], pidx))
                # Clear retx flag; if NACK again it will be set on feedback
                self._retx[u] = None
            else:
                # New TB: acquire free process
                pidx = self._acquire_proc(u)
                if pidx < 0:
                    # No free process (should not happen due to gating)
                    continue
                sinr_eff_db = eff_sinr_eesm_db(np.asarray(d['sinr_vec_db']), beta_db=float(d.get('eesm_beta_db', 1.0)))
                # Apply OLLA offset for selection
                sinr_sel_db = float(self._olla[u].apply(sinr_eff_db))
                mcs = choose_mcs_from_sinr(
                    sinr_sel_db,
                    table_kind=str(self.cfg.get('mcs_table_kind', self.cfg.get('csi_mcs_table', 'table_1_64qam'))),
                    target_bler=float(self.cfg.get('harq_target_bler', 0.1)),
                    slope_db=float(self.cfg.get('bler_slope_db', 1.0)),
                    margin_db=float(self.cfg.get('bler_margin_db', 1.5)),
                )
                # Stats: record chosen MCS index
                self._mcs_counts[mcs.idx] = self._mcs_counts.get(mcs.idx, 0) + 1
                tbs_bits = calc_tbs_bits(
                    int(d['n_prb']), mcs,
                    n_layers=int(self.cfg.get('n_layers', 1)),
                    cp_type=str(self.cfg.get('cp_type', 'normal')),
                    dmrs_sym_per_slot=int(self.cfg.get('pusch_dmrs_sym_per_slot', 1)),
                    dmrs_re_per_sym_per_prb=int(self.cfg.get('dmrs_re_per_sym_per_prb', 6)),
                    oh_prb=int(self.cfg.get('oh_prb', 0)),
                )
                tb = {
                    'tb_id': int(self._tb_next[u]),
                    'rv_idx': 0,            # RV sequence start
                    'n_retx': 0,
                    'sinr_eff_db': float(sinr_eff_db),  # base eff SINR, combined on retx
                    'mcs_idx': int(mcs.idx),
                    'mcs_Qm': int(mcs.Qm),
                    'mcs_R': float(mcs.R),
                    'tbs_bits': int(tbs_bits),
                }
                self._tb_next[u] += 1
                self._procs[u][pidx] = tb
                # Schedule feedback
                self._acks[u].append((self.t + self.D, tb['tb_id'], pidx))

    # ------------------------
    # Internal helpers
    # ------------------------
    def _acquire_proc(self, ue: int) -> int:
        if self._outstanding[ue] >= self.P:
            return -1
        slots = self._procs[ue]
        for i in range(self.P):
            if slots[i] is None:
                slots[i] = {}
                self._outstanding[ue] += 1
                return i
        return -1

    def _decide_ack(self, ue: int, tb: Dict) -> bool:
        from link_adapt import bler_awgn_sigmoid, MCS
        # Effective SINR for current combined attempts
        sinr_eff_db = float(tb.get('sinr_eff_db', 0.0))
        m = MCS(idx=0, Qm=int(tb.get('mcs_Qm', 2)), R=float(tb.get('mcs_R', 0.1)))
        p = float(bler_awgn_sigmoid(
            sinr_eff_db,
            m,
            slope_db=float(self.cfg.get('bler_slope_db', 1.0)),
            margin_db=float(self.cfg.get('bler_margin_db', 1.5)),
        ))
        # Maximum retransmissions
        n_retx = int(tb.get('n_retx', 0))
        if n_retx >= int(self.cfg.get('harq_max_retx', 4)):
            # Force drop outcome with probability 1 (treat as NACK final -> drop)
            return False
        # Stochastic decision
        return (np.random.random() > p)

    def get_stats(self) -> Dict:
        """Return collected HARQ/link-adaptation statistics."""
        return {
            'ack_count': int(self._ack_count),
            'nack_count': int(self._nack_count),
            'mcs_counts': dict(self._mcs_counts),
            'olla_offset_avg': list(self._olla_hist_avg),
            'olla_last_per_ue': [float(o.offset_db) for o in self._olla],
        }
