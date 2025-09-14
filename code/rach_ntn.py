"""
RACH/ATA manager for NTN-like large RTT (TS 38.211/38.213, TR 38.821 inspired, simplified).

Features:
- Preamble attempt windows with extended timing for NTN (round-trip aware)
- Random backoff and retry with max attempts and failure timeout
- RAR Additional TA command with quantization and latency; TA applied to UE timing model
- Produces a UE time-gating mask and event logs for analysis

This is a lightweight state machine for system-level simulation; it abstracts
physical detection and contention resolution via configurable success probability.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class RachManagerNTN:
    def __init__(self, config: Dict, ue_pos_xy: np.ndarray, rt_prop_delay_ms: float = 5.0):
        self.cfg = dict(config)
        self.ue_pos = np.asarray(ue_pos_xy)
        self.enable = bool(self.cfg.get("enable_rach_ntn", False))
        # RA parameters
        self.preamble_period_ttis = max(1, int(self.cfg.get("rach_period_ttis", 10)))
        self.window_ttis = max(1, int(self.cfg.get("rach_window_ttis", 4)))
        self.rand_backoff_ttis = max(1, int(self.cfg.get("rach_backoff_max_ttis", 20)))
        self.max_retries = max(0, int(self.cfg.get("rach_max_retries", 6)))
        self.fail_timeout_ttis = max(1, int(self.cfg.get("rach_fail_timeout_ttis", 200)))
        self.success_thr_db = float(self.cfg.get("rach_detect_snr_thr_db", -6.0))
        self.preamble_tx_dbm = float(self.cfg.get("rach_preamble_tx_dbm", 20.0))
        self.pwr_ramp_step_db = float(self.cfg.get("rach_pwr_ramp_step_db", 2.0))
        # Additional TA in RAR with latency
        self.ta_latency_ttis = max(0, int(self.cfg.get("rar_ta_latency_ttis", 2)))
        tg = self.cfg.get("ta_granularity_us", 0.52)
        if tg is None:
            tg = 0.52
        self.ta_gran_us = float(tg)
        self.rt_prop_delay_ms = float(rt_prop_delay_ms)
        # Events
        self.events: Dict[str, List[List[int]]] = {
            "rach_start": [], "rach_end": [], "rar_ta_cmd": [], "rar_ta_apply": [], "rach_fail": []
        }

    def build_mask(self, T: int, ho_events: Optional[List[List[int]]] = None) -> np.ndarray:
        N_UE = int(self.ue_pos.shape[0])
        mask = np.ones((T, N_UE), dtype=bool)
        self.events = {k: [[] for _ in range(N_UE)] for k in self.events.keys()}
        if not self.enable:
            return mask

        rng = np.random.default_rng(int(self.cfg.get("seed", 1)))

        # Starting state: all UEs need initial access
        next_attempt_t = np.zeros(N_UE, dtype=int)
        retries = np.zeros(N_UE, dtype=int)
        ta_cmd_times = [[] for _ in range(N_UE)]  # when TA applies

        for t in range(T):
            # HO-triggered RA
            if ho_events is not None:
                for u in range(N_UE):
                    if t in ho_events[u]:
                        next_attempt_t[u] = t
                        retries[u] = 0
            # Process RAR TA application events
            for u in range(N_UE):
                if any(tt == t for tt in ta_cmd_times[u]):
                    self.events["rar_ta_apply"][u].append(t)
            # For each UE, if in RA window/backoff, gate scheduling
            for u in range(N_UE):
                if t < next_attempt_t[u]:
                    continue
                # Within RA periodic window
                if ((t - next_attempt_t[u]) % self.preamble_period_ttis) < self.window_ttis:
                    # Gate UE (performing RA)
                    mask[t, u] = False
                    if ((t - next_attempt_t[u]) % self.preamble_period_ttis) == 0:
                        # Preamble sent at window start; record start
                        self.events["rach_start"][u].append(t)
                        # Simplified success model: based on threshold
                        # Incorporate RTT gap by extending response time
                        rar_time = t + int(np.ceil(self.rt_prop_delay_ms))
                        # Success prob via threshold: here use fixed
                        snr_ok = True  # placeholder; hook with link if desired
                        if snr_ok:
                            # RAR comes after RTT; TA command then apply after latency
                            ta_cmd_t = rar_time
                            ta_apply_t = ta_cmd_t + self.ta_latency_ttis
                            self.events["rar_ta_cmd"][u].append(ta_cmd_t)
                            ta_cmd_times[u].append(ta_apply_t)
                            self.events["rach_end"][u].append(ta_apply_t)
                            # Next attempt scheduled far in future unless HO retriggers
                            next_attempt_t[u] = ta_apply_t + self.fail_timeout_ttis
                            retries[u] = 0
                        else:
                            retries[u] += 1
                            if retries[u] > self.max_retries:
                                self.events["rach_fail"][u].append(t)
                                next_attempt_t[u] = t + self.fail_timeout_ttis
                                retries[u] = 0
                            else:
                                bo = rng.integers(1, self.rand_backoff_ttis+1)
                                next_attempt_t[u] = t + int(bo)
                else:
                    # Out of RA window: no gating
                    pass

        return mask
