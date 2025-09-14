"""
Handover/beam-switch manager (Stage-3 A3-like with measurement windows).

Implements two modes:
- Legacy (off-axis TTT): threshold on off-axis beyond (half_bw - margin) over TTT TTIs.
- A3-like with SSB/CSI-RS measurement windows: every ssb_period TTIs, compare
  serving vs. best neighbor beam RSRP (approx. via beam gain). If neighbor exceeds
  serving by hysteresis for a3_ttt measurement windows, start HO.

For A3 mode, if a multi-beam grid is available (OrbitModelMultiBeam with .grid),
neighbor beams are selected from k nearest grid centers around the UE; otherwise,
falls back to legacy off-axis metric but sampled at measurement period.

Outputs:
- ue_mask_time[T, UE] masked during HO interruption windows.
- events: {"ho_start": List[List[int]]}
"""

from typing import Dict, List, Tuple
import numpy as np
from orbit import simple_beam_gain_db


class HOManager:
    def __init__(self, config: Dict, orbit_model, ue_pos_xy: np.ndarray):
        self.cfg = dict(config)
        self.orbit = orbit_model
        self.ue_pos = np.asarray(ue_pos_xy)
        # Parameters
        self.enable = bool(self.cfg.get("enable_beam_ho", False)) and (self.orbit is not None)
        self.half_bw = float(self.cfg.get("beam_half_bw_deg", 4.0))
        self.margin = float(self.cfg.get("ho_trigger_margin_deg", 1.0))
        self.ttt = max(1, int(self.cfg.get("ho_ttt_ttis", 10)))
        self.int_len = max(1, int(self.cfg.get("ho_interrupt_ttis", 3)))
        # A3-like params
        self.a3_enable = bool(self.cfg.get("enable_a3_ho", True))
        self.meas_period = max(1, int(self.cfg.get("ssb_period_ttis", 20)))
        self.meas_offset = int(self.cfg.get("ssb_offset_ttis", 0))
        self.a3_hyst_db = float(self.cfg.get("a3_hysteresis_db", 3.0))
        self.a3_ttt_meas = max(1, int(self.cfg.get("a3_ttt_meas", 2)))
        self.nb_k = max(1, int(self.cfg.get("a3_neighbor_k", 4)))
        # State
        self._ttt_cnt = None
        self._in_ho = None
        self._rem = None
        self.events: Dict[str, List[List[int]]] = {"ho_start": []}
        self._a3_cnt = None

    def _current_alt_km(self, t: int) -> float:
        # Try OrbitSkyfield _subpoint_px to get time-varying altitude
        if hasattr(self.orbit, "_subpoint_px"):
            try:
                _, _, alt_km = self.orbit._subpoint_px(t)
                return float(alt_km)
            except Exception:
                pass
        # Fallback to configured altitude
        return float(getattr(self.orbit, "alt_km", self.cfg.get("sat_altitude_km", 600.0)))

    def _rsrp_gain_db(self, ue_xy: Tuple[float, float], center_xy: Tuple[float, float], alt_km: float) -> float:
        dx = (float(ue_xy[0]) - float(center_xy[0])) * float(self.orbit.cell_km)
        dy = (float(ue_xy[1]) - float(center_xy[1])) * float(self.orbit.cell_km)
        r_ground = np.hypot(dx, dy)
        offaxis = np.rad2deg(np.arctan2(r_ground, float(alt_km)))
        return float(simple_beam_gain_db(np.array([offaxis]), self.cfg.get("G_rx_db", 32.0), self.cfg.get("beam_half_bw_deg", 4.0), self.cfg.get("beam_edge_drop_db", 3.0))[0])

    def build_mask(self, T: int) -> np.ndarray:
        N_UE = int(self.ue_pos.shape[0])
        mask = np.ones((T, N_UE), dtype=bool)
        if not self.enable:
            self.events["ho_start"] = [[] for _ in range(N_UE)]
            return mask
        self._ttt_cnt = np.zeros(N_UE, dtype=int)
        self._a3_cnt = np.zeros(N_UE, dtype=int)
        self._in_ho = np.zeros(N_UE, dtype=bool)
        self._rem = np.zeros(N_UE, dtype=int)
        ho_lists: List[List[int]] = [[] for _ in range(N_UE)]
        # Serving beam center per UE (px); initialize to nearest grid center to UE if grid exists, else to current boresight
        serv_c = np.zeros((N_UE, 2), dtype=float)
        if hasattr(self.orbit, 'grid') and isinstance(self.orbit.grid, np.ndarray) and self.orbit.grid.size > 0:
            for u in range(N_UE):
                dx = self.orbit.grid[:,0] - float(self.ue_pos[u,0])
                dy = self.orbit.grid[:,1] - float(self.ue_pos[u,1])
                i = int(np.argmin(dx*dx + dy*dy))
                serv_c[u,0] = float(self.orbit.grid[i,0])
                serv_c[u,1] = float(self.orbit.grid[i,1])
        else:
            cx0, cy0 = self.orbit.beam_center_at(0)
            serv_c[:,0] = float(cx0)
            serv_c[:,1] = float(cy0)
        pending_c = np.zeros_like(serv_c)
        centers_time = np.zeros((T, N_UE, 2), dtype=float)
        thr = max(0.0, self.half_bw - self.margin)
        # Iterate time
        for t in range(T):
            _, offaxis = self.orbit.get_slant_and_offaxis(self.ue_pos, t)
            # When in HO, decrement remaining time and keep masked
            active = ~self._in_ho
            if self.a3_enable:
                # Measurement instant?
                if ((t - self.meas_offset) % self.meas_period) == 0:
                    alt_km = self._current_alt_km(t)
                    # Serving center per UE; evaluate best neighbor
                    # A3 condition per UE
                    for u in np.where(active)[0].tolist():
                        ue_xy = (self.ue_pos[u,0], self.ue_pos[u,1])
                        g_serv = self._rsrp_gain_db(ue_xy, (serv_c[u,0], serv_c[u,1]), alt_km)
                        margin_db = -1e9
                        best_center = (serv_c[u,0], serv_c[u,1])
                        # Neighbor search if grid available
                        if hasattr(self.orbit, 'grid') and isinstance(self.orbit.grid, np.ndarray) and self.orbit.grid.size > 0:
                            # pick k nearest beam centers to UE
                            dx = self.orbit.grid[:,0] - float(ue_xy[0])
                            dy = self.orbit.grid[:,1] - float(ue_xy[1])
                            d2 = dx*dx + dy*dy
                            idx = np.argsort(d2)[:max(self.nb_k,1)]
                            g_best = g_serv
                            for i in idx:
                                cxy = (self.orbit.grid[i,0], self.orbit.grid[i,1])
                                # skip current serving center
                                if abs(cxy[0]-serv_c[u,0])<1e-6 and abs(cxy[1]-serv_c[u,1])<1e-6:
                                    continue
                                g = self._rsrp_gain_db(ue_xy, cxy, alt_km)
                                if g > g_best:
                                    g_best = g
                                    best_center = cxy
                            margin_db = g_best - g_serv
                        else:
                            # Fallback: off-axis margin vs threshold
                            margin_db = (offaxis[u] - thr)
                        if margin_db > self.a3_hyst_db:
                            self._a3_cnt[u] += 1
                            pending_c[u,0] = best_center[0]
                            pending_c[u,1] = best_center[1]
                        else:
                            self._a3_cnt[u] = 0
                to_start = active & (self._a3_cnt >= self.a3_ttt_meas)
                self._a3_cnt[to_start] = 0
            else:
                # Legacy off-axis TTT per TTI
                need_ho = (offaxis > thr)
                self._ttt_cnt[active & need_ho] += 1
                self._ttt_cnt[active & (~need_ho)] = 0
                to_start = active & (self._ttt_cnt >= self.ttt)
                self._ttt_cnt[to_start] = 0
            if np.any(to_start):
                self._in_ho[to_start] = True
                self._rem[to_start] = self.int_len
                # log events
                for u in np.where(to_start)[0].tolist():
                    ho_lists[u].append(t)
            # Apply masking and progress HO timers
            mask[t, :] = ~self._in_ho
            self._rem[self._in_ho] -= 1
            done = self._in_ho & (self._rem <= 0)
            if np.any(done):
                # finalize new serving center for UEs that completed HO
                for u in np.where(done)[0].tolist():
                    if pending_c[u,0] != 0.0 or pending_c[u,1] != 0.0:
                        serv_c[u,0] = pending_c[u,0]
                        serv_c[u,1] = pending_c[u,1]
                self._in_ho[done] = False
                self._rem[done] = 0
            centers_time[t,:,:] = serv_c
        self.events["ho_start"] = ho_lists
        self.events["serving_centers_time"] = centers_time
        return mask
