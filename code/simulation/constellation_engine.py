# -*- coding: utf-8 -*-
"""
Multi-Satellite Constellation Simulation Engine.

Encapsulates the run_constellation() logic with:
- Per-satellite geometry and capacity computation
- UE-satellite association with handover
- Independent per-satellite scheduling
- Progress callbacks for Web integration
"""

import logging
from typing import Dict, Optional, List, Any, Callable
import numpy as np
from tqdm import tqdm

from .state import ConstellationState
from .callbacks import (
    CallbackManager, ProgressCallback, TTIMetrics, PhaseEvent, SimulationPhase
)
from .helpers import (
    generate_ue_positions,
    resolve_noise_and_prb_bw,
    apply_open_loop_power_control,
    compute_caps,
)

from data_io.radiomap import select_radio_map
from ntn import ConstellationOrbit
from logging_utils import get_logger

logger = get_logger(__name__)


class ConstellationEngine:
    """Multi-satellite constellation simulation engine.

    Encapsulates the `run_constellation()` logic with explicit state
    management, progress callbacks, and Web-friendly design.

    Example:
        >>> engine = ConstellationEngine(config)
        >>> engine.on_progress(lambda tti, m: print(f"TTI {tti}"))
        >>> result = engine.run()
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize engine with configuration.

        Args:
            config: Simulation configuration dict
        """
        self.config = config
        self._callbacks = CallbackManager()
        self._state: Optional[ConstellationState] = None
        self._rng: Optional[np.random.Generator] = None
        self._trace: Optional[Dict[str, Any]] = None

    @property
    def state(self) -> Optional[ConstellationState]:
        """Access current simulation state."""
        return self._state

    def add_callback(self, callback: ProgressCallback) -> "ConstellationEngine":
        """Add a progress callback. Returns self for chaining."""
        self._callbacks.add(callback)
        return self

    def on_progress(self, handler: Callable[[int, Dict], None]) -> "ConstellationEngine":
        """Register a simple progress handler function."""
        from .callbacks import FunctionCallback
        self._callbacks.add(FunctionCallback(
            on_tti=lambda m: handler(m.tti, m.to_dict())
        ))
        return self

    def initialize(self) -> ConstellationState:
        """Initialize constellation simulation state."""
        self._notify_phase(SimulationPhase.INITIALIZING, "Loading Radio Map and constellation")

        config = self.config
        seed = config["seed"]
        self._rng = np.random.default_rng(seed)

        N_UE = int(config["N_UE"])
        T = int(config["T"])

        # Load Radio Map
        R_xyz_dbm, X, Y, Z = select_radio_map(config)

        # Generate UE positions
        ue_pos = generate_ue_positions(N_UE, X, Y, self._rng)

        # Resolve noise and power
        noise_dbm, prb_bw_hz = resolve_noise_and_prb_bw(config)
        P_tx_dbm = apply_open_loop_power_control(config, 0.0, 0.0)

        # Initialize constellation orbit
        orbit = ConstellationOrbit(config, X, Y)

        # Create state
        self._state = ConstellationState(
            config=config,
            seed=seed,
            X=X, Y=Y, Z=Z,
            N_UE=N_UE, T=T,
            R_xyz_dbm=R_xyz_dbm,
            R_t=R_xyz_dbm.copy(),
            ue_pos=ue_pos,
            noise_dbm=noise_dbm,
            prb_bw_hz=prb_bw_hz,
            P_tx_dbm=float(P_tx_dbm) if np.isscalar(P_tx_dbm) else float(P_tx_dbm[0]),
            num_satellites=len(orbit.sats),
            serving=np.full(N_UE, -1, dtype=int),
            ho_timer=np.zeros(N_UE, dtype=int),
            curr_metric_db=np.full(N_UE, -1e9, dtype=float),
            ho_events=[[] for _ in range(N_UE)],
            outage_ttis=np.zeros(N_UE, dtype=int),
            serving_trace=[] if bool(config.get("include_serving_trace", False)) else None,
            orbit=orbit,
            phase="initialized",
        )

        self._notify_phase(SimulationPhase.INITIALIZED, "Constellation initialized")
        return self._state

    def run(self) -> Dict[str, Any]:
        """Execute the complete constellation simulation."""
        if self._state is None or not self._state.is_initialized():
            self.initialize()

        self._notify_phase(SimulationPhase.RUNNING, "Starting constellation simulation")

        # Register MCS tables
        self._register_mcs_tables()

        # Run TTI loop
        self._run_tti_loop()

        self._notify_phase(SimulationPhase.FINALIZING, "Computing final metrics")

        # Finalize
        result = self._finalize()

        self._state.phase = "complete"
        self._notify_phase(SimulationPhase.COMPLETE, "Constellation simulation complete")

        return result

    def _register_mcs_tables(self) -> None:
        """Register MCS tables from files."""
        from link import register_mcs_tables_from_file, register_bler_curves_from_file

        config = self.config
        try:
            if config.get("mcs_3gpp_table_path"):
                register_mcs_tables_from_file(config.get("mcs_3gpp_table_path"))
        except Exception as e:
            logger.warning(f"Failed to load 3GPP MCS tables: {e}")

        try:
            if config.get("bler_curve_path"):
                register_bler_curves_from_file(config.get("bler_curve_path"))
        except Exception as e:
            logger.warning(f"Failed to load BLER curves: {e}")

    def _run_tti_loop(self) -> None:
        """Execute the main TTI loop."""
        from scheduler.radiomap import pf_schedule_radiomap_blocks
        from link import HarqManagerFull
        from ntn_csi import snr_to_se_sched

        config = self.config
        state = self._state
        T = state.T
        N_UE = state.N_UE
        Z = state.Z

        # Configuration
        assoc_metric_kind = str(config.get("association_metric", "snr_wb")).lower()
        ho_enabled = bool(config.get("ho_enabled", True))
        ho_hyst_db = float(config.get("ho_hyst_db", 2.0))
        ho_ttt = int(config.get("ho_ttt_ttis", 20))
        min_elev = float(config.get("min_elev_deg", 5.0))
        enable_tv = bool(config.get("enable_time_varying", False))
        vx, vy = config.get("rm_drift_px", (0, 0))
        flicker = float(config.get("rm_flicker_db_std", 0.0))
        trace_enabled = bool(config.get("enable_trace", False)) or str(config.get("trace_level", "")).lower() in ("ui", "full", "kpi")

        # Optional trace buffers for UI playback
        if trace_enabled:
            self._trace = {
                "tti": np.arange(T, dtype=int),
                "se_sched_base_per_prb": np.zeros(T, dtype=float),
                "se_ack_base_per_prb": np.zeros(T, dtype=float),
                "se_sched_rm_per_prb": np.zeros(T, dtype=float),
                "se_ack_rm_per_prb": np.zeros(T, dtype=float),
                "active_ues": np.zeros(T, dtype=int),
                "candidate_sats": [],
                "assignments_base_by_tti": [],
                "assignments_rm_by_tti": [],
                "ue_thr_sched_base": np.zeros((T, N_UE), dtype=float),
                "ue_thr_ack_base": np.zeros((T, N_UE), dtype=float),
                "ue_thr_sched_rm": np.zeros((T, N_UE), dtype=float),
                "ue_thr_ack_rm": np.zeros((T, N_UE), dtype=float),
                "serving_trace": np.zeros((T, N_UE), dtype=int),
            }
        else:
            self._trace = None

        # Unified parameters
        prb_cap_unified = int(config.get("constellation_prb_cap", config.get("rm_max_prbs_per_ue", 20)))
        dlpm = str(config.get("rm_dl_power_model", "equal_prb"))
        Ptot = config.get("rm_P_tot_dbm")
        pmin = config.get("rm_p_min_dbm")
        pmax = config.get("rm_p_max_dbm")

        mcs_params = {
            "olla_offset_db": config.get("csi_olla_offset_db", 0.0),
            "mcs_table": config.get("csi_mcs_table", "legacy"),
            "residual_freq_hz": config.get("residual_freq_hz", 0.0),
            "scs_khz": config.get("scs_khz", 30),
        }

        # Progress bar
        show_progress = bool(config.get("show_progress", True))
        pbar = tqdm(range(T), desc="Constellation", unit="TTI", disable=not show_progress,
                    bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for t_idx in pbar:
            state.current_tti = t_idx

            # Time-varying Radio Map update
            if t_idx > 0 and enable_tv:
                if vx or vy:
                    state.R_t = np.roll(state.R_t, shift=(int(vx), int(vy), 0), axis=(0, 1, 2))
                if flicker > 0.0:
                    state.R_t = state.R_t + self._rng.normal(0.0, flicker, size=state.R_t.shape)

            # Get candidate satellites
            cand = state.orbit.candidate_indices_at(t_idx)
            if not cand:
                # No visible satellites this TTI -> treat as outage
                self._update_association([], assoc_metric_kind, min_elev, ho_enabled, ho_hyst_db, ho_ttt, t_idx)
                if state.serving_trace is not None:
                    state.serving_trace.append(np.array(state.serving, copy=True))
                state.outage_ttis += (state.serving < 0).astype(int)
                if trace_enabled and self._trace is not None:
                    self._trace["candidate_sats"].append([])
                    self._trace["assignments_base_by_tti"].append({})
                    self._trace["assignments_rm_by_tti"].append({})
                    self._trace["active_ues"][t_idx] = int(np.sum(state.serving >= 0))
                    self._trace["serving_trace"][t_idx] = np.asarray(state.serving, dtype=int)
                self._emit_tti_metrics(t_idx, T)
                continue
            if trace_enabled and self._trace is not None:
                self._trace["candidate_sats"].append([int(x) for x in cand])

            # Clear per-TTI caches
            state.clear_per_tti_caches()

            # Compute geometry and capacity for each candidate satellite
            for si in cand:
                L_fs, G_rx, tau_s_arr, fd_hz_arr, elev = state.orbit.geometry_for_sat(state.ue_pos, si, t_idx)

                cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb = compute_caps(
                    state.R_t, state.ue_pos,
                    P_tx_dbm=state.P_tx_dbm,
                    L_fs_db=L_fs,
                    G_rx_db=G_rx,
                    shadow_db_std=config["shadow_std_db"],
                    N0_dbm=state.noise_dbm,
                    rx_nf_db=config.get("rx_nf_db", 0.0),
                    impl_loss_db=config.get("impl_loss_db", 0.0),
                    seed=config["seed"],
                    elevation_deg=elev,
                    channel_model=config.get("channel_model", "3gpp_ntn"),
                    channel_params=config.get("channel_params"),
                    channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
                )

                state.snr_lin_per_sat[si] = snr_lin
                state.snr_wb_per_sat[si] = snr_lin_wb
                state.cap_per_sat[si] = cap
                state.prx_dbm_per_sat[si] = P_rx_dbm
                state.elev_per_sat[si] = elev

            # Association and handover
            self._update_association(cand, assoc_metric_kind, min_elev, ho_enabled, ho_hyst_db, ho_ttt, t_idx)

            # Record serving trace
            if state.serving_trace is not None:
                state.serving_trace.append(np.array(state.serving, copy=True))
            if trace_enabled and self._trace is not None:
                self._trace["serving_trace"][t_idx] = np.asarray(state.serving, dtype=int)

            # Update outage counter
            state.outage_ttis += (state.serving < 0).astype(int)

            # Per-satellite scheduling
            base_thr_sched_t = np.zeros(N_UE, dtype=float)
            base_thr_ack_t = np.zeros(N_UE, dtype=float)
            rm_thr_sched_t = np.zeros(N_UE, dtype=float)
            rm_thr_ack_t = np.zeros(N_UE, dtype=float)
            base_assignments_by_sat: Dict[int, np.ndarray] = {}
            rm_assignments_by_sat: Dict[int, np.ndarray] = {}

            for si in cand:
                ue_idx = np.flatnonzero(state.serving == si)
                if ue_idx.size == 0:
                    continue

                cap = state.cap_per_sat[si]
                snr_lin = state.snr_lin_per_sat[si]
                snr_wb = state.snr_wb_per_sat[si]

                # UE mask for this satellite
                mask = np.zeros(N_UE, dtype=bool)
                mask[ue_idx] = True
                ue_mask = mask[None, :]

                # Baseline metric
                se_base_prb = snr_to_se_sched(
                    snr_lin, config.get("use_mcs", False), mcs_params,
                    enable_cqi_quant=True,
                    cqi_table=config.get("csi_mcs_table", "nr_256qam")
                )

                # Get or create HARQ managers
                harq_base = state.harq_base_by_sat.get(si)
                harq_rm = state.harq_rm_by_sat.get(si)

                if harq_base is None and bool(config.get("enable_harq_full", False)):
                    harq_base = HarqManagerFull(
                        num_ue=N_UE,
                        num_procs=int(config.get("harq_max_procs", 16)),
                        ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                        config=config,
                    )
                    state.harq_base_by_sat[si] = harq_base

                if harq_rm is None and bool(config.get("enable_harq_full", False)):
                    harq_rm = HarqManagerFull(
                        num_ue=N_UE,
                        num_procs=int(config.get("harq_max_procs", 16)),
                        ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                        config=config,
                    )
                    state.harq_rm_by_sat[si] = harq_rm

                # Scheduler calls
                _cfg = dict(config)
                _cfg['harq_flush_tail'] = False
                # Avoid nested tqdm bars for per-satellite per-TTI calls
                _cfg['show_progress'] = False

                # Local trace outputs (1 TTI)
                _a_base: Optional[list] = [] if trace_enabled else None
                _t_base: Optional[list] = [] if trace_enabled else None
                _ack_base: Optional[list] = [] if trace_enabled else None
                _a_rm: Optional[list] = [] if trace_enabled else None
                _t_rm: Optional[list] = [] if trace_enabled else None
                _ack_rm: Optional[list] = [] if trace_enabled else None

                base = pf_schedule_radiomap_blocks(
                    cap, 1, beta=config["pf_beta"],
                    snr_lin=snr_lin,
                    overhead_eff=config.get("overhead_eff", 1.0),
                    use_mcs=config.get("use_mcs", False),
                    power_split=config.get("power_split", False),
                    se_metric_override=None,
                    max_prbs_per_ue=prb_cap_unified,
                    mcs_params=mcs_params,
                    se_metric_time=se_base_prb[None, ...],
                    snr_lin_time=None,
                    eesm_beta_db=float(config.get("baseline_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
                    require_contiguous=bool(config.get("sched_require_contiguous", True)),
                    rng=self._rng,
                    ue_mask_time=ue_mask,
                    harq_mgr=harq_base,
                    dl_power_model=dlpm,
                    P_tot_dbm=Ptot,
                    P_ref_dbm=config.get("P_tx_dbm"),
                    p_min_dbm=pmin,
                    p_max_dbm=pmax,
                    record_assignments=trace_enabled,
                    assignments_out=_a_base,
                    record_ue_thr=trace_enabled,
                    ue_thr_out=_t_base,
                    record_ue_ack_thr=trace_enabled,
                    ue_ack_thr_out=_ack_base,
                    config=_cfg,
                )

                rm = pf_schedule_radiomap_blocks(
                    cap, 1, beta=config["pf_beta"],
                    snr_lin=snr_lin,
                    overhead_eff=config.get("overhead_eff", 1.0),
                    use_mcs=config.get("use_mcs", False),
                    power_split=config.get("power_split", False),
                    se_metric_override=None,
                    max_prbs_per_ue=prb_cap_unified,
                    mcs_params=mcs_params,
                    se_metric_time=None,
                    snr_lin_time=None,
                    eesm_beta_db=float(config.get("rm_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
                    require_contiguous=bool(config.get("sched_require_contiguous", True)),
                    rng=self._rng,
                    ue_mask_time=ue_mask,
                    harq_mgr=harq_rm,
                    dl_power_model=dlpm,
                    P_tot_dbm=Ptot,
                    P_ref_dbm=config.get("P_tx_dbm"),
                    p_min_dbm=pmin,
                    p_max_dbm=pmax,
                    record_assignments=trace_enabled,
                    assignments_out=_a_rm,
                    record_ue_thr=trace_enabled,
                    ue_thr_out=_t_rm,
                    record_ue_ack_thr=trace_enabled,
                    ue_ack_thr_out=_ack_rm,
                    config=_cfg,
                )

                state.sum_rate_base_def += base * Z
                state.sum_rate_rm += rm * Z

                if trace_enabled:
                    try:
                        if _t_base and len(_t_base) > 0:
                            base_thr_sched_t += np.asarray(_t_base[0], dtype=float)
                        if _ack_base and len(_ack_base) > 0:
                            base_thr_ack_t += np.asarray(_ack_base[0], dtype=float)
                        if _t_rm and len(_t_rm) > 0:
                            rm_thr_sched_t += np.asarray(_t_rm[0], dtype=float)
                        if _ack_rm and len(_ack_rm) > 0:
                            rm_thr_ack_t += np.asarray(_ack_rm[0], dtype=float)
                        if _a_base and len(_a_base) > 0:
                            base_assignments_by_sat[int(si)] = np.asarray(_a_base[0], dtype=int)
                        if _a_rm and len(_a_rm) > 0:
                            rm_assignments_by_sat[int(si)] = np.asarray(_a_rm[0], dtype=int)
                    except Exception:
                        pass

                # Update per-satellite KPI
                k = state.kpi_per_sat.get(si)
                if k is None:
                    k = {
                        "name": getattr(state.orbit.sats[si], 'name', f"SAT-{int(si)}"),
                        "ttis_active": 0,
                        "served_ue_sum": 0,
                        "served_ue_max": 0,
                        "sum_se_base_def": 0.0,
                        "sum_se_rm": 0.0,
                    }
                    state.kpi_per_sat[si] = k

                k["ttis_active"] += 1
                k["served_ue_sum"] += int(ue_idx.size)
                k["served_ue_max"] = max(int(k["served_ue_max"]), int(ue_idx.size))
                k["sum_se_base_def"] += float(base * Z)
                k["sum_se_rm"] += float(rm * Z)

            if trace_enabled and self._trace is not None:
                self._trace["active_ues"][t_idx] = int(np.sum(state.serving >= 0))
                self._trace["ue_thr_sched_base"][t_idx] = base_thr_sched_t
                self._trace["ue_thr_ack_base"][t_idx] = base_thr_ack_t
                self._trace["ue_thr_sched_rm"][t_idx] = rm_thr_sched_t
                self._trace["ue_thr_ack_rm"][t_idx] = rm_thr_ack_t
                self._trace["se_sched_base_per_prb"][t_idx] = float(np.sum(base_thr_sched_t) / max(1, Z))
                self._trace["se_ack_base_per_prb"][t_idx] = float(np.sum(base_thr_ack_t) / max(1, Z))
                self._trace["se_sched_rm_per_prb"][t_idx] = float(np.sum(rm_thr_sched_t) / max(1, Z))
                self._trace["se_ack_rm_per_prb"][t_idx] = float(np.sum(rm_thr_ack_t) / max(1, Z))
                self._trace["assignments_base_by_tti"].append(base_assignments_by_sat)
                self._trace["assignments_rm_by_tti"].append(rm_assignments_by_sat)

            # Emit TTI metrics
            self._emit_tti_metrics(t_idx, T)

    def _update_association(self, cand: List[int], metric_kind: str, min_elev: float,
                            ho_enabled: bool, ho_hyst_db: float, ho_ttt: int, t_idx: int) -> None:
        """Update UE-satellite association with handover logic."""
        state = self._state
        N_UE = state.N_UE

        # Find best satellite per UE
        best_sat = np.full(N_UE, -1, dtype=int)
        best_metric_db = np.full(N_UE, -1e9, dtype=float)

        for si in cand:
            elev = state.elev_per_sat[si]
            vis_mask = elev >= min_elev

            if metric_kind == 'snr_wb':
                met = state.snr_wb_per_sat[si]
                met_db = 10.0 * np.log10(np.maximum(1e-12, met))
            elif metric_kind == 'prx_dbm':
                met_db = state.prx_dbm_per_sat[si]
            else:
                met = state.snr_wb_per_sat[si]
                met_db = 10.0 * np.log10(np.maximum(1e-12, met))

            met_db = np.where(vis_mask, met_db, -1e9)
            take = met_db > best_metric_db
            best_metric_db = np.where(take, met_db, best_metric_db)
            best_sat = np.where(take, si, best_sat)

        # Update serving with HO logic
        for ue in range(N_UE):
            s_old = int(state.serving[ue])
            met_old = float(state.curr_metric_db[ue])
            b = int(best_sat[ue])

            if b < 0:
                # No visible satellite
                state.serving[ue] = -1
                state.ho_timer[ue] = 0
                state.curr_metric_db[ue] = -1e9
                if s_old >= 0:
                    state.ho_events[ue].append({
                        "t": int(t_idx), "type": "outage_start",
                        "from": int(s_old), "to": -1,
                        "prev_metric_db": met_old,
                    })
                continue

            if state.serving[ue] < 0:
                # Initial attachment
                state.serving[ue] = b
                state.curr_metric_db[ue] = best_metric_db[ue]
                state.ho_timer[ue] = 0
                state.ho_events[ue].append({
                    "t": int(t_idx), "type": "attach",
                    "from": -1, "to": int(b),
                    "metric_db": float(best_metric_db[ue]),
                })
                continue

            if b == state.serving[ue]:
                # Same serving satellite
                state.curr_metric_db[ue] = best_metric_db[ue]
                state.ho_timer[ue] = 0
                continue

            # Different candidate - apply HO policy
            if not ho_enabled:
                state.serving[ue] = b
                state.curr_metric_db[ue] = best_metric_db[ue]
                state.ho_timer[ue] = 0
                state.ho_events[ue].append({
                    "t": int(t_idx), "type": "handover",
                    "from": int(s_old), "to": int(b),
                    "prev_metric_db": met_old,
                    "metric_db": float(best_metric_db[ue]),
                })
                continue

            diff_db = best_metric_db[ue] - state.curr_metric_db[ue]
            if diff_db > ho_hyst_db:
                state.ho_timer[ue] += 1
                if state.ho_timer[ue] >= ho_ttt:
                    state.serving[ue] = b
                    state.curr_metric_db[ue] = best_metric_db[ue]
                    state.ho_timer[ue] = 0
                    state.ho_events[ue].append({
                        "t": int(t_idx), "type": "handover",
                        "from": int(s_old), "to": int(b),
                        "prev_metric_db": met_old,
                        "metric_db": float(best_metric_db[ue]),
                        "diff_db": float(diff_db),
                        "hyst_db": float(ho_hyst_db),
                        "ttt_ttis": int(ho_ttt),
                    })
            else:
                state.ho_timer[ue] = 0

    def _emit_tti_metrics(self, t_idx: int, T: int) -> None:
        """Emit progress metrics for current TTI."""
        state = self._state
        Z = state.Z

        # Compute current averages
        avg_base = state.sum_rate_base_def / max(1, t_idx + 1) / max(1, Z)
        avg_rm = state.sum_rate_rm / max(1, t_idx + 1) / max(1, Z)

        metrics = TTIMetrics(
            tti=t_idx,
            progress=(t_idx + 1) / T,
            cumulative_se_baseline=avg_base,
            cumulative_se_radiomap=avg_rm,
            active_ues=int(np.sum(state.serving >= 0)),
        )
        self._callbacks.notify_tti(metrics)

    def _finalize(self) -> Dict[str, Any]:
        """Finalize simulation and build result dict."""
        import os
        import json
        from .trace import SimulationTrace, SchedulerTrace

        config = self.config
        state = self._state
        T = state.T
        Z = state.Z
        N_UE = state.N_UE

        # Average SE
        avg_se_base_def = state.sum_rate_base_def / max(1, T) / max(1, Z)
        avg_se_rm = state.sum_rate_rm / max(1, T) / max(1, Z)
        imp_pct = (avg_se_rm - avg_se_base_def) / max(1e-9, avg_se_base_def) * 100.0

        # Per-satellite summary
        per_sat_summary = []
        for si, k in sorted(state.kpi_per_sat.items(), key=lambda x: x[0]):
            tt = int(k["ttis_active"]) or 1
            per_sat_summary.append({
                "sat_index": int(si),
                "name": str(k["name"]),
                "ttis_active": int(k["ttis_active"]),
                "avg_served_ue": float(k["served_ue_sum"]) / float(tt),
                "max_served_ue": int(k["served_ue_max"]),
                "avg_se_base_default_per_prb": float(k["sum_se_base_def"]) / float(tt * max(1, Z)),
                "avg_se_rm_per_prb": float(k["sum_se_rm"]) / float(tt * max(1, Z)),
            })

        # Aggregate HARQ stats
        harq_stats_base = self._aggregate_harq_stats(state.harq_base_by_sat)
        harq_stats_map = self._aggregate_harq_stats(state.harq_rm_by_sat)

        report = {
            "avg_se_baseline_default": avg_se_base_def,
            "avg_se_radiomap": avg_se_rm,
            "improvement_vs_default_pct": imp_pct,
            "R_xyz_dbm": state.R_xyz_dbm,
            "ue_pos": state.ue_pos,
            "T": int(T),
            "Z": int(Z),
            "N_UE": int(N_UE),
            "prb_bw_hz": float(state.prb_bw_hz),
            "system_bandwidth_hz": float(state.prb_bw_hz) * float(Z),
            "total_throughput_baseline_bps": float(avg_se_base_def * state.prb_bw_hz * Z),
            "total_throughput_radiomap_bps": float(avg_se_rm * state.prb_bw_hz * Z),
            "avg_ue_throughput_baseline_bps": float((avg_se_base_def * state.prb_bw_hz * Z) / max(1, N_UE)),
            "avg_ue_throughput_radiomap_bps": float((avg_se_rm * state.prb_bw_hz * Z) / max(1, N_UE)),
            "ho_events_per_ue": state.ho_events,
            "handover_count_per_ue": [sum(1 for e in state.ho_events[i] if e.get("type") == "handover") for i in range(N_UE)],
            "outage_ttis_per_ue": state.outage_ttis.tolist(),
            "per_sat_kpis": per_sat_summary,
            "sat_index_to_name": {int(i): getattr(state.orbit.sats[i], 'name', f"SAT-{int(i)}") for i in range(len(state.orbit.sats))},
            "harq_stats_base": harq_stats_base,
            "harq_stats_map": harq_stats_map,
        }

        if state.serving_trace is not None:
            report["serving_trace"] = np.stack(state.serving_trace, axis=0)

        trace_enabled = bool(config.get("enable_trace", False)) or str(config.get("trace_level", "")).lower() in ("ui", "full", "kpi")
        if trace_enabled and self._trace is not None:
            baseline_trace = SchedulerTrace(
                assignments=None,
                ue_thr_scheduled=np.asarray(self._trace.get("ue_thr_sched_base"), dtype=float),
                ue_thr_acked=np.asarray(self._trace.get("ue_thr_ack_base"), dtype=float),
                sum_se_scheduled_per_prb=np.asarray(self._trace.get("se_sched_base_per_prb"), dtype=float),
                sum_se_acked_per_prb=np.asarray(self._trace.get("se_ack_base_per_prb"), dtype=float),
            )
            radiomap_trace = SchedulerTrace(
                assignments=None,
                ue_thr_scheduled=np.asarray(self._trace.get("ue_thr_sched_rm"), dtype=float),
                ue_thr_acked=np.asarray(self._trace.get("ue_thr_ack_rm"), dtype=float),
                sum_se_scheduled_per_prb=np.asarray(self._trace.get("se_sched_rm_per_prb"), dtype=float),
                sum_se_acked_per_prb=np.asarray(self._trace.get("se_ack_rm_per_prb"), dtype=float),
            )
            trace = SimulationTrace(
                mode="constellation",
                tti=np.asarray(self._trace.get("tti"), dtype=int),
                baseline=baseline_trace,
                radiomap=radiomap_trace,
                oals=None,
                extra={
                    "N_UE": int(N_UE),
                    "Z": int(Z),
                    "T": int(T),
                    "prb_bw_hz": float(state.prb_bw_hz),
                    "system_bandwidth_hz": float(state.prb_bw_hz) * float(Z),
                    "candidate_sats": list(self._trace.get("candidate_sats", [])),
                    "assignments_base_by_tti": list(self._trace.get("assignments_base_by_tti", [])),
                    "assignments_rm_by_tti": list(self._trace.get("assignments_rm_by_tti", [])),
                    "serving_trace": np.asarray(self._trace.get("serving_trace"), dtype=int),
                },
            )
            report["trace"] = trace.to_dict()

        # Write JSON report
        try:
            if bool(config.get("write_json_report", False)):
                out_dir = config.get("plot_dir", "output")
                os.makedirs(out_dir, exist_ok=True)
                name = str(config.get("report_basename", "constellation_summary"))
                path = os.path.join(out_dir, f"{name}.json")

                def serialize(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    raise TypeError

                with open(path, 'w') as f:
                    json.dump(report, f, default=serialize)
        except Exception as e:
            logger.warning(f"Constellation JSON report failed: {e}")

        logger.debug(
            "run_constellation complete: baseline=%.4f, radiomap=%.4f, improvement=%+.2f%%",
            float(avg_se_base_def),
            float(avg_se_rm),
            float(imp_pct),
        )

        return report

    def _aggregate_harq_stats(self, harq_dict: Dict[int, Any]) -> Optional[Dict]:
        """Aggregate HARQ statistics across satellites."""
        if not harq_dict:
            return None

        total_started = 0
        total_acked = 0
        total_dropped = 0
        total_init_ack = 0
        total_retx_weighted = 0.0

        for si, mgr in harq_dict.items():
            if mgr is None or not hasattr(mgr, 'get_stats'):
                continue
            hs = mgr.get_stats()
            tb_started = int(hs.get('tb_started', hs.get('initial_ack_count', 0) + hs.get('initial_nack_count', 0)))
            tb_acked = int(hs.get('tb_acked', hs.get('ack_count', 0)))
            tb_dropped = int(hs.get('tb_dropped', 0))
            init_ack = int(hs.get('initial_ack_count', 0))
            avg_retx = float(hs.get('avg_retx_per_acked', 0.0))

            total_started += tb_started
            total_acked += tb_acked
            total_dropped += tb_dropped
            total_init_ack += init_ack
            total_retx_weighted += avg_retx * max(0, tb_acked)

        if total_started <= 0:
            return None

        return {
            'tb_started': int(total_started),
            'tb_acked': int(total_acked),
            'tb_dropped': int(total_dropped),
            'ack_rate': float(total_acked / total_started),
            'first_try_ack_rate': float(total_init_ack / total_started),
            'avg_retx_per_acked': float(total_retx_weighted / total_acked) if total_acked > 0 else 0.0,
        }

    def _notify_phase(self, phase: SimulationPhase, message: str = "") -> None:
        """Notify callbacks of phase change."""
        prev_phase = None
        if self._state is not None:
            prev_phase = SimulationPhase(self._state.phase) if self._state.phase in [p.value for p in SimulationPhase] else None
            self._state.phase = phase.value

        event = PhaseEvent(phase=phase, previous_phase=prev_phase, message=message)
        self._callbacks.notify_phase(event)


def run_constellation(config: Dict) -> Dict:
    """Legacy-compatible wrapper for ConstellationEngine.

    Args:
        config: Simulation configuration dict

    Returns:
        Result dict with constellation simulation outputs
    """
    engine = ConstellationEngine(config)
    return engine.run()
