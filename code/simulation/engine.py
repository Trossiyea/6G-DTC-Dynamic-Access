# -*- coding: utf-8 -*-
"""
Single-Satellite Simulation Engine.

Encapsulates the run_once() logic into a class-based engine with:
- Explicit state management
- Progress callbacks for Web integration
- Testable, modular design
"""

import logging
from typing import Dict, Optional, List, Any, Callable
import numpy as np

from .state import SimulationState, GeometryResult, SchedulerResult, TimeSeriesData
from .callbacks import (
    CallbackManager, ProgressCallback, TTIMetrics, PhaseEvent, SimulationPhase
)
from .helpers import (
    generate_ue_positions,
    resolve_noise_and_prb_bw,
    apply_open_loop_power_control,
    compute_caps,
    compute_metric_override_static_if_needed,
    build_time_variation_if_enabled,
    delay_series,
    hold_series,
    compute_jain_fairness,
)

# Lazy imports to avoid circular dependencies
from data_io.radiomap import select_radio_map
from orbit import compute_geometry_and_beam, OrbitModel
from logging_utils import get_logger

logger = get_logger(__name__)


class SimulationEngine:
    """Single-satellite downlink simulation engine.

    Encapsulates the `run_once()` logic with explicit state management,
    progress callbacks, and Web-friendly design.

    Example:
        >>> engine = SimulationEngine(config)
        >>> engine.on_progress(lambda tti, m: print(f"TTI {tti}"))
        >>> result = engine.run()

    For Web integration:
        >>> engine = SimulationEngine(config)
        >>> engine.add_callback(websocket_callback)
        >>> result = engine.run()
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize engine with configuration.

        Args:
            config: Simulation configuration dict
        """
        self.config = config
        self._callbacks = CallbackManager()
        self._state: Optional[SimulationState] = None
        self._rng: Optional[np.random.Generator] = None

    @property
    def state(self) -> Optional[SimulationState]:
        """Access current simulation state."""
        return self._state

    def add_callback(self, callback: ProgressCallback) -> "SimulationEngine":
        """Add a progress callback. Returns self for chaining."""
        self._callbacks.add(callback)
        return self

    def on_progress(self, handler: Callable[[int, Dict], None]) -> "SimulationEngine":
        """Register a simple progress handler function.

        Args:
            handler: Function(tti, metrics_dict) called after each TTI

        Returns:
            Self for chaining
        """
        from .callbacks import FunctionCallback
        self._callbacks.add(FunctionCallback(
            on_tti=lambda m: handler(m.tti, m.to_dict())
        ))
        return self

    def initialize(self) -> SimulationState:
        """Initialize simulation state (load data, compute geometry).

        Call this to pre-initialize before run(), or let run() call it.

        Returns:
            Initialized SimulationState
        """
        self._notify_phase(SimulationPhase.INITIALIZING, "Loading Radio Map and initializing UEs")

        config = self.config
        seed = config["seed"]
        self._rng = np.random.default_rng(seed)

        N_UE = config["N_UE"]
        T = config["T"]

        # Load Radio Map
        R_xyz_dbm, X, Y, Z = select_radio_map(config)

        # Generate UE positions
        ue_pos = generate_ue_positions(N_UE, X, Y, self._rng)

        # Compute geometry and beam
        L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue = compute_geometry_and_beam(config, X, Y, ue_pos)

        # Resolve noise and bandwidth
        noise_dbm, prb_bw_hz = resolve_noise_and_prb_bw(config)

        # Apply power control (DL simplification)
        P_tx_per_ue_dbm = apply_open_loop_power_control(config, L_fs_per_ue, G_rx_per_ue)

        # Compute static SINR/capacity snapshot
        cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb = compute_caps(
            R_xyz_dbm, ue_pos,
            P_tx_dbm=P_tx_per_ue_dbm,
            L_fs_db=L_fs_per_ue,
            G_rx_db=G_rx_per_ue,
            shadow_db_std=config["shadow_std_db"],
            N0_dbm=noise_dbm,
            rx_nf_db=config.get("rx_nf_db", 0.0),
            impl_loss_db=config.get("impl_loss_db", 0.0),
            seed=seed,
            elevation_deg=elev_deg_per_ue,
            channel_model=config.get("channel_model", "3gpp_ntn"),
            channel_params=config.get("channel_params"),
            channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
        )

        # Optional orbit model for dynamics
        orbit_model = None
        if config.get("enable_orbit_dynamics", False):
            orbit_model = OrbitModel(config, X, Y)

        # Create state object
        self._state = SimulationState(
            config=config,
            seed=seed,
            X=X, Y=Y, Z=Z,
            N_UE=N_UE, T=T,
            R_xyz_dbm=R_xyz_dbm,
            ue_pos=ue_pos,
            geometry=GeometryResult(
                L_fs_db=np.atleast_1d(L_fs_per_ue),
                G_rx_db=np.atleast_1d(G_rx_per_ue),
                elev_deg=np.atleast_1d(elev_deg_per_ue) if elev_deg_per_ue is not None else None,
            ),
            cap=cap,
            cap_wb=cap_wb,
            snr_lin=snr_lin,
            snr_lin_wb=snr_lin_wb,
            P_rx_dbm=P_rx_dbm,
            I_total_dbm=I_total_dbm,
            noise_dbm=noise_dbm,
            prb_bw_hz=prb_bw_hz,
            P_tx_per_ue_dbm=P_tx_per_ue_dbm,
            orbit_model=orbit_model,
            phase="initialized",
        )

        # Compute metric override if estimation error/blur configured
        self._state.metric_override = compute_metric_override_static_if_needed(
            config, R_xyz_dbm, ue_pos,
            L_fs_per_ue, G_rx_per_ue, noise_dbm, elev_deg_per_ue
        )

        # Build time series if time-varying enabled
        time_series_dict = build_time_variation_if_enabled(
            config, R_xyz_dbm, ue_pos,
            L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue,
            P_tx_per_ue_dbm, noise_dbm, self._rng,
            None if config.get("enable_time_varying", False) else self._state.metric_override,
            orbit_model=orbit_model,
        )
        if time_series_dict is not None:
            self._state.time_series = TimeSeriesData(
                se_time_rm=time_series_dict["se_time_rm"],
                se_time_wb=time_series_dict["se_time_wb"],
                snr_time=time_series_dict["snr_time"],
                snr_wb_time=time_series_dict["snr_wb_time"],
                tau_time=time_series_dict.get("tau_time"),
                fd_time=time_series_dict.get("fd_time"),
            )

        self._notify_phase(SimulationPhase.INITIALIZED, "Initialization complete")
        return self._state

    def run(self) -> Dict[str, Any]:
        """Execute the complete simulation.

        Returns:
            Result dict compatible with legacy run_once() output
        """
        if self._state is None or not self._state.is_initialized():
            self.initialize()

        self._notify_phase(SimulationPhase.RUNNING, "Starting scheduler execution")

        # Register MCS tables
        self._register_mcs_tables()

        # Run schedulers
        result = self._run_schedulers()

        self._notify_phase(SimulationPhase.FINALIZING, "Computing final metrics")

        # Finalize and return
        final_result = self._finalize(result)

        self._state.phase = "complete"
        self._notify_phase(SimulationPhase.COMPLETE, "Simulation complete")

        return final_result

    def _register_mcs_tables(self) -> None:
        """Register 3GPP MCS tables from file if configured."""
        from link_adapt import register_mcs_tables_from_file, register_bler_curves_from_file

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

    def _run_schedulers(self) -> Dict[str, Any]:
        """Execute baseline and RadioMap schedulers."""
        from scheduler.radiomap import pf_schedule_radiomap_blocks
        from harq import HarqManager, HarqManagerFull
        from ntn_csi import snr_to_se_sched

        config = self.config
        state = self._state
        T = config["T"]
        N_UE = config["N_UE"]

        mcs_params = {
            "olla_offset_db": config.get("csi_olla_offset_db", 0.0),
            "mcs_table": config.get("csi_mcs_table", "legacy"),
            "residual_freq_hz": config.get("residual_freq_hz", 0.0),
            "scs_khz": config.get("scs_khz", 30),
        }

        # Result containers
        result = {
            "harq_stats_base": None,
            "harq_stats_map": None,
            "assignments_base": None,
            "assignments_rm": None,
            "ue_thr_base": None,
            "ue_thr_rm": None,
        }

        if state.time_series is not None:
            # Time-varying mode
            result.update(self._run_time_varying_schedulers(mcs_params))
        else:
            # Static mode
            result.update(self._run_static_schedulers(mcs_params))

        return result

    def _run_static_schedulers(self, mcs_params: Dict) -> Dict[str, Any]:
        """Run schedulers in static (non-time-varying) mode."""
        from scheduler.radiomap import pf_schedule_radiomap_blocks

        config = self.config
        state = self._state
        T = config["T"]

        # Recording options
        _rec_rm = bool(config.get("record_assignments", False))
        _rec_base = _rec_rm and str(config.get("record_assignments_target", "rm")).lower() in ("base", "both", "all")
        assignments_base = [] if _rec_base else None
        assignments_rm = [] if _rec_rm else None

        # Baseline scheduler
        base_se_default = pf_schedule_radiomap_blocks(
            state.cap, T, beta=config["pf_beta"],
            snr_lin=state.snr_lin,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            se_metric_override=None,
            max_prbs_per_ue=config.get("baseline_max_prbs_per_ue", config.get("max_prbs_per_ue")),
            mcs_params=mcs_params,
            se_metric_time=None,
            snr_lin_time=None,
            eesm_beta_db=float(config.get("baseline_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
            require_contiguous=bool(config.get("sched_require_contiguous", True)),
            rng=self._rng,
            dl_power_model=str(config.get("baseline_dl_power_model", "equal_prb")),
            P_tot_dbm=config.get("baseline_P_tot_dbm"),
            P_ref_dbm=config.get("P_tx_dbm"),
            p_min_dbm=config.get("baseline_p_min_dbm"),
            p_max_dbm=config.get("baseline_p_max_dbm"),
            record_assignments=_rec_base,
            assignments_out=assignments_base,
            config=config,
        )

        # RadioMap scheduler
        map_se = pf_schedule_radiomap_blocks(
            state.cap, T, beta=config["pf_beta"],
            snr_lin=state.snr_lin,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            se_metric_override=state.metric_override,
            max_prbs_per_ue=config.get("rm_max_prbs_per_ue", config.get("max_prbs_per_ue")),
            mcs_params=mcs_params,
            se_metric_time=None,
            snr_lin_time=None,
            eesm_beta_db=float(config.get("rm_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
            require_contiguous=bool(config.get("sched_require_contiguous", True)),
            rng=self._rng,
            ue_mask_time=None,
            dl_power_model=str(config.get("rm_dl_power_model", "equal_prb")),
            P_tot_dbm=config.get("rm_P_tot_dbm"),
            P_ref_dbm=config.get("P_tx_dbm"),
            p_min_dbm=config.get("rm_p_min_dbm"),
            p_max_dbm=config.get("rm_p_max_dbm"),
            record_assignments=_rec_rm,
            assignments_out=assignments_rm,
            config=config,
        )

        # Store results in state
        state.result_baseline = SchedulerResult(avg_se=base_se_default)
        state.result_radiomap = SchedulerResult(avg_se=map_se)

        return {
            "base_se_default": base_se_default,
            "map_se": map_se,
            "harq_stats_base": None,
            "harq_stats_map": None,
            "assignments_base": assignments_base,
            "assignments_rm": assignments_rm,
        }

    def _run_time_varying_schedulers(self, mcs_params: Dict) -> Dict[str, Any]:
        """Run schedulers in time-varying mode with HARQ."""
        from scheduler.radiomap import pf_schedule_radiomap_blocks
        from harq import HarqManager, HarqManagerFull
        from ntn_csi import snr_to_se_sched

        config = self.config
        state = self._state
        ts = state.time_series
        T = config["T"]
        N_UE = config["N_UE"]

        # Apply CSI delays
        baseline_delay = int(config.get("baseline_csi_delay_ttis", 0))
        rm_delay = int(config.get("rm_csi_delay_ttis", 0))

        se_time_wb = delay_series(ts.se_time_wb, baseline_delay)
        se_time_rm = delay_series(ts.se_time_rm, rm_delay)

        # Baseline per-PRB metric with CQI quantization
        se_time_base = np.empty_like(ts.se_time_rm)
        for tt in range(ts.snr_time.shape[0]):
            se_time_base[tt] = snr_to_se_sched(
                ts.snr_time[tt], config.get("use_mcs", False), mcs_params,
                enable_cqi_quant=True,
                cqi_table=config.get("csi_mcs_table", "nr_256qam")
            )
        se_time_base = delay_series(se_time_base, baseline_delay)

        # Optional CQI reporting periodicity
        period = int(config.get("cqi_period_ttis", 0) or 0)
        offset = int(config.get("cqi_offset_ttis", 0) or 0)
        if period and period > 1:
            if bool(config.get("enable_cqi_periodicity_base", False)):
                se_time_wb = hold_series(se_time_wb, period, offset)
                se_time_base = hold_series(se_time_base, period, offset)
            if bool(config.get("enable_cqi_periodicity_rm", False)):
                se_time_rm = hold_series(se_time_rm, period, offset)

        # HARQ managers
        harq_mgr_base = None
        harq_mgr_map = None
        if bool(config.get("enable_harq_full", False)):
            harq_mgr_base = HarqManagerFull(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                config=config,
            )
            harq_mgr_map = HarqManagerFull(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                config=config,
            )
        elif bool(config.get("enable_harq_deferral", False)):
            harq_mgr_base = HarqManager(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
            )
            harq_mgr_map = HarqManager(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
            )

        # Recording options
        _rec_base = bool(config.get("record_assignments", False)) and \
            str(config.get("record_assignments_target", "rm")).lower() in ("base", "both", "all")
        _rec_base_thr = bool(config.get("record_ue_thr", False)) and \
            str(config.get("record_assignments_target", "rm")).lower() in ("base", "both", "all")
        _rec_rm = bool(config.get("record_assignments", False)) and \
            str(config.get("record_assignments_target", "rm")).lower() in ("rm", "both", "all")
        _rec_rm_thr = bool(config.get("record_ue_thr", False)) and \
            str(config.get("record_assignments_target", "rm")).lower() in ("rm", "both", "all")

        assignments_base = [] if _rec_base else None
        ue_thr_base = [] if _rec_base_thr else None
        assignments_rm = [] if _rec_rm else None
        ue_thr_rm = [] if _rec_rm_thr else None

        # Baseline scheduler
        base_se_default = pf_schedule_radiomap_blocks(
            state.cap, T, beta=config["pf_beta"],
            snr_lin=state.snr_lin,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            se_metric_override=None,
            max_prbs_per_ue=config.get("baseline_max_prbs_per_ue", config.get("max_prbs_per_ue")),
            mcs_params=mcs_params,
            se_metric_time=se_time_base,
            snr_lin_time=ts.snr_time,
            eesm_beta_db=float(config.get("baseline_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
            require_contiguous=bool(config.get("sched_require_contiguous", True)),
            rng=self._rng,
            ue_mask_time=None,
            harq_mgr=harq_mgr_base,
            dl_power_model=str(config.get("baseline_dl_power_model", "equal_prb")),
            P_tot_dbm=config.get("baseline_P_tot_dbm"),
            P_ref_dbm=config.get("P_tx_dbm"),
            p_min_dbm=config.get("baseline_p_min_dbm"),
            p_max_dbm=config.get("baseline_p_max_dbm"),
            record_assignments=_rec_base,
            assignments_out=assignments_base,
            record_ue_thr=_rec_base_thr,
            ue_thr_out=ue_thr_base,
            config=config,
        )

        # RadioMap scheduler
        map_se = pf_schedule_radiomap_blocks(
            state.cap, T, beta=config["pf_beta"],
            snr_lin=state.snr_lin,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            se_metric_override=None if state.metric_override is None else state.metric_override,
            max_prbs_per_ue=config.get("rm_max_prbs_per_ue", config.get("max_prbs_per_ue")),
            mcs_params=mcs_params,
            se_metric_time=se_time_rm,
            snr_lin_time=ts.snr_time,
            eesm_beta_db=float(config.get("rm_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
            require_contiguous=bool(config.get("sched_require_contiguous", True)),
            rng=self._rng,
            ue_mask_time=None,
            harq_mgr=harq_mgr_map,
            dl_power_model=str(config.get("rm_dl_power_model", "equal_prb")),
            P_tot_dbm=config.get("rm_P_tot_dbm"),
            P_ref_dbm=config.get("P_tx_dbm"),
            p_min_dbm=config.get("rm_p_min_dbm"),
            p_max_dbm=config.get("rm_p_max_dbm"),
            record_assignments=_rec_rm,
            assignments_out=assignments_rm,
            record_ue_thr=_rec_rm_thr,
            ue_thr_out=ue_thr_rm,
            config=config,
        )

        # Collect HARQ stats
        harq_stats_base = None
        harq_stats_map = None
        if harq_mgr_base is not None and hasattr(harq_mgr_base, 'get_stats'):
            try:
                harq_stats_base = harq_mgr_base.get_stats()
            except Exception:
                pass
        if harq_mgr_map is not None and hasattr(harq_mgr_map, 'get_stats'):
            try:
                harq_stats_map = harq_mgr_map.get_stats()
            except Exception:
                pass

        # Store results in state
        state.result_baseline = SchedulerResult(
            avg_se=base_se_default,
            harq_stats=harq_stats_base
        )
        state.result_radiomap = SchedulerResult(
            avg_se=map_se,
            harq_stats=harq_stats_map
        )

        return {
            "base_se_default": base_se_default,
            "map_se": map_se,
            "harq_stats_base": harq_stats_base,
            "harq_stats_map": harq_stats_map,
            "assignments_base": assignments_base,
            "assignments_rm": assignments_rm,
            "ue_thr_base": ue_thr_base,
            "ue_thr_rm": ue_thr_rm,
        }

    def _finalize(self, sched_result: Dict) -> Dict[str, Any]:
        """Finalize simulation and build result dict."""
        from link_adapt import re_per_prb_from_config

        config = self.config
        state = self._state

        base_se_default = sched_result["base_se_default"]
        map_se = sched_result["map_se"]

        # System bandwidth
        try:
            sys_bw_hz = float(state.prb_bw_hz) * float(state.Z)
        except Exception:
            sys_bw_hz = float(config.get("scs_khz", 30.0)) * 1e3 * 12.0 * float(state.Z)

        # Build result dict
        report = {
            "avg_se_baseline_default": base_se_default,
            "avg_se_radiomap": map_se,
            "improvement_vs_default_pct": (map_se - base_se_default) / max(1e-9, base_se_default) * 100.0,
            "R_xyz_dbm": state.R_xyz_dbm,
            "ue_pos": state.ue_pos,
            "cap": state.cap,
            "cap_wb": state.cap_wb,
            "snr_lin": state.snr_lin,
            "snr_lin_wb": state.snr_lin_wb,
            "prb_bw_hz": float(state.prb_bw_hz),
            "system_bandwidth_hz": float(sys_bw_hz),
            "total_throughput_baseline_bps": float(base_se_default * sys_bw_hz),
            "total_throughput_radiomap_bps": float(map_se * sys_bw_hz),
            "harq_stats_base": sched_result.get("harq_stats_base"),
            "harq_stats_map": sched_result.get("harq_stats_map"),
        }

        # Time series data
        if state.time_series is not None:
            report["tau_time"] = state.time_series.tau_time
            report["fd_time"] = state.time_series.fd_time

        # Assignments and per-UE throughput
        self._attach_assignment_data(report, sched_result)

        # Per-UE metrics
        self._compute_per_ue_metrics(report, sched_result, sys_bw_hz)

        # Write JSON report if configured
        self._write_json_report(report)

        logger.debug(
            "run complete: baseline=%.4f, radiomap=%.4f, improvement=%+.2f%%",
            float(report.get("avg_se_baseline_default", float("nan"))),
            float(report.get("avg_se_radiomap", float("nan"))),
            float(report.get("improvement_vs_default_pct", float("nan"))),
        )

        return report

    def _attach_assignment_data(self, report: Dict, sched_result: Dict) -> None:
        """Attach PRB assignments and per-UE throughput to report."""
        config = self.config
        state = self._state
        T = config["T"]

        try:
            assignments_rm = sched_result.get("assignments_rm")
            if assignments_rm is not None and len(assignments_rm) > 0:
                report["assignments_rm"] = np.stack(assignments_rm, axis=0)

            assignments_base = sched_result.get("assignments_base")
            if assignments_base is not None and len(assignments_base) > 0:
                report["assignments_base"] = np.stack(assignments_base, axis=0)

            ue_thr_rm = sched_result.get("ue_thr_rm")
            if ue_thr_rm is not None and len(ue_thr_rm) > 0:
                thr_mat = np.stack(ue_thr_rm, axis=0)
                report["ue_thr_time_rm"] = thr_mat
                report["per_ue_se_rm_avg"] = (np.sum(thr_mat, axis=0) / float(max(1, T) * state.Z)).tolist()

            ue_thr_base = sched_result.get("ue_thr_base")
            if ue_thr_base is not None and len(ue_thr_base) > 0:
                thr_mat_b = np.stack(ue_thr_base, axis=0)
                report["ue_thr_time_base"] = thr_mat_b
                report["per_ue_se_base_avg"] = (np.sum(thr_mat_b, axis=0) / float(max(1, T) * state.Z)).tolist()
        except Exception:
            pass

    def _compute_per_ue_metrics(self, report: Dict, sched_result: Dict, sys_bw_hz: float) -> None:
        """Compute per-UE throughput and fairness metrics."""
        from link_adapt import re_per_prb_from_config

        config = self.config
        state = self._state
        T = config["T"]
        N_UE = config["N_UE"]

        harq_stats_base = sched_result.get("harq_stats_base")
        harq_stats_map = sched_result.get("harq_stats_map")
        base_se_default = report["avg_se_baseline_default"]
        map_se = report["avg_se_radiomap"]

        try:
            re_per_prb = re_per_prb_from_config(config)
            T_total = int(T)
            Z_total = state.Z

            def per_ue_avg_se(hs):
                if not hs or not isinstance(hs, dict) or 'acked_bits_per_ue' not in hs:
                    return None
                bits = np.asarray(hs['acked_bits_per_ue'], dtype=float)
                return (bits / float(max(1, re_per_prb) * T_total * Z_total)).tolist()

            se_ue_base = per_ue_avg_se(harq_stats_base)
            se_ue_map = per_ue_avg_se(harq_stats_map)

            report["per_ue_avg_se_base"] = se_ue_base
            report["per_ue_avg_se_map"] = se_ue_map

            # Per-UE throughput
            if se_ue_base is not None:
                report["per_ue_throughput_baseline_bps"] = (np.asarray(se_ue_base, dtype=float) * sys_bw_hz).tolist()
                report["avg_ue_throughput_baseline_bps"] = float(np.mean(report["per_ue_throughput_baseline_bps"]))
            else:
                N_UE_eff = max(1, N_UE)
                report["avg_ue_throughput_baseline_bps"] = float((base_se_default * sys_bw_hz) / N_UE_eff)
                if 'per_ue_se_base_avg' in report:
                    p = np.asarray(report['per_ue_se_base_avg'], dtype=float) * sys_bw_hz
                    report["per_ue_throughput_baseline_bps"] = p.tolist()
                    report["avg_ue_throughput_baseline_bps"] = float(np.mean(p))

            if se_ue_map is not None:
                report["per_ue_throughput_radiomap_bps"] = (np.asarray(se_ue_map, dtype=float) * sys_bw_hz).tolist()
                report["avg_ue_throughput_radiomap_bps"] = float(np.mean(report["per_ue_throughput_radiomap_bps"]))
            else:
                N_UE_eff = max(1, N_UE)
                report["avg_ue_throughput_radiomap_bps"] = float((map_se * sys_bw_hz) / N_UE_eff)
                if 'per_ue_se_rm_avg' in report:
                    p = np.asarray(report['per_ue_se_rm_avg'], dtype=float) * sys_bw_hz
                    report["per_ue_throughput_radiomap_bps"] = p.tolist()
                    report["avg_ue_throughput_radiomap_bps"] = float(np.mean(p))

            # Fairness
            report["fairness_jain_base"] = compute_jain_fairness(se_ue_base)
            report["fairness_jain_map"] = compute_jain_fairness(se_ue_map)
        except Exception:
            pass

    def _write_json_report(self, report: Dict) -> None:
        """Write JSON report to output directory if configured."""
        import os
        import json

        config = self.config
        try:
            if bool(config.get("write_json_report", False)):
                out_dir = config.get("plot_dir", "output")
                os.makedirs(out_dir, exist_ok=True)
                name = config.get("report_basename", "summary")
                path = os.path.join(out_dir, f"{name}.json")

                def serialize(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    raise TypeError

                with open(path, 'w') as f:
                    json.dump(report, f, default=serialize)
        except Exception as e:
            logger.warning(f"Failed to write JSON report: {e}")

    def _notify_phase(self, phase: SimulationPhase, message: str = "") -> None:
        """Notify callbacks of phase change."""
        prev_phase = None
        if self._state is not None:
            prev_phase = SimulationPhase(self._state.phase) if self._state.phase in [p.value for p in SimulationPhase] else None
            self._state.phase = phase.value

        event = PhaseEvent(
            phase=phase,
            previous_phase=prev_phase,
            message=message,
        )
        self._callbacks.notify_phase(event)


def run_once(config: Dict) -> Dict:
    """Legacy-compatible wrapper for SimulationEngine.

    This function provides backward compatibility with the original
    run_once() interface.

    Args:
        config: Simulation configuration dict

    Returns:
        Result dict with simulation outputs
    """
    engine = SimulationEngine(config)
    return engine.run()
