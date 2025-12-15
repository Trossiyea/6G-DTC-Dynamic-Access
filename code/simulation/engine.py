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

# Optional QoS integration
try:
    from scheduler.integration import (
        compute_mlwdf_metric,
        compute_exppf_metric,
        QoSSchedulerType,
    )
    QOS_AVAILABLE = True
except ImportError:
    QOS_AVAILABLE = False

# Optional OALS integration (Phase 11 - Patent)
try:
    from scheduler.lookahead import (
        OALSConfig,
        OALSScheduler,
        create_oals_scheduler,
    )
    OALS_AVAILABLE = True
except ImportError:
    OALS_AVAILABLE = False

# Lazy imports to avoid circular dependencies
from data_io.radiomap import select_radio_map
from ntn import compute_geometry_and_beam, OrbitModel, StaticOrbitModel
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
        elif OALS_AVAILABLE and config.get("enable_oals", False):
            orbit_model = StaticOrbitModel(config, X, Y)

        # Optional OALS scheduler (Phase 11 - Patent)
        oals_scheduler = None
        if OALS_AVAILABLE and config.get("enable_oals", False) and orbit_model is not None:
            oals_scheduler = create_oals_scheduler(
                n_ue=N_UE,
                orbit_model=orbit_model,
                config=config,
            )
            logger.info("OALS scheduler initialized (Phase 11 - Patent)")

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
            oals_scheduler=oals_scheduler,
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

        # Auto-enable per-UE throughput recording for traffic simulation
        traffic_model = self.config.get("traffic_model", "full_buffer")
        if traffic_model != "full_buffer":
            if not self.config.get("record_ue_thr", False):
                logger.debug("Auto-enabling record_ue_thr for traffic simulation")
                self.config["record_ue_thr"] = True

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

    def _run_schedulers(self) -> Dict[str, Any]:
        """Execute baseline and RadioMap schedulers."""
        from scheduler.radiomap import pf_schedule_radiomap_blocks
        from link import HarqManager, HarqManagerFull
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
        trace_enabled = bool(config.get("enable_trace", False)) or str(config.get("trace_level", "")).lower() in ("ui", "full", "kpi")
        target = str(config.get("record_assignments_target", "rm")).lower()
        if trace_enabled:
            target = "both"

        # Recording options
        _rec_base = (trace_enabled or bool(config.get("record_assignments", False))) and target in ("base", "both", "all")
        _rec_rm = (trace_enabled or bool(config.get("record_assignments", False))) and target in ("rm", "both", "all")
        assignments_base = [] if _rec_base else None
        assignments_rm = [] if _rec_rm else None
        _rec_thr = trace_enabled or bool(config.get("record_ue_thr", False))
        _rec_base_thr = _rec_thr and target in ("base", "both", "all")
        _rec_rm_thr = _rec_thr and target in ("rm", "both", "all")
        ue_thr_base = [] if _rec_base_thr else None
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
            record_ue_thr=_rec_rm_thr,
            ue_thr_out=ue_thr_rm,
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
            "ue_thr_base": ue_thr_base,
            "ue_thr_rm": ue_thr_rm,
        }

    def _run_time_varying_schedulers(self, mcs_params: Dict) -> Dict[str, Any]:
        """Run schedulers in time-varying mode with HARQ."""
        from scheduler.radiomap import pf_schedule_radiomap_blocks
        from link import HarqManager, HarqManagerFull
        from ntn_csi import snr_to_se_sched

        config = self.config
        state = self._state
        ts = state.time_series
        T = config["T"]
        N_UE = config["N_UE"]
        trace_enabled = bool(config.get("enable_trace", False)) or str(config.get("trace_level", "")).lower() in ("ui", "full", "kpi")
        target = str(config.get("record_assignments_target", "rm")).lower()
        if trace_enabled:
            target = "both"

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

        # QoS metric weighting (optional)
        qos_algorithm = str(config.get("scheduler_algorithm", config.get("qos_algorithm", "pf"))).lower()
        enable_qos = bool(config.get("enable_qos", qos_algorithm not in ("pf",)))
        if enable_qos and QOS_AVAILABLE and qos_algorithm in ("m-lwdf", "mlwdf", "exp-pf", "exppf"):
            # Initialize QoS state
            hol_delay_ms = np.zeros(N_UE)
            qos_params = np.column_stack([
                np.full(N_UE, config.get("mlwdf_delta", config.get("qos_delta", 0.01))),
                np.full(N_UE, config.get("mlwdf_tau", config.get("qos_tau_ms", 100.0))),
            ])
            avg_thr = np.full(N_UE, 1e-3)

            # Apply QoS weighting to SE metrics per TTI
            for tt in range(T):
                # Simulate HoL delay growth (simplified model)
                hol_delay_ms = hol_delay_ms + config.get("tti_ms", 1.0)

                if qos_algorithm in ("m-lwdf", "mlwdf"):
                    # Wideband metric for baseline
                    se_time_base[tt] = compute_mlwdf_metric(
                        se_time_base[tt], avg_thr, hol_delay_ms, qos_params
                    )
                    se_time_rm[tt] = compute_mlwdf_metric(
                        se_time_rm[tt], avg_thr, hol_delay_ms, qos_params
                    )
                elif qos_algorithm in ("exp-pf", "exppf"):
                    se_time_base[tt] = compute_exppf_metric(
                        se_time_base[tt],
                        avg_thr,
                        hol_delay_ms,
                        qos_params,
                        beta=float(config.get("exppf_beta", 1.0)),
                        c=float(config.get("exppf_c", 1.0)),
                    )
                    se_time_rm[tt] = compute_exppf_metric(
                        se_time_rm[tt],
                        avg_thr,
                        hol_delay_ms,
                        qos_params,
                        beta=float(config.get("exppf_beta", 1.0)),
                        c=float(config.get("exppf_c", 1.0)),
                    )

                # Update average throughput (simplified)
                avg_thr = 0.9 * avg_thr + 0.1 * np.mean(se_time_base[tt], axis=-1)
                # Reset delay for scheduled UEs (simplified: reset all)
                hol_delay_ms = hol_delay_ms * 0.5

        # OALS metric correction (Phase 11 - Patent)
        oals_phi_time = None
        oals_trend_time = None
        oals_urgency_time = None
        oals_correction_time = None
        oals_update_ttis = None
        if OALS_AVAILABLE and state.oals_scheduler is not None:
            from scheduler.lookahead import compute_correction_factor_batch
            oals = state.oals_scheduler
            update_interval = int(config.get("lookahead_update_interval", 10))

            # Initialize urgency. In production, urgency should come from Traffic/QoS manager.
            # For UI/benchmarking, we support simple synthetic urgency models via config.
            urgency_dist = str(config.get("urgency_distribution", "ramp") or "ramp").lower().strip()
            if urgency_dist in ("zero", "none", "off"):
                urgency = np.zeros(N_UE, dtype=float)
            elif urgency_dist in ("uniform",):
                urgency = np.asarray(self._rng.random(N_UE), dtype=float)
            elif urgency_dist in ("bursty",):
                frac = float(config.get("urgency_burst_frac", 0.2))
                frac = float(np.clip(frac, 0.0, 1.0))
                n_hi = int(round(frac * N_UE))
                urgency = 0.2 * np.asarray(self._rng.random(N_UE), dtype=float)
                if n_hi > 0:
                    idx = self._rng.choice(N_UE, size=n_hi, replace=False)
                    urgency[idx] = 0.8 + 0.4 * np.asarray(self._rng.random(n_hi), dtype=float)
            elif urgency_dist in ("mixed",):
                urgency = np.asarray(self._rng.random(N_UE), dtype=float)
            else:
                # "ramp" and any unknown kind: keep the legacy ramp+reset model.
                urgency = np.zeros(N_UE, dtype=float)
            if trace_enabled:
                oals_phi_time = np.zeros((T, N_UE), dtype=float)
                oals_trend_time = np.zeros((T, N_UE), dtype=float)
                oals_urgency_time = np.zeros((T, N_UE), dtype=float)
                oals_correction_time = np.ones((T, N_UE), dtype=float)
                oals_update_ttis = []

            for tt in range(T):
                # Update lookahead predictions periodically
                if tt % update_interval == 0:
                    did_update = oals.update_lookahead(state.ue_pos, tt)
                    if trace_enabled and did_update and oals_update_ttis is not None:
                        oals_update_ttis.append(int(tt))

                # Compute and apply OALS correction to RadioMap metrics
                # Get QoS weight (use ones if QoS not enabled)
                qos_weight = np.ones(N_UE)

                # Apply OALS correction to per-PRB metrics
                oals.set_urgency(urgency)
                if trace_enabled:
                    phi = np.asarray(oals.get_phi_array(), dtype=float).reshape(-1)
                    trend = np.asarray(oals.get_trend_array(), dtype=float).reshape(-1)
                    if phi.size != N_UE:
                        phi = np.ones(N_UE, dtype=float)
                    if trend.size != N_UE:
                        trend = np.zeros(N_UE, dtype=float)
                    correction = compute_correction_factor_batch(phi, urgency, trend, oals.cfg)
                    if correction.size != N_UE:
                        correction = np.ones(N_UE, dtype=float)
                    oals_phi_time[tt] = phi
                    oals_trend_time[tt] = trend
                    oals_urgency_time[tt] = urgency
                    oals_correction_time[tt] = correction
                se_time_rm[tt] = oals.compute_oals_metric(
                    se_time_rm[tt], qos_weight
                )

                # Update urgency (synthetic model)
                if urgency_dist in ("ramp",):
                    urgency = np.clip(urgency + 0.01, 0.0, 1.0)
                    reset_mask = self._rng.random(N_UE) < 0.1
                    urgency[reset_mask] = 0.0
                elif urgency_dist in ("uniform",):
                    urgency = np.asarray(self._rng.random(N_UE), dtype=float)
                elif urgency_dist in ("bursty",):
                    frac = float(config.get("urgency_burst_frac", 0.2))
                    frac = float(np.clip(frac, 0.0, 1.0))
                    n_hi = int(round(frac * N_UE))
                    urgency = 0.2 * np.asarray(self._rng.random(N_UE), dtype=float)
                    if n_hi > 0:
                        idx = self._rng.choice(N_UE, size=n_hi, replace=False)
                        urgency[idx] = 0.8 + 0.4 * np.asarray(self._rng.random(n_hi), dtype=float)
                elif urgency_dist in ("mixed",):
                    jitter = float(config.get("urgency_jitter_std", 0.05))
                    urgency = np.clip(
                        urgency + self._rng.normal(0.0, jitter, size=N_UE),
                        0.0,
                        None,
                    )
                    refresh = self._rng.random(N_UE) < float(config.get("urgency_refresh_prob", 0.05))
                    urgency[refresh] = np.asarray(self._rng.random(np.count_nonzero(refresh)), dtype=float)
                else:
                    # "zero"/"none"/"off" or unknown: keep constant.
                    pass

            logger.debug("OALS metric correction applied for %d TTIs", T)

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
        _rec_base = (trace_enabled or bool(config.get("record_assignments", False))) and target in ("base", "both", "all")
        _rec_base_thr = (trace_enabled or bool(config.get("record_ue_thr", False))) and target in ("base", "both", "all")
        _rec_rm = (trace_enabled or bool(config.get("record_assignments", False))) and target in ("rm", "both", "all")
        _rec_rm_thr = (trace_enabled or bool(config.get("record_ue_thr", False))) and target in ("rm", "both", "all")
        _rec_base_ack = (trace_enabled or bool(config.get("record_ue_ack_thr", False))) and target in ("base", "both", "all")
        _rec_rm_ack = (trace_enabled or bool(config.get("record_ue_ack_thr", False))) and target in ("rm", "both", "all")

        assignments_base = [] if _rec_base else None
        ue_thr_base = [] if _rec_base_thr else None
        ue_ack_base = [] if _rec_base_ack else None
        assignments_rm = [] if _rec_rm else None
        ue_thr_rm = [] if _rec_rm_thr else None
        ue_ack_rm = [] if _rec_rm_ack else None

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
            record_ue_ack_thr=_rec_base_ack,
            ue_ack_thr_out=ue_ack_base,
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
            record_ue_ack_thr=_rec_rm_ack,
            ue_ack_thr_out=ue_ack_rm,
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
            "ue_ack_base": ue_ack_base,
            "ue_ack_rm": ue_ack_rm,
            "oals_phi_time": oals_phi_time,
            "oals_trend_time": oals_trend_time,
            "oals_urgency_time": oals_urgency_time,
            "oals_correction_time": oals_correction_time,
            "oals_update_ttis": oals_update_ttis,
        }

    def _finalize(self, sched_result: Dict) -> Dict[str, Any]:
        """Finalize simulation and build result dict."""
        from link import re_per_prb_from_config
        from .trace import SimulationTrace, SchedulerTrace, OALSTrace

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

        # Traffic layer simulation (post-processing)
        self._apply_traffic_simulation(report)

        # OALS statistics (Phase 11 - Patent)
        if OALS_AVAILABLE and state.oals_scheduler is not None:
            oals_stats = state.oals_scheduler.get_statistics()
            report["oals_stats"] = oals_stats
            report["oals_enabled"] = True
            logger.debug("OALS stats: %s", oals_stats)
        else:
            report["oals_enabled"] = False

        # UI/trace export (optional)
        trace_enabled = bool(config.get("enable_trace", False)) or str(config.get("trace_level", "")).lower() in ("ui", "full", "kpi")
        if trace_enabled:
            T = int(state.T)
            Z = int(state.Z)

            base_assign = report.get("assignments_base")
            rm_assign = report.get("assignments_rm")
            base_thr = report.get("ue_thr_time_base")
            rm_thr = report.get("ue_thr_time_rm")
            base_ack = report.get("ue_ack_time_base")
            rm_ack = report.get("ue_ack_time_rm")

            def _sum_se_per_prb(arr: Any) -> Any:
                if arr is None:
                    return None
                try:
                    return np.sum(np.asarray(arr, dtype=float), axis=1) / float(max(1, Z))
                except Exception:
                    return None

            baseline_trace = SchedulerTrace(
                assignments=base_assign,
                ue_thr_scheduled=base_thr,
                ue_thr_acked=base_ack,
                sum_se_scheduled_per_prb=_sum_se_per_prb(base_thr),
                sum_se_acked_per_prb=_sum_se_per_prb(base_ack),
            )
            radiomap_trace = SchedulerTrace(
                assignments=rm_assign,
                ue_thr_scheduled=rm_thr,
                ue_thr_acked=rm_ack,
                sum_se_scheduled_per_prb=_sum_se_per_prb(rm_thr),
                sum_se_acked_per_prb=_sum_se_per_prb(rm_ack),
            )

            oals_trace = None
            try:
                phi = sched_result.get("oals_phi_time")
                trend = sched_result.get("oals_trend_time")
                urg = sched_result.get("oals_urgency_time")
                corr = sched_result.get("oals_correction_time")
                upd = sched_result.get("oals_update_ttis") or []
                if phi is not None and trend is not None and urg is not None and corr is not None:
                    oals_trace = OALSTrace(
                        phi=np.asarray(phi, dtype=float),
                        trend=np.asarray(trend, dtype=float),
                        urgency=np.asarray(urg, dtype=float),
                        correction=np.asarray(corr, dtype=float),
                        update_ttis=[int(x) for x in list(upd)],
                    )
            except Exception:
                oals_trace = None

            trace = SimulationTrace(
                mode="single",
                tti=np.arange(T, dtype=int),
                baseline=baseline_trace,
                radiomap=radiomap_trace,
                oals=oals_trace,
                extra={
                    "N_UE": int(state.N_UE),
                    "Z": int(state.Z),
                    "T": int(state.T),
                    "prb_bw_hz": float(state.prb_bw_hz),
                    "system_bandwidth_hz": float(sys_bw_hz),
                    "scheduler_algorithm": str(config.get("scheduler_algorithm", config.get("qos_algorithm", "pf"))),
                },
            )
            report["trace"] = trace.to_dict()

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

            ue_ack_rm = sched_result.get("ue_ack_rm")
            if ue_ack_rm is not None and len(ue_ack_rm) > 0:
                report["ue_ack_time_rm"] = np.stack(ue_ack_rm, axis=0)

            ue_ack_base = sched_result.get("ue_ack_base")
            if ue_ack_base is not None and len(ue_ack_base) > 0:
                report["ue_ack_time_base"] = np.stack(ue_ack_base, axis=0)
        except Exception:
            pass

    def _compute_per_ue_metrics(self, report: Dict, sched_result: Dict, sys_bw_hz: float) -> None:
        """Compute per-UE throughput and fairness metrics."""
        from link import re_per_prb_from_config

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

    def _apply_traffic_simulation(self, report: Dict) -> None:
        """Apply traffic layer simulation if traffic_model != full_buffer.

        This uses post-processing approach: scheduler runs with full-buffer
        assumption, then we simulate traffic dynamics on top of the output.
        """
        config = self.config
        traffic_model = config.get("traffic_model", "full_buffer")

        # Skip for full buffer mode (backward compatible)
        if traffic_model == "full_buffer":
            return

        try:
            from .traffic_simulator import TrafficSimulator

            # Get propagation delay if available
            tau_s_per_ue = None
            if self._state and self._state.geometry:
                tau_s_per_ue = getattr(self._state.geometry, "tau_s", None)

            # Run traffic simulation
            simulator = TrafficSimulator(config)
            traffic_result = simulator.simulate(report, tau_s_per_ue)

            # Merge traffic KPIs into report
            if "traffic_kpi" in traffic_result:
                report["traffic_kpi"] = traffic_result["traffic_kpi"]

                # Log summary
                kpi = traffic_result["traffic_kpi"]
                if "error" not in kpi:
                    logger.info(
                        "Traffic simulation: model=%s, delivered=%d/%d (%.1f%%), "
                        "latency_p95=%.2fms",
                        traffic_model,
                        kpi.get("packets_delivered", 0),
                        kpi.get("packets_arrived", 0),
                        kpi.get("packet_delivery_rate", 0) * 100,
                        kpi.get("latency_cdf", {}).get("percentiles", {}).get("p95.0", 0),
                    )
                else:
                    logger.warning("Traffic simulation failed: %s", kpi.get("error"))

        except ImportError as e:
            logger.warning(f"Traffic simulation unavailable: {e}")
        except Exception as e:
            logger.warning(f"Traffic simulation failed: {e}")

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
