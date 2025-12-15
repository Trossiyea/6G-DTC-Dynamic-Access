# -*- coding: utf-8 -*-
"""OALS (Orbit-Aware Lookahead Scheduling) Performance Benchmark.

Evaluates OALS algorithm performance across different parameter configurations:
1. Lookahead horizon sensitivity (H = 50, 100, 200, 500 TTI)
2. Urgency threshold sensitivity (α = 0.7, 0.8, 0.9)
3. Decay exponent sensitivity (γ = 1.5, 2.0, 3.0)
4. Full parameter grid comparison

Patent Core Algorithm Testing - Phase 11
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Allow running as a script from repo root (imports expect `code/` on sys.path).
import os
import sys
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)


@dataclass
class OALSBenchmarkResult:
    """Single OALS benchmark run result."""
    # Configuration
    lookahead_horizon: int
    alpha_urgent: float
    beta_wait: float
    gamma_decay: float

    # SE metrics
    baseline_se: float
    radiomap_se: float
    oals_se: float  # Note: OALS modifies radiomap path

    # OALS statistics
    oals_updates: int = 0
    oals_schedule_now: int = 0
    oals_delay: int = 0
    oals_urgent: int = 0

    # Fairness
    baseline_fairness: float = 0.0
    radiomap_fairness: float = 0.0
    oals_fairness: float = 0.0

    # HARQ
    baseline_harq_ack_rate: float = 1.0
    oals_harq_ack_rate: float = 1.0

    @property
    def oals_vs_baseline_pct(self) -> float:
        """OALS vs Baseline gain (%)."""
        if self.baseline_se > 0:
            return (self.oals_se - self.baseline_se) / self.baseline_se * 100
        return 0.0

    @property
    def oals_vs_radiomap_pct(self) -> float:
        """OALS vs RadioMap (no OALS) gain (%)."""
        if self.radiomap_se > 0:
            return (self.oals_se - self.radiomap_se) / self.radiomap_se * 100
        return 0.0

    @property
    def delay_ratio(self) -> float:
        """Ratio of delayed scheduling decisions."""
        total = self.oals_schedule_now + self.oals_delay + self.oals_urgent
        return self.oals_delay / max(total, 1)

    @property
    def urgent_ratio(self) -> float:
        """Ratio of urgent scheduling decisions."""
        total = self.oals_schedule_now + self.oals_delay + self.oals_urgent
        return self.oals_urgent / max(total, 1)


@dataclass
class OALSBenchmarkConfig:
    """OALS Benchmark configuration."""
    # Simulation base parameters
    n_ue: int = 50
    n_tti: int = 1000
    seed: int = 42

    # OALS parameter sweep ranges
    lookahead_horizons: List[int] = field(
        default_factory=lambda: [50, 100, 200, 500]
    )
    alpha_values: List[float] = field(
        default_factory=lambda: [0.7, 0.8, 0.9]
    )
    beta_values: List[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.4]
    )
    gamma_values: List[float] = field(
        default_factory=lambda: [1.5, 2.0, 3.0]
    )

    # LEO scenario
    ntn_scenario: str = "leo_600km"
    enable_time_varying: bool = True
    # NOTE: orbit_dynamics requires valid TLE + a visible start time. The benchmark
    # sets auto_orbit_start_for_visibility=True to avoid starting in outage.
    enable_orbit_dynamics: bool = True

    # Traffic model (affects urgency distribution)
    urgency_distribution: str = "mixed"  # "uniform" / "bursty" / "mixed"


def create_oals_base_config(bench_cfg: OALSBenchmarkConfig) -> Dict[str, Any]:
    """Create base simulation config for OALS testing.

    Args:
        bench_cfg: Benchmark configuration

    Returns:
        Base config dict compatible with SimulationEngine
    """
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    rm_path = os.path.join(base_dir, "radio_map", "Toronto", "RadioMap", "RM_toronto125_dBm.mat")

    return {
        "seed": bench_cfg.seed,
        "N_UE": bench_cfg.n_ue,
        "T": bench_cfg.n_tti,

        # Radio map
        "radio_map_mat_path": rm_path,
        "radio_map_mat_var": "XdB_recon_tensor",
        "radio_map_units": "dBm",

        # LEO satellite parameters
        "sat_altitude_km": 600,
        "carrier_freq_GHz": 2.0,
        "beam_half_bw_deg": 8.0,
        "beam_edge_drop_db": 3.0,
        "cell_size_km": 0.125,

        # PHY parameters
        "scs_khz": 30,
        # Benchmark note:
        # The bundled Radio Maps have strong interference levels (median around -45 dBm/PRB).
        # For algorithm/UI verification we use a higher per-PRB EIRP to keep SE non-trivial.
        "P_tx_dbm": 60.0,
        "G_rx_db": 38.0,
        "rx_nf_db": 7.0,
        "impl_loss_db": 1.0,
        "overhead_eff": 0.85,

        # Channel - time varying REQUIRED for OALS
        "enable_time_varying": bench_cfg.enable_time_varying,
        "enable_orbit_dynamics": bench_cfg.enable_orbit_dynamics,
        # Use a deterministic channel for OALS algorithm benchmarking so that
        # geometry-driven lookahead is measurable (instead of being swamped by i.i.d fading).
        "channel_model": "legacy",
        "shadow_std_db": 0.0,

        # TLE for orbit prediction (required for OALS)
        "tle_name": "OALS-TEST-SAT",
        "tle_lines": [
            "1 59422C 24065B   25266.77548611  .00029064  00000+0  23954-3 0  2662",
            "2 59422  53.1572 196.6800 0001379  82.5466  65.8632 15.69667376    15",
        ],
        # Start time: use a deterministic epoch close to the TLE and then auto-adjust
        # to a visible pass so the run doesn't start with near-zero SE.
        "orbit_start_datetime": "2025-10-07T21:38:31.574982+00:00",
        "auto_orbit_start_for_visibility": True,
        "auto_orbit_start_search_hours": 48.0,
        "auto_orbit_start_min_elev_deg": 5.0,
        "ref_lat_deg": 43.65108,
        "ref_lon_deg": -79.34702,
        # Coarser time step so orbit/beam movement is visible within short benchmarks.
        "tti_ms": 10.0,

        # Scheduler
        "pf_beta": 0.1,
        "use_mcs": False,
        "sched_require_contiguous": True,
        "sched_eesm_beta_db": 2.5,
        "baseline_sched_eesm_beta_db": 2.7,
        "rm_sched_eesm_beta_db": 3.5,

        # CSI delay
        "baseline_csi_delay_ttis": 12,
        "rm_csi_delay_ttis": 0,
        "enable_cqi_periodicity_base": True,
        "enable_cqi_periodicity_rm": False,
        "cqi_period_ttis": 5,
        "cqi_offset_ttis": 0,

        # RadioMap estimation error
        "radiomap_est_error_db": 0.0,
        "radiomap_blur_sigma": 0.0,

        # Radio Map dynamics
        "rm_flicker_db_std": 0.0,

        # HARQ
        "enable_harq_full": True,
        "harq_max_procs": 16,
        "harq_ack_delay_ttis": 10,

        # OALS enabled (will be overridden per test)
        # NOTE: When enable_orbit_dynamics=False, SimulationEngine will fall back to
        # a static orbit model so OALS can still run end-to-end (for UI/testing).
        "enable_oals": True,
        "lookahead_horizon_ttis": 200,
        "lookahead_sample_interval": 5,
        "lookahead_update_interval": 10,
        "alpha_urgent": 0.8,
        "beta_wait": 0.3,
        "theta_lookahead": 0.7,
        "gamma_decay": 2.0,
        "boost_factor": 2.0,
        "enable_trend_correction": True,
        "trend_threshold_db": 0.5,
        "trend_epsilon": 0.1,

        # Recording
        "record_ue_thr": True,
        "enable_trace": True,
        "trace_level": "kpi",
        "show_progress": False,
        "write_json_report": False,

        # OALS synthetic urgency model (SimulationEngine supports this key)
        "urgency_distribution": bench_cfg.urgency_distribution,
    }


def run_single_oals_test(
    base_config: Dict[str, Any],
    lookahead_horizon: int,
    alpha_urgent: float,
    beta_wait: float,
    gamma_decay: float,
) -> OALSBenchmarkResult:
    """Run single OALS test with specified parameters.

    Args:
        base_config: Base simulation configuration
        lookahead_horizon: Lookahead window in TTIs
        alpha_urgent: Urgency threshold
        beta_wait: Wait threshold
        gamma_decay: Decay exponent

    Returns:
        OALSBenchmarkResult with metrics
    """
    from simulation import SimulationEngine

    def _run_engine(cfg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return SimulationEngine(cfg).run()
        except RuntimeError as e:
            msg = str(e)
            if ("Skyfield" in msg) or ("sgp4" in msg):
                cfg2 = cfg.copy()
                cfg2["enable_orbit_dynamics"] = False
                cfg2["enable_oals"] = True
                if cfg2.get("show_progress", False):
                    print(f"[WARN] Orbit dynamics unavailable; fallback to StaticOrbitModel: {msg}")
                return SimulationEngine(cfg2).run()
            raise

    config = base_config.copy()

    # First run WITHOUT OALS to get a clean RadioMap baseline (same orbit/time-varying series).
    config_no_oals = config.copy()
    config_no_oals["enable_oals"] = False
    result_no_oals = _run_engine(config_no_oals)

    # Then run WITH OALS (same base config, only lookahead params differ).
    config_oals = config.copy()
    config_oals["enable_oals"] = True
    config_oals["lookahead_horizon_ttis"] = lookahead_horizon
    config_oals["alpha_urgent"] = alpha_urgent
    config_oals["beta_wait"] = beta_wait
    config_oals["gamma_decay"] = gamma_decay
    result_oals = _run_engine(config_oals)

    baseline_se = result_no_oals.get("avg_se_baseline_default", 0.0) or 0.0
    radiomap_se = result_no_oals.get("avg_se_radiomap", 0.0) or 0.0
    oals_se = result_oals.get("avg_se_radiomap", 0.0) or 0.0

    baseline_fairness = result_no_oals.get("fairness_jain_base", 0.0) or 0.0
    radiomap_fairness = result_no_oals.get("fairness_jain_map", 0.0) or 0.0
    oals_fairness = result_oals.get("fairness_jain_map", 0.0) or 0.0

    # Extract OALS statistics (updates only; per-TTI recommendations are computed below).
    oals_stats = result_oals.get("oals_stats", {}) or {}

    # HARQ ACK rates (baseline uses baseline scheduler; OALS uses radiomap path)
    harq_base = result_no_oals.get("harq_stats_base") or {}
    harq_rm_base = result_no_oals.get("harq_stats_map") or {}
    harq_rm_oals = result_oals.get("harq_stats_map") or {}

    base_ack = harq_base.get("ack_count", 1) if harq_base else 1
    base_total = base_ack + (harq_base.get("nack_count", 0) if harq_base else 0)
    rm_ack_base = harq_rm_base.get("ack_count", 1) if harq_rm_base else 1
    rm_total_base = rm_ack_base + (harq_rm_base.get("nack_count", 0) if harq_rm_base else 0)
    rm_ack_oals = harq_rm_oals.get("ack_count", 1) if harq_rm_oals else 1
    rm_total_oals = rm_ack_oals + (harq_rm_oals.get("nack_count", 0) if harq_rm_oals else 0)

    # Derive "delay/urgent/now" ratios from the recorded phi/urgency series (conceptual).
    oals_schedule_now = 0
    oals_delay = 0
    oals_urgent = 0
    try:
        trace = result_oals.get("trace") or {}
        oals_trace = trace.get("oals") or {}
        phi = np.asarray(oals_trace.get("phi"), dtype=float)
        urg = np.asarray(oals_trace.get("urgency"), dtype=float)
        if phi.ndim == 2 and urg.ndim == 2 and phi.shape == urg.shape:
            urgent_mask = urg > float(alpha_urgent)
            delay_mask = (urg < float(beta_wait)) & (phi < float(config_oals.get("theta_lookahead", 0.7)))
            oals_urgent = int(np.count_nonzero(urgent_mask))
            oals_delay = int(np.count_nonzero(delay_mask & (~urgent_mask)))
            oals_schedule_now = int(phi.size - oals_urgent - oals_delay)
    except Exception:
        pass

    return OALSBenchmarkResult(
        lookahead_horizon=lookahead_horizon,
        alpha_urgent=alpha_urgent,
        beta_wait=beta_wait,
        gamma_decay=gamma_decay,
        baseline_se=baseline_se,
        radiomap_se=radiomap_se,
        oals_se=oals_se,
        oals_updates=oals_stats.get("updates", 0),
        oals_schedule_now=oals_schedule_now,
        oals_delay=oals_delay,
        oals_urgent=oals_urgent,
        baseline_fairness=baseline_fairness,
        radiomap_fairness=radiomap_fairness,
        oals_fairness=oals_fairness,
        baseline_harq_ack_rate=base_ack / max(base_total, 1),
        oals_harq_ack_rate=rm_ack_oals / max(rm_total_oals, 1),
    )


def run_oals_baseline_comparison(
    base_config: Dict[str, Any],
    oals_params: Dict[str, Any],
) -> tuple:
    """Run OALS vs non-OALS comparison.

    Args:
        base_config: Base simulation configuration
        oals_params: OALS parameters to test

    Returns:
        (oals_result, baseline_radiomap_se) tuple
    """
    from simulation import SimulationEngine

    def _run_engine(cfg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return SimulationEngine(cfg).run()
        except RuntimeError as e:
            msg = str(e)
            if ("Skyfield" in msg) or ("sgp4" in msg):
                cfg2 = cfg.copy()
                cfg2["enable_orbit_dynamics"] = False
                cfg2["enable_oals"] = bool(cfg.get("enable_oals", False))
                if cfg2.get("show_progress", False):
                    print(f"[WARN] Orbit dynamics unavailable; fallback to StaticOrbitModel: {msg}")
                return SimulationEngine(cfg2).run()
            raise

    # First run without OALS to get pure RadioMap baseline
    config_no_oals = base_config.copy()
    config_no_oals["enable_oals"] = False

    result_no_oals = _run_engine(config_no_oals)
    radiomap_baseline_se = result_no_oals.get("avg_se_radiomap", 0.0) or 0.0

    # Then run with OALS
    config_oals = base_config.copy()
    config_oals["enable_oals"] = True
    for key, val in oals_params.items():
        config_oals[key] = val

    result_oals = _run_engine(config_oals)

    return result_oals, radiomap_baseline_se


def run_benchmark(
    bench_cfg: Optional[OALSBenchmarkConfig] = None,
    test_type: str = "horizon",
    verbose: bool = True,
) -> List[OALSBenchmarkResult]:
    """Run OALS benchmark.

    Args:
        bench_cfg: Benchmark configuration
        test_type: Type of test to run:
            - "horizon": Lookahead horizon sensitivity
            - "alpha": Urgency threshold sensitivity
            - "gamma": Decay exponent sensitivity
            - "full": Full parameter grid (reduced)
        verbose: Print progress

    Returns:
        List of OALSBenchmarkResult
    """
    if bench_cfg is None:
        bench_cfg = OALSBenchmarkConfig()

    base_config = create_oals_base_config(bench_cfg)
    results = []

    if test_type == "horizon":
        # Lookahead horizon sensitivity
        for H in bench_cfg.lookahead_horizons:
            if verbose:
                print(f"Testing lookahead_horizon = {H} TTIs...")

            result = run_single_oals_test(
                base_config,
                lookahead_horizon=H,
                alpha_urgent=0.8,
                beta_wait=0.3,
                gamma_decay=2.0,
            )
            results.append(result)

            if verbose:
                print(
                    f"  Base SE: {result.baseline_se:.4f}, RM SE: {result.radiomap_se:.4f}, OALS SE: {result.oals_se:.4f}"
                )
                print(
                    f"  Gain vs RM: {result.oals_vs_radiomap_pct:+.2f}%, vs Base: {result.oals_vs_baseline_pct:+.2f}%"
                )
                print(f"  Updates: {result.oals_updates}, Delay ratio: {result.delay_ratio:.2%}")

    elif test_type == "alpha":
        # Urgency threshold sensitivity
        for alpha in bench_cfg.alpha_values:
            if verbose:
                print(f"Testing alpha_urgent = {alpha}...")

            result = run_single_oals_test(
                base_config,
                lookahead_horizon=200,
                alpha_urgent=alpha,
                beta_wait=0.3,
                gamma_decay=2.0,
            )
            results.append(result)

            if verbose:
                print(
                    f"  Base SE: {result.baseline_se:.4f}, RM SE: {result.radiomap_se:.4f}, OALS SE: {result.oals_se:.4f}"
                )
                print(
                    f"  Gain vs RM: {result.oals_vs_radiomap_pct:+.2f}%, vs Base: {result.oals_vs_baseline_pct:+.2f}%"
                )
                print(f"  Urgent ratio: {result.urgent_ratio:.2%}, Delay ratio: {result.delay_ratio:.2%}")

    elif test_type == "gamma":
        # Decay exponent sensitivity
        for gamma in bench_cfg.gamma_values:
            if verbose:
                print(f"Testing gamma_decay = {gamma}...")

            result = run_single_oals_test(
                base_config,
                lookahead_horizon=200,
                alpha_urgent=0.8,
                beta_wait=0.3,
                gamma_decay=gamma,
            )
            results.append(result)

            if verbose:
                print(
                    f"  Base SE: {result.baseline_se:.4f}, RM SE: {result.radiomap_se:.4f}, OALS SE: {result.oals_se:.4f}"
                )
                print(
                    f"  Gain vs RM: {result.oals_vs_radiomap_pct:+.2f}%, vs Base: {result.oals_vs_baseline_pct:+.2f}%"
                )

    elif test_type == "full":
        # Full parameter grid (reduced for efficiency)
        for H in [100, 200]:
            for alpha in [0.7, 0.9]:
                for gamma in [1.5, 2.5]:
                    if verbose:
                        print(f"Testing H={H}, α={alpha}, γ={gamma}...")

                    result = run_single_oals_test(
                        base_config,
                        lookahead_horizon=H,
                        alpha_urgent=alpha,
                        beta_wait=0.3,
                        gamma_decay=gamma,
                    )
                    results.append(result)

                    if verbose:
                        print(
                            f"  Base SE: {result.baseline_se:.4f}, RM SE: {result.radiomap_se:.4f}, OALS SE: {result.oals_se:.4f}"
                        )
                        print(
                            f"  Gain vs RM: {result.oals_vs_radiomap_pct:+.2f}%, vs Base: {result.oals_vs_baseline_pct:+.2f}%"
                        )

    return results


def print_summary(results: List[OALSBenchmarkResult]) -> None:
    """Print benchmark summary table."""
    print("\n" + "=" * 90)
    print("OALS (Orbit-Aware Lookahead Scheduling) Benchmark Results")
    print("=" * 90)
    print(
        f"{'H(TTI)':<8} {'α':<6} {'γ':<6} {'Base SE':<10} {'RM SE':<10} {'OALS SE':<10} "
        f"{'GainRM(%)':<10} {'Delay%':<8} {'Updates':<8}"
    )
    print("-" * 90)

    for r in results:
        delay_pct = r.delay_ratio * 100
        print(
            f"{r.lookahead_horizon:<8} {r.alpha_urgent:<6.2f} {r.gamma_decay:<6.2f} "
            f"{r.baseline_se:<10.4f} {r.radiomap_se:<10.4f} {r.oals_se:<10.4f} "
            f"{r.oals_vs_radiomap_pct:<+10.2f} {delay_pct:<8.1f} {r.oals_updates:<8}"
        )

    print("-" * 90)

    # Summary statistics
    if len(results) > 0:
        gains_rm = [r.oals_vs_radiomap_pct for r in results]
        gains_base = [r.oals_vs_baseline_pct for r in results]
        print(
            f"Average Gain vs RM: {np.mean(gains_rm):+.2f}%  |  "
            f"Max: {np.max(gains_rm):+.2f}%  |  "
            f"Min: {np.min(gains_rm):+.2f}%"
        )
        print(f"Average Gain vs Base: {np.mean(gains_base):+.2f}%")

        # Best configuration
        best_idx = int(np.argmax(gains_rm))
        best = results[best_idx]
        print(f"\nBest Config: H={best.lookahead_horizon}, α={best.alpha_urgent}, γ={best.gamma_decay}")
        print(f"  -> Gain vs RM: {best.oals_vs_radiomap_pct:+.2f}%, Updates: {best.oals_updates}")


def export_results(results: List[OALSBenchmarkResult], path: str) -> None:
    """Export results to CSV.

    Args:
        results: List of benchmark results
        path: Output CSV path
    """
    import csv

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'lookahead_horizon', 'alpha_urgent', 'beta_wait', 'gamma_decay',
            'baseline_se', 'radiomap_se', 'oals_se',
            'oals_vs_baseline_pct', 'oals_vs_radiomap_pct',
            'oals_updates', 'oals_schedule_now', 'oals_delay', 'oals_urgent',
            'delay_ratio', 'urgent_ratio',
            'baseline_fairness', 'oals_fairness',
            'baseline_harq_ack_rate', 'oals_harq_ack_rate'
        ])
        for r in results:
            writer.writerow([
                r.lookahead_horizon, r.alpha_urgent, r.beta_wait, r.gamma_decay,
                r.baseline_se, r.radiomap_se, r.oals_se,
                r.oals_vs_baseline_pct, r.oals_vs_radiomap_pct,
                r.oals_updates, r.oals_schedule_now, r.oals_delay, r.oals_urgent,
                r.delay_ratio, r.urgent_ratio,
                r.baseline_fairness, r.oals_fairness,
                r.baseline_harq_ack_rate, r.oals_harq_ack_rate
            ])

    print(f"Results exported to {path}")


if __name__ == "__main__":
    # Quick test with reduced parameters
    print("Running OALS Benchmark (quick test)...")
    cfg = OALSBenchmarkConfig(
        n_ue=20,
        n_tti=200,
        lookahead_horizons=[50, 100, 200],
    )
    results = run_benchmark(cfg, test_type="horizon")
    print_summary(results)
