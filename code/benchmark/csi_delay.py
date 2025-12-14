# -*- coding: utf-8 -*-
"""CSI Delay Sensitivity Benchmark.

Compares RadioMap-aware scheduling vs traditional CQI-based scheduling
under varying CSI feedback delays for LEO NTN scenarios.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    csi_delay_ms: float
    baseline_se: float
    radiomap_se: float
    baseline_fairness: float = 0.0
    radiomap_fairness: float = 0.0
    baseline_harq_ack_rate: float = 1.0
    radiomap_harq_ack_rate: float = 1.0

    @property
    def improvement_pct(self) -> float:
        """RadioMap improvement over baseline (%)."""
        if self.baseline_se > 0:
            return (self.radiomap_se - self.baseline_se) / self.baseline_se * 100
        return 0.0


@dataclass
class CSIDelayBenchmarkConfig:
    """Benchmark configuration."""
    # CSI delays to test (in TTIs, 1 TTI = 1ms for 15kHz SCS)
    csi_delays_ttis: List[int] = field(default_factory=lambda: [0, 2, 4, 8, 16, 32])

    # Simulation parameters
    n_ue: int = 50
    n_tti: int = 500
    seed: int = 42

    # LEO scenario defaults
    ntn_scenario: str = "leo_600km"
    enable_time_varying: bool = True

    # RadioMap uses position-based prediction (no CSI delay impact)
    rm_csi_delay_ttis: int = 0


def create_base_config(bench_cfg: CSIDelayBenchmarkConfig) -> Dict[str, Any]:
    """Create base simulation config for LEO scenario."""
    import os
    # Find radio map file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    rm_path = os.path.join(base_dir, "radio_map", "Shanghai", "RadioMap", "RM_shanghai125_dBm.mat")

    return {
        "seed": bench_cfg.seed,
        "N_UE": bench_cfg.n_ue,
        "T": bench_cfg.n_tti,

        # Radio map (地面基站同频干扰功率图)
        "radio_map_mat_path": rm_path,
        "radio_map_mat_var": "XdB_recon_tensor",
        "radio_map_units": "dBm",

        # LEO satellite parameters
        "sat_alt_km": 600,
        "sat_elev_deg": 45,
        "ntn_scenario": bench_cfg.ntn_scenario,

        # PHY parameters
        "scs_khz": 30,
        "P_tx_dbm": 30.0,

        # Channel
        "enable_time_varying": bench_cfg.enable_time_varying,
        "channel_model": "3gpp_ntn",
        "ntn_channel_profile": "s_band_handheld_urban",
        "shadow_std_db": 3.0,

        # Scheduler
        "pf_beta": 0.9,
        "use_mcs": True,
        "sched_require_contiguous": True,

        # RadioMap uses prior info (no delay)
        "rm_csi_delay_ttis": bench_cfg.rm_csi_delay_ttis,

        # HARQ
        "enable_harq_full": True,
        "harq_max_procs": 16,
        "harq_ack_delay_ttis": 8,  # LEO RTT ~8ms

        # Recording
        "record_ue_thr": True,
        "record_assignments_target": "both",
    }


def run_single_delay(
    base_config: Dict[str, Any],
    csi_delay_ttis: int,
    tti_ms: float = 1.0,
) -> BenchmarkResult:
    """Run simulation with specific CSI delay.

    Compares:
    - Baseline: Wideband CQI scheduler (pf_schedule_baseline)
    - RadioMap: Per-PRB interference-aware scheduler (pf_schedule_radiomap_blocks)

    Args:
        base_config: Base simulation configuration
        csi_delay_ttis: CSI feedback delay in TTIs
        tti_ms: TTI duration in ms

    Returns:
        BenchmarkResult with SE comparison
    """
    from simulation import SimulationEngine
    from scheduler.baseline import pf_schedule_baseline
    from scheduler.radiomap import pf_schedule_radiomap_blocks

    config = base_config.copy()
    config["show_progress"] = False

    # Initialize engine to get state
    engine = SimulationEngine(config)
    engine.initialize()
    state = engine.state

    T = config["T"]
    Z = state.cap.shape[1]

    # Baseline: Wideband CQI scheduler
    baseline_se = pf_schedule_baseline(
        cap_wb=state.cap_wb,
        Z=Z,
        T=T,
        beta=config.get("pf_beta", 0.9),
        snr_lin_wb=state.snr_lin_wb,
        snr_lin_prb=state.snr_lin,
        cap_prb=state.cap,
        use_mcs=config.get("use_mcs", False),
        config=config,
    )

    # RadioMap: Per-PRB interference-aware scheduler
    radiomap_se = pf_schedule_radiomap_blocks(
        cap=state.cap,
        T=T,
        beta=config.get("pf_beta", 0.9),
        snr_lin=state.snr_lin,
        overhead_eff=1.0,
        use_mcs=config.get("use_mcs", False),
        power_split=False,
        se_metric_override=None,
        max_prbs_per_ue=None,
        mcs_params=None,
        se_metric_time=None,
        snr_lin_time=None,
        config=config,
    )

    return BenchmarkResult(
        csi_delay_ms=csi_delay_ttis * tti_ms,
        baseline_se=baseline_se,
        radiomap_se=radiomap_se,
    )


def run_benchmark(
    bench_cfg: Optional[CSIDelayBenchmarkConfig] = None,
    verbose: bool = True,
) -> List[BenchmarkResult]:
    """Run full CSI delay sensitivity benchmark.

    Args:
        bench_cfg: Benchmark configuration
        verbose: Print progress

    Returns:
        List of results for each CSI delay
    """
    if bench_cfg is None:
        bench_cfg = CSIDelayBenchmarkConfig()

    base_config = create_base_config(bench_cfg)
    results = []

    for delay in bench_cfg.csi_delays_ttis:
        if verbose:
            print(f"Running CSI delay = {delay} TTIs...")

        result = run_single_delay(base_config, delay)
        results.append(result)

        if verbose:
            print(f"  Baseline SE: {result.baseline_se:.3f} bits/s/Hz")
            print(f"  RadioMap SE: {result.radiomap_se:.3f} bits/s/Hz")
            print(f"  Improvement: {result.improvement_pct:+.1f}%")

    return results


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print benchmark summary table."""
    print("\n" + "=" * 70)
    print("CSI Delay Sensitivity Benchmark Results")
    print("=" * 70)
    print(f"{'Delay(ms)':<10} {'Baseline SE':<12} {'RadioMap SE':<12} {'Gain(%)':<10} {'HARQ(B/R)':<12}")
    print("-" * 70)

    for r in results:
        print(f"{r.csi_delay_ms:<10.0f} {r.baseline_se:<12.3f} {r.radiomap_se:<12.3f} "
              f"{r.improvement_pct:<+10.1f} {r.baseline_harq_ack_rate:.2f}/{r.radiomap_harq_ack_rate:.2f}")

    print("-" * 70)

    # Summary statistics
    if len(results) > 1:
        delays = [r.csi_delay_ms for r in results]
        gains = [r.improvement_pct for r in results]

        # Gain at zero delay vs max delay
        gain_0 = results[0].improvement_pct
        gain_max = results[-1].improvement_pct

        print(f"Gain at 0ms delay:   {gain_0:+.1f}%")
        print(f"Gain at {delays[-1]:.0f}ms delay: {gain_max:+.1f}%")
        print(f"Additional gain from CSI robustness: {gain_max - gain_0:+.1f}%")


def export_results(results: List[BenchmarkResult], path: str) -> None:
    """Export results to CSV."""
    import csv

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'csi_delay_ms', 'baseline_se', 'radiomap_se', 'improvement_pct',
            'baseline_fairness', 'radiomap_fairness',
            'baseline_harq_ack_rate', 'radiomap_harq_ack_rate'
        ])
        for r in results:
            writer.writerow([
                r.csi_delay_ms, r.baseline_se, r.radiomap_se, r.improvement_pct,
                r.baseline_fairness, r.radiomap_fairness,
                r.baseline_harq_ack_rate, r.radiomap_harq_ack_rate
            ])


if __name__ == "__main__":
    # Quick test with reduced parameters
    cfg = CSIDelayBenchmarkConfig(
        csi_delays_ttis=[0, 4, 8, 16],
        n_ue=20,
        n_tti=100,
    )
    results = run_benchmark(cfg)
    print_summary(results)
