# -*- coding: utf-8 -*-
"""
Radio Map–aware dynamic access simulation for direct-to-satellite downlink (DL, FDD).

This module provides the main entry points for simulation:
- run_once(): Single-satellite simulation
- run_constellation(): Multi-satellite constellation simulation
- run_many(): Multiple seeds batch run

The simulation logic has been refactored into the `simulation/` package:
- SimulationEngine: Single-satellite engine with callback support
- ConstellationEngine: Multi-satellite engine with callback support
- SimulationState: Serializable state for Web integration

This file maintains backward compatibility with the original API.
"""

import logging
import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

# Import the refactored simulation engines
from simulation import (
    SimulationEngine,
    ConstellationEngine,
    SimulationState,
    # Re-export helpers for backward compatibility
    generate_ue_positions,
    resolve_noise_and_prb_bw,
    apply_open_loop_power_control,
    compute_caps,
    compute_metric_override_static_if_needed,
    build_time_variation_if_enabled,
)

from config import CONFIG
from logging_utils import get_logger
from result_schema import SimulationResult, to_serializable_result

logger = get_logger(__name__)


# -----------------------
# Main API Functions (backward compatible)
# -----------------------

def run_once(config: Dict) -> Dict:
    """Single experiment orchestration using SimulationEngine.

    This function provides backward compatibility with the original run_once().

    Args:
        config: Simulation configuration dict

    Returns:
        Result dict with simulation outputs including:
        - avg_se_baseline_default: Baseline average SE (bits/s/Hz)
        - avg_se_radiomap: RadioMap-aware average SE (bits/s/Hz)
        - improvement_vs_default_pct: Relative improvement (%)
        - cap, snr_lin, ue_pos: Intermediate data
        - harq_stats_*: HARQ statistics if enabled
    """
    engine = SimulationEngine(config)
    return engine.run()


def run_constellation(config: Dict) -> Dict:
    """Multi-satellite constellation simulation using ConstellationEngine.

    This function provides backward compatibility with the original run_constellation().

    Args:
        config: Simulation configuration dict with constellation settings

    Returns:
        Result dict with constellation simulation outputs including:
        - avg_se_baseline_default, avg_se_radiomap, improvement_vs_default_pct
        - ho_events_per_ue: Handover event logs
        - per_sat_kpis: Per-satellite KPI summary
        - outage_ttis_per_ue: Outage duration per UE
    """
    engine = ConstellationEngine(config)
    return engine.run()


def run_many(config: Dict, seeds: np.ndarray) -> Dict:
    """Run simulation with multiple seeds for statistical analysis.

    Args:
        config: Base configuration dict
        seeds: Array of seed values

    Returns:
        Dict with arrays of results:
        - baseline_default: Array of baseline SE values
        - radiomap: Array of RadioMap SE values
        - improvement_vs_default_pct: Array of improvement percentages
    """
    base_def_list, map_list, imp_def_list = [], [], []
    for s in seeds:
        c2 = dict(config)
        c2["seed"] = int(s)
        out = run_once(c2)
        base_def_list.append(out["avg_se_baseline_default"])
        map_list.append(out["avg_se_radiomap"])
        imp_def_list.append(out["improvement_vs_default_pct"])
    return {
        "baseline_default": np.array(base_def_list),
        "radiomap": np.array(map_list),
        "improvement_vs_default_pct": np.array(imp_def_list),
    }


def as_simulation_result(result: Dict, dataclass: bool = False):
    """Convert a raw result dict into a typed SimulationResult.

    Args:
        result: Raw result dict from run_once/run_constellation
        dataclass: If True, return SimulationResult dataclass;
                   if False, return JSON-friendly dict

    Returns:
        SimulationResult or serializable dict
    """
    res = SimulationResult.from_dict(result)
    return res if dataclass else to_serializable_result(result)


# -----------------------
# CLI Entry Point
# -----------------------

if __name__ == '__main__':
    # Run constellation or single-satellite experiment
    if bool(CONFIG.get("enable_constellation", False)):
        out = run_constellation(CONFIG)
        print("Constellation-run results (independent scheduling, no inter-sat interference)")
        print(f"  Baseline-Default avg SE (bits/s/Hz): {out['avg_se_baseline_default']:.3f}")
        print(f"  RadioMap         avg SE (bits/s/Hz): {out['avg_se_radiomap']:.3f}")
        try:
            print(f"  Gain vs Default (%): {out['improvement_vs_default_pct']:.2f}")
        except Exception:
            pass

        # Optional concise HARQ summary (constellation aggregate)
        if bool(CONFIG.get("print_harq_summary", True)):
            def _print_harq_const(label: str, hs: dict) -> None:
                if not hs:
                    print(f"\n[HARQ] {label}: no HARQ stats available.")
                    return
                tb_started = int(hs.get('tb_started', 0))
                tb_acked = int(hs.get('tb_acked', 0))
                tb_dropped = int(hs.get('tb_dropped', 0))
                ack_rate = float(hs.get('ack_rate', 0.0)) * 100.0
                first_try = float(hs.get('first_try_ack_rate', 0.0)) * 100.0
                avg_retx = float(hs.get('avg_retx_per_acked', 0.0))
                print(f"\n[HARQ] {label} (aggregate):")
                print(f"  TB started/ACKed/dropped: {tb_started}/{tb_acked}/{tb_dropped}  (ACK rate={ack_rate:.1f}%, first-try ACK={first_try:.1f}%)")
                print(f"  Avg retransmissions per ACKed TB: {avg_retx:.2f}")
            try:
                _print_harq_const("Baseline-Default", out.get("harq_stats_base"))
                _print_harq_const("RadioMap", out.get("harq_stats_map"))
            except Exception:
                pass
    else:
        single = run_once(CONFIG)
        print("Single-run results")
        print(f"  Baseline-Default avg SE (bits/s/Hz): {single['avg_se_baseline_default']:.3f}")
        print(f"  RadioMap        avg SE (bits/s/Hz): {single['avg_se_radiomap']:.3f}")
        print(f"  Gain vs Default (%): {single['improvement_vs_default_pct']:.2f}")
        try:
            bw_mhz = single.get('system_bandwidth_hz', 0.0) / 1e6
            th_base = single.get('total_throughput_baseline_bps', None)
            th_map = single.get('total_throughput_radiomap_bps', None)
            if th_base is not None and th_map is not None:
                print(f"  System Bandwidth: {bw_mhz:.3f} MHz")
                print(f"  Baseline-Default total throughput: {th_base/1e6:.3f} Mbps")
                print(f"  RadioMap        total throughput: {th_map/1e6:.3f} Mbps")
                if 'avg_ue_throughput_baseline_bps' in single and 'avg_ue_throughput_radiomap_bps' in single:
                    print(f"  Avg UE throughput (Baseline): {single['avg_ue_throughput_baseline_bps']/1e6:.3f} Mbps/UE")
                    print(f"  Avg UE throughput (RadioMap): {single['avg_ue_throughput_radiomap_bps']/1e6:.3f} Mbps/UE")
        except Exception:
            pass

        # Optional concise HARQ summary
        if bool(CONFIG.get("print_harq_summary", True)):
            def _print_harq(label: str, hs: dict, per_ue_key: str) -> None:
                if not hs:
                    print(f"\n[HARQ] {label}: no HARQ stats available.")
                    return
                tb_started = int(hs.get('tb_started', hs.get('initial_ack_count', 0) + hs.get('initial_nack_count', 0)))
                tb_acked = int(hs.get('tb_acked', hs.get('ack_count', 0)))
                tb_dropped = int(hs.get('tb_dropped', 0))
                init_ack = int(hs.get('initial_ack_count', 0))
                avg_retx = float(hs.get('avg_retx_per_acked', 0.0))
                olla_hist = hs.get('olla_offset_avg', []) or []
                olla_last = float(olla_hist[-1]) if len(olla_hist) > 0 else float(np.mean(hs.get('olla_last_per_ue', []) or [0.0]))
                print(f"\n[HARQ] {label}:")
                if tb_started > 0:
                    ack_rate = 100.0 * tb_acked / float(tb_started)
                    init_ack_rate = 100.0 * init_ack / float(tb_started)
                    print(f"  TB started/ACKed/dropped: {tb_started}/{tb_acked}/{tb_dropped}  (ACK rate={ack_rate:.1f}%, first-try ACK={init_ack_rate:.1f}%)")
                else:
                    print(f"  TB started/ACKed/dropped: {tb_started}/{tb_acked}/{tb_dropped}")
                print(f"  Avg retransmissions per ACKed TB: {avg_retx:.2f}")
                print(f"  OLLA avg offset (last): {olla_last:+.2f} dB")
                # Per-UE goodput (SE per PRB) if available
                per_ue = single.get(per_ue_key)
                if per_ue:
                    arr = np.asarray(per_ue, dtype=float)
                    print(f"  Per-UE avg SE: mean={arr.mean():.3f}, min={arr.min():.3f}, max={arr.max():.3f}")

            _print_harq("Baseline-Default", single.get("harq_stats_base"), "per_ue_avg_se_base")
            _print_harq("RadioMap", single.get("harq_stats_map"), "per_ue_avg_se_map")

        # Run multiple seeds to show robustness
        seeds = np.arange(1, 21)
        multi = run_many(CONFIG, seeds)
        print("\nMulti-seed summary (N=20)")
        print(f"  Baseline-Default avg SE: {multi['baseline_default'].mean():.3f} ± {multi['baseline_default'].std():.3f}")
        print(f"  RadioMap         avg SE: {multi['radiomap'].mean():.3f} ± {multi['radiomap'].std():.3f}")
        print(f"  Gain vs Default median: {np.median(multi['improvement_vs_default_pct']):.2f}% (min={multi['improvement_vs_default_pct'].min():.2f}%, max={multi['improvement_vs_default_pct'].max():.2f}%)")

        # Plots
        save_plots = CONFIG.get("save_plots", True)
        show_plots = CONFIG.get("show_plots", False)
        plot_dir = CONFIG.get("plot_dir", "output")
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        def maybe_finalize(fig_name: str):
            if save_plots:
                plt.savefig(os.path.join(plot_dir, fig_name), dpi=140, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()

        # 1) Improvement distribution
        plt.figure(figsize=(6, 4))
        plt.hist(multi["improvement_vs_default_pct"], bins=10, edgecolor='black')
        plt.title("Radio Map–aware gain vs Default baseline")
        plt.xlabel("Gain vs. Default baseline (%)")
        plt.ylabel("Count")
        plt.tight_layout()
        maybe_finalize("gain_distribution.png")

        # 2) Example Interference Map slice (median over frequency)
        R_med = np.median(single["R_xyz_dbm"], axis=2)
        plt.figure(figsize=(5, 5))
        plt.imshow(R_med.T, origin='lower', aspect='equal')
        plt.title("Interference Map (median over frequency), dBm")
        plt.colorbar(label='dBm')
        plt.tight_layout()
        maybe_finalize("interference_map_median.png")

        # 3) Example per-UE wideband vs best-PRB capacity (first 10 UEs)
        ue = np.arange(min(10, CONFIG["N_UE"]))
        best_prb = single["cap"][ue].max(axis=1)
        wb = single["cap_wb"][ue]
        x = np.arange(ue.size)
        plt.figure(figsize=(6, 4))
        plt.bar(x - 0.2, wb, width=0.4, label='Wideband (baseline)')
        plt.bar(x + 0.2, best_prb, width=0.4, label='Best PRB (RadioMap)')
        plt.xticks(x, [f"UE{int(i)}" for i in ue])
        plt.ylabel("Spectral efficiency (bits/s/Hz)")
        plt.title("Per-UE: wideband vs best PRB opportunity")
        plt.legend()
        plt.tight_layout()
        maybe_finalize("per_ue_wb_vs_best_prb.png")
