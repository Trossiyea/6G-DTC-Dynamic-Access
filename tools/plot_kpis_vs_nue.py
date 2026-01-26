#!/usr/bin/env python3
"""
Sweep the number of UEs and plot line charts for:
- Average SE (bits/s/Hz) vs N_UE
- Jain's Fairness Index vs N_UE

Uses code/config.py and code/main.py (run_once). Results are averaged over
optional multiple seeds for smoother curves.

Examples:
  # Default sweep 10..150 step 10 (single seed)
  python tools/plot_kpis_vs_nue.py

  # Custom sweep list with averaging over 3 seeds and smaller T for speed
  python tools/plot_kpis_vs_nue.py --nue-list 10,20,40,80,120 --n-seeds 3 --T 400

  # Show figures interactively
  python tools/plot_kpis_vs_nue.py --show
"""

from __future__ import annotations

import argparse
import os
import sys
import csv
import time
import importlib.util
from typing import Dict, List, Tuple
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt

# Mapping of scenario names to their config patch files
SCENARIOS = {
    'single': 'test/config_toronto_single.py',
    'constellation': 'test/config_toronto_constellation.py'
}

def load_config_patch(path: str) -> Dict:
    """Load the CONFIG dictionary from a python file."""
    if not os.path.exists(path):
        print(f"Warning: Config patch file {path} not found.")
        return {}
    
    spec = importlib.util.spec_from_file_location("patch_config", path)
    if spec is None or spec.loader is None:
        return {}
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "CONFIG", {})

def _ensure_style():
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        pass
    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.5,
        'grid.linestyle': '--',
        'legend.frameon': False,
    })



def resolve_constellation_satellite(cfg: Dict) -> Dict:
    """If constellation is enabled, pick the best satellite from catalog."""
    if not cfg.get("enable_constellation", False):
        return cfg
    if cfg.get("tle_lines") is not None:
        return cfg
        
    catalog_path = cfg.get("tle_catalog_path")
    if not catalog_path:
        return cfg
        
    try:
        from skyfield.api import load, EarthSatellite, wgs84
    except ImportError:
        return cfg 
    
    # Parse start time
    start_str = cfg.get("orbit_start_datetime")
    try:
        if not start_str:
            t0 = datetime.now(timezone.utc)
        else:
            s = str(start_str).replace('Z', '+00:00')
            t0 = datetime.fromisoformat(s)
    except ValueError:
         t0 = datetime.now(timezone.utc)

    # Load TLEs
    satellites = []
    if os.path.exists(catalog_path):
        with open(catalog_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        i = 0
        while i < len(lines)-1:
            l1 = lines[i]
            l2 = lines[i+1]
            if l1.startswith('1 ') and l2.startswith('2 '):
                name = f"SAT-{l1[2:7]}"
                if i > 0 and not lines[i-1].startswith('1 ') and not lines[i-1].startswith('2 '):
                    name = lines[i-1]
                satellites.append((name, l1, l2))
                i += 2
            else:
                i += 1
    
    if not satellites:
        return cfg
        
    # Evaluate at t0
    ts = load.timescale()
    t_sf = ts.from_datetime(t0)
    
    lat = float(cfg.get("ref_lat_deg", 0.0))
    lon = float(cfg.get("ref_lon_deg", 0.0))
    ground = wgs84.latlon(lat, lon)
    
    best_sat = None
    best_elev = -90.0
    
    for name, l1, l2 in satellites:
        sat = EarthSatellite(l1, l2, name)
        topoc = sat.at(t_sf) - ground.at(t_sf)
        alt, _, _ = topoc.altaz()
        el = alt.degrees
        if el > best_elev:
            best_elev = el
            best_sat = (name, l1, l2)
            
    if best_sat:
        name, l1, l2 = best_sat
        cfg["tle_name"] = name
        cfg["tle_lines"] = [l1, l2]
        print(f"[Constellation] Auto-selected best satellite: {name} (Elev: {best_elev:.2f} deg)")
        
    return cfg


def _run_once_with(cfg_base: Dict, N_UE: int, seed: int, T_override: int | None) -> Tuple[float, float, float, float, float, float]:
    # Ensure local code directory is prioritized
    code_path = os.path.abspath('code')
    if code_path not in sys.path:
        sys.path.insert(0, code_path)
        
    from main import run_once, run_constellation  # type: ignore
    cfg = dict(cfg_base)
    # Note: Do NOT call resolve_constellation_satellite here - constellation mode
    # should use all satellites from the catalog, not select a single best one.
    cfg['N_UE'] = int(N_UE)
    cfg['seed'] = int(seed)
    if T_override is not None:
        cfg['T'] = int(T_override)
    
    # Disable unnecessary recording for speed
    cfg['record_assignments'] = False
    cfg['record_ue_thr'] = True
    cfg['show_progress'] = True
    
    # Choose the correct runner based on constellation mode
    if cfg.get('enable_constellation', False):
        print(f"  [Constellation Mode] N_UE={N_UE}, seed={seed}")
        rep = run_constellation(cfg)
    else:
        print(f"  [Single Satellite Mode] N_UE={N_UE}, seed={seed}")
        rep = run_once(cfg)
    
    se_base = float(rep.get('avg_se_baseline_default', np.nan))
    se_mr = float(rep.get('avg_se_baseline_mr', np.nan))
    se_rm = float(rep.get('avg_se_radiomap', np.nan))
    # Retrieve per-UE SE arrays if available to recompute/verify Jain's index
    se_ue_b = rep.get('per_ue_avg_se_base', [])
    se_ue_mr = rep.get('per_ue_avg_se_mr', [])
    se_ue_m = rep.get('per_ue_avg_se_rm', [])

    def jain_index(arr):
        if not arr: return np.nan
        a = np.array(arr, dtype=float)
        # Filter out negligible values to avoid div-by-zero or numerical noise if needed
        # But standard Jain includes all. Zeros are fine.
        s = np.sum(a)
        s2 = np.sum(a * a)
        if s2 <= 1e-15: return 0.0
        n = float(len(a))
        return (s * s) / (n * s2)

    # Use local computation if available, otherwise fallback to reported value
    fj_b = jain_index(se_ue_b) if se_ue_b else float(rep.get('fairness_jain_base', np.nan) or np.nan)
    fj_mr = jain_index(se_ue_mr) if se_ue_mr else float(rep.get('fairness_jain_mr', np.nan) or np.nan)
    fj_m = jain_index(se_ue_m) if se_ue_m else float(rep.get('fairness_jain_map', np.nan) or np.nan)

    return se_base, se_mr, se_rm, fj_b, fj_mr, fj_m


def _parse_list(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(',') if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot Average SE and Jain fairness vs number of UEs')
    g = p.add_argument_group('Sweep range')
    g.add_argument('--nue-list', type=str, default=None, help='Comma-separated list of N_UE values (overrides min/max/step)')
    g.add_argument('--nue-min', type=int, default=25, help='Minimum number of UEs (inclusive)')
    g.add_argument('--nue-max', type=int, default=100, help='Maximum number of UEs (inclusive)')
    g.add_argument('--nue-step', type=int, default=25, help='Step in UEs')#原先40-200，step=10
    g2 = p.add_argument_group('Averaging / speed')
    g2.add_argument('--n-seeds', type=int, default=1, help='Number of seeds to average over (>=1)')
    g2.add_argument('--base-seed', type=int, default=None, help='Base seed; when set, seeds are base, base+1, ...')
    g2.add_argument('--T', type=int, default=None, help='Override T (TTIs) for speed')
    out = p.add_argument_group('Output')
    out.add_argument('--outfile-prefix', type=str, default='kpis_vs_nue', help='Output file prefix under output/')
    out.add_argument('--scenario', type=str, choices=['single', 'constellation', 'all'], default='single',
                     help='Scenario to run: single (default), constellation, or all (both)')
    out.add_argument('--show', action='store_true', help='Also show figures')
    return p.parse_args()


def run_scenario_sweep(scenario_name: str, cfg_base: Dict, args: argparse.Namespace, sweeps: List[int]) -> None:
    """Run the sweep for a specific scenario."""
    print(f"\n=== Running Scenario: {scenario_name} ===")
    
    # Prepare output directories
    out_dir = cfg_base.get('plot_dir', 'output')
    out_dir_jain = os.path.join(out_dir, 'jain')
    os.makedirs(out_dir_jain, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    base_seed = int(args.base_seed) if args.base_seed is not None else int(cfg_base.get('seed', 101))
    n_seeds = max(1, int(args.n_seeds))

    se_base_vals: List[float] = []
    se_mr_vals: List[float] = []
    se_rm_vals: List[float] = []
    fj_base_vals: List[float] = []
    fj_mr_vals: List[float] = []
    fj_rm_vals: List[float] = []

    print(f"Running sweep over N_UE = {sweeps} (seeds={n_seeds}, base_seed={base_seed}, T_override={args.T})")
    for N in sweeps:
        se_b_acc = []
        se_mr_acc = []
        se_m_acc = []
        fj_b_acc = []
        fj_mr_acc = []
        fj_m_acc = []
        for i in range(n_seeds):
            seed_i = base_seed + i
            se_b, se_mr, se_m, fj_b, fj_mr, fj_m = _run_once_with(cfg_base, N, seed_i, args.T)
            se_b_acc.append(se_b)
            se_mr_acc.append(se_mr)
            se_m_acc.append(se_m)
            fj_b_acc.append(fj_b)
            fj_mr_acc.append(fj_mr)
            fj_m_acc.append(fj_m)
        se_base_vals.append(float(np.nanmean(se_b_acc)))
        se_mr_vals.append(float(np.nanmean(se_mr_acc)))
        se_rm_vals.append(float(np.nanmean(se_m_acc)))
        fj_base_vals.append(float(np.nanmean(fj_b_acc)))
        fj_mr_vals.append(float(np.nanmean(fj_mr_acc)))
        fj_rm_vals.append(float(np.nanmean(fj_m_acc)))
        print(f"  N_UE={N:3d}: SE(base/MR/RM)={se_base_vals[-1]:.3f}/{se_mr_vals[-1]:.3f}/{se_rm_vals[-1]:.3f}, "
              f"Jain(base/MR/RM)={fj_base_vals[-1]:.3f}/{fj_mr_vals[-1]:.3f}/{fj_rm_vals[-1]:.3f}")

    sweeps_arr = np.array(sweeps, dtype=int)
    
    # Prefix for outputs
    file_prefix = f"{scenario_name}_{args.outfile_prefix}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Plot Average SE vs N_UE (three lines)
    fig1 = plt.figure(figsize=(7.8, 4.6))
    ax1 = fig1.gca()
    ax1.plot(sweeps_arr, se_base_vals, '-o', color='#4C78A8', label='3GPP PF', linewidth=1.8, markersize=4)
    ax1.plot(sweeps_arr, se_mr_vals, '-^', color='#E15759', label='3GPP MR', linewidth=1.8, markersize=4)
    ax1.plot(sweeps_arr, se_rm_vals, '-s', color='#F58518', label='RadioMap', linewidth=1.8, markersize=4)
    ax1.set_xlabel('Number of UEs')
    ax1.set_ylabel('Average SE (bits/s/Hz)')
    ax1.set_title(f'Average Spectral Efficiency vs Number of UEs\n({scenario_name})')
    ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax1.legend()
    fig1.tight_layout()
    p1 = os.path.join(out_dir_jain, f"{file_prefix}_avg_se_vs_nue_{timestamp}.png")
    fig1.savefig(p1, dpi=160, bbox_inches='tight')
    if args.show:
        plt.show()
    else:
        plt.close(fig1)

    # Plot Jain's fairness vs N_UE (three lines)
    fig2 = plt.figure(figsize=(7.8, 4.6))
    ax2 = fig2.gca()
    ax2.plot(sweeps_arr, fj_base_vals, '-o', color='#4C78A8', label='3GPP PF', linewidth=1.8, markersize=4)
    ax2.plot(sweeps_arr, fj_mr_vals, '-^', color='#E15759', label='3GPP MR', linewidth=1.8, markersize=4)
    ax2.plot(sweeps_arr, fj_rm_vals, '-s', color='#F58518', label='RadioMap', linewidth=1.8, markersize=4)
    ax2.set_xlabel('Number of UEs')
    ax2.set_ylabel("Jain's Fairness Index (0–1)")
    ax2.set_title(f"Jain's Fairness vs Number of UEs\n({scenario_name})")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax2.legend()
    fig2.tight_layout()
    p2 = os.path.join(out_dir_jain, f"{file_prefix}_jain_vs_nue_{timestamp}.png")
    fig2.savefig(p2, dpi=160, bbox_inches='tight')
    if args.show:
        plt.show()
    else:
        plt.close(fig2)

    print('Saved figures:')
    print(' -', p1)
    print(' -', p2)

    # Save raw data to CSV with timestamp
    csv_filename = f"{file_prefix}_data_{timestamp}.csv"
    csv_path = os.path.join(out_dir_jain, csv_filename)
    
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Add metadata header
            writer.writerow(['# Timestamp', timestamp])
            writer.writerow(['# Scenario', scenario_name])
            writer.writerow(['# Seeds', n_seeds])
            writer.writerow(['# Base Seed', base_seed])
            writer.writerow([]) # Empty line
            
            # Data header
            writer.writerow(['N_UE', 'SE_3GPP_PF', 'SE_3GPP_MR', 'SE_RadioMap', 'Jain_3GPP_PF', 'Jain_3GPP_MR', 'Jain_RadioMap'])
            for i, n in enumerate(sweeps):
                writer.writerow([
                    n,
                    f"{se_base_vals[i]:.6f}", f"{se_mr_vals[i]:.6f}", f"{se_rm_vals[i]:.6f}",
                    f"{fj_base_vals[i]:.6f}", f"{fj_mr_vals[i]:.6f}", f"{fj_rm_vals[i]:.6f}"
                ])
        print('Saved data log:')
        print(' -', csv_path)
    except Exception as e:
        print(f"Failed to save CSV log: {e}")


def main() -> None:
    _ensure_style()
    args = parse_args()
    
    # Ensure local code directory is prioritized
    code_path = os.path.abspath('code')
    if code_path not in sys.path:
        sys.path.insert(0, code_path)
        
    # Import the default config (as the base)
    from config import CONFIG as BASE_CONFIG_ORIG  # type: ignore

    # Build sweep list
    if args.nue_list:
        sweeps = _parse_list(args.nue_list)
    else:
        sweeps = list(range(int(args.nue_min), int(args.nue_max) + 1, int(args.nue_step)))

    # Determine which scenarios to run
    if args.scenario == 'all':
        scenarios_to_run = ['single', 'constellation']
    else:
        scenarios_to_run = [args.scenario]
    
    for scenario_key in scenarios_to_run:
        patch_file = SCENARIOS.get(scenario_key)
        if not patch_file:
            print(f"Error: Unknown scenario '{scenario_key}'")
            continue
            
        print(f"\n--- Setting up {scenario_key} scenario from {patch_file} ---")
        
        # Load the base config and apply patch
        scen_config = dict(BASE_CONFIG_ORIG)
        patch = load_config_patch(patch_file)
        if patch:
            scen_config.update(patch)
            print(f"Loaded {len(patch)} override keys.")
        else:
            print(f"Warning: Could not load patch from {patch_file}, using base config.")

        # Run the sweep for this scenario
        run_scenario_sweep(scenario_key, scen_config, args, sweeps)


if __name__ == '__main__':
    main()

