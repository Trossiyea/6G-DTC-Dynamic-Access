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
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


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


def _run_once_with(cfg_base: Dict, N_UE: int, seed: int, T_override: int | None) -> Tuple[float, float, float, float]:
    sys.path.append('code')
    from main import run_once  # type: ignore
    cfg = dict(cfg_base)
    cfg['N_UE'] = int(N_UE)
    cfg['seed'] = int(seed)
    if T_override is not None:
        cfg['T'] = int(T_override)
    rep = run_once(cfg)
    se_base = float(rep.get('avg_se_baseline_default', np.nan))
    se_rm = float(rep.get('avg_se_radiomap', np.nan))
    fj_b = rep.get('fairness_jain_base', None)
    fj_m = rep.get('fairness_jain_map', None)
    fj_b = float(fj_b) if fj_b is not None else np.nan
    fj_m = float(fj_m) if fj_m is not None else np.nan
    return se_base, se_rm, fj_b, fj_m


def _parse_list(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(',') if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot Average SE and Jain fairness vs number of UEs')
    g = p.add_argument_group('Sweep range')
    g.add_argument('--nue-list', type=str, default=None, help='Comma-separated list of N_UE values (overrides min/max/step)')
    g.add_argument('--nue-min', type=int, default=40, help='Minimum number of UEs (inclusive)')
    g.add_argument('--nue-max', type=int, default=150, help='Maximum number of UEs (inclusive)')
    g.add_argument('--nue-step', type=int, default=10, help='Step in UEs')
    g2 = p.add_argument_group('Averaging / speed')
    g2.add_argument('--n-seeds', type=int, default=1, help='Number of seeds to average over (>=1)')
    g2.add_argument('--base-seed', type=int, default=None, help='Base seed; when set, seeds are base, base+1, ...')
    g2.add_argument('--T', type=int, default=None, help='Override T (TTIs) for speed')
    out = p.add_argument_group('Output')
    out.add_argument('--outfile-prefix', type=str, default='kpis_vs_nue', help='Output file prefix under output/')
    out.add_argument('--show', action='store_true', help='Also show figures')
    return p.parse_args()


def main() -> None:
    _ensure_style()
    args = parse_args()
    sys.path.append('code')
    from config import CONFIG  # type: ignore

    # Build sweep list
    if args.nue_list:
        sweeps = _parse_list(args.nue_list)
    else:
        sweeps = list(range(int(args.nue_min), int(args.nue_max) + 1, int(args.nue_step)))

    # Prepare config baseline; ensure HARQ stays enabled to compute fairness from acked bits
    cfg_base = dict(CONFIG)
    out_dir = cfg_base.get('plot_dir', 'output')
    os.makedirs(out_dir, exist_ok=True)

    base_seed = int(args.base_seed) if args.base_seed is not None else int(cfg_base.get('seed', 101))
    n_seeds = max(1, int(args.n_seeds))

    se_base_vals: List[float] = []
    se_rm_vals: List[float] = []
    fj_base_vals: List[float] = []
    fj_rm_vals: List[float] = []

    print(f"Running sweep over N_UE = {sweeps} (seeds={n_seeds}, base_seed={base_seed}, T_override={args.T})")
    for N in sweeps:
        se_b_acc = []
        se_m_acc = []
        fj_b_acc = []
        fj_m_acc = []
        for i in range(n_seeds):
            seed_i = base_seed + i
            se_b, se_m, fj_b, fj_m = _run_once_with(cfg_base, N, seed_i, args.T)
            se_b_acc.append(se_b)
            se_m_acc.append(se_m)
            fj_b_acc.append(fj_b)
            fj_m_acc.append(fj_m)
        se_base_vals.append(float(np.nanmean(se_b_acc)))
        se_rm_vals.append(float(np.nanmean(se_m_acc)))
        fj_base_vals.append(float(np.nanmean(fj_b_acc)))
        fj_rm_vals.append(float(np.nanmean(fj_m_acc)))
        print(f"  N_UE={N:3d}: SE(base/RM)={se_base_vals[-1]:.3f}/{se_rm_vals[-1]:.3f}, Jain(base/RM)={fj_base_vals[-1]:.3f}/{fj_rm_vals[-1]:.3f}")

    sweeps_arr = np.array(sweeps, dtype=int)

    # Plot Average SE vs N_UE (two lines)
    fig1 = plt.figure(figsize=(7.8, 4.6))
    ax1 = fig1.gca()
    ax1.plot(sweeps_arr, se_base_vals, '-o', color='#4C78A8', label='Baseline-Default', linewidth=1.8, markersize=4)
    ax1.plot(sweeps_arr, se_rm_vals, '-s', color='#F58518', label='RadioMap', linewidth=1.8, markersize=4)
    ax1.set_xlabel('Number of UEs')
    ax1.set_ylabel('Average SE (bits/s/Hz)')
    ax1.set_title('Average Spectral Efficiency vs Number of UEs')
    ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax1.legend()
    fig1.tight_layout()
    p1 = os.path.join(out_dir, f"{args.outfile_prefix}_avg_se_vs_nue.png")
    fig1.savefig(p1, dpi=160, bbox_inches='tight')
    if args.show:
        plt.show()
    else:
        plt.close(fig1)

    # Plot Jain's fairness vs N_UE (two lines)
    fig2 = plt.figure(figsize=(7.8, 4.6))
    ax2 = fig2.gca()
    ax2.plot(sweeps_arr, fj_base_vals, '-o', color='#4C78A8', label='Baseline-Default', linewidth=1.8, markersize=4)
    ax2.plot(sweeps_arr, fj_rm_vals, '-s', color='#F58518', label='RadioMap', linewidth=1.8, markersize=4)
    ax2.set_xlabel('Number of UEs')
    ax2.set_ylabel("Jain's Fairness Index (0–1)")
    ax2.set_title("Jain's Fairness vs Number of UEs")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax2.legend()
    fig2.tight_layout()
    p2 = os.path.join(out_dir, f"{args.outfile_prefix}_jain_vs_nue.png")
    fig2.savefig(p2, dpi=160, bbox_inches='tight')
    if args.show:
        plt.show()
    else:
        plt.close(fig2)

    print('Saved figures:')
    print(' -', p1)
    print(' -', p2)


if __name__ == '__main__':
    main()

