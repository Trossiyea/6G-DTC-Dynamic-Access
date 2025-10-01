#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scenario runner and visualizer for Baseline-Default vs RadioMap (RM).

Usage examples:
  python test/run_scenarios.py --scenario shanghai_single --power equal
  python test/run_scenarios.py --scenario shanghai_constellation --power waterfill

Env overrides:
  SEEDS=20 SEED_START=1 T=2000 N_UE=100 python test/run_scenarios.py ...
"""

import os
import sys
import json
import math
from copy import deepcopy
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Make code/ importable
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_BASE, 'code'))

from config import CONFIG  # type: ignore
from main import run_once, run_constellation  # type: ignore


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _load_overrides(name: str) -> Dict:
    here = os.path.dirname(os.path.abspath(__file__))
    mod_map = {
        'shanghai_single': 'configs.shanghai_single',
        'shanghai_constellation': 'configs.shanghai_constellation',
        'calgary_single': 'configs.calgary_single',
        'calgary_constellation': 'configs.calgary_constellation',
    }
    if name not in mod_map:
        raise SystemExit(f"Unknown scenario '{name}'.")
    mod = __import__(mod_map[name], fromlist=['overrides'])
    return dict(mod.overrides())


def _apply_power_model(cfg: Dict, power: str) -> Dict:
    power = str(power).lower()
    out = dict(cfg)
    if power == 'equal':
        out.update({
            'baseline_dl_power_model': 'equal_prb',
            'rm_dl_power_model': 'equal_prb',
            'baseline_P_tot_dbm': None,
            'rm_P_tot_dbm': None,
        })
    elif power == 'waterfill':
        out.update({
            'baseline_dl_power_model': 'waterfill',
            'rm_dl_power_model': 'waterfill',
            'baseline_P_tot_dbm': 50.0,
            'rm_P_tot_dbm': 50.0,
        })
        # Keep table min/max from base CONFIG to mimic real constraints
    else:
        raise SystemExit("--power must be 'equal' or 'waterfill'")
    return out


def _bootstrap_ci(a: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(0)
    if a.size == 0:
        return (math.nan, math.nan)
    n = a.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.mean(a[idx], axis=1)
    lo = np.percentile(means, 100.0 * (alpha / 2.0))
    hi = np.percentile(means, 100.0 * (1.0 - alpha / 2.0))
    return (float(lo), float(hi))


def _plot_gain_hist(imp: np.ndarray, out_dir: str):
    plt.figure(figsize=(6, 4))
    plt.hist(imp, bins=12, edgecolor='black')
    plt.title('Gain (RM vs Baseline-Default)')
    plt.xlabel('Gain (%)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'gain_histogram.png'), dpi=140)
    plt.close()


def _plot_se_vs_seed(base: np.ndarray, rm: np.ndarray, out_dir: str):
    x = np.arange(base.size)
    plt.figure(figsize=(7, 4))
    plt.plot(x, base, 'o-', label='Baseline-Default')
    plt.plot(x, rm, 's-', label='RadioMap')
    plt.xlabel('Seed index')
    plt.ylabel('Average SE (bits/s/Hz)')
    plt.title('Average SE vs seed')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'se_vs_seed.png'), dpi=140)
    plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, choices=['shanghai_single', 'shanghai_constellation', 'calgary_single', 'calgary_constellation'])
    ap.add_argument('--power', required=True, choices=['equal', 'waterfill'])
    args = ap.parse_args()

    # Env overrides
    N_SEEDS = int(os.getenv('SEEDS', '20'))
    SEED_START = int(os.getenv('SEED_START', '1'))
    T = int(os.getenv('T', '2000'))
    N_UE = int(os.getenv('N_UE', '100'))

    seeds = list(range(SEED_START, SEED_START + N_SEEDS))
    base_cfg = deepcopy(CONFIG)
    base_cfg.update(_load_overrides(args.scenario))
    # Core knobs aligned with test goal
    base_cfg.update({
        'use_mcs': True,
        'mcs_table_kind': '3gpp_table_2',
        'sched_require_contiguous': True,
        'N_UE': N_UE,
        'T': T,
        # Let runner own outputs
        'save_plots': False,
        'show_plots': False,
        'write_json_report': False,
    })
    base_cfg = _apply_power_model(base_cfg, args.power)

    # Output dir
    out_dir = _ensure_dir(os.path.join('test', 'out', args.scenario, args.power))

    vals_def: List[float] = []
    vals_rm: List[float] = []
    imps: List[float] = []

    # Per-seed JSON for constellation also saved
    for i, s in enumerate(seeds):
        cfg = deepcopy(base_cfg)
        cfg['seed'] = int(s)
        if bool(cfg.get('enable_constellation', False)):
            rep = run_constellation(cfg)
            # Save raw report per seed for constellation
            path = os.path.join(out_dir, f'constellation_summary_seed{s}.json')
            with open(path, 'w') as f:
                json.dump(rep, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
        else:
            rep = run_once(cfg)
            # Save raw report per seed (optional)
            path = os.path.join(out_dir, f'summary_seed{s}.json')
            with open(path, 'w') as f:
                json.dump(rep, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)

        vals_def.append(float(rep['avg_se_baseline_default']))
        vals_rm.append(float(rep['avg_se_radiomap']))
        imps.append((float(rep['avg_se_radiomap']) - float(rep['avg_se_baseline_default'])) / max(1e-9, float(rep['avg_se_baseline_default'])) * 100.0)

    base_arr = np.array(vals_def)
    rm_arr = np.array(vals_rm)
    imp_arr = np.array(imps)

    # Write CSV
    with open(os.path.join(out_dir, 'summary.csv'), 'w') as f:
        f.write('seed,avg_se_baseline_default,avg_se_radiomap,improvement_vs_default_pct\n')
        for i, s in enumerate(seeds):
            f.write(f"{s},{base_arr[i]:.6f},{rm_arr[i]:.6f},{imp_arr[i]:.2f}\n")

    # Aggregate JSON
    lo, hi = _bootstrap_ci(imp_arr, n_boot=2000, alpha=0.05)
    agg = {
        'scenario': args.scenario,
        'power_model': args.power,
        'seeds': {'start': SEED_START, 'count': N_SEEDS},
        'N_UE': N_UE,
        'T': T,
        'avg_se_baseline_default_mean': float(base_arr.mean()),
        'avg_se_baseline_default_std': float(base_arr.std()),
        'avg_se_radiomap_mean': float(rm_arr.mean()),
        'avg_se_radiomap_std': float(rm_arr.std()),
        'imp_vs_default_mean_pct': float(imp_arr.mean()),
        'imp_vs_default_std_pct': float(imp_arr.std()),
        'imp_vs_default_median_pct': float(np.median(imp_arr)),
        'imp_vs_default_p10_pct': float(np.percentile(imp_arr, 10)),
        'imp_vs_default_p90_pct': float(np.percentile(imp_arr, 90)),
        'imp_vs_default_bootstrap95_mean_ci_pct': [lo, hi],
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(agg, f, indent=2)

    # Plots
    _plot_gain_hist(imp_arr, out_dir)
    _plot_se_vs_seed(base_arr, rm_arr, out_dir)

    print(f"Done. Results written to: {out_dir}")


if __name__ == '__main__':
    main()
