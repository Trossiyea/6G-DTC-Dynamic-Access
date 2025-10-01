#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radio Map resolution sensitivity experiment.

Runs Baseline-Default vs Radio Map (RM) across multiple Radio Map spatial
resolutions (e.g., 25x25, 35x35, 50x50) and compares improvement statistics.

Two modes to provide maps:
  1) --base-mat <path>: resample XY to requested sizes and write temporary
     MAT files under test/out/resolutions/.
  2) --mats <p1,p2,...>: comma-separated MAT file paths already at desired
     resolutions (skips resampling).

Outputs per scenario/power combo under test/out/rm_resolution/<scenario>/<power>/:
  - resolution_summary.csv: per-resolution mean/std/median/P10/P90 and CI
  - imp_vs_resolution.png: mean improvement vs resolution with error bars
  - gain_hist_res_<X>x<Y>.png: improvement histogram per resolution
  - (optional) per-seed CSVs under subfolders for debugging
"""

import os
import sys
import json
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# Import simulation modules
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_BASE, 'code'))
from config import CONFIG  # type: ignore
from main import run_once, run_constellation, load_radio_map_from_mat  # type: ignore


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _resample_xy_mw(P_mw: np.ndarray, X_new: int, Y_new: int) -> np.ndarray:
    """Bilinear separable interpolation on XY axes in linear mW domain.
    P_mw: [X,Y,Z] -> [X_new,Y_new,Z]
    """
    X0, Y0, Z = P_mw.shape
    if X_new == X0 and Y_new == Y0:
        return np.array(P_mw, copy=True)
    old_x = np.linspace(0.0, float(X0 - 1), num=X0)
    old_y = np.linspace(0.0, float(Y0 - 1), num=Y0)
    new_x = np.linspace(0.0, float(X0 - 1), num=int(X_new))
    new_y = np.linspace(0.0, float(Y0 - 1), num=int(Y_new))
    # First pass: X
    tmp = np.empty((int(X_new), Y0, Z), dtype=float)
    for y in range(Y0):
        for z in range(Z):
            tmp[:, y, z] = np.interp(new_x, old_x, P_mw[:, y, z])
    # Second pass: Y
    out = np.empty((int(X_new), int(Y_new), Z), dtype=float)
    for x in range(int(X_new)):
        for z in range(Z):
            out[x, :, z] = np.interp(new_y, old_y, tmp[x, :, z])
    return out


def _write_mat_mw(path: str, P_mw: np.ndarray, var: str = 'X_true') -> None:
    savemat(path, {var: P_mw})


def _power_overrides(power: str) -> Dict:
    p = str(power).lower()
    if p == 'equal':
        return {
            'baseline_dl_power_model': 'equal_prb',
            'rm_dl_power_model': 'equal_prb',
            'baseline_P_tot_dbm': None,
            'rm_P_tot_dbm': None,
        }
    if p == 'waterfill':
        return {
            'baseline_dl_power_model': 'waterfill',
            'rm_dl_power_model': 'waterfill',
            'baseline_P_tot_dbm': 50.0,
            'rm_P_tot_dbm': 50.0,
        }
    raise SystemExit("--power must be 'equal' or 'waterfill'")


def _scenario_overrides(name: str) -> Dict:
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


def _bootstrap_ci(a: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(0)
    n = a.size
    if n == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    m = np.mean(a[idx], axis=1)
    lo = np.percentile(m, 100.0 * (alpha / 2.0))
    hi = np.percentile(m, 100.0 * (1.0 - alpha / 2.0))
    return (float(lo), float(hi))


def _plot_imp_vs_resolution(labels: List[str], means: List[float], stds: List[float], out_dir: str):
    x = np.arange(len(labels))
    plt.figure(figsize=(7, 4))
    plt.errorbar(x, means, yerr=stds, fmt='o-', capsize=4)
    plt.xticks(x, labels)
    plt.ylabel('Improvement vs Baseline (%)')
    plt.title('RM gain vs Radio Map XY resolution')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'imp_vs_resolution.png'), dpi=140)
    plt.close()


def _plot_gain_hist(imp: np.ndarray, label: str, out_dir: str):
    plt.figure(figsize=(6, 4))
    plt.hist(imp, bins=12, edgecolor='black')
    plt.title(f'Gain dist @ {label}')
    plt.xlabel('Gain (%)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'gain_hist_{label.replace(" ", "_")}.png'), dpi=140)
    plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, choices=['shanghai_single', 'shanghai_constellation', 'calgary_single', 'calgary_constellation'])
    ap.add_argument('--power', required=True, choices=['equal', 'waterfill'])
    ap.add_argument('--sizes', default='25,35,50', help='Comma-separated XY sizes (square), e.g., 25,35,50')
    ap.add_argument('--base-mat', default=None, help='Base MAT to resample; if omitted, use CONFIG[radio_map_mat_path]')
    ap.add_argument('--mats', default=None, help='Comma-separated MAT paths (skip resampling)')
    args = ap.parse_args()

    # Seeds and scale from env
    N_SEEDS = int(os.getenv('SEEDS', '20'))
    SEED_START = int(os.getenv('SEED_START', '1'))
    T = int(os.getenv('T', '2000'))
    N_UE = int(os.getenv('N_UE', '100'))
    seeds = list(range(SEED_START, SEED_START + N_SEEDS))

    sizes = [int(s.strip()) for s in str(args.sizes).split(',') if s.strip()]

    # Output root
    out_root = _ensure_dir(os.path.join('test', 'out', 'rm_resolution', args.scenario, args.power))

    # Prepare maps
    maps: List[Tuple[str, str]] = []  # (label, path)
    if args.mats:
        for p in str(args.mats).split(','):
            p = p.strip()
            if not os.path.exists(p):
                raise SystemExit(f"MAT path does not exist: {p}")
            # Try to infer label from filename
            lab = os.path.splitext(os.path.basename(p))[0]
            maps.append((lab, p))
    else:
        base_mat = args.base_mat or CONFIG.get('radio_map_mat_path')
        if not base_mat or not os.path.exists(base_mat):
            raise SystemExit("Base MAT not found. Provide --base-mat or --mats.")
        # Load base as dBm then convert to mW
        R_dbm = load_radio_map_from_mat(base_mat, var_name=CONFIG.get('radio_map_mat_var', 'X_true'), units=CONFIG.get('radio_map_units', 'mW'))
        P_mw = 10.0 ** (R_dbm / 10.0)
        X0, Y0, Z0 = P_mw.shape
        res_dir = _ensure_dir(os.path.join(out_root, 'resampled'))
        for s in sizes:
            P_rs = _resample_xy_mw(P_mw, s, s)
            out_path = os.path.join(res_dir, f'R_{s}x{s}_Z{Z0}.mat')
            _write_mat_mw(out_path, P_rs, var=CONFIG.get('radio_map_mat_var', 'X_true'))
            maps.append((f'{s}x{s}', out_path))

    # Base config
    base_cfg = dict(CONFIG)
    base_cfg.update(_scenario_overrides(args.scenario))
    base_cfg.update(_power_overrides(args.power))
    base_cfg.update({
        'use_mcs': True,
        'mcs_table_kind': '3gpp_table_2',
        'sched_require_contiguous': True,
        'N_UE': N_UE,
        'T': T,
        'save_plots': False,
        'show_plots': False,
        'write_json_report': False,
    })

    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []

    # Main loop
    for label, mat_path in maps:
        print(f"Running resolution {label} with map: {mat_path}")
        # Per-resolution output dir
        out_dir = _ensure_dir(os.path.join(out_root, label))
        vals_def: List[float] = []
        vals_rm: List[float] = []
        imps: List[float] = []
        for i, s in enumerate(seeds):
            cfg = dict(base_cfg)
            cfg['seed'] = int(s)
            cfg['radio_map_mat_path'] = mat_path
            if bool(cfg.get('enable_constellation', False)):
                rep = run_constellation(cfg)
            else:
                rep = run_once(cfg)
            vals_def.append(float(rep['avg_se_baseline_default']))
            vals_rm.append(float(rep['avg_se_radiomap']))
            imps.append((float(rep['avg_se_radiomap']) - float(rep['avg_se_baseline_default'])) / max(1e-9, float(rep['avg_se_baseline_default'])) * 100.0)
        base_arr = np.array(vals_def)
        rm_arr = np.array(vals_rm)
        imp_arr = np.array(imps)
        # Save per-seed CSV
        with open(os.path.join(out_dir, 'summary.csv'), 'w') as f:
            f.write('seed,avg_se_baseline_default,avg_se_radiomap,improvement_vs_default_pct\n')
            for i, s in enumerate(seeds):
                f.write(f"{s},{base_arr[i]:.6f},{rm_arr[i]:.6f},{imp_arr[i]:.2f}\n")
        # Hist plot
        _plot_gain_hist(imp_arr, label, out_dir)
        # Aggregate metrics for top-level plot
        labels.append(label)
        means.append(float(imp_arr.mean()))
        stds.append(float(imp_arr.std()))

    # Overall plot across resolutions
    _plot_imp_vs_resolution(labels, means, stds, out_root)
    # Overall CSV
    with open(os.path.join(out_root, 'resolution_summary.csv'), 'w') as f:
        f.write('label,imp_mean_pct,imp_std_pct\n')
        for lab, m, s in zip(labels, means, stds):
            f.write(f"{lab},{m:.2f},{s:.2f}\n")

    print(f"Done. See results under: {out_root}")


if __name__ == '__main__':
    main()

