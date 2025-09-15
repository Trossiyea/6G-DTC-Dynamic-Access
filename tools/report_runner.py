#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formal report generator for NR-NTN DL (DtC) experiments.

Outputs to output/:
- gain_distribution_rm_vs_baseline.png
- rm_waterfill_vs_equal.png
- prb_assignment_heatmap_rm.png
- fd_time_summary.png, tau_time_summary.png
- formal_summary.csv (per-seed metrics and summary)

Default scenario (can override via env):
- N_UE=50, T=200, seeds=1..20
- use_mcs=True, 3GPP table_2, contiguous blocks
- enable_time_varying=True, enable_orbit_dynamics=True
- Interference heterogeneity: K_interferers=14, rm_flicker_db_std=2.5
- Residual Doppler fraction: 0.5
- Baseline: CSI delay=30 TTI, CQI periodicity=20 (quantized)
- RM: rm_delay=0, CQI periodicity off
- Power: Baseline equal_prb, RM waterfill(Ptot=50 dBm)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from config import CONFIG  # type: ignore
from main import run_once  # type: ignore


def ensure_out(dirp: str = 'output') -> str:
    os.makedirs(dirp, exist_ok=True)
    return dirp


def pct(a, p):
    return np.percentile(a, p, axis=1)


def main():
    out_dir = ensure_out(CONFIG.get('plot_dir', 'output'))

    N_UE = int(os.getenv('REP_N_UE', '50'))
    T = int(os.getenv('REP_T', '200'))
    N_SEEDS = int(os.getenv('REP_SEEDS', '20'))
    SEED_START = int(os.getenv('REP_SEED_START', '1'))
    seeds = list(range(SEED_START, SEED_START + N_SEEDS))

    base_cfg = deepcopy(CONFIG)
    base_cfg.update({
        'use_mcs': True,
        'mcs_3gpp_table_path': 'docs/mcs_tables_38_214.json',
        'mcs_table_kind': '3gpp_table_2',
        'sched_require_contiguous': True,
        'enable_time_varying': True,
        'enable_orbit_dynamics': True,
        'K_interferers': 14,
        'rm_flicker_db_std': 2.5,
        'baseline_csi_delay_ttis': 30,
        'rm_csi_delay_ttis': 0,
        'enable_cqi_periodicity': False,
        'enable_cqi_periodicity_base': True,
        'enable_cqi_periodicity_rm': False,
        'cqi_period_ttis': 20,
        'enable_cqi_quantization': True,
        'doppler_residual_fraction': 0.5,
        'baseline_dl_power_model': 'equal_prb',
        'rm_dl_power_model': 'waterfill',
        'rm_P_tot_dbm': 50.0,
        'N_UE': N_UE,
        'T': T,
        'save_plots': False,
        'show_plots': False,
    })

    print('Report scenario: N_UE={}, T={}, seeds={}..{}'.format(N_UE, T, seeds[0], seeds[-1]))

    base_def, rm_wf, rm_eq = [], [], []
    imp_wf, imp_eq = [], []
    seed0_assign = None
    seed0_fd = None
    seed0_tau = None
    per_ue_base = None
    per_ue_rm = None

    for i, s in enumerate(seeds):
        # RM waterfill
        cfg1 = deepcopy(base_cfg)
        cfg1['seed'] = int(s)
        cfg1['rm_dl_power_model'] = 'waterfill'
        cfg1['rm_P_tot_dbm'] = 50.0
        # Record assignments/time series on first seed
        if i == 0:
            cfg1['record_assignments'] = True
            cfg1['record_assignments_target'] = 'both'
            cfg1['record_ue_thr'] = True
        r1 = run_once(cfg1)

        # RM equal_prb
        cfg2 = deepcopy(base_cfg)
        cfg2['seed'] = int(s)
        cfg2['rm_dl_power_model'] = 'equal_prb'
        cfg2['rm_P_tot_dbm'] = None
        r2 = run_once(cfg2)

        base_def.append(float(r1['avg_se_baseline_default']))
        rm_wf.append(float(r1['avg_se_radiomap']))
        rm_eq.append(float(r2['avg_se_radiomap']))
        imp_wf.append((float(r1['avg_se_radiomap']) - float(r1['avg_se_baseline_default'])) / max(1e-9, float(r1['avg_se_baseline_default'])) * 100.0)
        imp_eq.append((float(r2['avg_se_radiomap']) - float(r2['avg_se_baseline_default'])) / max(1e-9, float(r2['avg_se_baseline_default'])) * 100.0)

        if i == 0:
            # Save PRB assignment heatmap and fd/τ time series
            A = r1.get('assignments_rm', None)
            seed0_assign = np.array(A) if A is not None else None
            seed0_fd = r1.get('fd_time', None)
            seed0_tau = r1.get('tau_time', None)
            per_ue_rm = np.array(r1.get('per_ue_se_rm_avg', []), dtype=float) if r1.get('per_ue_se_rm_avg') is not None else None
            per_ue_base = np.array(r1.get('per_ue_se_base_avg', []), dtype=float) if r1.get('per_ue_se_base_avg') is not None else None

    base_def = np.array(base_def)
    rm_wf = np.array(rm_wf)
    rm_eq = np.array(rm_eq)
    imp_wf = np.array(imp_wf)
    imp_eq = np.array(imp_eq)

    # 1) Gain distribution (waterfill RM vs baseline)
    plt.figure(figsize=(6,4))
    plt.hist(imp_wf, bins=12, edgecolor='black')
    plt.title('Gain (RM waterfill vs Baseline-Default)')
    plt.xlabel('Gain (%)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'gain_distribution_rm_vs_baseline.png'), dpi=140)
    plt.close()

    # 2) RM waterfill vs equal_prb comparison
    x = np.arange(len(seeds))
    plt.figure(figsize=(7,4))
    plt.plot(x, rm_eq, 'o-', label='RM equal_prb')
    plt.plot(x, rm_wf, 's-', label='RM waterfill')
    plt.xlabel('Seed index')
    plt.ylabel('Average SE (bits/s/Hz)')
    plt.title('RM: waterfill vs equal_prb by seed')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rm_waterfill_vs_equal.png'), dpi=140)
    plt.close()

    # 3) PRB assignment heatmap (RM, first seed)
    if seed0_assign is not None:
        plt.figure(figsize=(7,4))
        plt.imshow(seed0_assign.T, aspect='auto', origin='lower', interpolation='nearest')
        plt.colorbar(label='UE index')
        plt.xlabel('TTI')
        plt.ylabel('PRB index')
        plt.title('PRB assignment (RM, winners per PRB)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'prb_assignment_heatmap_rm.png'), dpi=140)
        plt.close()

    # 4) fd/τ time series (median with 10–90% bands, first seed)
    def plot_time_series_band(arr, title, ylabel, fname):
        if arr is None:
            return
        A = np.asarray(arr, dtype=float)  # [T,UE]
        T0 = A.shape[0]
        t = np.arange(T0)
        med = np.median(A, axis=1)
        p10 = np.percentile(A, 10, axis=1)
        p90 = np.percentile(A, 90, axis=1)
        plt.figure(figsize=(7,4))
        plt.plot(t, med, label='median')
        plt.fill_between(t, p10, p90, alpha=0.25, label='P10–P90')
        plt.xlabel('TTI')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=140)
        plt.close()

    plot_time_series_band(seed0_fd, 'Residual Doppler fd over time', 'fd (Hz)', 'fd_time_summary.png')
    plot_time_series_band(seed0_tau, 'Propagation delay τ over time', 'τ (s)', 'tau_time_summary.png')

    # 5) Per-UE SE comparison (RM vs Baseline-Default, first seed)
    if per_ue_base is not None and per_ue_rm is not None and per_ue_base.size > 0 and per_ue_rm.size > 0:
        idx = np.arange(per_ue_base.size)
        plt.figure(figsize=(9,4))
        plt.bar(idx - 0.2, per_ue_base, width=0.4, label='Baseline-Default')
        plt.bar(idx + 0.2, per_ue_rm, width=0.4, label='Radio Map')
        plt.xlabel('UE index')
        plt.ylabel('Average SE per UE (bits/s/Hz)')
        plt.title('Per-UE SE comparison (first seed)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'per_ue_se_comparison.png'), dpi=140)
        plt.close()

    # 6) Formal CSV summary
    out_csv = os.path.join(out_dir, 'formal_summary.csv')
    with open(out_csv, 'w') as f:
        f.write('seed,baseline_default,rm_equal,rm_waterfill,gain_rm_equal_pct,gain_rm_waterfill_pct\n')
        for i, s in enumerate(seeds):
            f.write(f'{s},{base_def[i]:.6f},{rm_eq[i]:.6f},{rm_wf[i]:.6f},{imp_eq[i]:.2f},{imp_wf[i]:.2f}\n')
        f.write('\n')
        f.write('summary,mean,std,median\n')
        f.write('baseline_default,{:.6f},{:.6f},{:.6f}\n'.format(base_def.mean(), base_def.std(), np.median(base_def)))
        f.write('rm_equal,{:.6f},{:.6f},{:.6f}\n'.format(rm_eq.mean(), rm_eq.std(), np.median(rm_eq)))
        f.write('rm_waterfill,{:.6f},{:.6f},{:.6f}\n'.format(rm_wf.mean(), rm_wf.std(), np.median(rm_wf)))
        f.write('gain_rm_equal_pct,{:.2f},{:.2f},{:.2f}\n'.format(imp_eq.mean(), imp_eq.std(), np.median(imp_eq)))
        f.write('gain_rm_waterfill_pct,{:.2f},{:.2f},{:.2f}\n'.format(imp_wf.mean(), imp_wf.std(), np.median(imp_wf)))

    print('Saved figures and CSV to', out_dir)


if __name__ == '__main__':
    main()
