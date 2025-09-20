#!/usr/bin/env python3
"""
Visualize key KPIs from a single simulation run:
- UE average throughput (Mbps)
- Average SE (bits/s/Hz)
- Jain's fairness index
- ACK rate (%)

References the repository config (code/config.py) and executes run_once from code/main.py.
Each KPI is saved as a separate figure under output/ by default.

Examples:
  python tools/plot_kpis.py                      # use CONFIG as-is
  python tools/plot_kpis.py --outfile-prefix run # customize output prefix
  python tools/plot_kpis.py --show               # also show figures
  python tools/plot_kpis.py --no-sort            # keep UE order (no sorting)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_config_and_run() -> Tuple[Dict, Dict]:
    sys.path.append('code')
    from config import CONFIG  # type: ignore
    from main import run_once  # type: ignore
    # Ensure HARQ / per-UE metrics are available
    cfg = dict(CONFIG)
    cfg.setdefault('record_assignments', False)
    cfg.setdefault('record_assignments_target', 'rm')
    cfg.setdefault('record_ue_thr', False)
    # Single run
    rep = run_once(cfg)
    return cfg, rep


def _total_bw_hz(cfg: Dict) -> float:
    # PRB bandwidth from explicit value or SCS (PRB = 12 subcarriers)
    if 'prb_bw_hz' in cfg and cfg['prb_bw_hz'] is not None:
        prb_bw_hz = float(cfg['prb_bw_hz'])
    else:
        scs_khz = float(cfg.get('scs_khz', 30.0) or 30.0)
        prb_bw_hz = 12.0 * scs_khz * 1e3
    Z = int(cfg.get('Z', 51))
    return prb_bw_hz * Z


def _bar(ax, labels, values, title: str, ylabel: str):
    x = np.arange(len(labels))
    colors = ['#4C78A8', '#F58518', '#54A24B']
    ax.bar(x, values, color=[colors[i % len(colors)] for i in range(len(values))], edgecolor='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.5)


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
    })


def plot_kpis(outfile_prefix: str, show: bool, sort_ues: bool) -> None:
    cfg, rep = _load_config_and_run()

    out_dir = cfg.get('plot_dir', 'output')
    os.makedirs(out_dir, exist_ok=True)

    # 1) UE average throughput (Mbps) — UE index on x-axis
    se_b = rep.get('per_ue_avg_se_base') or rep.get('per_ue_se_base_avg')
    se_m = rep.get('per_ue_avg_se_map') or rep.get('per_ue_se_rm_avg')
    se_b = np.asarray(se_b, dtype=float) if se_b is not None else None
    se_m = np.asarray(se_m, dtype=float) if se_m is not None else None
    B_tot = _total_bw_hz(cfg)
    # Throughput = SE * total bandwidth
    thr_b = se_b * B_tot / 1e6 if se_b is not None else None  # Mbps
    thr_m = se_m * B_tot / 1e6 if se_m is not None else None

    if thr_b is not None and thr_m is not None:
        N = max(thr_b.size, thr_m.size)
        ue_idx = np.arange(N)
        # Align sizes if needed
        if thr_b.size != N:
            thr_b = np.pad(thr_b, (0, N - thr_b.size), constant_values=np.nan)
        if thr_m.size != N:
            thr_m = np.pad(thr_m, (0, N - thr_m.size), constant_values=np.nan)
        # Optional sorting by RadioMap throughput
        order = np.argsort(thr_m) if sort_ues else ue_idx
        fig1 = plt.figure(figsize=(10.5, 4.2))
        ax1 = fig1.gca()
        ax1.plot(ue_idx, thr_b[order], label='Baseline-Default', color='#4C78A8', marker='o', markersize=3, linewidth=1.2, alpha=0.9)
        ax1.plot(ue_idx, thr_m[order], label='RadioMap', color='#F58518', marker='s', markersize=3, linewidth=1.2, alpha=0.9)
        ax1.set_xlabel('UE index' + (' (sorted by RadioMap throughput)' if sort_ues else ''))
        ax1.set_ylabel('Avg throughput per UE (Mbps)')
        ax1.set_title('UE Average Throughput')
        ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
        ax1.legend()
        fig1.tight_layout()
        p1 = os.path.join(out_dir, f"{outfile_prefix}_ue_throughput_mbps.png")
        fig1.savefig(p1, dpi=160, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig1)

    # 2) Average SE (global) — scheduler on x-axis
    se_base = float(rep.get('avg_se_baseline_default', np.nan))
    se_rm = float(rep.get('avg_se_radiomap', np.nan))
    fig2 = plt.figure(figsize=(5.4, 4.2))
    ax2 = fig2.gca()
    _bar(ax2, ['Baseline-Default', 'RadioMap'], [se_base, se_rm], title='Average Spectral Efficiency', ylabel='Avg SE (bits/s/Hz)')
    fig2.tight_layout()
    p2 = os.path.join(out_dir, f"{outfile_prefix}_avg_se.png")
    fig2.savefig(p2, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig2)

    # 3) Jain's fairness index — scheduler on x-axis
    fj_b = rep.get('fairness_jain_base', None)
    fj_m = rep.get('fairness_jain_map', None)
    fj_b = float(fj_b) if fj_b is not None else np.nan
    fj_m = float(fj_m) if fj_m is not None else np.nan
    fig3 = plt.figure(figsize=(5.4, 4.2))
    ax3 = fig3.gca()
    _bar(ax3, ['Baseline-Default', 'RadioMap'], [fj_b, fj_m], title="Jain's Fairness Index", ylabel='Jain index (0–1)')
    ax3.set_ylim(0.0, 1.05)
    fig3.tight_layout()
    p3 = os.path.join(out_dir, f"{outfile_prefix}_jain_fairness.png")
    fig3.savefig(p3, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig3)

    # 4) ACK rate — scheduler on x-axis
    def _ack_rate(hs: Optional[Dict]) -> float:
        if not hs or not isinstance(hs, dict):
            return float('nan')
        ts = float(hs.get('tb_started', hs.get('initial_ack_count', 0) + hs.get('initial_nack_count', 0) or 0))
        ta = float(hs.get('tb_acked', hs.get('ack_count', 0) or 0))
        return (ta / ts) * 100.0 if ts > 0 else float('nan')
    ack_b = _ack_rate(rep.get('harq_stats_base'))
    ack_m = _ack_rate(rep.get('harq_stats_map'))
    fig4 = plt.figure(figsize=(5.4, 4.2))
    ax4 = fig4.gca()
    _bar(ax4, ['Baseline-Default', 'RadioMap'], [ack_b, ack_m], title='ACK Rate', ylabel='ACK rate (%)')
    ax4.set_ylim(0.0, 100.0)
    fig4.tight_layout()
    p4 = os.path.join(out_dir, f"{outfile_prefix}_ack_rate.png")
    fig4.savefig(p4, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig4)

    print('Saved figures:')
    for p in (p1 if 'p1' in locals() else None, p2, p3, p4):
        if p:
            print(' -', p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Visualize key KPIs for a single simulation run.')
    p.add_argument('--outfile-prefix', type=str, default='kpis', help='Output file prefix, saved under output/.')
    p.add_argument('--show', action='store_true', help='Also show the figures')
    p.add_argument('--no-sort', action='store_true', help='Do not sort UEs by RadioMap throughput')
    return p.parse_args()


def main() -> None:
    _ensure_style()
    args = parse_args()
    plot_kpis(outfile_prefix=args.outfile_prefix, show=args.show, sort_ues=(not args.no_sort))


if __name__ == '__main__':
    main()

