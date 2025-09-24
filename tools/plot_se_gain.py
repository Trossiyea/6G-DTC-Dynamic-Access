#!/usr/bin/env python3
"""
Visualize SE gain of Radio Map over Baseline-Default from the simulation report.

Supports constellation report (with per-satellite KPIs) and single-run report.

Figures produced (as available):
  1) Overall SE bars: Baseline-Default vs Radio Map
  2) Per-satellite gain histogram (constellation only)
  3) Top-K satellites by gain (constellation only)

Usage examples:
  python tools/plot_se_gain.py --show
  python tools/plot_se_gain.py --report output/constellation_summary.json --top-k 20 --outfile-prefix output/se_gain
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot SE gain of Radio Map over Baseline-Default.')
    p.add_argument('--report', type=str, default='output/constellation_summary.json', help='Path to JSON report.')
    p.add_argument('--top-k', type=int, default=20, help='Top-K satellites for bar chart (constellation).')
    p.add_argument('--outfile-prefix', type=str, default='output/se_gain', help='Prefix for output PNGs.')
    p.add_argument('--show', action='store_true', help='Show plot windows.')
    return p.parse_args()


def _maybe_mkdir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _overall_fig(baseline: float, radiomap: float, outpath: str, show: bool):
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    xs = np.arange(2)
    vals = [baseline, radiomap]
    colors = ['#7aa6c2', '#d1495b']
    ax.bar(xs, vals, color=colors, width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(['Baseline-Default', 'Radio Map'])
    ax.set_ylabel('Avg SE (bits/s/Hz)')
    gain = (radiomap - baseline) / max(1e-9, baseline) * 100.0 if baseline > 0 else float('inf')
    ax.set_title(f'Overall SE gain: {gain:.2f}%')
    plt.tight_layout()
    _maybe_mkdir(outpath)
    plt.savefig(outpath, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f'Saved: {outpath}')


def _per_sat_figs(per_sat: List[Dict], name_map: Dict[int, str], top_k: int, prefix: str, show: bool):
    # Build arrays
    base = []
    rm = []
    names = []
    for k in per_sat:
        b = float(k.get('avg_se_base_default_per_prb', 0.0))
        r = float(k.get('avg_se_rm_per_prb', 0.0))
        base.append(b)
        rm.append(r)
        nm = k.get('name') or name_map.get(int(k.get('sat_index', -1)), f"SAT-{int(k.get('sat_index', -1))}")
        names.append(str(nm))
    base = np.asarray(base, dtype=float)
    rm = np.asarray(rm, dtype=float)
    gain = (rm - base) / np.maximum(1e-9, base) * 100.0

    # Histogram of per-sat gains
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.6))
    ax1.hist(gain[np.isfinite(gain)], bins=20, edgecolor='black')
    ax1.set_xlabel('Gain vs Baseline-Default (%)')
    ax1.set_ylabel('Satellite count')
    ax1.set_title('Per-satellite SE gain distribution')
    plt.tight_layout()
    out1 = f"{prefix}_per_sat_hist.png"
    _maybe_mkdir(out1)
    plt.savefig(out1, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig1)
    print(f'Saved: {out1}')

    # Top-K satellites by gain
    idx = np.argsort(-gain)  # descending
    k = min(top_k, idx.size)
    sel = idx[:k]
    fig2, ax2 = plt.subplots(figsize=(max(5.0, 0.35 * k + 3.0), 4.0))
    y = np.arange(k)
    ax2.barh(y, gain[sel], color='#d1495b', alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels([names[i] for i in sel])
    ax2.invert_yaxis()
    ax2.set_xlabel('Gain vs Baseline-Default (%)')
    ax2.set_title(f'Top-{k} satellites by SE gain')
    plt.tight_layout()
    out2 = f"{prefix}_per_sat_top{int(k)}.png"
    _maybe_mkdir(out2)
    plt.savefig(out2, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig2)
    print(f'Saved: {out2}')


def main():
    args = _parse_args()
    if not os.path.exists(args.report):
        raise SystemExit(f'Report not found: {args.report}')
    with open(args.report, 'r') as f:
        report = json.load(f)

    # Overall
    base = report.get('avg_se_baseline_default')
    rm = report.get('avg_se_radiomap')
    if base is None or rm is None:
        raise SystemExit('Report missing overall SE fields. Ensure you ran with constellation or single-run outputs.')
    _overall_fig(float(base), float(rm), f"{args.outfile_prefix}_overall.png", args.show)

    # Per-satellite (constellation only)
    per_sat = report.get('per_sat_kpis', None)
    if isinstance(per_sat, list) and len(per_sat) > 0:
        name_map = {int(k): str(v) for (k, v) in (report.get('sat_index_to_name') or {}).items()}
        _per_sat_figs(per_sat, name_map, int(args.top_k), args.outfile_prefix, args.show)
    else:
        print('No per-satellite KPIs in report; skipped per-sat plots.')


if __name__ == '__main__':
    main()

