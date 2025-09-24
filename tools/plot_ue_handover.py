#!/usr/bin/env python3
"""
Visualize handover timeline for a single UE from the constellation JSON report.

Reads the constellation_summary.json (or a provided JSON path), reconstructs the
serving satellite timeline for the chosen UE from event logs (or uses the
serving_trace if available), and plots a categorical step timeline with event
markers (attach/handover/outage-start/outage-end).

Usage examples:
  python tools/plot_ue_handover.py --ue 0 --show
  python tools/plot_ue_handover.py --report output/constellation_summary.json --ue 5 --outfile output/ue5_handover.png
  python tools/plot_ue_handover.py --unit sec --show
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_config():
    import sys
    sys.path.append('code')
    try:
        from config import CONFIG  # type: ignore
        return CONFIG
    except Exception:
        return {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot handover timeline for a single UE.')
    p.add_argument('--report', type=str, default='output/constellation_summary.json', help='Path to constellation JSON report.')
    p.add_argument('--ue', type=int, default=0, help='UE index to visualize.')
    p.add_argument('--unit', type=str, default='tti', choices=['tti', 'sec'], help='Time unit for x-axis (TTI or seconds).')
    p.add_argument('--outfile', type=str, default=None, help='Output PNG path (default: output/ue_<ue>_handover.png).')
    p.add_argument('--show', action='store_true', help='Show the plot window.')
    return p.parse_args()


def _reconstruct_serving_from_events(T: int, events: List[Dict]) -> np.ndarray:
    """Reconstruct piecewise-constant serving timeline from event list.
    Events are dicts with keys: 't', 'type' in {'attach','handover','outage_start','outage_end'}, 'to', 'from'.
    Returns array of length T with sat index, or -1 for outage.
    """
    serv = np.full(T, -1, dtype=int)
    if not events:
        return serv
    # Sort events by time
    ev = sorted(events, key=lambda e: int(e.get('t', 0)))
    curr = -1
    last_t = 0
    for e in ev:
        t = int(e.get('t', 0))
        t = max(0, min(T - 1, t))
        # Fill [last_t, t) with current
        if t > last_t:
            serv[last_t:t] = curr
        et = str(e.get('type', ''))
        if et == 'attach' or et == 'handover' or et == 'outage_end':
            curr = int(e.get('to', -1))
        elif et == 'outage_start':
            curr = -1
        last_t = t
    # Fill tail
    if last_t < T:
        serv[last_t:] = curr
    return serv


def _prepare_timeline(report: Dict, ue: int, unit: str, tti_ms: float) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    T = int(report.get('T', 0))
    if T <= 0:
        raise RuntimeError('Invalid or missing T in report.')
    # Prefer serving_trace if present
    serv = None
    if 'serving_trace' in report:
        try:
            st = np.asarray(report['serving_trace'], dtype=int)  # [T, N_UE]
            if st.ndim == 2 and 0 <= ue < st.shape[1]:
                serv = st[:, ue]
        except Exception:
            serv = None
    if serv is None:
        events_all = report.get('ho_events_per_ue', None)
        if not isinstance(events_all, list) or not (0 <= ue < len(events_all)):
            raise RuntimeError('No serving_trace and no ho_events_per_ue available for reconstruction.')
        serv = _reconstruct_serving_from_events(T, events_all[ue])

    x = np.arange(T, dtype=float)
    if unit == 'sec':
        x = x * max(1e-3, float(tti_ms) * 1e-3)
    name_map = {int(k): str(v) for (k, v) in (report.get('sat_index_to_name') or {}).items()}
    return x, np.asarray(serv, dtype=int), name_map


def _plot_handover(x: np.ndarray, serv: np.ndarray, name_map: Dict[int, str], ue: int, unit: str, events: List[Dict], outfile: str | None, show: bool):
    # Map satellite indices to categories; include outage (-1)
    unique_vals = [v for v in np.unique(serv) if v != -1]
    cats = [-1] + list(unique_vals)
    idx_map = {cats[i]: i for i in range(len(cats))}
    labels = ['Outage'] + [name_map.get(v, f'SAT-{int(v)}') for v in unique_vals]
    y = np.array([idx_map[int(v)] for v in serv])

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.step(x, y, where='post', color='#2a6f97', linewidth=1.5)
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time ({})'.format('s' if unit == 'sec' else 'TTI'))
    ax.set_title(f'UE {int(ue)} handover timeline')
    ax.grid(axis='x', color='#aaaaaa', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot event markers
    if events:
        for e in events:
            t = int(e.get('t', 0))
            xv = x[t if t < len(x) else -1]
            et = str(e.get('type', ''))
            color = {'attach': '#2ca02c', 'handover': '#d62728', 'outage_start': '#ff7f0e', 'outage_end': '#1f77b4'}.get(et, '#444444')
            ax.axvline(xv, color=color, linestyle=':', linewidth=1.0, alpha=0.8)
            # Small text label
            try:
                to = e.get('to', None)
                name = 'Outage' if (to is None or int(to) < 0) else name_map.get(int(to), f'SAT-{int(to)}')
                ax.text(xv, 0.95, f'{et}\n{name}', transform=ax.get_xaxis_transform(), fontsize=8, color=color, ha='center', va='top')
            except Exception:
                pass

    out = outfile or f'output/ue_{int(ue)}_handover.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f'Saved: {out}')


def main():
    args = _parse_args()
    if not os.path.exists(args.report):
        raise SystemExit(f'Report not found: {args.report}. Run constellation first or pass --report.')
    with open(args.report, 'r') as f:
        report = json.load(f)
    cfg = _load_config()
    tti_ms = float(cfg.get('tti_ms', 1.0))
    ue = int(args.ue)

    x, serv, name_map = _prepare_timeline(report, ue, args.unit, tti_ms)
    events_all = report.get('ho_events_per_ue', [])
    events = events_all[ue] if isinstance(events_all, list) and (0 <= ue < len(events_all)) else []
    _plot_handover(x, serv, name_map, ue, args.unit, events, args.outfile, args.show)


if __name__ == '__main__':
    main()

