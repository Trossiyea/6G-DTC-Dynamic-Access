# -*- coding: utf-8 -*-
"""
KPI analysis for Radio Map vs Baseline-Default.

Runs a single simulation (run_once) with analysis-friendly switches enabled
and prints:
  - Throughput: avg SE (baseline, RM) and improvement
  - HARQ reliability: ACK rate, first-try ACK, avg retrans/ACKed, drops
  - Per-UE goodput: mean/min/max, Jain fairness, improvements
  - MCS utilization: mean/quantiles, high-MCS fraction
  - Interference avoidance: mean interference on scheduled PRBs vs map median
  - Block quality consistency (approx.): within-block SINR(dB) variance

Notes
  - Interference/block metrics require PRB winners; we enable
    record_assignments for both baseline and RM.
  - Block SINR variance uses the static per-PRB SNR snapshot (snr_lin)
    as an approximation to avoid large time-series payloads.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional, List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from config import CONFIG as _CONFIG
from main import run_once


def _safe(arr):
    return np.asarray(arr, dtype=float) if arr is not None else None


def _harq_kpis(hs: Optional[Dict]) -> Dict[str, float]:
    out = {
        'tb_started': np.nan,
        'tb_acked': np.nan,
        'tb_dropped': np.nan,
        'ack_rate': np.nan,
        'first_try_ack_rate': np.nan,
        'avg_retx_per_acked': np.nan,
    }
    if not hs or not isinstance(hs, dict):
        return out
    ts = float(hs.get('tb_started', 0))
    ta = float(hs.get('tb_acked', 0))
    td = float(hs.get('tb_dropped', 0))
    fi = float(hs.get('initial_ack_count', 0))
    out.update({
        'tb_started': ts,
        'tb_acked': ta,
        'tb_dropped': td,
        'ack_rate': (ta / ts) if ts > 0 else np.nan,
        'first_try_ack_rate': (fi / ts) if ts > 0 else np.nan,
        'avg_retx_per_acked': float(hs.get('avg_retx_per_acked', np.nan)),
    })
    return out


def _mcs_stats(hs: Optional[Dict]) -> Dict[str, float]:
    out = {'mean_idx': np.nan, 'p50_idx': np.nan, 'p75_idx': np.nan, 'p90_idx': np.nan}
    if not hs or not isinstance(hs, dict):
        return out
    counts = hs.get('mcs_counts', {}) or {}
    if not counts:
        return out
    idx = np.array(sorted(int(k) for k in counts.keys()), dtype=float)
    cnt = np.array([int(counts[int(i)]) for i in idx], dtype=float)
    s = cnt.sum()
    if s <= 0:
        return out
    mean_idx = float((idx * cnt).sum() / s)
    cdf = np.cumsum(cnt) / s
    def q(p: float) -> float:
        j = np.searchsorted(cdf, p, side='left')
        j = min(j, idx.size - 1)
        return float(idx[j])
    out.update({'mean_idx': mean_idx, 'p50_idx': q(0.50), 'p75_idx': q(0.75), 'p90_idx': q(0.90)})
    return out


def _contiguous_segments(w: np.ndarray, ue: int) -> List[Tuple[int, int]]:
    seg = []
    z = 0
    Z = w.shape[0]
    while z < Z:
        if int(w[z]) == ue:
            l = z
            while z + 1 < Z and int(w[z + 1]) == ue:
                z += 1
            r = z
            seg.append((l, r))
        z += 1
    return seg


def _block_var_db(assignments: Optional[np.ndarray], snr_lin: np.ndarray) -> float:
    """Approximate within-block SINR(dB) variance using static snapshot snr_lin[UE,Z]."""
    if assignments is None:
        return np.nan
    T, Z = assignments.shape
    N_UE = snr_lin.shape[0]
    vals = []
    snr_db = 10.0 * np.log10(np.maximum(snr_lin, 1e-12))
    for t in range(T):
        w = assignments[t]
        for ue in range(N_UE):
            for l, r in _contiguous_segments(w, ue):
                seg = snr_db[ue, l:r + 1]
                if seg.size >= 2:
                    vals.append(float(np.var(seg)))
    if not vals:
        return np.nan
    return float(np.mean(vals))


def _assigned_interf_db(R_xyz_dbm: np.ndarray, ue_pos: np.ndarray, winners: Optional[np.ndarray]) -> float:
    if winners is None:
        return np.nan
    T, Z = winners.shape
    x_idx = ue_pos[:, 0].astype(int)
    y_idx = ue_pos[:, 1].astype(int)
    acc = []
    for z in range(Z):
        wz = winners[:, z].astype(int)
        rx = x_idx[wz]
        ry = y_idx[wz]
        rr = R_xyz_dbm[rx, ry, z]
        acc.append(rr)
    arr = np.concatenate([a.reshape(-1) for a in acc], axis=0)
    return float(np.mean(arr)) if arr.size > 0 else np.nan


def _global_map_median_db(R_xyz_dbm: np.ndarray) -> float:
    return float(np.median(R_xyz_dbm))


def analyze_and_print(config_overrides: Optional[Dict] = None) -> None:
    cfg = dict(_CONFIG)
    # Enable analysis aids
    cfg['record_assignments'] = True
    cfg['record_assignments_target'] = 'both'
    cfg['record_ue_thr'] = True
    cfg['write_json_report'] = False
    if config_overrides:
        cfg.update(config_overrides)

    rep = run_once(cfg)

    # Core SE
    se_base = float(rep.get('avg_se_baseline_default', np.nan))
    se_rm = float(rep.get('avg_se_radiomap', np.nan))
    imp_se = (se_rm - se_base) / max(1e-12, se_base) * 100.0

    # HARQ
    harq_b = _harq_kpis(rep.get('harq_stats_base'))
    harq_m = _harq_kpis(rep.get('harq_stats_map'))

    # Per-UE goodput and fairness
    per_base = rep.get('per_ue_avg_se_base') or rep.get('per_ue_se_base_avg')
    per_map = rep.get('per_ue_avg_se_map') or rep.get('per_ue_se_rm_avg')
    per_base = _safe(per_base)
    per_map = _safe(per_map)
    fair_b = rep.get('fairness_jain_base', None)
    fair_m = rep.get('fairness_jain_map', None)

    # MCS usage
    mcs_b = _mcs_stats(rep.get('harq_stats_base'))
    mcs_m = _mcs_stats(rep.get('harq_stats_map'))

    # Interference on scheduled PRBs
    R_xyz_dbm = _safe(rep.get('R_xyz_dbm'))
    ue_pos = _safe(rep.get('ue_pos')).astype(int) if rep.get('ue_pos') is not None else None
    winners_b = rep.get('assignments_base', None)
    winners_m = rep.get('assignments_rm', None)
    winners_b = _safe(winners_b).astype(int) if winners_b is not None else None
    winners_m = _safe(winners_m).astype(int) if winners_m is not None else None
    interf_b = _assigned_interf_db(R_xyz_dbm, ue_pos, winners_b) if R_xyz_dbm is not None and ue_pos is not None else np.nan
    interf_m = _assigned_interf_db(R_xyz_dbm, ue_pos, winners_m) if R_xyz_dbm is not None and ue_pos is not None else np.nan
    map_med = _global_map_median_db(R_xyz_dbm) if R_xyz_dbm is not None else np.nan

    # Block SINR variance (approx, static snapshot)
    snr_lin = _safe(rep.get('snr_lin'))
    var_b = _block_var_db(winners_b, snr_lin) if snr_lin is not None else np.nan
    var_m = _block_var_db(winners_m, snr_lin) if snr_lin is not None else np.nan

    # Print summary
    print('\n=== KPI Summary ===')
    print(f'Baseline-Default avg SE: {se_base:.3f} bits/s/Hz')
    print(f'RadioMap         avg SE: {se_rm:.3f} bits/s/Hz')
    print(f'Improvement vs Baseline: {imp_se:.2f}%')

    print('\n[HARQ] Baseline-Default:')
    print(f'  ACK rate: {harq_b["ack_rate"]*100.0:.1f}%  | first-try: {harq_b["first_try_ack_rate"]*100.0:.1f}%')
    print(f'  Avg retrans/ACKed: {harq_b["avg_retx_per_acked"]:.2f} | Dropped TBs: {harq_b["tb_dropped"]:.0f}')
    print('[HARQ] RadioMap:')
    print(f'  ACK rate: {harq_m["ack_rate"]*100.0:.1f}%  | first-try: {harq_m["first_try_ack_rate"]*100.0:.1f}%')
    print(f'  Avg retrans/ACKed: {harq_m["avg_retx_per_acked"]:.2f} | Dropped TBs: {harq_m["tb_dropped"]:.0f}')
    if np.isfinite(harq_b['ack_rate']) and np.isfinite(harq_m['ack_rate']):
        print(f'  ΔACK rate vs Baseline: {(harq_m["ack_rate"]-harq_b["ack_rate"]) * 100.0:+.2f}%')
    if np.isfinite(harq_b['tb_dropped']) and np.isfinite(harq_m['tb_dropped']):
        drop_red = (harq_b['tb_dropped'] - harq_m['tb_dropped']) / max(1e-9, harq_b['tb_dropped']) * 100.0
        print(f'  Drop reduction vs Baseline: {drop_red:+.2f}%')

    if per_base is not None and per_map is not None and per_base.size > 0 and per_map.size > 0:
        print('\n[Per-UE Goodput]')
        print(f'  Baseline mean/min/max: {per_base.mean():.3f}/{per_base.min():.3f}/{per_base.max():.3f}')
        print(f'  RadioMap mean/min/max: {per_map.mean():.3f}/{per_map.min():.3f}/{per_map.max():.3f}')
        m_imp = (per_map.mean() - per_base.mean()) / max(1e-12, per_base.mean()) * 100.0
        print(f'  Mean improvement vs Baseline: {m_imp:.2f}%')
    if fair_b is not None and fair_m is not None:
        print('\n[Fairness (Jain)]')
        print(f'  Baseline: {float(fair_b):.4f}  | RadioMap: {float(fair_m):.4f}')
        print(f'  ΔFairness vs Baseline: {float(fair_m) - float(fair_b):+.4f}')

    print('\n[MCS Utilization] (index stats)')
    print(f'  Baseline mean/p50/p75/p90: {mcs_b["mean_idx"]:.1f}/{mcs_b["p50_idx"]:.1f}/{mcs_b["p75_idx"]:.1f}/{mcs_b["p90_idx"]:.1f}')
    print(f'  RadioMap mean/p50/p75/p90: {mcs_m["mean_idx"]:.1f}/{mcs_m["p50_idx"]:.1f}/{mcs_m["p75_idx"]:.1f}/{mcs_m["p90_idx"]:.1f}')
    if np.isfinite(mcs_b['mean_idx']) and np.isfinite(mcs_m['mean_idx']):
        print(f'  ΔMean MCS idx vs Baseline: {mcs_m["mean_idx"]-mcs_b["mean_idx"]:+.2f}')

    if np.isfinite(interf_b) and np.isfinite(interf_m):
        print('\n[Interference on Assigned PRBs] (dBm)')
        print(f'  Baseline mean: {interf_b:.2f} dBm  | RadioMap mean: {interf_m:.2f} dBm')
        if np.isfinite(map_med):
            print(f'  Global map median: {map_med:.2f} dBm')
            print(f'  Δ vs median (Baseline/RM): {interf_b-map_med:+.2f} / {interf_m-map_med:+.2f} dB')
            print(f'  RM improvement (lower is better): {(interf_b - interf_m):+.2f} dB')

    if np.isfinite(var_b) and np.isfinite(var_m):
        print('\n[Block SINR variance] (approx, dB^2)')
        print(f'  Baseline: {var_b:.3f}  | RadioMap: {var_m:.3f}')
        if var_b > 0:
            print(f'  Reduction vs Baseline: {(1.0 - var_m/var_b)*100.0:+.2f}%')


if __name__ == '__main__':
    analyze_and_print()

