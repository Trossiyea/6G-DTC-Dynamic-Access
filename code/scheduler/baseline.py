# -*- coding: utf-8 -*-
"""
Baseline 3GPP-like proportional fair scheduler with wideband CQI.

This scheduler uses a single wideband CQI report to make scheduling
decisions, assigning PRBs across the top sqrt(N) UEs per TTI to avoid
monopolization.
"""

from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

from core.capacity import (
    se_from_snr_with_split,
    se_from_cap_shannon_with_split,
    se_metric_strategy,
)
from link import HarqManager, re_per_prb_from_config


def pf_schedule_baseline(
    cap_wb: np.ndarray,
    Z: int,
    T: int,
    beta: float = 0.1,
    snr_lin_wb: Optional[np.ndarray] = None,
    overhead_eff: float = 1.0,
    use_mcs: bool = False,
    power_split: bool = False,
    mcs_params: Optional[Dict] = None,
    se_metric_time: Optional[np.ndarray] = None,
    snr_lin_wb_time: Optional[np.ndarray] = None,
    snr_lin_prb: Optional[np.ndarray] = None,
    cap_prb: Optional[np.ndarray] = None,
    snr_lin_time_prb: Optional[np.ndarray] = None,
    force_wideband_throughput: bool = False,
    ue_mask_time: Optional[np.ndarray] = None,
    harq_mgr: Optional[HarqManager] = None,
    config: Optional[Dict] = None,
) -> float:
    """
    3GPP-like baseline: proportional fair with wideband CQI.

    Assigns PRBs in each TTI across the top sqrt(N) UEs per PF metric,
    equally split, to avoid one-UE monopolization.

    Args:
        cap_wb: Wideband Shannon capacity per UE [N_UE]
        Z: Number of PRBs
        T: Number of TTIs to simulate
        beta: PF averaging factor (default 0.1)
        snr_lin_wb: Wideband linear SNR per UE [N_UE]
        overhead_eff: Overhead efficiency factor
        use_mcs: If True, use MCS mapping instead of Shannon
        power_split: If True, apply power split penalty
        mcs_params: MCS parameters dict
        se_metric_time: Time-varying SE metric [T, N_UE]
        snr_lin_wb_time: Time-varying wideband SNR [T, N_UE]
        snr_lin_prb: Per-PRB SNR [N_UE, Z]
        cap_prb: Per-PRB Shannon capacity [N_UE, Z]
        snr_lin_time_prb: Time-varying per-PRB SNR [T, N_UE, Z]
        force_wideband_throughput: If True, use only wideband metrics
        ue_mask_time: UE scheduling mask per TTI [T, N_UE]
        harq_mgr: Optional HARQ manager
        config: Configuration dict

    Returns:
        Average sum spectral efficiency per PRB (bits/s/Hz)
    """
    cfg = config or {}
    harq_priority_bonus = float(cfg.get("harq_retx_priority_bonus", 0.0))
    re_per_prb_val: Optional[int] = None

    N_UE = cap_wb.shape[0]

    # Use Shannon cap as metric by default; if use_mcs, convert to MCS SE
    if se_metric_time is None:
        metric_se = se_metric_strategy(
            use_mcs, snr_lin=snr_lin_wb, cap_shannon=cap_wb, mcs_params=mcs_params
        )

    Rbar = np.full(N_UE, 1e-3)
    sum_rate = 0.0

    # Progress bar for TTI loop
    show_progress = bool(cfg.get("show_progress", True))
    pbar_iter = tqdm(
        range(T),
        desc="Baseline",
        unit="TTI",
        disable=not show_progress,
        bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )

    for t_idx in pbar_iter:
        # Apply HARQ feedback and credit goodput
        if harq_mgr is not None:
            ack_bits = None
            try:
                ack_bits = harq_mgr.advance_time()
            except TypeError:
                ack_bits = None
            if ack_bits is not None:
                if re_per_prb_val is None:
                    re_per_prb_val = max(1, re_per_prb_from_config(cfg))
                thr_ack = np.asarray(ack_bits, dtype=float) / float(re_per_prb_val)
                sum_rate += float(np.sum(thr_ack))
                Rbar = (1 - beta) * Rbar + beta * thr_ack

        # Get metric for this TTI
        if se_metric_time is not None:
            metric_se_t = se_metric_time[t_idx]
        else:
            metric_se_t = metric_se

        metric = metric_se_t / Rbar

        # Apply UE mask if provided
        if ue_mask_time is not None:
            mask_t = np.asarray(ue_mask_time[t_idx], dtype=bool)
            blocked = ~mask_t
            if np.any(blocked):
                metric = np.array(metric, copy=True)
                metric[blocked] = -1e9

        # Apply HARQ gating
        if harq_mgr is not None:
            mask_h = np.array(
                [harq_mgr.can_schedule(u) for u in range(N_UE)], dtype=bool
            )
            if 'mask_t' not in locals() or ue_mask_time is None:
                metric = np.array(metric, copy=True)
            metric[~mask_h] = -1e9

        # Select top sqrt(N) UEs
        U_select = min(N_UE, max(3, int(np.sqrt(N_UE))))
        if U_select < N_UE:
            top_idx = np.argpartition(-metric, U_select - 1)[:U_select]
            top_idx = top_idx[np.argsort(-metric[top_idx])]
            selected = top_idx
        else:
            selected = np.argsort(-metric)

        # Filter by mask if provided
        if ue_mask_time is not None:
            mask_t = np.asarray(ue_mask_time[t_idx], dtype=bool)
            sel = [u for u in selected if mask_t[u]]
            if len(sel) == 0:
                sel = [selected[0]]
            selected = np.array(sel, dtype=int)
            U_select = len(selected)

        # Allocate PRBs equally across selected UEs
        alloc_counts = np.full(U_select, Z // U_select, dtype=int)
        remainder = Z - int(alloc_counts.sum())
        if remainder > 0:
            alloc_counts[:remainder] += 1

        # Build per-PRB winners by round-robin
        winners = np.full(Z, -1, dtype=int)
        rem = alloc_counts.copy()
        k_ptr = 0
        for z in range(Z):
            for _ in range(U_select):
                if rem[k_ptr] > 0:
                    winners[z] = selected[k_ptr]
                    rem[k_ptr] -= 1
                    k_ptr = (k_ptr + 1) % U_select
                    break
                k_ptr = (k_ptr + 1) % U_select
            if winners[z] < 0:
                winners[z] = selected[0]

        # Compute throughput
        thr_i = np.zeros(N_UE)
        if power_split:
            counts = np.bincount(winners, minlength=N_UE)
        else:
            counts = np.ones(N_UE, dtype=int)

        for z in range(Z):
            ue = winners[z]
            k_prb = int(counts[ue]) if power_split else 1

            if force_wideband_throughput:
                if (snr_lin_wb is not None) or (snr_lin_wb_time is not None):
                    snr_base = (
                        snr_lin_wb_time[t_idx, ue]
                        if snr_lin_wb_time is not None
                        else snr_lin_wb[ue]
                    )
                    se = se_from_snr_with_split(
                        snr_base, k_prb if power_split else 1, use_mcs, mcs_params
                    )
                else:
                    se = se_from_cap_shannon_with_split(
                        metric_se_t[ue], k_prb if power_split else 1
                    )
            else:
                # Prefer per-PRB SNR/SE if available
                if (snr_lin_prb is not None) or (snr_lin_time_prb is not None):
                    if snr_lin_time_prb is not None:
                        snr_base = snr_lin_time_prb[t_idx, ue, z]
                    else:
                        snr_base = snr_lin_prb[ue, z]
                    se = se_from_snr_with_split(
                        snr_base, k_prb if power_split else 1, use_mcs, mcs_params
                    )
                elif cap_prb is not None and not use_mcs:
                    se = se_from_cap_shannon_with_split(
                        cap_prb[ue, z], k_prb if power_split else 1
                    )
                else:
                    # Fallback to wideband
                    if (snr_lin_wb is not None) or (snr_lin_wb_time is not None):
                        snr_base = (
                            snr_lin_wb_time[t_idx, ue]
                            if snr_lin_wb_time is not None
                            else snr_lin_wb[ue]
                        )
                        se = se_from_snr_with_split(
                            snr_base, k_prb if power_split else 1, use_mcs, mcs_params
                        )
                    else:
                        se = se_from_cap_shannon_with_split(
                            metric_se_t[ue], k_prb if power_split else 1
                        )

            thr_i[ue] += se * overhead_eff

        sum_rate += thr_i.sum()
        Rbar = (1 - beta) * Rbar + beta * thr_i

        # Update HARQ after scheduling
        if harq_mgr is not None and hasattr(harq_mgr, 'on_scheduled'):
            harq_mgr.on_scheduled(np.unique(winners))

    # Tail flush: realize ACKs that arrive after last TTI
    if harq_mgr is not None and bool(cfg.get("harq_flush_tail", True)):
        try:
            D = int(cfg.get("harq_ack_delay_ttis", 0) or 0)
        except Exception:
            D = 0
        if D > 0:
            if re_per_prb_val is None:
                re_per_prb_val = max(1, re_per_prb_from_config(cfg))
            for s in range(1, D + 1):
                try:
                    ack_bits_tail = harq_mgr.advance_time()
                except TypeError:
                    ack_bits_tail = None
                if ack_bits_tail is not None:
                    thr_ack = np.asarray(ack_bits_tail, dtype=float) / float(
                        re_per_prb_val
                    )
                    sum_rate += float(np.sum(thr_ack))
                    Rbar = (1 - beta) * Rbar + beta * thr_ack

    avg_sum_rate_per_prb = sum_rate / (T * Z)
    return avg_sum_rate_per_prb
