# -*- coding: utf-8 -*-
"""
Radio Map–aware proportional fair scheduler with contiguous RB blocks.

This scheduler uses per-PRB Radio Map interference information to make
frequency-selective scheduling decisions, allocating contiguous RB blocks
to UEs using a greedy marginal-gain PF algorithm.

Features:
- Contiguous block allocation for single-MCS transmission
- EESM-based effective SINR for MCS selection
- Optional water-filling power allocation
- HARQ integration with retransmission priority
"""

from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

from csi import sinr_to_se_mcs
from core.capacity import _block_se_from_snr_vec
from harq import HarqManager
from link_adapt import re_per_prb_from_config
from .power_alloc import apply_dl_power_allocation, waterfill_groups


def pf_schedule_radiomap_blocks(
    cap: np.ndarray,
    T: int,
    beta: float,
    snr_lin: Optional[np.ndarray],
    overhead_eff: float,
    use_mcs: bool,
    power_split: bool,
    se_metric_override: Optional[np.ndarray],
    max_prbs_per_ue: Optional[int],
    mcs_params: Optional[Dict],
    se_metric_time: Optional[np.ndarray],
    snr_lin_time: Optional[np.ndarray],
    eesm_beta_db: float = 1.0,
    require_contiguous: bool = True,
    rng: Optional[np.random.Generator] = None,
    ue_mask_time: Optional[np.ndarray] = None,
    harq_mgr: Optional[HarqManager] = None,
    # DL power allocation
    dl_power_model: str = "equal_prb",
    P_tot_dbm: Optional[float] = None,
    P_ref_dbm: Optional[float] = None,
    p_min_dbm: Optional[float] = None,
    p_max_dbm: Optional[float] = None,
    # Recording options
    record_assignments: bool = False,
    assignments_out: Optional[list] = None,
    record_ue_thr: bool = False,
    ue_thr_out: Optional[list] = None,
    config: Optional[Dict] = None,
) -> float:
    """
    Radio Map–aware PF with contiguous RB blocks (single-MCS via EESM).

    Uses greedy marginal ΔSE allocation with power split awareness and
    optional water-filling power allocation.

    Args:
        cap: Per-PRB Shannon capacity [N_UE, Z]
        T: Number of TTIs to simulate
        beta: PF averaging factor
        snr_lin: Per-PRB linear SNR [N_UE, Z]
        overhead_eff: Overhead efficiency factor
        use_mcs: If True, use MCS mapping instead of Shannon
        power_split: If True, apply power split penalty
        se_metric_override: Override SE metric for scheduler decisions
        max_prbs_per_ue: Maximum PRBs per UE per TTI
        mcs_params: MCS parameters dict
        se_metric_time: Time-varying SE metric [T, N_UE, Z]
        snr_lin_time: Time-varying per-PRB SNR [T, N_UE, Z]
        eesm_beta_db: EESM beta parameter (dB)
        require_contiguous: If True, enforce contiguous block allocation
        rng: Random number generator
        ue_mask_time: UE scheduling mask per TTI [T, N_UE]
        harq_mgr: Optional HARQ manager
        dl_power_model: Power allocation model ('equal_prb', 'waterfill')
        P_tot_dbm: Total DL power budget (dBm)
        P_ref_dbm: Reference power per PRB (dBm)
        p_min_dbm: Minimum power per PRB (dBm)
        p_max_dbm: Maximum power per PRB (dBm)
        record_assignments: If True, record PRB assignments
        assignments_out: Output list for assignments
        record_ue_thr: If True, record per-UE throughput
        ue_thr_out: Output list for per-UE throughput
        config: Configuration dict

    Returns:
        Average sum spectral efficiency per PRB (bits/s/Hz)
    """
    cfg = config or {}
    harq_priority_bonus = float(cfg.get("harq_retx_priority_bonus", 0.0))
    re_per_prb_val: Optional[int] = None

    N_UE, Z = cap.shape
    rng = np.random.default_rng(0) if rng is None else rng

    def to_robust_sinr_db(arr_snr_lin: np.ndarray) -> np.ndarray:
        sinr_db = 10.0 * np.log10(np.maximum(arr_snr_lin, 1e-12))
        return sinr_db

    Rbar = np.full(N_UE, 1e-3)
    sum_rate = 0.0

    # Progress bar for TTI loop
    show_progress = bool(cfg.get("show_progress", True))
    pbar_iter = tqdm(
        range(T),
        desc="RadioMap",
        unit="TTI",
        disable=not show_progress,
        bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )

    for t_idx in pbar_iter:
        # HARQ feedback processing
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

        mask_t = None
        if ue_mask_time is not None:
            mask_t = np.asarray(ue_mask_time[t_idx], dtype=bool)

        # Prepare predicted per-PRB seed scores
        if se_metric_time is not None:
            se_pred_k1 = np.asarray(se_metric_time[t_idx], dtype=float)
            sinr_db_pred = None
        else:
            if snr_lin_time is not None:
                snr_mat = np.asarray(snr_lin_time[t_idx], dtype=float)
            elif snr_lin is not None:
                snr_mat = np.asarray(snr_lin, dtype=float)
            else:
                gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap)) - 1.0)
                snr_mat = gamma

            sinr_db_pred = to_robust_sinr_db(snr_mat)

            if use_mcs:
                table = mcs_params.get("mcs_table", "legacy") if mcs_params else "legacy"
                se_pred_k1 = sinr_to_se_mcs(sinr_db_pred, table=table)
            else:
                se_pred_k1 = np.log2(1.0 + snr_mat)

        # Apply override if provided
        if se_metric_time is None and se_metric_override is not None:
            se_pred_k1 = np.asarray(se_metric_override, dtype=float)

        # Initialize block tracking
        winners = np.full(Z, -1, dtype=int)
        l_idx = np.full(N_UE, -1, dtype=int)
        r_idx = np.full(N_UE, -1, dtype=int)
        k_assigned = np.zeros(N_UE, dtype=int)
        block_se_pred = np.zeros(N_UE, dtype=float)

        # Precompute PRB order per UE for fast seeding
        order_per_ue = np.argsort(-se_pred_k1, axis=1)
        ptr_per_ue = np.zeros(N_UE, dtype=int)

        def next_unassigned_best(ue: int) -> Optional[int]:
            ptr = int(ptr_per_ue[ue])
            ord_row = order_per_ue[ue]
            while ptr < ord_row.size:
                z = int(ord_row[ptr])
                if winners[z] < 0:
                    ptr_per_ue[ue] = ptr + 1
                    return z
                ptr += 1
            return None

        def can_grow_left(ue: int) -> bool:
            return l_idx[ue] > 0 and winners[l_idx[ue] - 1] < 0

        def can_grow_right(ue: int) -> bool:
            return r_idx[ue] >= 0 and r_idx[ue] < (Z - 1) and winners[r_idx[ue] + 1] < 0

        def pred_block_se(ue: int, li: int, ri: int) -> float:
            """Predicted per-PRB SE for UE over [li..ri]."""
            if se_metric_time is None and se_metric_override is None and sinr_db_pred is not None:
                snr_lin_vec = 10.0 ** (sinr_db_pred[ue, li:ri + 1] / 10.0)
                return _block_se_from_snr_vec(
                    snr_lin_vec,
                    ri - li + 1 if power_split else 1,
                    use_mcs,
                    mcs_params,
                    eesm_beta_db,
                )
            elif se_metric_time is not None:
                return float(np.mean(se_metric_time[t_idx, ue, li:ri + 1]))
            else:
                return float(np.mean(se_pred_k1[ue, li:ri + 1]))

        def apply_assign(ue: int, z: int) -> None:
            nonlocal winners, l_idx, r_idx, k_assigned, block_se_pred
            winners[z] = ue
            if k_assigned[ue] == 0:
                l_idx[ue] = r_idx[ue] = z
            else:
                if z == l_idx[ue] - 1:
                    l_idx[ue] = z
                elif z == r_idx[ue] + 1:
                    r_idx[ue] = z
                else:
                    l_idx[ue] = min(l_idx[ue], z) if l_idx[ue] >= 0 else z
                    r_idx[ue] = max(r_idx[ue], z) if r_idx[ue] >= 0 else z
            k_assigned[ue] += 1
            li, ri = int(l_idx[ue]), int(r_idx[ue])
            block_se_pred[ue] = pred_block_se(ue, li, ri)

        # HARQ retransmission pre-assignment
        if harq_mgr is not None and hasattr(harq_mgr, 'get_retx_requirements'):
            try:
                reqs = harq_mgr.get_retx_requirements()
                for ue_req, (li0, ri0) in reqs.items():
                    li0 = max(0, int(li0))
                    ri0 = min(Z - 1, int(ri0))
                    if mask_t is not None and not mask_t[ue_req]:
                        continue
                    if not harq_mgr.can_schedule(ue_req):
                        continue
                    can_assign = all(winners[zz] < 0 for zz in range(li0, ri0 + 1))
                    if not can_assign:
                        continue
                    for zz in range(li0, ri0 + 1):
                        winners[zz] = int(ue_req)
                    l_idx[ue_req] = li0
                    r_idx[ue_req] = ri0
                    k_assigned[ue_req] = ri0 - li0 + 1
                    block_se_pred[ue_req] = pred_block_se(int(ue_req), li0, ri0)
            except Exception:
                pass

        # Greedy allocation
        assigned_cnt = int(np.sum(winners >= 0))

        while assigned_cnt < Z:
            best_delta = -1e9
            best_action = None

            for ue in range(N_UE):
                if harq_mgr is not None and not harq_mgr.can_schedule(ue):
                    continue
                if mask_t is not None and not mask_t[ue]:
                    continue
                if max_prbs_per_ue is not None and k_assigned[ue] >= int(max_prbs_per_ue):
                    continue

                k0 = int(k_assigned[ue])

                # Seed action
                if k0 == 0:
                    z0 = next_unassigned_best(ue)
                    if z0 is not None:
                        se_new = float(se_pred_k1[ue, z0])
                        delta = se_new
                        metric = delta / Rbar[ue]
                        # Retransmission priority boost
                        if harq_mgr is not None and hasattr(harq_mgr, 'get_retx_ues'):
                            try:
                                _retx_mask = np.asarray(harq_mgr.get_retx_ues(), dtype=bool)
                                if _retx_mask[ue]:
                                    metric += harq_priority_bonus
                            except Exception:
                                pass
                        if metric > best_delta:
                            best_delta = metric
                            best_action = (ue, int(z0))
                    continue

                # Grow actions
                li, ri = int(l_idx[ue]), int(r_idx[ue])
                se_old = float(block_se_pred[ue])

                # Left growth
                if not require_contiguous or can_grow_left(ue):
                    zl = li - 1
                    if zl >= 0 and winners[zl] < 0:
                        se_new = pred_block_se(ue, zl, ri)
                        k_new = k0 + 1
                        delta = k_new * se_new - k0 * se_old
                        metric = delta / Rbar[ue]
                        if harq_mgr is not None and hasattr(harq_mgr, 'get_retx_ues'):
                            try:
                                _retx_mask = np.asarray(harq_mgr.get_retx_ues(), dtype=bool)
                                if _retx_mask[ue]:
                                    metric += harq_priority_bonus
                            except Exception:
                                pass
                        if metric > best_delta:
                            best_delta = metric
                            best_action = (ue, int(zl))

                # Right growth
                if not require_contiguous or can_grow_right(ue):
                    zr = ri + 1
                    if zr < Z and winners[zr] < 0:
                        se_new = pred_block_se(ue, li, zr)
                        k_new = k0 + 1
                        delta = k_new * se_new - k0 * se_old
                        metric = delta / Rbar[ue]
                        if harq_mgr is not None and hasattr(harq_mgr, 'get_retx_ues'):
                            try:
                                _retx_mask = np.asarray(harq_mgr.get_retx_ues(), dtype=bool)
                                if _retx_mask[ue]:
                                    metric += harq_priority_bonus
                            except Exception:
                                pass
                        if metric > best_delta:
                            best_delta = metric
                            best_action = (ue, int(zr))

            # Fallback
            if best_action is None:
                remaining = np.flatnonzero(winners < 0)
                if remaining.size == 0:
                    break
                z = int(remaining[0])
                cand = (
                    np.arange(N_UE)
                    if max_prbs_per_ue is None
                    else np.flatnonzero(k_assigned < int(max_prbs_per_ue))
                )
                if mask_t is not None:
                    cand = cand[mask_t[cand]]
                if cand.size == 0:
                    break
                ue = int(cand[np.argmax(se_pred_k1[cand, z] / Rbar[cand])])
                best_action = (ue, z)

            ue_sel, z_sel = best_action
            apply_assign(int(ue_sel), int(z_sel))
            assigned_cnt += 1

        # Record assignments
        if record_assignments and assignments_out is not None:
            try:
                assignments_out.append(np.array(winners, copy=True))
            except Exception:
                pass

        # Compute throughput
        if snr_lin_time is not None:
            snr_true = np.asarray(snr_lin_time[t_idx], dtype=float)
        elif snr_lin is not None:
            snr_true = np.asarray(snr_lin, dtype=float)
        else:
            gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap)) - 1.0)
            snr_true = gamma

        if harq_mgr is not None and hasattr(harq_mgr, 'on_scheduled_blocks'):
            # Apply power allocation and register HARQ TBs
            snr_scaled = _apply_power_and_register_harq(
                snr_true, winners, l_idx, r_idx, k_assigned, N_UE, Z,
                dl_power_model, P_tot_dbm, P_ref_dbm, p_min_dbm, p_max_dbm,
                eesm_beta_db, harq_mgr
            )
        else:
            # Legacy throughput accumulation
            snr_scaled = apply_dl_power_allocation(
                snr_true, winners, l_idx, r_idx, k_assigned,
                dl_power_model, P_tot_dbm, P_ref_dbm, p_min_dbm, p_max_dbm,
                N_UE, Z
            )

            thr_i = np.zeros(N_UE, dtype=float)
            for ue in range(N_UE):
                k0 = int(k_assigned[ue])
                if k0 <= 0:
                    continue
                li, ri = int(l_idx[ue]), int(r_idx[ue])
                snr_vec = snr_scaled[ue, li:ri + 1]
                k_power = 1 if str(dl_power_model).lower() in ('equal_prb', 'waterfill') else (k0 if power_split else 1)
                se_per_prb = _block_se_from_snr_vec(
                    snr_vec, k_power, use_mcs, mcs_params, eesm_beta_db
                )
                thr_i[ue] = (ri - li + 1) * se_per_prb * overhead_eff

            if record_ue_thr and ue_thr_out is not None:
                try:
                    ue_thr_out.append(np.array(thr_i, copy=True))
                except Exception:
                    pass

            sum_rate += thr_i.sum()
            Rbar = (1 - beta) * Rbar + beta * thr_i

            if harq_mgr is not None and hasattr(harq_mgr, 'on_scheduled'):
                scheduled = np.flatnonzero(k_assigned > 0)
                harq_mgr.on_scheduled(scheduled)

    avg_sum_rate_per_prb = sum_rate / (T * Z)
    return avg_sum_rate_per_prb


def _apply_power_and_register_harq(
    snr_true: np.ndarray,
    winners: np.ndarray,
    l_idx: np.ndarray,
    r_idx: np.ndarray,
    k_assigned: np.ndarray,
    N_UE: int,
    Z: int,
    dl_power_model: str,
    P_tot_dbm: Optional[float],
    P_ref_dbm: Optional[float],
    p_min_dbm: Optional[float],
    p_max_dbm: Optional[float],
    eesm_beta_db: float,
    harq_mgr,
) -> np.ndarray:
    """Apply power allocation and register HARQ transport blocks."""
    # Build full winners array
    winners_full = np.full(Z, -1, dtype=int)
    for ue in range(N_UE):
        if int(k_assigned[ue]) <= 0:
            continue
        li, ri = int(l_idx[ue]), int(r_idx[ue])
        winners_full[li:ri + 1] = ue

    snr_scaled = np.array(snr_true, copy=True)

    if str(dl_power_model).lower() == 'waterfill' and P_tot_dbm is not None:
        # Group-level water-filling
        blocks = []
        for ue in range(N_UE):
            if int(k_assigned[ue]) <= 0:
                continue
            li, ri = int(l_idx[ue]), int(r_idx[ue])
            blocks.append((ue, li, ri))

        if blocks:
            P_ref_dbm_eff = float(P_ref_dbm) if P_ref_dbm is not None else 0.0
            P_ref_mW = 10.0 ** (P_ref_dbm_eff / 10.0)
            P_tot_mW = 10.0 ** (float(P_tot_dbm) / 10.0)
            pmin_mW = 0.0 if p_min_dbm is None else 10.0 ** (float(p_min_dbm) / 10.0)
            pmax_mW = float('inf') if p_max_dbm is None else 10.0 ** (float(p_max_dbm) / 10.0)

            a_g = []
            k_g = []
            for ue, li, ri in blocks:
                k = ri - li + 1
                P0 = P_ref_mW if P_ref_mW > 0.0 else 1.0
                a_vec = np.maximum(1e-12, snr_true[ue, li:ri + 1]) / P0
                a_g.append(float(np.mean(a_vec)))
                k_g.append(int(k))

            a_g = np.asarray(a_g, dtype=float)
            k_g = np.asarray(k_g, dtype=float)

            p_grp = waterfill_groups(a_g, k_g, P_tot_mW, pmin_mW, pmax_mW)

            for idx, (ue, li, ri) in enumerate(blocks):
                P0 = P_ref_mW if P_ref_mW > 0.0 else 1.0
                scale = p_grp[idx] / P0
                snr_scaled[ue, li:ri + 1] = snr_true[ue, li:ri + 1] * scale

    # Register HARQ transport blocks
    sched_info: Dict[int, Dict] = {}
    for ue in range(N_UE):
        k0 = int(k_assigned[ue])
        if k0 <= 0:
            continue
        li, ri = int(l_idx[ue]), int(r_idx[ue])
        snr_vec_db = 10.0 * np.log10(np.maximum(snr_scaled[ue, li:ri + 1], 1e-12))
        sched_info[int(ue)] = {
            'sinr_vec_db': snr_vec_db,
            'n_prb': ri - li + 1,
            'eesm_beta_db': float(eesm_beta_db),
            'li': li,
            'ri': ri,
        }

    harq_mgr.on_scheduled_blocks(sched_info)
    return snr_scaled
