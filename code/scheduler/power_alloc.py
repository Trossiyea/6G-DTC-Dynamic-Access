# -*- coding: utf-8 -*-
"""
DL power allocation strategies for NR-NTN scheduling.

Provides:
- Per-PRB water-filling with box constraints
- Group-level weighted water-filling for contiguous block allocation
- High-level power allocation dispatcher
"""

from typing import Dict, Optional, Tuple
import numpy as np


def waterfill_prb(
    channel_gains: np.ndarray,
    P_total: float,
    p_min: float = 0.0,
    p_max: float = float('inf'),
) -> np.ndarray:
    """
    Per-PRB water-filling power allocation with box constraints.

    Allocates total power P_total across PRBs to maximize sum capacity,
    subject to per-PRB power bounds [p_min, p_max].

    Uses bisection on the Lagrange multiplier (water level) with an
    active-set algorithm for saturated constraints.

    Args:
        channel_gains: Effective channel gain per PRB (a[i] = SNR_i / P_ref)
        P_total: Total power budget (mW or linear)
        p_min: Minimum power per PRB (mW)
        p_max: Maximum power per PRB (mW)

    Returns:
        Optimal power allocation per PRB (same shape as channel_gains)

    Notes:
        For SNR with reference power P_ref:
            a[i] = snr_true[i] / P_ref
        After allocation, scale SNR:
            snr_new[i] = snr_true[i] * (p[i] / P_ref)
    """
    a_vec = np.asarray(channel_gains, dtype=float)
    n = a_vec.size

    if n == 0:
        return np.array([], dtype=float)

    # Active set algorithm for box-constrained water-filling
    active = np.ones(n, dtype=bool)
    p = np.zeros(n, dtype=float)
    P_remain = float(P_total)

    max_iter = 100
    for iteration in range(max_iter):
        a_act = a_vec[active]
        if a_act.size == 0:
            break

        inv_a = 1.0 / np.maximum(a_act, 1e-30)

        # Bisection on water level nu: sum(max(0, 1/nu - 1/a)) = P_remain
        lo, hi = 1e-12, max(1.0, a_act.max() * 1e3)

        for _ in range(50):
            nu = (lo + hi) * 0.5
            p_tmp = np.maximum(0.0, 1.0 / nu - inv_a)
            s = p_tmp.sum()
            if s > P_remain:
                lo = nu
            else:
                hi = nu

        p_act = np.maximum(0.0, 1.0 / hi - inv_a)

        # Apply box constraints
        p_act = np.clip(p_act, p_min, p_max)

        # Update full vector
        p[:] = 0.0
        p[active] = p_act

        # Check total power
        total_alloc = p.sum()
        if abs(total_alloc - P_total) < 1e-6 * P_total:
            break

        # Handle saturated constraints
        if total_alloc > P_total + 1e-6:
            # Some at p_min saturated; remove from active and re-solve
            idxs = np.flatnonzero(active)
            sat_low = (p[idxs] <= p_min + 1e-12)
            if not np.any(sat_low):
                break
            active[idxs[sat_low]] = False
            P_remain = max(0.0, P_total - np.sum(p[idxs[sat_low]]))
            continue

        # total_alloc < P_total: some at p_max saturated
        residual = P_total - total_alloc
        if residual <= 1e-6:
            break

        idxs = np.flatnonzero(active)
        sat_high = (p[idxs] >= p_max - 1e-12)
        if not np.any(sat_high):
            # Distribute tiny residual equally
            p[idxs] += residual / float(len(idxs))
            break

        active[idxs[sat_high]] = False
        P_remain = residual

    return p


def waterfill_groups(
    group_gains: np.ndarray,
    group_sizes: np.ndarray,
    P_total: float,
    p_min: float = 0.0,
    p_max: float = float('inf'),
) -> np.ndarray:
    """
    Group-level weighted water-filling for contiguous block allocation.

    Each group (UE block) gets uniform power per PRB within the group.
    This preserves single-MCS per block and reduces EESM penalty.

    The objective is weighted capacity maximization:
        max sum_g k_g * log(1 + a_g * p_g)
    subject to sum_g k_g * p_g <= P_total and p_min <= p_g <= p_max

    Args:
        group_gains: Mean effective channel gain per group (a_bar per PRB)
        group_sizes: Number of PRBs in each group (k_g)
        P_total: Total power budget
        p_min: Minimum power per PRB
        p_max: Maximum power per PRB

    Returns:
        Power per PRB for each group
    """
    a_vec = np.asarray(group_gains, dtype=float)
    w_vec = np.asarray(group_sizes, dtype=float)

    if a_vec.size == 0:
        return np.array([], dtype=float)

    # Handle infeasible lower bound: if sum(k * p_min) > P, relax p_min
    sum_min = float(np.sum(w_vec) * p_min)
    p_min_eff = p_min
    if p_min > 0.0 and sum_min > P_total:
        p_min_eff = P_total / float(np.sum(w_vec))

    # Bisection on water level nu
    lo, hi = 1e-12, max(1.0, a_vec.max() * 1e3)

    def total_power(nu: float) -> float:
        p = np.maximum(0.0, 1.0 / nu - 1.0 / np.maximum(a_vec, 1e-30))
        if p_max < float('inf'):
            p = np.minimum(p, p_max)
        if p_min_eff > 0.0:
            p = np.maximum(p, p_min_eff)
        return float(np.sum(w_vec * p))

    # Bisection to find water level
    for _ in range(60):
        mid = (lo + hi) * 0.5
        s = total_power(mid)
        if s > P_total:
            lo = mid
        else:
            hi = mid

    # Final allocation
    nu = hi
    p = np.maximum(0.0, 1.0 / nu - 1.0 / np.maximum(a_vec, 1e-30))
    if p_max < float('inf'):
        p = np.minimum(p, p_max)
    if p_min_eff > 0.0:
        p = np.maximum(p, p_min_eff)

    # Normalize tiny residual due to clipping
    s = float(np.sum(w_vec * p))
    if s > 0 and abs(s - P_total) / max(P_total, 1e-12) > 1e-3:
        p *= (P_total / s)

    return p


def apply_dl_power_allocation(
    snr_true: np.ndarray,
    winners: np.ndarray,
    l_idx: np.ndarray,
    r_idx: np.ndarray,
    k_assigned: np.ndarray,
    dl_power_model: str,
    P_tot_dbm: Optional[float],
    P_ref_dbm: Optional[float],
    p_min_dbm: Optional[float],
    p_max_dbm: Optional[float],
    N_UE: int,
    Z: int,
) -> np.ndarray:
    """
    Apply DL power allocation to scale SNR based on power model.

    Supports:
    - 'equal_prb': No scaling (baseline)
    - 'waterfill': Group-level water-filling across UE blocks

    Args:
        snr_true: True SNR matrix [UE, Z]
        winners: PRB winners array [Z] (-1 for unassigned)
        l_idx: Left block boundary per UE
        r_idx: Right block boundary per UE
        k_assigned: Number of PRBs assigned per UE
        dl_power_model: Power model name
        P_tot_dbm: Total power budget in dBm
        P_ref_dbm: Reference power per PRB in dBm
        p_min_dbm: Min power per PRB in dBm
        p_max_dbm: Max power per PRB in dBm
        N_UE: Number of UEs
        Z: Number of PRBs

    Returns:
        Scaled SNR matrix [UE, Z]
    """
    snr_scaled = np.array(snr_true, copy=True)

    if str(dl_power_model).lower() != 'waterfill' or P_tot_dbm is None:
        return snr_scaled

    # Build blocks list
    blocks = []  # (ue, li, ri)
    for ue in range(N_UE):
        if int(k_assigned[ue]) <= 0:
            continue
        li, ri = int(l_idx[ue]), int(r_idx[ue])
        blocks.append((ue, li, ri))

    if not blocks:
        return snr_scaled

    # Convert power levels to linear
    P_ref_dbm_eff = float(P_ref_dbm) if P_ref_dbm is not None else 0.0
    P_ref_mW = 10.0 ** (P_ref_dbm_eff / 10.0)
    P_tot_mW = 10.0 ** (float(P_tot_dbm) / 10.0)
    pmin_mW = 0.0 if p_min_dbm is None else 10.0 ** (float(p_min_dbm) / 10.0)
    pmax_mW = float('inf') if p_max_dbm is None else 10.0 ** (float(p_max_dbm) / 10.0)

    # Build group gains and sizes
    a_g = []
    k_g = []
    for (ue, li, ri) in blocks:
        k = (ri - li + 1)
        P0 = P_ref_mW if P_ref_mW > 0.0 else 1.0
        a_vec = np.maximum(1e-12, snr_true[ue, li:ri + 1]) / P0
        a_g.append(float(np.mean(a_vec)))
        k_g.append(int(k))

    a_g = np.asarray(a_g, dtype=float)
    k_g = np.asarray(k_g, dtype=float)

    # Water-fill
    p_grp = waterfill_groups(a_g, k_g, P_tot_mW, pmin_mW, pmax_mW)

    # Apply per-UE uniform PRB power
    for idx, (ue, li, ri) in enumerate(blocks):
        P0 = P_ref_mW if P_ref_mW > 0.0 else 1.0
        scale = p_grp[idx] / P0
        snr_scaled[ue, li:ri + 1] = snr_true[ue, li:ri + 1] * scale

    return snr_scaled


def _dbm_to_mw(dbm: float) -> float:
    """Convert dBm to milliwatts."""
    return 10.0 ** (dbm / 10.0)
