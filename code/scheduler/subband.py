# -*- coding: utf-8 -*-
"""
Subband-level baseline proportional fair scheduler.

This scheduler splits the bandwidth into S groups (subbands) and uses
subband-level CQI reports for scheduling decisions. It supports both
simple per-group PF allocation and marginal ΔSE allocation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from link import sinr_to_se_mcs, effective_sinr_eesm
from core.capacity import _block_se_from_snr_vec


def group_ranges(Z: int, num_groups: int) -> List[Tuple[int, int]]:
    """
    Split Z PRBs into num_groups contiguous groups.

    The groups are sized as evenly as possible, with earlier groups
    receiving extra PRBs when Z is not evenly divisible.

    Args:
        Z: Total number of PRBs
        num_groups: Number of groups to create

    Returns:
        List of (start, end) tuples for each group (inclusive bounds)
    """
    num_groups = max(1, int(num_groups))
    base = Z // num_groups
    rem = Z % num_groups
    ranges = []
    start = 0
    for g in range(num_groups):
        size = base + (1 if g < rem else 0)
        end = start + size - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def pf_schedule_baseline_subband(
    Z: int,
    T: int,
    beta: float,
    overhead_eff: float,
    use_mcs: bool,
    power_split: bool,
    mcs_params: Optional[Dict],
    num_groups: int,
    eesm_beta_db: float = 1.0,
    max_groups_per_ue: Optional[int] = None,
    use_marginal_delta: bool = True,
    # Inputs for scheduling metric and throughput
    snr_lin_prb: Optional[np.ndarray] = None,
    cap_prb: Optional[np.ndarray] = None,
    snr_lin_time_prb_metric: Optional[np.ndarray] = None,
    snr_lin_time_prb_true: Optional[np.ndarray] = None,
    se_metric_time_subband: Optional[np.ndarray] = None,
) -> float:
    """
    Baseline scheduler with subband-level CQI.

    Splits Z PRBs into S groups, computes per-group SE metric
    (EESM+MCS if use_mcs else Shannon mean), then PF-allocates groups
    each TTI. Each assigned group uses a single MCS evaluated over
    the group's PRBs.

    Args:
        Z: Number of PRBs
        T: Number of TTIs
        beta: PF averaging factor
        overhead_eff: Overhead efficiency factor
        use_mcs: If True, use MCS mapping
        power_split: If True, apply power split penalty
        mcs_params: MCS parameters dict
        num_groups: Number of subbands
        eesm_beta_db: EESM beta parameter (dB)
        max_groups_per_ue: Maximum groups per UE per TTI
        use_marginal_delta: If True, use marginal ΔSE allocation
        snr_lin_prb: Static per-PRB SNR [N_UE, Z]
        cap_prb: Static per-PRB Shannon capacity [N_UE, Z]
        snr_lin_time_prb_metric: Time-varying metric SNR [T, N_UE, Z]
        snr_lin_time_prb_true: Time-varying true SNR [T, N_UE, Z]
        se_metric_time_subband: Precomputed subband metric [T, N_UE, S]

    Returns:
        Average sum SE per PRB (bits/s/Hz)
    """
    # Build groups
    groups = group_ranges(Z, num_groups)
    S = len(groups)

    def _metric_from_snr_mat(snr_mat: np.ndarray) -> np.ndarray:
        """Compute subband metric from per-PRB SNR matrix."""
        out = np.zeros((snr_mat.shape[0], S), dtype=float)
        for gi, (li, ri) in enumerate(groups):
            vec = snr_mat[:, li:ri + 1]
            if use_mcs:
                sinr_db_vec = 10.0 * np.log10(np.maximum(vec, 1e-12))
                sinr_eff_db = effective_sinr_eesm(
                    sinr_db_vec, beta_db=float(eesm_beta_db), axis=-1
                )
                table = mcs_params.get("mcs_table", "legacy") if mcs_params else "legacy"
                out[:, gi] = sinr_to_se_mcs(sinr_eff_db, table=table)
            else:
                out[:, gi] = np.log2(1.0 + np.maximum(vec, 0.0)).mean(axis=1)
        return out

    # Prepare static subband metric if no time series given
    se_metric_sub = None
    if se_metric_time_subband is None:
        if snr_lin_prb is not None:
            se_metric_sub = _metric_from_snr_mat(np.asarray(snr_lin_prb, dtype=float))
        elif cap_prb is not None and not use_mcs:
            se_metric_sub = np.zeros((cap_prb.shape[0], S), dtype=float)
            for gi, (li, ri) in enumerate(groups):
                se_metric_sub[:, gi] = np.asarray(cap_prb[:, li:ri + 1]).mean(axis=1)
        else:
            if cap_prb is None:
                raise ValueError(
                    "pf_schedule_baseline_subband requires snr_lin_prb or cap_prb"
                )
            gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap_prb)) - 1.0)
            se_metric_sub = _metric_from_snr_mat(gamma)

    # Resolve N_UE
    if se_metric_time_subband is not None:
        N_UE = se_metric_time_subband.shape[1]
    elif snr_lin_prb is not None:
        N_UE = int(np.asarray(snr_lin_prb).shape[0])
    elif cap_prb is not None:
        N_UE = int(np.asarray(cap_prb).shape[0])
    else:
        if snr_lin_time_prb_metric is None:
            raise ValueError(
                "Need snr_lin_time_prb_metric or se_metric_time_subband or static snr/cap"
            )
        N_UE = int(np.asarray(snr_lin_time_prb_metric).shape[1])

    Rbar = np.full(N_UE, 1e-3)
    sum_rate = 0.0

    for t_idx in range(T):
        # Get scheduling metric for this TTI
        if se_metric_time_subband is not None:
            metric_se = se_metric_time_subband[t_idx]
            snr_metric_mat = None
        elif snr_lin_time_prb_metric is not None:
            snr_metric_mat = np.asarray(snr_lin_time_prb_metric[t_idx], dtype=float)
            metric_se = _metric_from_snr_mat(snr_metric_mat)
        else:
            metric_se = se_metric_sub
            snr_metric_mat = None

        # Allocate groups
        winners = np.full(S, -1, dtype=int)

        if not use_marginal_delta:
            # Simple PF allocation
            metric = metric_se / Rbar.reshape(-1, 1)
            if max_groups_per_ue is None:
                winners = np.argmax(metric, axis=0)
            else:
                counts = np.zeros(N_UE, dtype=int)
                best_vals = metric.max(axis=0)
                order_g = np.argsort(-best_vals)
                for idx in order_g:
                    ue_best = int(np.argmax(metric[:, idx]))
                    if counts[ue_best] < int(max_groups_per_ue):
                        winners[idx] = ue_best
                        counts[ue_best] += 1
                    else:
                        sorted_ues = np.argsort(-metric[:, idx])
                        chosen = -1
                        for u in sorted_ues:
                            if counts[u] < int(max_groups_per_ue):
                                chosen = int(u)
                                break
                        if chosen < 0:
                            chosen = int(np.argmax(metric[:, idx]))
                        winners[idx] = chosen
                        counts[chosen] += 1
        else:
            # Marginal ΔSE allocation
            assigned_groups: List[List[int]] = [[] for _ in range(N_UE)]
            k_prb_assigned = np.zeros(N_UE, dtype=int)
            block_se_pred = np.zeros(N_UE, dtype=float)
            counts = np.zeros(N_UE, dtype=int)
            remaining = set(range(S))

            while remaining:
                best_delta = -1e9
                best_pair = None

                for gi in list(remaining):
                    li, ri = groups[gi]
                    size_g = ri - li + 1

                    for ue in range(N_UE):
                        if max_groups_per_ue is not None and counts[ue] >= int(max_groups_per_ue):
                            continue

                        k0 = int(k_prb_assigned[ue])

                        if snr_metric_mat is not None:
                            # Build SNR vectors for old and new allocations
                            if k0 > 0:
                                mask = np.zeros(Z, dtype=bool)
                                for gprev in assigned_groups[ue]:
                                    l0, r0 = groups[gprev]
                                    mask[l0:r0 + 1] = True
                                snr_vec_old = snr_metric_mat[ue, mask]
                            else:
                                snr_vec_old = None

                            mask_new = np.zeros(Z, dtype=bool)
                            if k0 > 0:
                                for gprev in assigned_groups[ue]:
                                    l0, r0 = groups[gprev]
                                    mask_new[l0:r0 + 1] = True
                            mask_new[li:ri + 1] = True
                            snr_vec_new = snr_metric_mat[ue, mask_new]

                            se_old = (
                                _block_se_from_snr_vec(
                                    snr_vec_old,
                                    k0 if (k0 > 0 and power_split) else 1,
                                    use_mcs,
                                    mcs_params,
                                    eesm_beta_db,
                                )
                                if k0 > 0
                                else 0.0
                            )
                            k_new = k0 + size_g
                            se_new = _block_se_from_snr_vec(
                                snr_vec_new,
                                k_new if power_split else 1,
                                use_mcs,
                                mcs_params,
                                eesm_beta_db,
                            )
                        else:
                            # Fallback: use per-group metric
                            se_old = float(block_se_pred[ue]) if k0 > 0 else 0.0
                            k_new = k0 + size_g
                            se_g = float(metric_se[ue, gi])
                            se_new = (k0 * se_old + size_g * se_g) / float(k_new)

                        delta = k_new * se_new - k0 * se_old
                        metric_pf = delta / Rbar[ue]

                        if metric_pf > best_delta:
                            best_delta = metric_pf
                            best_pair = (ue, gi, k_new, se_new)

                if best_pair is None:
                    break

                ue_sel, gi_sel, k_new_sel, se_new_sel = best_pair
                winners[gi_sel] = ue_sel
                remaining.remove(gi_sel)
                counts[ue_sel] += 1
                assigned_groups[ue_sel].append(gi_sel)
                k_prb_assigned[ue_sel] = k_new_sel
                block_se_pred[ue_sel] = se_new_sel

        # Throughput accumulation
        thr_i = np.zeros(N_UE, dtype=float)

        if power_split:
            prbs_per_ue = np.zeros(N_UE, dtype=int)
            for gi, (li, ri) in enumerate(groups):
                prbs_per_ue[winners[gi]] += ri - li + 1
        else:
            prbs_per_ue = np.ones(N_UE, dtype=int)

        # Get true SNR for throughput
        if snr_lin_time_prb_true is not None:
            snr_true = np.asarray(snr_lin_time_prb_true[t_idx], dtype=float)
        elif snr_lin_prb is not None:
            snr_true = np.asarray(snr_lin_prb, dtype=float)
        else:
            gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap_prb)) - 1.0)
            snr_true = gamma

        for gi, (li, ri) in enumerate(groups):
            ue = int(winners[gi])
            k_prb = int(prbs_per_ue[ue]) if power_split else 1
            snr_vec = snr_true[ue, li:ri + 1]
            se_per_prb = _block_se_from_snr_vec(
                snr_vec, k_prb, use_mcs, mcs_params, eesm_beta_db
            )
            thr_i[ue] += (ri - li + 1) * se_per_prb * overhead_eff

        sum_rate += thr_i.sum()
        Rbar = (1 - beta) * Rbar + beta * thr_i

    avg_sum_rate_per_prb = sum_rate / (T * Z)
    return avg_sum_rate_per_prb
