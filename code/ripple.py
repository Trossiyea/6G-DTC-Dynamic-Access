# -*- coding: utf-8 -*-
"""
RIPPLE: Radio‑map guided Interval Partitioning & dual‑Projected PF
with Lightweight power Equalization.

Milestone-1 implementation notes:
- Frequency-domain adaptive segmentation from interference statistics.
- Per-segment O(1) EESM via exp(-gamma/beta) prefix sums.
- Dualized greedy selection (one segment per UE preferred), then fill leftovers.
- Segment-internal equal power, no segment-level water-filling (hook left for future).
- HARQ integration mirrors existing PF path (advance_time + optional on_scheduled).

This module is self-contained and called optionally from main.run_once()
so baseline and RadioMap schedulers remain unchanged.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from tqdm import tqdm

from csi import sinr_to_se_mcs, effective_sinr_eesm


def _apply_ici_penalty_lin(snr_lin: np.ndarray, mcs_params: Optional[Dict]) -> np.ndarray:
    if mcs_params is None:
        return np.asarray(snr_lin)
    eps_f = float(mcs_params.get("residual_freq_hz", 0.0) or 0.0)
    if eps_f <= 0.0:
        return np.asarray(snr_lin)
    scs_khz = float(mcs_params.get("scs_khz", 30.0) or 30.0)
    T_sym = 1.0 / (scs_khz * 1e3)
    ici_factor = 1.0 + (2.0 * np.pi * eps_f * T_sym) ** 2
    return np.asarray(snr_lin) / ici_factor


def _seg_from_metric_1d(metric_db: np.ndarray,
                        thresh_db: float = 3.0,
                        max_K: Optional[int] = None,
                        min_len: int = 1,
                        max_len: int = 0) -> List[Tuple[int, int]]:
    """
    Build non-overlapping frequency segments [l,r] from a 1D metric in dB.
    - Candidate boundaries where |diff| > thresh_db.
    - Enforce min_len; cap number of segments to max_K by dropping weakest diffs.
    Returns a list of (l, r) covering [0..Z-1] without overlaps.
    """
    m = np.asarray(metric_db, dtype=float)
    Z = m.size
    if Z <= 0:
        return []
    if min_len <= 0:
        min_len = 1
    diffs = np.zeros(Z, dtype=float)
    diffs[1:] = np.abs(m[1:] - m[:-1])
    cand = [i for i in range(1, Z) if diffs[i] >= float(thresh_db)]
    # Start with naive segments using all candidates
    bounds = [0] + cand + [Z]
    segs: List[Tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        l = int(bounds[i])
        r = int(bounds[i + 1] - 1)
        segs.append((l, r))
    # Enforce min_len by merging tiny neighbors greedily
    def _length(s):
        return s[1] - s[0] + 1
    changed = True
    while changed and any(_length(s) < min_len for s in segs) and len(segs) > 1:
        changed = False
        for i, s in enumerate(list(segs)):
            if _length(s) >= min_len:
                continue
            # merge with neighbor having smaller boundary diff
            if i == 0:
                j = 1
            elif i == len(segs) - 1:
                j = i - 1
            else:
                # choose weaker boundary
                dl = diffs[segs[i][0]] if i > 0 else 1e9
                dr = diffs[segs[i + 1][0]] if i + 1 < len(segs) else 1e9
                j = i - 1 if dl <= dr else i + 1
            a = segs[min(i, j)][0]
            b = segs[max(i, j)][1]
            segs[min(i, j):max(i, j) + 1] = [(a, b)]
            changed = True
            break
    # Cap number of segments to max_K by dropping weakest boundaries
    if isinstance(max_K, int) and max_K > 0 and len(segs) > max_K:
        # Reconstruct boundaries and remove the weakest ones
        # Each internal boundary position is segs[i][1] + 1 for i in 0..len-2
        bpos = [segs[i][1] + 1 for i in range(len(segs) - 1)]
        bstr = [diffs[p] for p in bpos]
        # Indices to keep: drop the smallest (len-K) boundaries
        order = np.argsort(bstr)  # ascending by strength
        drop = set(int(order[i]) for i in range(len(segs) - max_K))
        kept_bounds = [0]
        for i in range(len(bpos)):
            if i not in drop:
                kept_bounds.append(bpos[i])
        kept_bounds.append(Z)
        segs = [(int(kept_bounds[i]), int(kept_bounds[i + 1] - 1)) for i in range(len(kept_bounds) - 1)]
    # Cap segment length if requested
    if isinstance(max_len, int) and max_len > 0:
        out: List[Tuple[int, int]] = []
        for (l, r) in segs:
            L = r - l + 1
            if L <= max_len:
                out.append((l, r))
            else:
                start = l
                while start <= r:
                    end = min(r, start + max_len - 1)
                    out.append((start, end))
                    start = end + 1
        segs = out
    return segs


def segment_band_from_interference(I_uez_dbm: np.ndarray,
                                   thresh_db: float = 3.0,
                                   max_K: Optional[int] = None,
                                   min_len: int = 1,
                                   method: str = "median",
                                   max_len: int = 0) -> List[Tuple[int, int]]:
    """Segment PRBs guided by cross-UE interference statistics (1D metric over PRB).
    I_uez_dbm: [UE, Z] in dBm. method in {"median","mean"}.
    """
    if I_uez_dbm is None:
        return [(0, I_uez_dbm.shape[1] - 1)] if I_uez_dbm is not None else []
    I = np.asarray(I_uez_dbm, dtype=float)
    if I.ndim != 2:
        raise ValueError("I_uez_dbm must be 2D [UE,Z]")
    metric = np.median(I, axis=0) if method == "median" else np.mean(I, axis=0)
    return _seg_from_metric_1d(metric, thresh_db=float(thresh_db), max_K=max_K, min_len=int(min_len), max_len=int(max_len))


def build_eesm_prefix(sinr_db: np.ndarray, beta_db: float) -> np.ndarray:
    """Prefix sums of exp(-sinr/beta) per UE for O(1) segment EESM.
    sinr_db: [UE,Z]
    returns prefix [UE,Z] of exp(-sinr/beta) cumulative along PRB axis.
    """
    s = np.asarray(sinr_db, dtype=float)
    z = np.exp(-s / float(beta_db))
    return np.cumsum(z, axis=1)


def seg_eesm_db(prefix: np.ndarray, l: int, r: int, beta_db: float) -> np.ndarray:
    """Compute per-UE effective SINR(dB) for segment [l,r] using EESM and prefix sums."""
    P = np.asarray(prefix, dtype=float)
    if l <= 0:
        s = P[:, r]
    else:
        s = P[:, r] - P[:, l - 1]
    k = max(1, int(r - l + 1))
    val = np.maximum(s / float(k), 1e-12)
    return -float(beta_db) * np.log(val)


def _se_from_eff_sinr_db(eff_db: np.ndarray, use_mcs: bool, mcs_params: Optional[Dict]) -> np.ndarray:
    if use_mcs:
        e = np.asarray(eff_db, dtype=float)
        if mcs_params is not None:
            e = e + float(mcs_params.get("olla_offset_db", 0.0))
        table = mcs_params.get("mcs_table", "legacy") if mcs_params else "legacy"
        return np.asarray(sinr_to_se_mcs(e, table=table), dtype=float)
    # Shannon fallback
    lin = 10.0 ** (np.asarray(eff_db, dtype=float) / 10.0)
    return np.log2(1.0 + np.maximum(lin, 0.0))


def _block_se_from_snr_vec_simple(snr_lin_vec: np.ndarray, use_mcs: bool, mcs_params: Optional[Dict], beta_db: float) -> float:
    """Compute per-PRB SE for a contiguous block using EESM over the vector.
    No power-split penalty; caller should scale snr_lin_vec if power model applies.
    Returns per-PRB SE (single MCS assumed across the block).
    """
    s = np.asarray(snr_lin_vec, dtype=float)
    if use_mcs:
        sinr_db_vec = 10.0 * np.log10(np.maximum(s, 1e-12))
        sinr_eff_db = effective_sinr_eesm(sinr_db_vec, beta_db=float(beta_db), axis=-1)
        if mcs_params is not None:
            sinr_eff_db = sinr_eff_db + float(mcs_params.get("olla_offset_db", 0.0))
        table = mcs_params.get("mcs_table", "legacy") if mcs_params else "legacy"
        se = float(np.asarray(sinr_to_se_mcs(sinr_eff_db, table=table)))
        return se
    else:
        return float(np.mean(np.log2(1.0 + np.maximum(s, 0.0))))


def ripple_schedule(
    cap: np.ndarray,
    T: int,
    beta: float,
    snr_lin: Optional[np.ndarray],
    overhead_eff: float,
    use_mcs: bool,
    mcs_params: Optional[Dict],
    snr_lin_time: Optional[np.ndarray] = None,
    I_total_dbm: Optional[np.ndarray] = None,
    eesm_beta_db: float = 3.0,
    rng: Optional[np.random.Generator] = None,
    ue_mask_time: Optional[np.ndarray] = None,
    harq_mgr=None,
    record_assignments: bool = False,
    assignments_out: Optional[list] = None,
    config: Optional[Dict] = None,
) -> float:
    """
    RIPPLE scheduling over T TTIs; returns avg sum SE per PRB.
    - cap: [UE,Z] shannon SE (unused if use_mcs=True; kept for fallback)
    - snr_lin: [UE,Z] or None; snr_lin_time: [T,UE,Z] optional
    - I_total_dbm: [UE,Z] for segmentation guidance (median over UE)
    """
    cfg = config or {}
    N_UE, Z = cap.shape
    rng = np.random.default_rng(0) if rng is None else rng

    # Segmentation (A): build once (slowly varying)
    seg_thresh = float(cfg.get("ripple_seg_thresh_db", 3.0))
    seg_max_K = int(cfg.get("ripple_seg_max_K", min(64, Z)))
    seg_min_len = int(cfg.get("ripple_seg_min_len", 2))
    seg_method = str(cfg.get("ripple_seg_from", "interference")).lower()
    seg_max_len = int(cfg.get("ripple_max_seg_len", 0) or 0)
    if seg_method.startswith("snr"):
        # Use SNR median in dB
        base = np.asarray(snr_lin if snr_lin is not None else cap, dtype=float)
        if snr_lin is not None:
            mdb = 10.0 * np.log10(np.maximum(np.median(base, axis=0), 1e-12))
        else:
            # invert shannon to SNR
            gamma = np.maximum(0.0, np.power(2.0, np.median(base, axis=0)) - 1.0)
            mdb = 10.0 * np.log10(np.maximum(gamma, 1e-12))
        segments = _seg_from_metric_1d(mdb, seg_thresh, seg_max_K, seg_min_len, seg_max_len)
    else:
        segments = segment_band_from_interference(I_total_dbm, seg_thresh, seg_max_K, seg_min_len, method="median", max_len=seg_max_len)
        if not segments:
            segments = [(0, Z - 1)]

    # Options
    strict_one_segment = bool(cfg.get("ripple_strict_one_segment", False))
    fill_leftover = bool(cfg.get("ripple_fill_leftover", True))
    iter_steps = int(cfg.get("ripple_dual_iters", 5))
    step_size = float(cfg.get("ripple_dual_step", 0.5))
    dp_enabled = bool(cfg.get("ripple_dp_enabled", False))
    merge_M = int(cfg.get("ripple_candidate_merge", 0) or 0)

    # PF average throughput
    Rbar = np.full(N_UE, 1e-3, dtype=float)
    sum_rate = 0.0

    show_progress = bool(cfg.get("show_progress", True))
    pbar = tqdm(range(T), desc="RIPPLE", unit="TTI", disable=not show_progress,
                bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    reseg_period = int(cfg.get("ripple_reseg_period_ttis", 0) or 0)
    for t_idx in pbar:
        # HARQ crediting
        if harq_mgr is not None:
            ack_bits = None
            try:
                ack_bits = harq_mgr.advance_time()
            except TypeError:
                ack_bits = None
            if ack_bits is not None:
                # Convert bits->SE via RE per PRB from config if available
                from link_adapt import re_per_prb_from_config
                re_per_prb = max(1, re_per_prb_from_config(cfg))
                thr_ack = np.asarray(ack_bits, dtype=float) / float(re_per_prb)
                sum_rate += float(np.sum(thr_ack))
                Rbar = (1 - beta) * Rbar + beta * thr_ack

        # Masking (UE gating)
        mask_t = None
        if ue_mask_time is not None:
            mask_t = np.asarray(ue_mask_time[t_idx], dtype=bool)

        # Pick SNRs for this TTI
        if snr_lin_time is not None:
            snr_now = np.asarray(snr_lin_time[t_idx], dtype=float)
        elif snr_lin is not None:
            snr_now = np.asarray(snr_lin, dtype=float)
        else:
            # Invert shannon
            gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap, dtype=float)) - 1.0)
            snr_now = gamma
        snr_now = _apply_ici_penalty_lin(snr_now, mcs_params)

        # Optional slow re-segmentation using current SNR (if enabled)
        if reseg_period > 0 and (t_idx > 0) and (t_idx % reseg_period == 0):
            # Use current SNR median for segmentation (robust proxy for interference shifts)
            mdb_now = 10.0 * np.log10(np.maximum(np.median(snr_now, axis=0), 1e-12))
            segments = _seg_from_metric_1d(mdb_now, seg_thresh, seg_max_K, seg_min_len, seg_max_len)
            if not segments:
                segments = [(0, Z - 1)]

        # EESM prefix (B)
        sinr_db_now = 10.0 * np.log10(np.maximum(snr_now, 1e-12))
        prefix = build_eesm_prefix(sinr_db_now, float(eesm_beta_db))

        # Precompute per-UE segment EESM (dB) and per-PRB SE for all segments (baseline equal-power)
        K = len(segments)
        se_seg = np.zeros((N_UE, K), dtype=float)
        eff_seg_db = np.zeros((N_UE, K), dtype=float)
        seg_len = np.zeros(K, dtype=int)
        use_mcs_flag = bool(cfg.get('ripple_use_mcs', use_mcs))
        for k, (l, r) in enumerate(segments):
            eff_db = seg_eesm_db(prefix, int(l), int(r), float(eesm_beta_db))
            eff_seg_db[:, k] = eff_db
            se_seg[:, k] = _se_from_eff_sinr_db(eff_db, use_mcs_flag, mcs_params)
            seg_len[k] = int(r - l + 1)

        # Dualized selection (C): greedy or DP
        w = 1.0 / np.maximum(Rbar, 1e-9)
        lamb = np.zeros(N_UE, dtype=float)
        assigned: Dict[int, Tuple[int, float]] = {}
        def _build_candidates() -> List[Tuple[int,int]]:
            # Merge up to M adjacent base segments to form candidates
            if merge_M <= 1:
                return list(segments)
            cand: List[Tuple[int,int]] = []
            for i in range(K):
                l0 = segments[i][0]
                r0 = segments[i][1]
                cand.append((l0, r0))
                for m in range(2, merge_M + 1):
                    j = i + m - 1
                    if j >= K:
                        break
                    cand.append((l0, segments[j][1]))
            # Dedup and sort by (l,r)
            cand = sorted(list(set(cand)))
            return cand

        def _dp_select(cands: List[Tuple[int,int]], lamb_arr: np.ndarray) -> Dict[Tuple[int,int], Tuple[int,float]]:
            # Compute phi for each candidate and associated best UE
            nC = len(cands)
            L = np.array([c[0] for c in cands], dtype=int)
            Rr = np.array([c[1] for c in cands], dtype=int)
            # Precompute per-UE per-candidate SE using prefix (O(U*nC))
            # Use MCS or Shannon according to use_mcs_flag
            se_uc = np.zeros((N_UE, nC), dtype=float)
            for j in range(nC):
                eff = seg_eesm_db(prefix, int(L[j]), int(Rr[j]), float(eesm_beta_db))
                se_uc[:, j] = _se_from_eff_sinr_db(eff, use_mcs_flag, mcs_params)
            blk_len = (Rr - L + 1).astype(float)
            # Compute phi_j and u*_j under current lambda
            phi = np.full(nC, -1e9, dtype=float)
            ustar = np.full(nC, -1, dtype=int)
            for j in range(nC):
                gain = w * (se_uc[:, j] * blk_len[j]) - lamb_arr
                if mask_t is not None:
                    gain = np.array(gain, copy=True)
                    gain[~mask_t] = -1e9
                if harq_mgr is not None:
                    try:
                        can = np.array([harq_mgr.can_schedule(u) for u in range(N_UE)], dtype=bool)
                        gain[~can] = -1e9
                    except Exception:
                        pass
                u = int(np.argmax(gain))
                val = float(gain[u])
                phi[j] = val
                ustar[j] = u
            # Weighted interval scheduling DP (non-overlapping by (L,Rr))
            order = np.argsort(Rr)
            Ls = L[order]; Rs = Rr[order]; phi_s = phi[order]; u_s = ustar[order]
            # prev index via binary search
            import bisect
            prev = []
            ends = list(Rs)
            for i in range(len(order)):
                li = int(Ls[i])
                # last interval that ends < li
                j = bisect.bisect_left(ends, li) - 1
                prev.append(j)
            dp = np.zeros(len(order), dtype=float)
            take = np.zeros(len(order), dtype=bool)
            for i in range(len(order)):
                notake = dp[i-1] if i > 0 else 0.0
                takev = phi_s[i]
                if prev[i] >= 0:
                    takev += dp[prev[i]]
                if takev > notake:
                    dp[i] = takev
                    take[i] = True
                else:
                    dp[i] = notake
                    take[i] = False
            # Recover chosen set
            chosen: Dict[Tuple[int,int], Tuple[int,float]] = {}
            i = len(order) - 1
            while i >= 0:
                if take[i]:
                    j = i
                    seg = (int(Ls[j]), int(Rs[j]))
                    chosen[seg] = (int(u_s[j]), float(phi_s[j]))
                    i = prev[j]
                else:
                    i -= 1
            return chosen

        for _ in range(max(1, iter_steps)):
            if dp_enabled and merge_M >= 1:
                cands = _build_candidates()
                chosen_dp = _dp_select(cands, lamb)
                # Update assignment map into base-segment index coordinates for downstream throughput
                assigned = {}
                # Map candidate segments back to leftmost base segment index for bookkeeping
                for (l, r), (u, val) in chosen_dp.items():
                    # find first base segment index whose l matches
                    k0 = None
                    for k in range(K):
                        if segments[k][0] == l:
                            k0 = k; break
                    if k0 is None:
                        # fallback: closest left index
                        k0 = 0
                    assigned[k0] = (int(u), float(val))
            else:
                # Greedy fallback on base segments
                phi_list: List[Tuple[float, int, int]] = []  # (phi, k, u*)
                for k in range(K):
                    gain = w * (se_seg[:, k] * float(seg_len[k])) - lamb
                    if mask_t is not None:
                        gain = np.array(gain, copy=True)
                        gain[~mask_t] = -1e9
                    if harq_mgr is not None:
                        try:
                            can = np.array([harq_mgr.can_schedule(u) for u in range(N_UE)], dtype=bool)
                            gain[~can] = -1e9
                        except Exception:
                            pass
                    u_star = int(np.argmax(gain))
                    phi = float(gain[u_star])
                    phi_list.append((phi, k, u_star))
                phi_list.sort(key=lambda x: -x[0])
                chosen_u = set()
                assigned = {}
                for phi, k, u in phi_list:
                    if phi <= 0:
                        continue
                    if strict_one_segment and u in chosen_u:
                        continue
                    if strict_one_segment:
                        chosen_u.add(u)
                    assigned[k] = (u, phi)
            # Dual update: penalize multi-assignments (n_u-1)
            if strict_one_segment:
                cnt = np.zeros(N_UE, dtype=int)
                for _, (u, _) in assigned.items():
                    cnt[int(u)] += 1
                lamb = np.maximum(0.0, lamb + float(step_size) * (cnt - 1))
            else:
                break

        winners = np.full(Z, -1, dtype=int)
        # Assign chosen segments
        for k, (u, _) in assigned.items():
            l, r = segments[k]
            winners[l:r + 1] = int(u)

        # Fill leftovers if requested
        if fill_leftover:
            # Per-PRB best UE by w*SE
            sinr_db_full = 10.0 * np.log10(np.maximum(snr_now, 1e-12))
            table = mcs_params.get("mcs_table", "legacy") if mcs_params else "legacy"
            se_prb_pred = sinr_to_se_mcs(sinr_db_full, table=table) if use_mcs_flag else np.log2(1.0 + np.maximum(snr_now, 0.0))
            for k, (l, r) in enumerate(segments):
                if winners[l] >= 0:
                    continue
                for z in range(l, r + 1):
                    gain = w * se_prb_pred[:, z]
                    if mask_t is not None:
                        gain = np.array(gain, copy=True)
                        gain[~mask_t] = -1e9
                    if harq_mgr is not None:
                        try:
                            can = np.array([harq_mgr.can_schedule(u) for u in range(N_UE)], dtype=bool)
                            gain[~can] = -1e9
                        except Exception:
                            pass
                    winners[z] = int(np.argmax(gain))

        # Optional segment-level power allocation (D)
        dlpm = str(cfg.get("ripple_dl_power_model", "equal")).lower()
        p_scales: Dict[int, float] = {}  # per segment index -> linear scale (p/P_ref)
        if dlpm == 'segment_wf' and (cfg.get('ripple_P_tot_dbm') is not None):
            P_tot_dbm = float(cfg.get('ripple_P_tot_dbm'))
            P_ref_dbm = float(cfg.get('P_tx_dbm', 30.0))
            p_min_dbm = cfg.get('ripple_p_min_dbm', None)
            p_max_dbm = cfg.get('ripple_p_max_dbm', None)

            def dR_dp_for_seg(kidx: int, p_mw: float) -> float:
                # derivative of block rate R w.r.t per-PRB power p (mW)
                u = winners[segments[kidx][0]]
                if u < 0:
                    return 0.0
                L = float(seg_len[kidx])
                p0 = max(1e-9, float(p_mw))
                # central difference
                h = max(1e-6, 0.05 * p0)
                p_lo = max(1e-9, p0 - h)
                p_hi = p0 + h
                def R_of(p_lin: float) -> float:
                    s = p_lin / max(1e-12, 10.0 ** (P_ref_dbm / 10.0))
                    delta_db = 10.0 * np.log10(max(s, 1e-12))
                    eff_db_scaled = eff_seg_db[u, kidx] + delta_db
                    se = float(_se_from_eff_sinr_db(eff_db_scaled, use_mcs, mcs_params))
                    return L * se * float(overhead_eff)
                return (R_of(p_hi) - R_of(p_lo)) / (p_hi - p_lo)

            # Boundaries per PRB in mW
            pmin_mW = 10.0 ** (float(p_min_dbm) / 10.0) if p_min_dbm is not None else 10.0 ** (P_ref_dbm / 10.0)
            pmax_mW = 10.0 ** (float(p_max_dbm) / 10.0) if p_max_dbm is not None else 10.0 ** (P_ref_dbm / 10.0)
            # Ensure pmin <= pmax
            if pmin_mW > pmax_mW:
                pmin_mW, pmax_mW = pmax_mW, pmin_mW

            # Clamp minimal total power feasibility
            P_tot_mW = 10.0 ** (P_tot_dbm / 10.0)
            P_min_need = float(np.sum(seg_len[[k for k in range(K) if winners[segments[k][0]] >= 0]])) * pmin_mW
            if P_min_need > P_tot_mW:
                # Not enough power to meet pmin: fallback equal to pmin across assigned; scale to budget
                for k in range(K):
                    if winners[segments[k][0]] < 0:
                        continue
                    p_scales[k] = pmin_mW / max(1e-12, 10.0 ** (P_ref_dbm / 10.0))
            else:
                # Water-fill via dual bisection on mu
                # mu in units of dR/dP (bits/Hz per mW)
                # bracket mu
                mu_lo = 0.0
                # max derivative at pmin across segments
                mu_hi = 0.0
                for k in range(K):
                    if winners[segments[k][0]] < 0:
                        continue
                    d = dR_dp_for_seg(k, pmin_mW) / float(seg_len[k])
                    if d > mu_hi:
                        mu_hi = d
                if mu_hi <= 0.0:
                    mu_hi = 1.0
                # bisection
                for _ in range(40):
                    mu = 0.5 * (mu_lo + mu_hi)
                    total = 0.0
                    p_opt: Dict[int, float] = {}
                    for k in range(K):
                        if winners[segments[k][0]] < 0:
                            continue
                        L = float(seg_len[k])
                        # Decide p_k by solving dR/dp = L*mu with box constraints
                        # Evaluate derivative at bounds
                        g_lo = dR_dp_for_seg(k, pmin_mW) - L * mu
                        g_hi = dR_dp_for_seg(k, pmax_mW) - L * mu
                        if g_lo <= 0.0 and g_hi <= 0.0:
                            p = pmin_mW
                        elif g_lo >= 0.0 and g_hi >= 0.0:
                            p = pmax_mW
                        else:
                            # Find root in [pmin,pmax]
                            lo, hi = pmin_mW, pmax_mW
                            for _ in range(25):
                                mid = 0.5 * (lo + hi)
                                g_mid = dR_dp_for_seg(k, mid) - L * mu
                                if g_mid > 0.0:
                                    lo = mid
                                else:
                                    hi = mid
                            p = 0.5 * (lo + hi)
                        p_opt[k] = p
                        total += L * p
                    if total > P_tot_mW:
                        mu_lo = mu
                    else:
                        mu_hi = mu
                # finalize per-segment scales
                for k, p in p_opt.items():
                    p_scales[k] = p / max(1e-12, 10.0 ** (P_ref_dbm / 10.0))

        # Throughput accumulation per UE (with optional power scaling)
        thr_i = np.zeros(N_UE, dtype=float)
        # Compute throughput using per-TTI block SINR vectors (closer to RM accounting)
        for k, (l, r) in enumerate(segments):
            ue = winners[l]
            if ue < 0:
                continue
            snr_vec = snr_now[ue, l:r + 1]
            scale = float(p_scales.get(k, 1.0))
            if scale != 1.0:
                snr_vec = snr_vec * max(scale, 1e-12)
            se_per_prb = _block_se_from_snr_vec_simple(snr_vec, use_mcs_flag, mcs_params, float(eesm_beta_db))
            thr_i[ue] += int(seg_len[k]) * se_per_prb * float(overhead_eff)

        # record assignments
        if record_assignments and assignments_out is not None:
            try:
                assignments_out.append(np.array(winners, copy=True))
            except Exception:
                pass

        sum_rate += thr_i.sum()
        Rbar = (1 - beta) * Rbar + beta * thr_i

        # HARQ notification: try full-block interface if one segment per UE, else fallback
        if harq_mgr is not None:
            # Build per-UE assigned segments
            per_ue_segments: Dict[int, Tuple[int, int, float]] = {}
            for k, (l, r) in enumerate(segments):
                u = winners[l]
                if u < 0:
                    continue
                if u in per_ue_segments:
                    # multiple segments; cannot form a single contiguous block reliably
                    per_ue_segments[u] = None
                else:
                    per_ue_segments[u] = (l, r, float(p_scales.get(k, 1.0)))
            use_block_api = hasattr(harq_mgr, 'on_scheduled_blocks') and all(v is not None for v in per_ue_segments.values())
            if use_block_api:
                info = {}
                for u, (l, r, s) in per_ue_segments.items():
                    delta_db = 10.0 * np.log10(max(s, 1e-12))
                    sinr_vec_db = sinr_db_now[int(u), l:r+1] + delta_db
                    info[int(u)] = {
                        'sinr_vec_db': np.asarray(sinr_vec_db, dtype=float),
                        'n_prb': int(r - l + 1),
                        'eesm_beta_db': float(eesm_beta_db),
                        'li': int(l), 'ri': int(r),
                    }
                try:
                    harq_mgr.on_scheduled_blocks(info)
                except Exception:
                    pass
            elif hasattr(harq_mgr, 'on_scheduled'):
                try:
                    harq_mgr.on_scheduled(np.unique(winners[winners >= 0]))
                except Exception:
                    pass

    # Tail flush if the HARQ manager supports it and config requests
    try:
        if harq_mgr is not None and bool(cfg.get("harq_flush_tail", True)):
            D = int(cfg.get("harq_ack_delay_ttis", 0) or 0)
            if D > 0:
                from link_adapt import re_per_prb_from_config
                re_per_prb = max(1, re_per_prb_from_config(cfg))
                for _ in range(D):
                    try:
                        ack_bits_tail = harq_mgr.advance_time()
                    except TypeError:
                        ack_bits_tail = None
                    if ack_bits_tail is not None:
                        thr_ack = np.asarray(ack_bits_tail, dtype=float) / float(re_per_prb)
                        sum_rate += float(np.sum(thr_ack))
                        Rbar = (1 - beta) * Rbar + beta * thr_ack
    except Exception:
        pass

    return sum_rate / float(T * Z)
