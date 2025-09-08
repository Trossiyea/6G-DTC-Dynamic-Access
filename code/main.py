# -*- coding: utf-8 -*-
"""
Radio Map–aware dynamic access simulation for direct-to-satellite (FDD) uplink.
- We compare a simple 3GPP-like baseline (wideband PF, no subband awareness) 
  against a Radio Map–aware proportional fair (per-PRB) scheduler.
- The Radio Map is a 3D tensor R[x, y, z] (dBm) measuring terrestrial interference.
- Output: average spectral efficiency (bits/s/Hz), relative gain, and a few plots.

Notes for reproducibility:
- You can edit the parameters under 'CONFIG' to stress-test different regimes.
- Charts use matplotlib only, one per figure, and no specific colors are set.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# -----------------------
# Utility conversions
# -----------------------
def dbm_to_mw(dbm: np.ndarray) -> np.ndarray:
    return 10.0 ** (dbm / 10.0)

def mw_to_dbm(mw: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(mw)

# -----------------------
# Radio Map generator
# -----------------------
def gen_radio_map(X: int, Y: int, Z: int, 
                  K: int = 7, base_noise_dbm: float = -121.45, seed: int = 1) -> np.ndarray:
    """
    Create a synthetic Radio Map R[x,y,z] in dBm.
    We place K Gaussian interference 'lobes' in the 3D space-frequency volume, 
    added on top of the thermal noise floor.
    """
    rng = np.random.default_rng(seed)
    power_mw = np.full((X, Y, Z), dbm_to_mw(base_noise_dbm), dtype=float)

    xs = np.arange(X).reshape(-1, 1, 1)
    ys = np.arange(Y).reshape(1, -1, 1)
    zs = np.arange(Z).reshape(1, 1, -1)

    for _ in range(K):
        cx, cy, cz = rng.uniform(0, X), rng.uniform(0, Y), rng.uniform(0, Z)
        ax, ay, az = rng.uniform(5, 15), rng.uniform(5, 15), rng.uniform(2, 8)
        peak_rel_db = rng.uniform(10, 35)  # peak above noise (dB)
        peak_rel_mw = 10 ** (peak_rel_db / 10.0)

        weight = np.exp(-(((xs - cx) ** 2) / (2 * ax ** 2)
                          + ((ys - cy) ** 2) / (2 * ay ** 2)
                          + ((zs - cz) ** 2) / (2 * az ** 2)))
        power_mw += dbm_to_mw(base_noise_dbm) * peak_rel_mw * weight

    return mw_to_dbm(power_mw)

# -----------------------
# Link & scheduling
# -----------------------
def compute_caps(R_xyz_dbm: np.ndarray,
                 ue_pos_xy: np.ndarray,
                 P_tx_dbm: float,
                 L_fs_db: float,
                 G_rx_db: float,
                 shadow_db_std: float = 5.0,
                 N0_dbm: float = -121.45,
                 seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-UE per-PRB spectral efficiency based on the Radio Map.
    Returns:
      cap[UE,Z], cap_wb[UE], P_rx_dbm[UE], I_total_dbm[UE,Z]
    """
    rng = np.random.default_rng(seed)
    X, Y, Z = R_xyz_dbm.shape
    N_UE = ue_pos_xy.shape[0]

    # UE positions
    x_idx = ue_pos_xy[:, 0]
    y_idx = ue_pos_xy[:, 1]

    # Link budget: received power per UE (dBm)
    shadow_db = rng.normal(0.0, shadow_db_std, size=N_UE)
    P_rx_dbm = P_tx_dbm - L_fs_db + G_rx_db + shadow_db  # dBm

    # Interference + thermal noise per UE per PRB (dBm)
    I_uez_dbm = R_xyz_dbm[x_idx, y_idx, :]  # [UE,Z]
    I_total_mw = dbm_to_mw(I_uez_dbm) + dbm_to_mw(N0_dbm)
    I_total_dbm = mw_to_dbm(I_total_mw)

    # Per-PRB SNR and capacity (bits/s/Hz)
    gamma_db = (P_rx_dbm.reshape(-1, 1) - I_total_dbm)            # [UE,Z]
    gamma_lin = 10.0 ** (gamma_db / 10.0)
    cap = np.log2(1.0 + gamma_lin)                                # [UE,Z]

    # Wideband (3GPP-like) interference (median across subbands)
    I_wb_mw = np.median(I_total_mw, axis=1)                       # [UE]
    I_wb_dbm = mw_to_dbm(I_wb_mw)
    gamma_db_wb = P_rx_dbm - I_wb_dbm
    gamma_lin_wb = 10.0 ** (gamma_db_wb / 10.0)
    cap_wb = np.log2(1.0 + gamma_lin_wb)                          # [UE]

    return cap, cap_wb, P_rx_dbm, I_total_dbm

def pf_schedule_baseline(cap_wb: np.ndarray, Z: int, T: int, beta: float = 0.1) -> float:
    """
    3GPP-like baseline: proportional fair with wideband CQI (same cap on every PRB).
    To avoid one-UE monopolization, assign PRBs in each TTI across the top sqrt(N) UEs 
    per PF metric, equally split.
    Returns average sum spectral efficiency per PRB (bits/s/Hz).
    """
    N_UE = cap_wb.shape[0]
    Rbar = np.full(N_UE, 1e-3)
    sum_rate = 0.0
    for _ in range(T):
        metric = cap_wb / Rbar
        order = np.argsort(-metric)
        U_select = min(N_UE, max(3, int(np.sqrt(N_UE))))
        selected = order[:U_select]

        alloc_counts = np.full(U_select, Z // U_select)
        remainder = Z - alloc_counts.sum()
        if remainder > 0:
            alloc_counts[:remainder] += 1

        thr_i = np.zeros(N_UE)
        for k, ue in enumerate(selected):
            thr_i[ue] = alloc_counts[k] * cap_wb[ue]
        sum_rate += thr_i.sum()
        Rbar = (1 - beta) * Rbar + beta * thr_i

    avg_sum_rate_per_prb = sum_rate / (T * Z)
    return avg_sum_rate_per_prb

def pf_schedule_radiomap(cap: np.ndarray, T: int, beta: float = 0.1) -> float:
    """
    Radio Map–aware PF: per-PRB scheduling using cap[UE,Z].
    Returns average sum spectral efficiency per PRB (bits/s/Hz).
    """
    N_UE, Z = cap.shape
    Rbar = np.full(N_UE, 1e-3)
    sum_rate = 0.0
    for _ in range(T):
        metric = cap / Rbar.reshape(-1, 1)  # [UE,Z]
        winners = np.argmax(metric, axis=0)  # [Z]
        thr_i = np.zeros(N_UE)
        for z in range(Z):
            ue = winners[z]
            thr_i[ue] += cap[ue, z]
        sum_rate += thr_i.sum()
        Rbar = (1 - beta) * Rbar + beta * thr_i

    avg_sum_rate_per_prb = sum_rate / (T * Z)
    return avg_sum_rate_per_prb

# -----------------------
# Experiment harness
# -----------------------
def run_once(config: Dict) -> Dict:
    rng = np.random.default_rng(config["seed"])
    X, Y, Z = config["X"], config["Y"], config["Z"]
    N_UE, T = config["N_UE"], config["T"]

    R_xyz_dbm = gen_radio_map(X, Y, Z, K=config["K_interferers"],
                              base_noise_dbm=config["noise_dbm"], seed=config["seed"])
    ue_pos = np.stack([rng.integers(0, X, size=N_UE),
                       rng.integers(0, Y, size=N_UE)], axis=1)

    cap, cap_wb, P_rx_dbm, I_total_dbm = compute_caps(
        R_xyz_dbm, ue_pos,
        P_tx_dbm=config["P_tx_dbm"],
        L_fs_db=config["L_fs_db"],
        G_rx_db=config["G_rx_db"],
        shadow_db_std=config["shadow_std_db"],
        N0_dbm=config["noise_dbm"],
        seed=config["seed"]
    )
    base_se = pf_schedule_baseline(cap_wb, Z, T, beta=config["pf_beta"])
    map_se = pf_schedule_radiomap(cap, T, beta=config["pf_beta"])
    return {
        "avg_se_baseline": base_se,
        "avg_se_radiomap": map_se,
        "improvement_pct": (map_se - base_se) / max(1e-9, base_se) * 100.0,
        "R_xyz_dbm": R_xyz_dbm,
        "ue_pos": ue_pos,
        "cap": cap,
        "cap_wb": cap_wb,
    }

def run_many(config: Dict, seeds: np.ndarray) -> Dict:
    base_list, map_list, imp_list = [], [], []
    for s in seeds:
        c2 = dict(config)
        c2["seed"] = int(s)
        out = run_once(c2)
        base_list.append(out["avg_se_baseline"])
        map_list.append(out["avg_se_radiomap"])
        imp_list.append(out["improvement_pct"])
    return {
        "baseline": np.array(base_list),
        "radiomap": np.array(map_list),
        "improvement_pct": np.array(imp_list)
    }

# -----------------------
# CONFIG (edit as needed)
# -----------------------
CONFIG = {
    "X": 60,              # map width
    "Y": 60,              # map height
    "Z": 64,              # number of PRBs/subbands
    "N_UE": 40,           # number of UEs in footprint
    "T": 200,             # number of TTIs
    "K_interferers": 7,   # number of terrestrial interference lobes
    "noise_dbm": -121.45, # per-PRB thermal noise ~180 kHz
    "P_tx_dbm": 23.0,     # UE EIRP per PRB (dBm)
    "L_fs_db": 154.0,     # free-space loss (dB) ≈ 600 km @ ~2 GHz
    "G_rx_db": 32.0,      # effective satellite RX gain + link margin (dB)
    "shadow_std_db": 5.0, # UE shadowing (dB, std. dev.)
    "pf_beta": 0.1,       # PF averaging factor
    "seed": 1             # random seed
}

# -----------------------
# Run single experiment
# -----------------------
single = run_once(CONFIG)
print("Single-run results")
print(f"  Baseline avg SE (bits/s/Hz): {single['avg_se_baseline']:.3f}")
print(f"  RadioMap avg SE (bits/s/Hz): {single['avg_se_radiomap']:.3f}")
print(f"  Gain (%): {single['improvement_pct']:.2f}")

# -----------------------
# Run multiple seeds to show robustness
# -----------------------
seeds = np.arange(1, 21)
multi = run_many(CONFIG, seeds)
print("\nMulti-seed summary (N=20)")
print(f"  Baseline avg SE: {multi['baseline'].mean():.3f} ± {multi['baseline'].std():.3f}")
print(f"  RadioMap avg SE: {multi['radiomap'].mean():.3f} ± {multi['radiomap'].std():.3f}")
print(f"  Gain median: {np.median(multi['improvement_pct']):.2f}% (min={multi['improvement_pct'].min():.2f}%, max={multi['improvement_pct'].max():.2f}%)")

# -----------------------
# Plots
# -----------------------

# 1) Improvement distribution
plt.figure(figsize=(6,4))
plt.hist(multi["improvement_pct"], bins=10, edgecolor='black')
plt.title("Radio Map–aware gain distribution across seeds")
plt.xlabel("Gain vs. baseline (%)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2) Example Radio Map slice (median over frequency)
R_med = np.median(single["R_xyz_dbm"], axis=2)
plt.figure(figsize=(5,5))
plt.imshow(R_med.T, origin='lower', aspect='equal')
plt.title("Radio Map (median over frequency), dBm")
plt.colorbar(label='dBm')
plt.tight_layout()
plt.show()

# 3) Example per-UE wideband vs best-subband capacity (first 10 UEs)
ue = np.arange(min(10, CONFIG["N_UE"]))
best_subband = single["cap"][ue].max(axis=1)
wb = single["cap_wb"][ue]
x = np.arange(ue.size)
plt.figure(figsize=(6,4))
plt.bar(x - 0.2, wb, width=0.4, label='Wideband (baseline)')
plt.bar(x + 0.2, best_subband, width=0.4, label='Best subband (RadioMap)')
plt.xticks(x, [f"UE{int(i)}" for i in ue])
plt.ylabel("Spectral efficiency (bits/s/Hz)")
plt.title("Per-UE: wideband vs best subband opportunity")
plt.legend()
plt.tight_layout()
plt.show()