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
from typing import Tuple, Dict, Optional, Union
import os
from scipy.io import loadmat
from csi import sinr_to_se_mcs
from config import CONFIG
from orbit import compute_geometry_and_beam, OrbitModel
from pc import pusch_open_loop_power

# -----------------------
# Utility conversions
# -----------------------
def dbm_to_mw(dbm: np.ndarray) -> np.ndarray:
    return 10.0 ** (dbm / 10.0)

def mw_to_dbm(mw: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(mw)

def thermal_noise_dbm(bw_hz: float, temp_K: float = 290.0) -> float:
    """
    Thermal noise (dBm) in bandwidth bw_hz at temperature temp_K.
    Uses kTB with -174 dBm/Hz reference at 290 K.
    """
    # If temp deviates from 290 K, adjust: -174 dBm/Hz + 10*log10(T/290)
    per_hz_dbm = -174.0 + 10.0 * np.log10(max(temp_K, 1e-9) / 290.0)
    return per_hz_dbm + 10.0 * np.log10(max(bw_hz, 1.0))

# -----------------------
# Common smoothing util
# -----------------------
def blur1d(a: np.ndarray, k: int, axis: int = 0) -> np.ndarray:
    """
    Separable box blur of radius k (window size 2k) along a specified axis.
    Edge handling via 'edge' pad; returns array with same shape as input.
    """
    if k <= 0:
        return a
    a = np.asarray(a)
    if axis < 0:
        axis = a.ndim + axis
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (k, k)
    padded = np.pad(a, pad_width, mode='edge').cumsum(axis=axis)
    slicer_hi = [slice(None)] * a.ndim
    slicer_lo = [slice(None)] * a.ndim
    slicer_hi[axis] = slice(2 * k, None)
    slicer_lo[axis] = slice(None, -2 * k)
    window_sum = padded[tuple(slicer_hi)] - padded[tuple(slicer_lo)]
    return window_sum / float(2 * k)

# -----------------------
# Radio Map generator
# -----------------------
def gen_radio_map(X: int, Y: int, Z: int,
                  K: int = 7, base_noise_dbm: float = -121.45, seed: int = 1) -> np.ndarray:
    """
    Create a synthetic terrestrial interference map R[x,y,z] in dBm (no thermal noise).
    We place K Gaussian interference 'lobes' in the 3D space-frequency volume. The
    lobe peak power is scaled relative to a reference thermal noise floor (base_noise_dbm),
    but the returned tensor contains INTERFERENCE ONLY. Thermal noise is added later
    in compute_caps().
    """
    rng = np.random.default_rng(seed)
    # Interference-only power (mW), initialized to 0 to avoid double-counting noise.
    interf_mw = np.zeros((X, Y, Z), dtype=float)

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
        interf_mw += dbm_to_mw(base_noise_dbm) * peak_rel_mw * weight

    # Avoid -inf when converting 0 mW to dBm for visualization convenience.
    interf_mw = np.maximum(interf_mw, 1e-30)
    return mw_to_dbm(interf_mw)

def load_radio_map_from_mat(path: str,
                            var_name: str = "X_true",
                            units: str = "mW") -> np.ndarray:
    """
    Load a Radio Map from a MATLAB .mat file and return dBm tensor R[x,y,z].
    - units: one of {"mW", "W", "dBm"}
    - var_name: variable name inside MAT file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Radio Map file not found: {path}")
    data = loadmat(path)
    # MATLAB loader brings meta keys like __header__/__version__/__globals__
    keys = [k for k in data.keys() if not k.startswith("__")]
    pick_key = var_name if var_name in data else (keys[0] if keys else None)
    if pick_key is None:
        raise KeyError(f"No data variables found in {path}. Raw keys: {list(data.keys())}")
    if var_name not in data:
        print(f"[load_radio_map_from_mat] '{var_name}' not found. Using '{pick_key}' instead.")
    X = np.array(data[pick_key], dtype=float)
    if units.lower() == "dbm":
        R_dbm = X
    elif units.lower() == "mw":
        R_dbm = 10.0 * np.log10(np.maximum(X, 1e-30))
    elif units.lower() == "w":
        R_dbm = 10.0 * np.log10(np.maximum(X * 1e3, 1e-30))
    else:
        raise ValueError(f"Unsupported units: {units} (use 'mW', 'W', or 'dBm')")
    return R_dbm
 

 

# -----------------------
# Helper blocks (refactor run_once)
# -----------------------
def se_from_snr(snr_lin: np.ndarray, use_mcs: bool, mcs_params: Optional[Dict] = None) -> np.ndarray:
    """
    Strategy function: map SNR (linear) to SE (bits/s/Hz).
    - If use_mcs: use an MCS mapping (approximate table for now).
    - Else: Shannon log2(1+SNR).
    mcs_params reserved for future (BLER targets, code rates, etc.).
    """
    # Apply optional frequency-offset induced ICI penalty (first-order approximation)
    if mcs_params is not None:
        eps_f = float(mcs_params.get("residual_freq_hz", 0.0) or 0.0)
        if eps_f > 0.0:
            scs_khz = float(mcs_params.get("scs_khz", 30.0) or 30.0)
            T_sym = 1.0 / (scs_khz * 1e3)
            ici_factor = 1.0 + (2.0 * np.pi * eps_f * T_sym) ** 2
            snr_lin = np.asarray(snr_lin) / ici_factor
    if use_mcs:
        sinr_db = 10.0 * np.log10(np.maximum(snr_lin, 1e-12))
        if mcs_params is not None:
            sinr_db = sinr_db + float(mcs_params.get("olla_offset_db", 0.0))
            table = mcs_params.get("mcs_table", "legacy")
        else:
            table = "legacy"
        return sinr_to_se_mcs(sinr_db, table=table)
    return np.log2(1.0 + np.maximum(snr_lin, 0.0))

def se_from_snr_with_split(snr_base: Union[float, np.ndarray], k_prb: int, use_mcs: bool,
                           mcs_params: Optional[Dict] = None) -> np.ndarray:
    """Apply power-split (if k_prb>1) and map via strategy."""
    k = max(1, int(k_prb))
    return se_from_snr(np.asarray(snr_base) / float(k), use_mcs, mcs_params)

def se_from_cap_shannon_with_split(cap_se: Union[float, np.ndarray], k_prb: int) -> np.ndarray:
    """
    Fallback when only Shannon SE is available (no SNR): adjust for power split.
    gamma = (2^SE - 1)/k; SE' = log2(1+gamma)
    """
    k = max(1, int(k_prb))
    gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap_se)) - 1.0) / float(k)
    return np.log2(1.0 + gamma)

def se_metric_strategy(use_mcs: bool,
              snr_lin: Optional[np.ndarray] = None,
              cap_shannon: Optional[np.ndarray] = None,
              mcs_params: Optional[Dict] = None) -> np.ndarray:
    """
    Strategy for scheduling metric SE:
    - prefer mapping from snr_lin if provided;
    - otherwise, if not using MCS and Shannon SE is provided, return it.
    """
    if snr_lin is not None:
        return se_from_snr(snr_lin, use_mcs, mcs_params)
    if (not use_mcs) and (cap_shannon is not None):
        return cap_shannon
    raise ValueError("se_metric_strategy requires snr_lin or (cap_shannon with use_mcs=False)")

def select_radio_map(config: Dict) -> Tuple[np.ndarray, int, int, int]:
    """
    Load external Radio Map if provided; else generate synthetic interference-only map.
    Returns (R_xyz_dbm, X, Y, Z).
    """
    if config.get("radio_map_mat_path"):
        R_xyz_dbm = load_radio_map_from_mat(
            config["radio_map_mat_path"],
            var_name=config.get("radio_map_mat_var", "X_true"),
            units=config.get("radio_map_units", "mW")
        )
        if R_xyz_dbm.ndim != 3:
            raise ValueError(f"Loaded Radio Map must be 3D, got shape {R_xyz_dbm.shape}")
        X, Y, Z = R_xyz_dbm.shape
    else:
        X, Y, Z = config["X"], config["Y"], config["Z"]
        R_xyz_dbm = gen_radio_map(X, Y, Z, K=config["K_interferers"],
                                  base_noise_dbm=config["noise_dbm"], seed=config["seed"])
    return R_xyz_dbm, X, Y, Z

def generate_ue_positions(N_UE: int, X: int, Y: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random UE grid indices of shape [N_UE, 2]."""
    return np.stack([rng.integers(0, X, size=N_UE), rng.integers(0, Y, size=N_UE)], axis=1)

 

def resolve_noise_and_prb_bw(config: Dict) -> Tuple[float, Optional[float]]:
    """
    Resolve thermal noise level (dBm) and PRB bandwidth (Hz) based on config.
    Prefers explicit PRB bandwidth; else SCS; else fixed noise_dbm.
    """
    if "prb_bw_hz" in config and config["prb_bw_hz"] is not None:
        prb_bw_hz = float(config["prb_bw_hz"])
        noise_dbm = thermal_noise_dbm(prb_bw_hz, temp_K=config.get("noise_temp_K", 290.0))
    elif "scs_khz" in config and config["scs_khz"] is not None:
        prb_bw_hz = float(config["scs_khz"]) * 1e3 * 12.0
        noise_dbm = thermal_noise_dbm(prb_bw_hz, temp_K=config.get("noise_temp_K", 290.0))
    else:
        prb_bw_hz = None
        noise_dbm = config["noise_dbm"]
    return noise_dbm, prb_bw_hz

def apply_open_loop_power_control(config: Dict,
                                  L_fs_per_ue: Union[np.ndarray, float],
                                  G_rx_per_ue: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Compute per-UE P_tx if PC enabled; else return configured P_tx_dbm (scalar or array)."""
    if config.get("enable_power_control", False):
        PL_eff_db = np.asarray(L_fs_per_ue, dtype=float) - np.asarray(G_rx_per_ue, dtype=float)
        return pusch_open_loop_power(
            P_cmax_dbm=config.get("P_max_dbm", config.get("P_tx_dbm", 23.0)),
            P0_dbm=config.get("pc_P0_dbm", -90.0),
            alpha=config.get("pc_alpha", 0.8),
            PL_db=PL_eff_db,
            M_prb=int(config.get("pc_M_ref", 1)),
            delta_tf_db=0.0,
        )
    return config["P_tx_dbm"]

def compute_metric_override_static_if_needed(config: Dict,
                                             R_xyz_dbm: np.ndarray,
                                             ue_pos: np.ndarray,
                                             L_fs_per_ue,
                                             G_rx_per_ue,
                                             noise_dbm: float) -> Optional[np.ndarray]:
    """
    If estimation error/blur configured, compute a static predicted per-PRB SE metric to
    override instantaneous metric in RadioMap scheduler (matches original behavior).
    """
    if config.get("radiomap_est_error_db", 0.0) <= 0.0 and config.get("radiomap_blur_sigma", 0.0) <= 0.0:
        return None
    rng = np.random.default_rng(config["seed"])
    R_hat_dbm = R_xyz_dbm.copy()
    err_db = rng.normal(0.0, config.get("radiomap_est_error_db", 0.0), size=R_hat_dbm.shape)
    R_hat_dbm = R_hat_dbm + err_db
    sig = config.get("radiomap_blur_sigma", 0.0)
    if sig and sig > 0.0:
        k = int(max(1, round(sig)))
        if k > 1:
            R_hat_dbm = blur1d(R_hat_dbm, k, axis=0)
            R_hat_dbm = blur1d(R_hat_dbm, k, axis=1)
    cap_pred, _, _, _, snr_lin_pred, _ = compute_caps(
        R_hat_dbm, ue_pos,
        P_tx_dbm=config["P_tx_dbm"],  # keep original semantics (no PC here)
        L_fs_db=L_fs_per_ue,
        G_rx_db=G_rx_per_ue,
        shadow_db_std=config["shadow_std_db"],
        N0_dbm=noise_dbm,
        rx_nf_db=config.get("rx_nf_db", 0.0),
        impl_loss_db=config.get("impl_loss_db", 0.0),
        seed=config["seed"]
    )
    mcs_params = {
        "olla_offset_db": config.get("csi_olla_offset_db", 0.0),
        "mcs_table": config.get("csi_mcs_table", "legacy"),
        "residual_freq_hz": config.get("residual_freq_hz", 0.0),
        "scs_khz": config.get("scs_khz", 30),
    }
    return se_from_snr(snr_lin_pred, config.get("use_mcs", False), mcs_params=mcs_params) if snr_lin_pred is not None else cap_pred

def build_time_variation_if_enabled(config: Dict,
                                    R_xyz_dbm: np.ndarray,
                                    ue_pos: np.ndarray,
                                    L_fs_per_ue,
                                    G_rx_per_ue,
                                    P_tx_per_ue_dbm,
                                    noise_dbm: float,
                                    rng: np.random.Generator,
                                    metric_override: Optional[np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    """
    If time variation is enabled, build time series of per-PRB and wideband metrics
    (both SE and SNR). Mirrors the original behavior including optional prediction
    under estimation error/blur.
    Returns a dict with keys: 'se_time_rm', 'se_time_wb', 'snr_time', 'snr_wb_time'.
    """
    if not config.get("enable_time_varying", False):
        return None
    T = config["T"]
    vx, vy = config.get("rm_drift_px", (0, 0))
    flicker = float(config.get("rm_flicker_db_std", 0.0))
    se_time_rm: list = []
    se_time_wb: list = []
    snr_time: list = []
    snr_wb_time: list = []
    tau_time: list = []
    fd_time: list = []
    R_t = R_xyz_dbm.copy()
    orbit_model = OrbitModel(config, R_xyz_dbm.shape[0], R_xyz_dbm.shape[1]) if config.get("enable_orbit_dynamics", False) else None
    for t in range(T):
        if t > 0:
            if vx or vy:
                R_t = np.roll(R_t, shift=(int(vx), int(vy), 0), axis=(0, 1, 2))
            if flicker > 0.0:
                R_t = R_t + rng.normal(0.0, flicker, size=R_t.shape)

        # Predictive schedule metric under imperfect map (optional, matches original gating)
        cap_pred_t = cap_wb_pred_t = snr_lin_pred_t = snr_lin_wb_pred_t = None
        if (metric_override is None) and (config.get("radiomap_est_error_db", 0.0) > 0.0 or config.get("radiomap_blur_sigma", 0.0) > 0.0):
            R_hat_dbm = R_t.copy()
            err_db = rng.normal(0.0, config.get("radiomap_est_error_db", 0.0), size=R_hat_dbm.shape)
            R_hat_dbm = R_hat_dbm + err_db
            sig = config.get("radiomap_blur_sigma", 0.0)
            if sig and sig > 0.0:
                k = int(max(1, round(sig)))
                if k > 1:
                    R_hat_dbm = blur1d(R_hat_dbm, k, axis=0)
                    R_hat_dbm = blur1d(R_hat_dbm, k, axis=1)
            if orbit_model is not None:
                L_fs_hat, G_rx_hat, _, _ = orbit_model.get_geometry(ue_pos, t)
            else:
                L_fs_hat, G_rx_hat = L_fs_per_ue, G_rx_per_ue
            cap_pred_t, cap_wb_pred_t, _, _, snr_lin_pred_t, snr_lin_wb_pred_t = compute_caps(
                R_hat_dbm, ue_pos,
                P_tx_dbm=P_tx_per_ue_dbm,
                L_fs_db=L_fs_hat,
                G_rx_db=G_rx_hat,
                shadow_db_std=config["shadow_std_db"],
                N0_dbm=noise_dbm,
                rx_nf_db=config.get("rx_nf_db", 0.0),
                impl_loss_db=config.get("impl_loss_db", 0.0),
                seed=config["seed"]
            )

        if orbit_model is not None:
            L_fs_t, G_rx_t, tau_s_t, fd_hz_t = orbit_model.get_geometry(ue_pos, t)
            tau_time.append(tau_s_t)
            fd_time.append(fd_hz_t)
        else:
            L_fs_t, G_rx_t = L_fs_per_ue, G_rx_per_ue
        cap_t, cap_wb_t, _, _, snr_lin_t, snr_lin_wb_t = compute_caps(
            R_t, ue_pos,
            P_tx_dbm=P_tx_per_ue_dbm,
            L_fs_db=L_fs_t,
            G_rx_db=G_rx_t,
            shadow_db_std=config["shadow_std_db"],
            N0_dbm=noise_dbm,
            rx_nf_db=config.get("rx_nf_db", 0.0),
            impl_loss_db=config.get("impl_loss_db", 0.0),
            seed=config["seed"]
        )
        if orbit_model is not None and config.get("doppler_residual_fraction", 0.0) > 0.0:
            eps_f = np.abs(fd_hz_t) * float(config.get("doppler_residual_fraction", 0.0))
            T_sym = 1.0 / (float(config.get("scs_khz", 30)) * 1e3)
            ici_fac = 1.0 + (2.0 * np.pi * eps_f * T_sym) ** 2
            snr_lin_t = snr_lin_t / ici_fac.reshape(-1, 1)
            snr_lin_wb_t = snr_lin_wb_t / ici_fac
        mcs_params = {
            "olla_offset_db": config.get("csi_olla_offset_db", 0.0),
            "mcs_table": config.get("csi_mcs_table", "legacy"),
            "residual_freq_hz": config.get("residual_freq_hz", 0.0),
            "scs_khz": config.get("scs_khz", 30),
        }
        if snr_lin_pred_t is not None:
            se_time_rm.append(se_from_snr(snr_lin_pred_t, config.get("use_mcs", False), mcs_params=mcs_params))
            se_time_wb.append(se_from_snr(snr_lin_wb_pred_t, config.get("use_mcs", False), mcs_params=mcs_params))
        else:
            se_time_rm.append(se_from_snr(snr_lin_t, config.get("use_mcs", False), mcs_params=mcs_params))
            se_time_wb.append(se_from_snr(snr_lin_wb_t, config.get("use_mcs", False), mcs_params=mcs_params))
        snr_time.append(snr_lin_t)
        snr_wb_time.append(snr_lin_wb_t)

    return {
        "se_time_rm": np.stack(se_time_rm, axis=0),      # [T, UE, Z]
        "se_time_wb": np.stack(se_time_wb, axis=0),      # [T, UE]
        "snr_time": np.stack(snr_time, axis=0),          # [T, UE, Z]
        "snr_wb_time": np.stack(snr_wb_time, axis=0),    # [T, UE]
        "tau_time": None if len(tau_time) == 0 else np.stack(tau_time, axis=0),   # [T, UE]
        "fd_time": None if len(fd_time) == 0 else np.stack(fd_time, axis=0),      # [T, UE]
    }
def compute_caps(R_xyz_dbm: np.ndarray,
                 ue_pos_xy: np.ndarray,
                 P_tx_dbm: Union[np.ndarray, float],
                 L_fs_db: Union[np.ndarray, float],
                 G_rx_db: Union[np.ndarray, float],
                 shadow_db_std: float = 5.0,
                 N0_dbm: float = -121.45,
                 rx_nf_db: float = 0.0,
                 impl_loss_db: float = 0.0,
                 seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-UE per-PRB spectral efficiency based on the Radio Map.
    Returns:
      cap_shannon[UE,Z], cap_wb_shannon[UE], P_rx_dbm[UE], I_total_dbm[UE,Z],
      snr_lin[UE,Z], snr_lin_wb[UE]
    """
    rng = np.random.default_rng(seed)
    X, Y, Z = R_xyz_dbm.shape
    N_UE = ue_pos_xy.shape[0]

    # UE positions
    x_idx = ue_pos_xy[:, 0]
    y_idx = ue_pos_xy[:, 1]

    # Link budget: received power per UE (dBm)
    shadow_db = rng.normal(0.0, shadow_db_std, size=N_UE)
    # Broadcast-compatible operations for scalar or per-UE arrays
    L_fs = np.asarray(L_fs_db, dtype=float)
    G_rx = np.asarray(G_rx_db, dtype=float)
    P_tx = np.asarray(P_tx_dbm, dtype=float)
    if L_fs.ndim == 0:
        L_fs = np.full(N_UE, float(L_fs))
    if G_rx.ndim == 0:
        G_rx = np.full(N_UE, float(G_rx))
    if P_tx.ndim == 0:
        P_tx = np.full(N_UE, float(P_tx))
    P_rx_dbm = P_tx - L_fs + G_rx + shadow_db  # dBm

    # Interference + thermal noise per UE per PRB (dBm)
    I_uez_dbm = R_xyz_dbm[x_idx, y_idx, :]  # [UE,Z]
    # Effective thermal noise incl. receiver NF and implementation loss (modeled as noise rise)
    N0_eff_dbm = N0_dbm + rx_nf_db + impl_loss_db
    I_total_mw = dbm_to_mw(I_uez_dbm) + dbm_to_mw(N0_eff_dbm)
    I_total_dbm = mw_to_dbm(I_total_mw)

    # Per-PRB SNR and capacity (bits/s/Hz)
    gamma_db = (P_rx_dbm.reshape(-1, 1) - I_total_dbm)            # [UE,Z]
    snr_lin = 10.0 ** (gamma_db / 10.0)
    cap = np.log2(1.0 + snr_lin)                                  # [UE,Z]

    # Wideband (3GPP-like) interference. Use mean across subbands for a more
    # conservative and realistic CQI statistic vs median.
    I_wb_mw = np.mean(I_total_mw, axis=1)                          # [UE]
    I_wb_dbm = mw_to_dbm(I_wb_mw)
    gamma_db_wb = P_rx_dbm - I_wb_dbm
    snr_lin_wb = 10.0 ** (gamma_db_wb / 10.0)
    cap_wb = np.log2(1.0 + snr_lin_wb)                            # [UE]

    return cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb

def pf_schedule_baseline(cap_wb: np.ndarray,
                         Z: int,
                         T: int,
                         beta: float = 0.1,
                         snr_lin_wb: np.ndarray = None,
                         overhead_eff: float = 1.0,
                         use_mcs: bool = False,
                         power_split: bool = False,
                         mcs_params: Optional[Dict] = None,
                         se_metric_time: Optional[np.ndarray] = None,
                         snr_lin_wb_time: Optional[np.ndarray] = None,
                         snr_lin_prb: Optional[np.ndarray] = None,
                         cap_prb: Optional[np.ndarray] = None,
                         snr_lin_time_prb: Optional[np.ndarray] = None) -> float:
    """
    3GPP-like baseline: proportional fair with wideband CQI (same cap on every PRB).
    To avoid one-UE monopolization, assign PRBs in each TTI across the top sqrt(N) UEs 
    per PF metric, equally split.
    Returns average sum spectral efficiency per PRB (bits/s/Hz).
    """
    N_UE = cap_wb.shape[0]
    # Use Shannon cap as metric by default; if use_mcs, convert to MCS SE (k=1) for metric
    if se_metric_time is None:
        metric_se = se_metric_strategy(use_mcs, snr_lin=snr_lin_wb, cap_shannon=cap_wb, mcs_params=mcs_params)
    Rbar = np.full(N_UE, 1e-3)
    sum_rate = 0.0
    for t_idx in range(T):
        if se_metric_time is not None:
            metric_se_t = se_metric_time[t_idx]
        else:
            metric_se_t = metric_se
        metric = metric_se_t / Rbar
        U_select = min(N_UE, max(3, int(np.sqrt(N_UE))))
        if U_select < N_UE:
            top_idx = np.argpartition(-metric, U_select - 1)[:U_select]
            top_idx = top_idx[np.argsort(-metric[top_idx])]
            selected = top_idx
        else:
            selected = np.argsort(-metric)

        # Determine number of PRBs per selected UE
        alloc_counts = np.full(U_select, Z // U_select, dtype=int)
        remainder = Z - int(alloc_counts.sum())
        if remainder > 0:
            alloc_counts[:remainder] += 1

        # Build per-PRB winners by round-robin across selected set (no subband awareness)
        winners = np.full(Z, -1, dtype=int)
        rem = alloc_counts.copy()
        k_ptr = 0
        for z in range(Z):
            # find next selected UE with remaining quota
            for _ in range(U_select):
                if rem[k_ptr] > 0:
                    winners[z] = selected[k_ptr]
                    rem[k_ptr] -= 1
                    k_ptr = (k_ptr + 1) % U_select
                    break
                k_ptr = (k_ptr + 1) % U_select
            if winners[z] < 0:
                winners[z] = selected[0]

        thr_i = np.zeros(N_UE)
        # Count PRBs per UE for power split
        if power_split:
            counts = np.bincount(winners, minlength=N_UE)
        else:
            counts = np.ones(N_UE, dtype=int)
        for z in range(Z):
            ue = winners[z]
            k_prb = int(counts[ue]) if power_split else 1
            # Prefer per-PRB SNR/SE if available for fairness
            if (snr_lin_prb is not None) or (snr_lin_time_prb is not None):
                if snr_lin_time_prb is not None:
                    snr_base = snr_lin_time_prb[t_idx, ue, z]
                else:
                    snr_base = snr_lin_prb[ue, z]
                se = se_from_snr_with_split(snr_base, k_prb if power_split else 1, use_mcs, mcs_params=mcs_params)
            elif cap_prb is not None and (not use_mcs):
                se = se_from_cap_shannon_with_split(cap_prb[ue, z], k_prb if power_split else 1)
            else:
                # Fallback to wideband
                if (snr_lin_wb is not None) or (snr_lin_wb_time is not None):
                    snr_base = snr_lin_wb_time[t_idx, ue] if snr_lin_wb_time is not None else snr_lin_wb[ue]
                    se = se_from_snr_with_split(snr_base, k_prb if power_split else 1, use_mcs, mcs_params=mcs_params)
                else:
                    se = se_from_cap_shannon_with_split(metric_se_t[ue], k_prb if power_split else 1)
            thr_i[ue] += se * overhead_eff
        sum_rate += thr_i.sum()
        Rbar = (1 - beta) * Rbar + beta * thr_i

    avg_sum_rate_per_prb = sum_rate / (T * Z)
    return avg_sum_rate_per_prb

def pf_schedule_radiomap(cap: np.ndarray,
                         T: int,
                         beta: float = 0.1,
                         snr_lin: np.ndarray = None,
                         overhead_eff: float = 1.0,
                         use_mcs: bool = False,
                         power_split: bool = False,
                         se_metric_override: Optional[np.ndarray] = None,
                         max_prbs_per_ue: Optional[int] = None,
                         mcs_params: Optional[Dict] = None,
                         se_metric_time: Optional[np.ndarray] = None,
                         snr_lin_time: Optional[np.ndarray] = None) -> float:
    """
    Radio Map–aware PF: per-PRB scheduling using cap[UE,Z].
    Returns average sum spectral efficiency per PRB (bits/s/Hz).
    """
    N_UE, Z = cap.shape
    # Metric per PRB: use strategy (Shannon or MCS); allow override
    if se_metric_time is None:
        if se_metric_override is not None:
            se_metric_arr = se_metric_override
        else:
            se_metric_arr = se_metric_strategy(use_mcs, snr_lin=snr_lin, cap_shannon=cap, mcs_params=mcs_params)
    Rbar = np.full(N_UE, 1e-3)
    sum_rate = 0.0
    for t_idx in range(T):
        metric_base = se_metric_time[t_idx] if se_metric_time is not None else se_metric_arr
        metric = metric_base / Rbar.reshape(-1, 1)  # [UE,Z]
        # Winner selection with optional per-UE PRB cap
        if max_prbs_per_ue is None:
            winners = np.argmax(metric, axis=0)  # [Z]
        else:
            winners = np.full(Z, -1, dtype=int)
            counts = np.zeros(N_UE, dtype=int)
            best_vals = metric.max(axis=0)
            order_z = np.argsort(-best_vals)
            top_k = min(N_UE, max(8, int(np.sqrt(N_UE))))
            kth = max(0, N_UE - top_k)
            topk_idx = np.argpartition(metric, kth, axis=0)[kth:, :] if top_k < N_UE else np.tile(np.arange(N_UE).reshape(-1, 1), (1, Z))
            for idx in order_z:
                cands = topk_idx[:, idx]
                vals = metric[cands, idx]
                cands = cands[np.argsort(-vals)]
                chosen = -1
                for ue in cands:
                    if counts[ue] < max_prbs_per_ue:
                        chosen = int(ue)
                        break
                if chosen < 0:
                    avail = np.flatnonzero(counts < max_prbs_per_ue)
                    if avail.size > 0:
                        chosen = int(avail[np.argmax(metric[avail, idx])])
                    else:
                        chosen = int(np.argmax(metric[:, idx]))
                winners[idx] = chosen
                counts[chosen] += 1
        thr_i = np.zeros(N_UE)
        # Count PRBs per UE for power split
        if power_split:
            counts = np.bincount(winners, minlength=N_UE)
        else:
            counts = np.ones(N_UE, dtype=int)
        for z in range(Z):
            ue = winners[z]
            k_prb = int(counts[ue]) if power_split else 1
            if snr_lin is not None or snr_lin_time is not None:
                snr_base = snr_lin_time[t_idx, ue, z] if snr_lin_time is not None else snr_lin[ue, z]
                se = se_from_snr_with_split(snr_base, k_prb if power_split else 1, use_mcs, mcs_params=mcs_params)
            else:
                se = se_from_cap_shannon_with_split(cap[ue, z], k_prb if power_split else 1)
            thr_i[ue] += se * overhead_eff
        sum_rate += thr_i.sum()
        Rbar = (1 - beta) * Rbar + beta * thr_i

    avg_sum_rate_per_prb = sum_rate / (T * Z)
    return avg_sum_rate_per_prb

# -----------------------
# Experiment harness
# -----------------------
def run_once(config: Dict) -> Dict:
    """Single experiment orchestration with lower cyclomatic complexity."""
    rng = np.random.default_rng(config["seed"])
    N_UE, T = config["N_UE"], config["T"]

    # Radio map and UEs
    R_xyz_dbm, X, Y, Z = select_radio_map(config)
    ue_pos = generate_ue_positions(N_UE, X, Y, rng)

    # Geometry/beam, noise/bandwidth, power control
    L_fs_per_ue, G_rx_per_ue = compute_geometry_and_beam(config, X, Y, ue_pos)
    noise_dbm, prb_bw_hz = resolve_noise_and_prb_bw(config)
    P_tx_per_ue_dbm = apply_open_loop_power_control(config, L_fs_per_ue, G_rx_per_ue)

    # Static snapshot (also used as reference when no time-variation)
    cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb = compute_caps(
        R_xyz_dbm, ue_pos,
        P_tx_dbm=P_tx_per_ue_dbm,
        L_fs_db=L_fs_per_ue,
        G_rx_db=G_rx_per_ue,
        shadow_db_std=config["shadow_std_db"],
        N0_dbm=noise_dbm,
        rx_nf_db=config.get("rx_nf_db", 0.0),
        impl_loss_db=config.get("impl_loss_db", 0.0),
        seed=config["seed"]
    )

    # Estimation error: optional static predicted metric for RadioMap scheduler
    metric_override = compute_metric_override_static_if_needed(
        config, R_xyz_dbm, ue_pos, L_fs_per_ue, G_rx_per_ue, noise_dbm
    )

    # Optional time variation
    time_series = build_time_variation_if_enabled(
        config, R_xyz_dbm, ue_pos, L_fs_per_ue, G_rx_per_ue, P_tx_per_ue_dbm, noise_dbm, rng, metric_override
    )
    mcs_params = {
        "olla_offset_db": config.get("csi_olla_offset_db", 0.0),
        "mcs_table": config.get("csi_mcs_table", "legacy"),
        "residual_freq_hz": config.get("residual_freq_hz", 0.0),
        "scs_khz": config.get("scs_khz", 30),
    }

    if time_series is not None:
        # Apply optional CSI delay to scheduler metric (not to actual SNR)
        csi_delay = int(config.get("csi_delay_ttis", 0))
        if csi_delay > 0:
            def delay_series(arr: np.ndarray, d: int) -> np.ndarray:
                T0 = arr.shape[0]
                out = np.empty_like(arr)
                for t in range(T0):
                    src = max(0, t - d)
                    out[t] = arr[src]
                return out
            se_time_wb = delay_series(time_series["se_time_wb"], csi_delay)
            se_time_rm = delay_series(time_series["se_time_rm"], csi_delay)
        else:
            se_time_wb = time_series["se_time_wb"]
            se_time_rm = time_series["se_time_rm"]
        # Keep tau/fd for downstream users (HARQ/deferral to be added)
        tau_time = time_series.get("tau_time")
        fd_time = time_series.get("fd_time")
        base_se = pf_schedule_baseline(
            cap_wb, Z, T, beta=config["pf_beta"],
            snr_lin_wb=snr_lin_wb,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            mcs_params=mcs_params,
            se_metric_time=se_time_wb,
            snr_lin_wb_time=time_series["snr_wb_time"],
            snr_lin_prb=snr_lin,
            cap_prb=cap,
            snr_lin_time_prb=time_series["snr_time"]
        )
        map_se = pf_schedule_radiomap(
            cap, T, beta=config["pf_beta"],
            snr_lin=snr_lin,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            se_metric_override=None if metric_override is None else metric_override,
            max_prbs_per_ue=config.get("max_prbs_per_ue"),
            mcs_params=mcs_params,
            se_metric_time=se_time_rm,
            snr_lin_time=time_series["snr_time"]
        )
    else:
        base_se = pf_schedule_baseline(
            cap_wb, Z, T, beta=config["pf_beta"],
            snr_lin_wb=snr_lin_wb,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            mcs_params=mcs_params,
            snr_lin_prb=snr_lin,
            cap_prb=cap,
        )
        map_se = pf_schedule_radiomap(
            cap, T, beta=config["pf_beta"],
            snr_lin=snr_lin,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            mcs_params=mcs_params,
            se_metric_override=metric_override,
            max_prbs_per_ue=config.get("max_prbs_per_ue")
        )

    return {
        "avg_se_baseline": base_se,
        "avg_se_radiomap": map_se,
        "improvement_pct": (map_se - base_se) / max(1e-9, base_se) * 100.0,
        "R_xyz_dbm": R_xyz_dbm,
        "ue_pos": ue_pos,
        "cap": cap,
        "cap_wb": cap_wb,
        "snr_lin": snr_lin,
        "snr_lin_wb": snr_lin_wb,
        # Optional dynamics for downstream consumers
        "tau_time": None if time_series is None else time_series.get("tau_time"),
        "fd_time": None if time_series is None else time_series.get("fd_time"),
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

# CONFIG is provided by code/config.py

if __name__ == '__main__':
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
    save_plots = CONFIG.get("save_plots", True)
    show_plots = CONFIG.get("show_plots", False)
    plot_dir = CONFIG.get("plot_dir", "output")
    if save_plots and not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    def maybe_finalize(fig_name: str):
        if save_plots:
            plt.savefig(os.path.join(plot_dir, fig_name), dpi=140, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 1) Improvement distribution
    plt.figure(figsize=(6,4))
    plt.hist(multi["improvement_pct"], bins=10, edgecolor='black')
    plt.title("Radio Map–aware gain distribution across seeds")
    plt.xlabel("Gain vs. baseline (%)")
    plt.ylabel("Count")
    plt.tight_layout()
    maybe_finalize("gain_distribution.png")

    # 2) Example Interference Map slice (median over frequency)
    R_med = np.median(single["R_xyz_dbm"], axis=2)
    plt.figure(figsize=(5,5))
    plt.imshow(R_med.T, origin='lower', aspect='equal')
    plt.title("Interference Map (median over frequency), dBm")
    plt.colorbar(label='dBm')
    plt.tight_layout()
    maybe_finalize("interference_map_median.png")

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
    maybe_finalize("per_ue_wb_vs_best_subband.png")
