# -*- coding: utf-8 -*-
"""
Radio Map–aware dynamic access simulation for direct-to-satellite downlink (DL, FDD).
- Baselines: a 3GPP-like wideband PF vs a Radio Map–aware per‑PRB/block PF.
- Radio Map R[x,y,z] (dBm) represents terrestrial interference at the UE receiver.
- Output: average spectral efficiency (bits/s/Hz), relative gain, and plots.

This repository has been simplified to DL only:
- All uplink-specific mechanics (UL open-loop PC, TA, UL Doppler pre‑comp) are removed.
- Transmit power is interpreted as DL EIRP (per‑PRB) or a total DL power budget.
"""

import logging
import math
import os
import json
from typing import Tuple, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Core utilities (refactored)
from core.units import dbm_to_mw, mw_to_dbm, thermal_noise_dbm, blur1d
from core.capacity import (
    SEMapper,
    se_from_snr,
    se_from_snr_with_split,
    se_from_cap_shannon_with_split,
    block_se_from_snr_vec as _block_se_from_snr_vec,
    se_metric_strategy,
    _apply_ici_penalty_lin,
)
from data_io.radiomap import load_radio_map_from_mat, select_radio_map

# Scheduler modules (refactored)
from scheduler.baseline import pf_schedule_baseline
from scheduler.radiomap import pf_schedule_radiomap_blocks
from scheduler.subband import pf_schedule_baseline_subband, group_ranges as _group_ranges

from csi import sinr_to_se_mcs, effective_sinr_eesm
from config import CONFIG
from constellation import ConstellationOrbit
from harq import HarqManager, HarqManagerFull
from link_adapt import re_per_prb_from_config, register_mcs_tables_from_file, register_bler_curves_from_file
from logging_utils import get_logger
from ntn_channel import sample_3gpp_ntn_fading
from ntn_csi import snr_to_se_sched
from orbit import compute_geometry_and_beam, OrbitModel, simple_beam_gain_db
from result_schema import SimulationResult, to_serializable_result

logger = get_logger(__name__)

# -----------------------
# Helper blocks (refactor run_once)
# -----------------------
def generate_ue_positions(N_UE: int, X: int, Y: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random UE grid indices of shape [N_UE, 2]."""
    return np.stack([rng.integers(0, X, size=N_UE), rng.integers(0, Y, size=N_UE)], axis=1)

 

def resolve_noise_and_prb_bw(config: Dict) -> Tuple[float, Optional[float]]:
    """
    Resolve thermal noise level (dBm) from SCS -> PRB BW; require SCS.
    """
    if "scs_khz" not in config or config["scs_khz"] is None:
        raise ValueError("scs_khz must be set to compute PRB bandwidth for noise")
    prb_bw_hz = float(config["scs_khz"]) * 1e3 * 12.0
    noise_dbm = thermal_noise_dbm(prb_bw_hz, temp_K=config.get("noise_temp_K", 290.0))
    return noise_dbm, prb_bw_hz

def apply_open_loop_power_control(config: Dict,
                                  L_fs_per_ue: Union[np.ndarray, float],
                                  G_rx_per_ue: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    DL-only simplification: return configured DL per‑PRB EIRP `P_tx_dbm` as-is.
    This function remains for interface compatibility.
    """
    return config["P_tx_dbm"]

def compute_metric_override_static_if_needed(config: Dict,
                                             R_xyz_dbm: np.ndarray,
                                             ue_pos: np.ndarray,
                                             L_fs_per_ue,
                                             G_rx_per_ue,
                                             noise_dbm: float,
                                             elev_deg_per_ue: Optional[np.ndarray]) -> Optional[np.ndarray]:
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
        seed=config["seed"],
        elevation_deg=elev_deg_per_ue,
        channel_model=config.get("channel_model", "3gpp_ntn"),
        channel_params=config.get("channel_params"),
        channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
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
                                    elev_deg_per_ue,
                                    P_tx_per_ue_dbm,
                                    noise_dbm: float,
                                    rng: np.random.Generator,
                                    metric_override: Optional[np.ndarray],
                                    orbit_model: Optional[object] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    If time variation is enabled, build time series of per-PRB and wideband metrics
    (both SE and SNR). Mirrors the original behavior including optional prediction
    under estimation error/blur.
    Returns a dict with keys: 'se_time_rm', 'se_time_wb', 'snr_time', 'snr_wb_time'.
    """
    if not config.get("enable_time_varying", False):
        return None
    T = config["T"]
    N_UE = int(ue_pos.shape[0])
    vx, vy = config.get("rm_drift_px", (0, 0))
    flicker = float(config.get("rm_flicker_db_std", 0.0))
    se_time_rm: list = []
    se_time_wb: list = []
    snr_time: list = []
    snr_wb_time: list = []
    tau_time: list = []
    fd_time: list = []
    R_t = R_xyz_dbm.copy()
    if orbit_model is None:
        orbit_model = OrbitModel(config, R_xyz_dbm.shape[0], R_xyz_dbm.shape[1]) if config.get("enable_orbit_dynamics", False) else None

    # DL-only: UL-specific pre-compensation and TA models removed.
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
                L_fs_hat, G_rx_hat, _, _, elev_hat = orbit_model.get_geometry(ue_pos, t)
            else:
                L_fs_hat, G_rx_hat, elev_hat = L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue
            cap_pred_t, cap_wb_pred_t, _, _, snr_lin_pred_t, snr_lin_wb_pred_t = compute_caps(
                R_hat_dbm, ue_pos,
                P_tx_dbm=P_tx_per_ue_dbm,
                L_fs_db=L_fs_hat,
                G_rx_db=G_rx_hat,
                shadow_db_std=config["shadow_std_db"],
                N0_dbm=noise_dbm,
                rx_nf_db=config.get("rx_nf_db", 0.0),
                impl_loss_db=config.get("impl_loss_db", 0.0),
                seed=config["seed"],
                elevation_deg=elev_hat,
                channel_model=config.get("channel_model", "3gpp_ntn"),
                channel_params=config.get("channel_params"),
                channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
            )

        if orbit_model is not None:
            L_fs_t, G_rx_t, tau_s_t, fd_hz_t, elev_t = orbit_model.get_geometry(ue_pos, t)
            tau_time.append(tau_s_t)
            fd_time.append(fd_hz_t)
        else:
            L_fs_t, G_rx_t, elev_t = L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue
        cap_t, cap_wb_t, _, _, snr_lin_t, snr_lin_wb_t = compute_caps(
            R_t, ue_pos,
            P_tx_dbm=P_tx_per_ue_dbm,
            L_fs_db=L_fs_t,
            G_rx_db=G_rx_t,
            shadow_db_std=config["shadow_std_db"],
            N0_dbm=noise_dbm,
            rx_nf_db=config.get("rx_nf_db", 0.0),
            impl_loss_db=config.get("impl_loss_db", 0.0),
            seed=config["seed"],
            elevation_deg=elev_t,
            channel_model=config.get("channel_model", "3gpp_ntn"),
            channel_params=config.get("channel_params"),
            channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
        )
        # DL-only: generic residual Doppler fraction -> ICI penalty (optional)
        dop_frac = float(config.get("doppler_residual_fraction", 0.0) or 0.0)
        if (orbit_model is not None) and (dop_frac > 0.0):
            eps_f = np.abs(fd_hz_t) * dop_frac
            scs_khz = float(config.get("scs_khz", 30))
            T_sym = 1.0 / (scs_khz * 1e3)
            ici_base = 1.0 + (2.0 * np.pi * eps_f * T_sym) ** 2
            ici_fac = np.maximum(1.0, ici_base * (1.0 + 2.0 * dop_frac))
            snr_lin_t = snr_lin_t / ici_fac.reshape(-1, 1)
            snr_lin_wb_t = snr_lin_wb_t / ici_fac
            if snr_lin_pred_t is not None:
                snr_lin_pred_t = snr_lin_pred_t / ici_fac.reshape(-1, 1)
            if snr_lin_wb_pred_t is not None:
                snr_lin_wb_pred_t = snr_lin_wb_pred_t / ici_fac
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
                 seed: int = 1,
                 elevation_deg: Optional[np.ndarray] = None,
                 channel_model: str = "3gpp_ntn",
                 channel_params: Optional[Dict] = None,
                 channel_profile: str = "s_band_handheld_urban") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-UE per-PRB spectral efficiency based on the Radio Map.
    Returns:
      cap_shannon[UE,Z], cap_wb_shannon[UE], P_rx_dbm[UE], I_total_dbm[UE,Z],
      snr_lin[UE,Z], snr_lin_wb[UE]
    channel_model:
      - '3gpp_ntn': three-state NTN fading (default)
      - 'lognormal': legacy independent lognormal fading with Shannon capacity
    """
    rng = np.random.default_rng(seed)
    X, Y, Z = R_xyz_dbm.shape
    N_UE = ue_pos_xy.shape[0]

    # UE positions
    x_idx = ue_pos_xy[:, 0]
    y_idx = ue_pos_xy[:, 1]

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
    channel_kind = (channel_model or "3gpp_ntn").lower().strip()
    if elevation_deg is None:
        elev_use = np.full(N_UE, 90.0, dtype=float)
    else:
        elev_use = np.asarray(elevation_deg, dtype=float)
        if elev_use.ndim == 0:
            elev_use = np.full(N_UE, float(elev_use))
    overrides = channel_params if isinstance(channel_params, dict) else None

    if channel_kind == "3gpp_ntn":
        large_scale_db, fading_lin, _ = sample_3gpp_ntn_fading(
            rng,
            elev_use,
            Z,
            profile_name=str(channel_profile or "s_band_handheld_urban"),
            overrides=overrides,
        )
        P_rx_dbm = P_tx - L_fs + G_rx + large_scale_db
    elif channel_kind in ("lognormal", "legacy"):
        shadow_db = rng.normal(0.0, shadow_db_std, size=N_UE)
        fading_lin = np.ones((N_UE, Z), dtype=float)
        P_rx_dbm = P_tx - L_fs + G_rx + shadow_db
    else:
        raise ValueError(f"Unsupported channel_model '{channel_model}'.")

    P_rx_prb_dbm = P_rx_dbm.reshape(-1, 1) + 10.0 * np.log10(np.maximum(fading_lin, 1e-12))

    # Interference + thermal noise per UE per PRB (dBm)
    I_uez_dbm = R_xyz_dbm[x_idx, y_idx, :]  # [UE,Z]
    # Effective thermal noise incl. receiver NF and implementation loss (modeled as noise rise)
    N0_eff_dbm = N0_dbm + rx_nf_db + impl_loss_db
    I_total_mw = dbm_to_mw(I_uez_dbm) + dbm_to_mw(N0_eff_dbm)
    I_total_dbm = mw_to_dbm(I_total_mw)

    # Per-PRB SNR and capacity (bits/s/Hz)
    gamma_db = P_rx_prb_dbm - I_total_dbm                          # [UE,Z]
    snr_lin = 10.0 ** (gamma_db / 10.0)
    cap = np.log2(1.0 + snr_lin)                                  # [UE,Z]

    # Wideband (3GPP-like) interference. Use mean across subbands for a more
    # conservative and realistic CQI statistic vs median.
    I_wb_mw = np.mean(I_total_mw, axis=1)                          # [UE]
    I_wb_dbm = mw_to_dbm(I_wb_mw)
    P_rx_mw = dbm_to_mw(P_rx_dbm)
    mean_fading = np.mean(fading_lin, axis=1)
    snr_lin_wb = (P_rx_mw * mean_fading) / np.maximum(I_wb_mw, 1e-30)
    snr_lin_wb = np.maximum(snr_lin_wb, 1e-12)
    cap_wb = np.log2(1.0 + snr_lin_wb)                            # [UE]

    return cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb

# -----------------------
# Experiment harness
# -----------------------
def run_once(config: Dict) -> Dict:
    """Single experiment orchestration with lower cyclomatic complexity."""
    logger.debug(
        "run_once start: N_UE=%s, T=%s, seed=%s, constellation=%s",
        config.get("N_UE"),
        config.get("T"),
        config.get("seed"),
        bool(config.get("enable_constellation", False)),
    )
    rng = np.random.default_rng(config["seed"])
    N_UE, T = config["N_UE"], config["T"]

    # Radio map and UEs
    R_xyz_dbm, X, Y, Z = select_radio_map(config)
    ue_pos = generate_ue_positions(N_UE, X, Y, rng)

    # Geometry/beam, noise/bandwidth, power settings
    L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue = compute_geometry_and_beam(config, X, Y, ue_pos)
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
        seed=config["seed"],
        elevation_deg=elev_deg_per_ue,
        channel_model=config.get("channel_model", "3gpp_ntn"),
        channel_params=config.get("channel_params"),
        channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
    )

    # Estimation error: optional static predicted metric for RadioMap scheduler
    metric_override = compute_metric_override_static_if_needed(
        config, R_xyz_dbm, ue_pos, L_fs_per_ue, G_rx_per_ue, noise_dbm, elev_deg_per_ue
    )

    # Orbit dynamics (measurement geometry) if enabled; no HO gating in minimal DL
    ue_mask_time = None
    events = None
    if config.get("enable_orbit_dynamics", False):
        orbit_model_meas = OrbitModel(config, X, Y)
    else:
        orbit_model_meas = None

    # One-time orbit mode prompt for clarity
    try:
        if orbit_model_meas is None:
            print("[Orbit] Dynamics disabled: using static geometry (no time-varying orbit).")
        else:
            tle_name = str(config.get("tle_name", "SAT"))
            start = str(config.get("orbit_start_datetime", "t0"))
            auto_ref = bool(config.get("auto_ref_from_tle", False))
            ref_lat = config.get("ref_lat_deg", None)
            ref_lon = config.get("ref_lon_deg", None)
            ref_txt = f", ref=({ref_lat:.4f}, {ref_lon:.4f})" if (isinstance(ref_lat, (int, float)) and isinstance(ref_lon, (int, float))) else ""
            print(f"[Orbit] Using Skyfield/TLE orbit: {tle_name} (start={start}), auto_ref={auto_ref}{ref_txt}.")
    except Exception:
        pass

    # Optional time variation
    time_series = build_time_variation_if_enabled(
        config, R_xyz_dbm, ue_pos, L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue, P_tx_per_ue_dbm, noise_dbm, rng,
        None if config.get("enable_time_varying", False) else metric_override,
        orbit_model=orbit_model_meas,
    )
    # Register 3GPP MCS tables from file if provided
    try:
        if config.get("mcs_3gpp_table_path"):
            register_mcs_tables_from_file(config.get("mcs_3gpp_table_path"))
    except Exception as e:
        print(f"[WARN] Failed to load 3GPP MCS tables: {e}")
    # Register BLER curves if provided
    try:
        if config.get("bler_curve_path"):
            register_bler_curves_from_file(config.get("bler_curve_path"))
    except Exception as e:
        print(f"[WARN] Failed to load BLER curves: {e}")
    mcs_params = {
        "olla_offset_db": config.get("csi_olla_offset_db", 0.0),
        "mcs_table": config.get("csi_mcs_table", "legacy"),
        "residual_freq_hz": config.get("residual_freq_hz", 0.0),
        "scs_khz": config.get("scs_khz", 30),
    }

    if time_series is not None:
        # Apply CSI delay to scheduler metrics (not to actual SNR)
        def delay_series(arr: np.ndarray, d: int) -> np.ndarray:
            if d <= 0:
                return arr
            T0 = arr.shape[0]
            out = np.empty_like(arr)
            for t in range(T0):
                src = max(0, t - d)
                out[t] = arr[src]
            return out
        def hold_series(arr: np.ndarray, period: int, offset: int = 0) -> np.ndarray:
            """Hold-last across time axis 0 given a reporting period/offset."""
            if period is None or period <= 1:
                return arr
            T0 = arr.shape[0]
            out = np.empty_like(arr)
            last = None
            for t in range(T0):
                if ((t - offset) % period) == 0:
                    out[t] = arr[t]
                    last = arr[t]
                else:
                    out[t] = arr[t] if last is None else last
            return out

        baseline_delay = int(config.get("baseline_csi_delay_ttis", 0))
        rm_delay = int(config.get("rm_csi_delay_ttis", 0))

        # Wideband metric (baseline)
        se_time_wb = delay_series(time_series["se_time_wb"], baseline_delay)

        # RadioMap metric (per PRB)
        se_time_rm = delay_series(time_series["se_time_rm"], rm_delay)

        # Baseline per-PRB SE metric derived from instantaneous SNR, then delay and optional CQI quant
        se_time_base = np.empty_like(time_series["se_time_rm"])  # [T, UE, Z]
        for tt in range(time_series["snr_time"].shape[0]):
            # Baseline fixed: use CQI quantization for metric
            se_time_base[tt] = snr_to_se_sched(
                time_series["snr_time"][tt], config.get("use_mcs", False), mcs_params,
                enable_cqi_quant=True,
                cqi_table=config.get("csi_mcs_table", "nr_256qam")
            )
        se_time_base = delay_series(se_time_base, baseline_delay)
        # Optional CQI reporting periodicity (hold-last), decoupled per-path
        period = int(config.get("cqi_period_ttis", 0) or 0)
        offset = int(config.get("cqi_offset_ttis", 0) or 0)
        if period and period > 1:
            if bool(config.get("enable_cqi_periodicity_base", False)):
                se_time_wb = hold_series(se_time_wb, period, offset)
                se_time_base = hold_series(se_time_base, period, offset)
            if bool(config.get("enable_cqi_periodicity_rm", False)):
                se_time_rm = hold_series(se_time_rm, period, offset)
        # Keep tau/fd for downstream users
        tau_time = time_series.get("tau_time")
        fd_time = time_series.get("fd_time")
        # No HO/RACH gating events in minimal DL

        # Optional HARQ (Stage-2 deferral or full Stage-3-like)
        harq_stats_base = None
        harq_stats_map = None
        harq_mgr_base = None
        harq_mgr_map = None
        if bool(config.get("enable_harq_full", False)):
            harq_mgr_base = HarqManagerFull(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                config=config,
            )
            harq_mgr_map = HarqManagerFull(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                config=config,
            )
        elif bool(config.get("enable_harq_deferral", False)):
            harq_mgr_base = HarqManager(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
            )
            harq_mgr_map = HarqManager(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
            )
        # Baseline: contiguous-block PF using per-PRB metric
        _rec_base = bool(config.get("record_assignments", False)) and (str(config.get("record_assignments_target", "rm")).lower() in ("base", "both", "all"))
        _rec_base_thr = bool(config.get("record_ue_thr", False)) and (str(config.get("record_assignments_target", "rm")).lower() in ("base", "both", "all"))
        assignments_base = [] if _rec_base else None
        ue_thr_base = [] if _rec_base_thr else None
        base_se_default = pf_schedule_radiomap_blocks(
                cap, T, beta=config["pf_beta"],
                snr_lin=snr_lin,
                overhead_eff=config.get("overhead_eff", 1.0),
                use_mcs=config.get("use_mcs", False),
                power_split=config.get("power_split", False),
                se_metric_override=None,
                max_prbs_per_ue=config.get("baseline_max_prbs_per_ue", config.get("max_prbs_per_ue")),
                mcs_params=mcs_params,
                se_metric_time=se_time_base,
                snr_lin_time=time_series["snr_time"],
                eesm_beta_db=float(config.get("baseline_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
                require_contiguous=bool(config.get("sched_require_contiguous", True)),
                rng=rng,
                ue_mask_time=None,
                harq_mgr=harq_mgr_base,
                dl_power_model=str(config.get("baseline_dl_power_model", "equal_prb")),
                P_tot_dbm=config.get("baseline_P_tot_dbm", None),
                P_ref_dbm=config.get("P_tx_dbm"),
                p_min_dbm=config.get("baseline_p_min_dbm", None),
                p_max_dbm=config.get("baseline_p_max_dbm", None),
                record_assignments=_rec_base,
                assignments_out=assignments_base,
                record_ue_thr=_rec_base_thr,
                ue_thr_out=ue_thr_base,
                config=config,
            )
        

        base_se_subband = None
        # RadioMap: contiguous-block PF with per-PRB metric
        if True:
            _rec_rm = bool(config.get("record_assignments", False)) and (str(config.get("record_assignments_target", "rm")).lower() in ("rm", "both", "all"))
            _rec_rm_thr = bool(config.get("record_ue_thr", False)) and (str(config.get("record_assignments_target", "rm")).lower() in ("rm", "both", "all"))
            assignments_rm = [] if _rec_rm else None
            ue_thr_rm = [] if _rec_rm_thr else None
            map_se = pf_schedule_radiomap_blocks(
                cap, T, beta=config["pf_beta"],
                snr_lin=snr_lin,
                overhead_eff=config.get("overhead_eff", 1.0),
                use_mcs=config.get("use_mcs", False),
                power_split=config.get("power_split", False),
                se_metric_override=None if metric_override is None else metric_override,
                max_prbs_per_ue=config.get("rm_max_prbs_per_ue", config.get("max_prbs_per_ue")),
                mcs_params=mcs_params,
                se_metric_time=se_time_rm,
                snr_lin_time=time_series["snr_time"],
                eesm_beta_db=float(config.get("rm_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
                require_contiguous=bool(config.get("sched_require_contiguous", True)),
                rng=rng,
                ue_mask_time=None,
                harq_mgr=harq_mgr_map,
                dl_power_model=str(config.get("rm_dl_power_model", "equal_prb")),
                P_tot_dbm=config.get("rm_P_tot_dbm", None),
                P_ref_dbm=config.get("P_tx_dbm"),
                p_min_dbm=config.get("rm_p_min_dbm", None),
                p_max_dbm=config.get("rm_p_max_dbm", None),
                record_assignments=_rec_rm,
                assignments_out=assignments_rm,
                record_ue_thr=_rec_rm_thr,
                ue_thr_out=ue_thr_rm,
                config=config,
            )
            sched_stats = None
        else:
            pass
        # Collect HARQ statistics if available
        if harq_mgr_base is not None and hasattr(harq_mgr_base, 'get_stats'):
            try:
                harq_stats_base = harq_mgr_base.get_stats()
            except Exception:
                harq_stats_base = None
        if harq_mgr_map is not None and hasattr(harq_mgr_map, 'get_stats'):
            try:
                harq_stats_map = harq_mgr_map.get_stats()
            except Exception:
                harq_stats_map = None
    else:
        # Baseline: contiguous-block PF using per-PRB metric
        base_se_default = pf_schedule_radiomap_blocks(
            cap, T, beta=config["pf_beta"],
            snr_lin=snr_lin,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            se_metric_override=None,
            max_prbs_per_ue=config.get("baseline_max_prbs_per_ue", config.get("max_prbs_per_ue")),
            mcs_params=mcs_params,
            se_metric_time=None,
            snr_lin_time=None,
            eesm_beta_db=float(config.get("baseline_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
            require_contiguous=bool(config.get("sched_require_contiguous", True)),
            rng=rng,
            dl_power_model=str(config.get("baseline_dl_power_model", "equal_prb")),
            P_tot_dbm=config.get("baseline_P_tot_dbm", None),
            P_ref_dbm=config.get("P_tx_dbm"),
            p_min_dbm=config.get("baseline_p_min_dbm", None),
            p_max_dbm=config.get("baseline_p_max_dbm", None),
            config=config,
        )
        
        base_se_subband = None
        # RadioMap: contiguous-block PF with per-PRB metric
        _rec_rm2 = bool(config.get("record_assignments", False)) and (str(config.get("record_assignments_target", "rm")).lower() in ("rm", "both", "all"))
        assignments_rm2 = [] if _rec_rm2 else None
        map_se = pf_schedule_radiomap_blocks(
            cap, T, beta=config["pf_beta"],
            snr_lin=snr_lin,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            se_metric_override=metric_override,
            max_prbs_per_ue=config.get("rm_max_prbs_per_ue", config.get("max_prbs_per_ue")),
            mcs_params=mcs_params,
            se_metric_time=None,
            snr_lin_time=None,
            eesm_beta_db=float(config.get("rm_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
            require_contiguous=bool(config.get("sched_require_contiguous", True)),
            rng=rng,
            ue_mask_time=None,
            dl_power_model=str(config.get("rm_dl_power_model", "equal_prb")),
            P_tot_dbm=config.get("rm_P_tot_dbm", None),
            P_ref_dbm=config.get("P_tx_dbm"),
            p_min_dbm=config.get("rm_p_min_dbm", None),
            p_max_dbm=config.get("rm_p_max_dbm", None),
            record_assignments=_rec_rm2,
            assignments_out=assignments_rm2,
            config=config,
        )
        sched_stats = None
        harq_stats_base = None
        harq_stats_map = None

    # Optional JSON report with per-UE throughput/fairness and events
    # Compute system bandwidth for throughput reporting
    try:
        sys_bw_hz = float(prb_bw_hz) * float(cap.shape[1])
    except Exception:
        sys_bw_hz = float(config.get("scs_khz", 30.0)) * 1e3 * 12.0 * float(cap.shape[1])

    report = {
        "avg_se_baseline_default": base_se_default,
        "avg_se_radiomap": map_se,
        "improvement_vs_default_pct": (map_se - base_se_default) / max(1e-9, base_se_default) * 100.0,
        "R_xyz_dbm": R_xyz_dbm,
        "ue_pos": ue_pos,
        "cap": cap,
        "cap_wb": cap_wb,
        "snr_lin": snr_lin,
        "snr_lin_wb": snr_lin_wb,
        # Bandwidth/throughput metrics
        "prb_bw_hz": float(prb_bw_hz),
        "system_bandwidth_hz": float(sys_bw_hz),
        "total_throughput_baseline_bps": float(base_se_default * sys_bw_hz),
        "total_throughput_radiomap_bps": float(map_se * sys_bw_hz),
        # Optional dynamics for downstream consumers
        "tau_time": None if time_series is None else time_series.get("tau_time"),
        "fd_time": None if time_series is None else time_series.get("fd_time"),
        "sched_stats": None,
        "harq_stats_base": harq_stats_base,
        "harq_stats_map": harq_stats_map,
    }

    # Attach PRB assignment timeline and per-UE throughput if recorded
    try:
        if 'assignments_rm' in locals() and assignments_rm is not None and len(assignments_rm) > 0:
            report["assignments_rm"] = np.stack(assignments_rm, axis=0)
        if 'assignments_base' in locals() and assignments_base is not None and len(assignments_base) > 0:
            report["assignments_base"] = np.stack(assignments_base, axis=0)
        if 'ue_thr_rm' in locals() and ue_thr_rm is not None and len(ue_thr_rm) > 0:
            thr_mat = np.stack(ue_thr_rm, axis=0)  # [T,UE]
            report["ue_thr_time_rm"] = thr_mat
            # Convert to average per-UE SE per PRB: sum over time / (T*Z)
            report["per_ue_se_rm_avg"] = (np.sum(thr_mat, axis=0) / float(max(1, config.get("T", T)) * cap.shape[1])).tolist()
        if 'ue_thr_base' in locals() and ue_thr_base is not None and len(ue_thr_base) > 0:
            thr_mat_b = np.stack(ue_thr_base, axis=0)
            report["ue_thr_time_base"] = thr_mat_b
            report["per_ue_se_base_avg"] = (np.sum(thr_mat_b, axis=0) / float(max(1, config.get("T", T)) * cap.shape[1])).tolist()
    except Exception:
        pass

    # Attach PRB assignment timeline for RM if recorded
    try:
        if 'assignments_rm' in locals() and assignments_rm is not None and len(assignments_rm) > 0:
            report["assignments_rm"] = np.stack(assignments_rm, axis=0)
        elif 'assignments_rm2' in locals() and assignments_rm2 is not None and len(assignments_rm2) > 0:
            report["assignments_rm"] = np.stack(assignments_rm2, axis=0)
    except Exception:
        pass

    # Compute per-UE avg SE (goodput) from acked bits if available
    try:
        re_per_prb = re_per_prb_from_config(config)
        T_total = int(config.get("T", T))
        Z_total = cap.shape[1]
        def per_ue_avg_se(hs):
            if not hs or not isinstance(hs, dict) or 'acked_bits_per_ue' not in hs:
                return None
            bits = np.asarray(hs['acked_bits_per_ue'], dtype=float)
            return (bits / float(max(1, re_per_prb) * T_total * Z_total)).tolist()
        se_ue_base = per_ue_avg_se(harq_stats_base)
        se_ue_map = per_ue_avg_se(harq_stats_map)
        report["per_ue_avg_se_base"] = se_ue_base
        report["per_ue_avg_se_map"] = se_ue_map
        # Map per-UE SE to throughput (bps) using system bandwidth
        if se_ue_base is not None:
            report["per_ue_throughput_baseline_bps"] = (np.asarray(se_ue_base, dtype=float) * sys_bw_hz).tolist()
            report["avg_ue_throughput_baseline_bps"] = float(np.mean(report["per_ue_throughput_baseline_bps"]))
        else:
            # Fallback: equal-share approximation
            N_UE_eff = max(1, int(config.get("N_UE", cap.shape[0])))
            report["avg_ue_throughput_baseline_bps"] = float((base_se_default * sys_bw_hz) / N_UE_eff)
            # Try recorded per-UE SE if available
            try:
                if 'per_ue_se_base_avg' in report:
                    p = np.asarray(report['per_ue_se_base_avg'], dtype=float) * sys_bw_hz
                    report["per_ue_throughput_baseline_bps"] = p.tolist()
                    report["avg_ue_throughput_baseline_bps"] = float(np.mean(p))
            except Exception:
                pass
        if se_ue_map is not None:
            report["per_ue_throughput_radiomap_bps"] = (np.asarray(se_ue_map, dtype=float) * sys_bw_hz).tolist()
            report["avg_ue_throughput_radiomap_bps"] = float(np.mean(report["per_ue_throughput_radiomap_bps"]))
        else:
            N_UE_eff = max(1, int(config.get("N_UE", cap.shape[0])))
            report["avg_ue_throughput_radiomap_bps"] = float((map_se * sys_bw_hz) / N_UE_eff)
            try:
                if 'per_ue_se_rm_avg' in report:
                    p = np.asarray(report['per_ue_se_rm_avg'], dtype=float) * sys_bw_hz
                    report["per_ue_throughput_radiomap_bps"] = p.tolist()
                    report["avg_ue_throughput_radiomap_bps"] = float(np.mean(p))
            except Exception:
                pass
        # Jain's fairness index
        def jain(x):
            if not x:
                return None
            arr = np.asarray(x, dtype=float)
            s = np.sum(arr)
            s2 = np.sum(arr * arr)
            n = arr.size
            return float((s * s) / max(1e-12, n * s2)) if s2 > 0 else 0.0
        report["fairness_jain_base"] = jain(se_ue_base) if se_ue_base is not None else None
        report["fairness_jain_map"] = jain(se_ue_map) if se_ue_map is not None else None
    except Exception:
        pass

    # Optionally write JSON to output directory
    try:
        if bool(config.get("write_json_report", False)):
            out_dir = config.get("plot_dir", "output")
            os.makedirs(out_dir, exist_ok=True)
            name = config.get("report_basename", "summary")
            path = os.path.join(out_dir, f"{name}.json")
            def serialize(obj):
                import numpy as _np
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                raise TypeError
            with open(path, 'w') as f:
                json.dump(report, f, default=serialize)
    except Exception as e:
        print(f"[WARN] Failed to write JSON report: {e}")

    logger.debug(
        "run_once complete: baseline=%.4f, radiomap=%.4f, improvement=%+.2f%%",
        float(report.get("avg_se_baseline_default", float("nan"))),
        float(report.get("avg_se_radiomap", float("nan"))),
        float(report.get("improvement_vs_default_pct", float("nan"))),
    )
    return report

def run_many(config: Dict, seeds: np.ndarray) -> Dict:
    base_def_list, map_list, imp_def_list = [], [], []
    for s in seeds:
        c2 = dict(config)
        c2["seed"] = int(s)
        out = run_once(c2)
        base_def_list.append(out["avg_se_baseline_default"])
        map_list.append(out["avg_se_radiomap"])
        imp_def_list.append(out["improvement_vs_default_pct"])
    return {
        "baseline_default": np.array(base_def_list),
        "radiomap": np.array(map_list),
        "improvement_vs_default_pct": np.array(imp_def_list),
    }


def as_simulation_result(result: Dict, dataclass: bool = False):
    """
    Convert a raw result dict into a typed SimulationResult for downstream
    consumers (web API, serialization). If dataclass=False, returns a
    JSON-friendly dict.
    """
    res = SimulationResult.from_dict(result)
    return res if dataclass else to_serializable_result(result)

# -----------------------
# Constellation (multi-satellite) runner
# -----------------------
def run_constellation(config: Dict) -> Dict:
    """Multi-satellite coverage with independent per-satellite scheduling.

    - Reads a TLE catalog (docs/DTC_tle.txt) and builds a Skyfield constellation.
    - At each TTI: compute geometry per candidate sat; associate UEs (with optional HO);
      run per-satellite PF (RadioMap blocks + wideband baseline) on the served UE subset.
    - Inter-satellite interference: not modeled.
    """
    logger.debug(
        "run_constellation start: N_UE=%s, T=%s, seed=%s, sats=max? %s",
        config.get("N_UE"),
        config.get("T"),
        config.get("seed"),
        config.get("constellation_max_ground_radius_km"),
    )
    rng = np.random.default_rng(config["seed"])
    N_UE, T = int(config["N_UE"]), int(config["T"]) 

    # Radio map and UEs
    R_xyz_dbm, X, Y, Z = select_radio_map(config)
    ue_pos = generate_ue_positions(N_UE, X, Y, rng)

    # Noise and power
    noise_dbm, prb_bw_hz = resolve_noise_and_prb_bw(config)
    P_tx_dbm = apply_open_loop_power_control(config, 0.0, 0.0)  # returns config["P_tx_dbm"]

    # Register 3GPP MCS tables and optional BLER curves for HARQ/MCS selection
    try:
        if config.get("mcs_3gpp_table_path"):
            register_mcs_tables_from_file(config.get("mcs_3gpp_table_path"))
    except Exception as e:
        print(f"[WARN] Failed to load 3GPP MCS tables: {e}")
    try:
        if config.get("bler_curve_path"):
            register_bler_curves_from_file(config.get("bler_curve_path"))
    except Exception as e:
        print(f"[WARN] Failed to load BLER curves: {e}")

    # Constellation orbit
    orbit = ConstellationOrbit(config, X, Y)

    # HO/association state
    assoc_metric_kind = str(config.get("association_metric", "snr_wb")).lower()
    ho_enabled = bool(config.get("ho_enabled", True))
    ho_hyst_db = float(config.get("ho_hyst_db", 2.0))
    ho_ttt = int(config.get("ho_ttt_ttis", 20))
    min_elev = float(config.get("min_elev_deg", 5.0))
    serving = np.full(N_UE, -1, dtype=int)
    ho_timer = np.zeros(N_UE, dtype=int)
    # Current metric in dB scale for HO comparison
    curr_metric_db = np.full(N_UE, -1e9, dtype=float)
    # HO/outage logs
    ho_events: list = [[] for _ in range(N_UE)]
    outage_ttis = np.zeros(N_UE, dtype=int)
    include_trace = bool(config.get("include_serving_trace", False))
    serving_trace = [] if include_trace else None

    # Time-varying Radio Map (optional)
    R_t = R_xyz_dbm.copy()
    vx, vy = config.get("rm_drift_px", (0, 0))
    flicker = float(config.get("rm_flicker_db_std", 0.0))
    enable_tv = bool(config.get("enable_time_varying", False))

    # KPI accumulators
    sum_rate_rm = 0.0
    sum_rate_base_def = 0.0
    kpi_per_sat: Dict[int, Dict] = {}
    # Per-satellite HARQ managers (persist across TTIs)
    harq_base_by_sat: Dict[int, HarqManagerFull] = {}
    harq_rm_by_sat: Dict[int, HarqManagerFull] = {}

    # For optional per-UE throughput (debug): not recording HARQ here
    # Progress bar for constellation TTI loop
    show_progress = bool(config.get("show_progress", True))
    pbar_iter = tqdm(range(T), desc="Constellation", unit="TTI", disable=not show_progress,
                     bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    for t_idx in pbar_iter:
        if t_idx > 0 and enable_tv:
            if vx or vy:
                R_t = np.roll(R_t, shift=(int(vx), int(vy), 0), axis=(0, 1, 2))
            if flicker > 0.0:
                R_t = R_t + rng.normal(0.0, flicker, size=R_t.shape)

        cand = orbit.candidate_indices_at(t_idx)
        if not cand:
            continue

        # Per-sat caches
        caps: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        # (cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb), but we only keep some
        snr_lin_s: Dict[int, np.ndarray] = {}
        snr_wb_s: Dict[int, np.ndarray] = {}
        cap_s: Dict[int, np.ndarray] = {}
        prx_dbm_s: Dict[int, np.ndarray] = {}
        elev_s: Dict[int, np.ndarray] = {}

        # Build geometry and capacities per candidate sat
        # Unified PRB cap and power model for fair baseline vs RM in constellation mode
        prb_cap_unified = int(config.get("constellation_prb_cap", config.get("rm_max_prbs_per_ue", 20)))
        dlpm = str(config.get("rm_dl_power_model", "equal_prb"))
        Ptot = config.get("rm_P_tot_dbm", None)
        pmin = config.get("rm_p_min_dbm", None)
        pmax = config.get("rm_p_max_dbm", None)

        for si in cand:
            L_fs, G_rx, tau_s_arr, fd_hz_arr, elev = orbit.geometry_for_sat(ue_pos, si, t_idx)
            # Cap/SNR (per UE, PRB). Channel per sat per t; no inter-sat interference.
            cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb = compute_caps(
                R_t, ue_pos,
                P_tx_dbm=P_tx_dbm,
                L_fs_db=L_fs,
                G_rx_db=G_rx,
                shadow_db_std=config["shadow_std_db"],
                N0_dbm=noise_dbm,
                rx_nf_db=config.get("rx_nf_db", 0.0),
                impl_loss_db=config.get("impl_loss_db", 0.0),
                seed=config["seed"],
                elevation_deg=elev,
                channel_model=config.get("channel_model", "3gpp_ntn"),
                channel_params=config.get("channel_params"),
                channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
            )
            snr_lin_s[si] = snr_lin
            snr_wb_s[si] = snr_lin_wb
            cap_s[si] = cap
            prx_dbm_s[si] = P_rx_dbm
            elev_s[si] = elev

        # Association + (optional) HO
        # Compute best sat per UE based on metric and min elevation
        best_sat = np.full(N_UE, -1, dtype=int)
        best_metric_db = np.full(N_UE, -1e9, dtype=float)
        for si in cand:
            elev = elev_s[si]
            vis_mask = elev >= min_elev
            if assoc_metric_kind == 'snr_wb':
                met = snr_wb_s[si]
                met_db = 10.0 * np.log10(np.maximum(1e-12, met))
            elif assoc_metric_kind == 'prx_dbm':
                met_db = prx_dbm_s[si]
            else:
                # default to snr_wb
                met = snr_wb_s[si]
                met_db = 10.0 * np.log10(np.maximum(1e-12, met))
            # Apply visibility mask
            met_db = np.where(vis_mask, met_db, -1e9)
            take = met_db > best_metric_db
            best_metric_db = np.where(take, met_db, best_metric_db)
            best_sat = np.where(take, si, best_sat)

        # Update serving with HO policy
        for ue in range(N_UE):
            s_old = int(serving[ue])
            met_old = float(curr_metric_db[ue])
            b = int(best_sat[ue])
            if b < 0:
                # No visible satellite
                serving[ue] = -1
                ho_timer[ue] = 0
                curr_metric_db[ue] = -1e9
                # Log outage start
                if s_old >= 0:
                    ho_events[ue].append({
                        "t": int(t_idx), "type": "outage_start",
                        "from": int(s_old), "to": -1,
                        "prev_metric_db": met_old,
                    })
                continue
            if serving[ue] < 0:
                serving[ue] = b
                curr_metric_db[ue] = best_metric_db[ue]
                ho_timer[ue] = 0
                ho_events[ue].append({
                    "t": int(t_idx), "type": "attach",
                    "from": -1, "to": int(b),
                    "metric_db": float(best_metric_db[ue]),
                })
                continue
            if b == serving[ue]:
                # Same serving; refresh metric
                curr_metric_db[ue] = best_metric_db[ue]
                ho_timer[ue] = 0
                # If continuing after outage end
                if s_old < 0 and serving[ue] >= 0:
                    ho_events[ue].append({
                        "t": int(t_idx), "type": "outage_end",
                        "from": -1, "to": int(serving[ue]),
                        "metric_db": float(best_metric_db[ue]),
                    })
                continue
            # Candidate different than serving
            if not ho_enabled:
                serving[ue] = b
                curr_metric_db[ue] = best_metric_db[ue]
                ho_timer[ue] = 0
                ho_events[ue].append({
                    "t": int(t_idx), "type": "handover",
                    "from": int(s_old), "to": int(b),
                    "prev_metric_db": met_old,
                    "metric_db": float(best_metric_db[ue]),
                })
                continue
            diff_db = best_metric_db[ue] - curr_metric_db[ue]
            if diff_db > ho_hyst_db:
                ho_timer[ue] += 1
                if ho_timer[ue] >= ho_ttt:
                    serving[ue] = b
                    curr_metric_db[ue] = best_metric_db[ue]
                    ho_timer[ue] = 0
                    ho_events[ue].append({
                        "t": int(t_idx), "type": "handover",
                        "from": int(s_old), "to": int(b),
                        "prev_metric_db": met_old,
                        "metric_db": float(best_metric_db[ue]),
                        "diff_db": float(diff_db),
                        "hyst_db": float(ho_hyst_db),
                        "ttt_ttis": int(ho_ttt),
                    })
            else:
                ho_timer[ue] = 0

        # Outage accumulation and optional serving trace
        outage_ttis += (serving < 0).astype(int)
        if include_trace and serving_trace is not None:
            serving_trace.append(np.array(serving, copy=True))

        # Build per-satellite UE subsets and schedule independently
        mcs_params = {
            "olla_offset_db": config.get("csi_olla_offset_db", 0.0),
            "mcs_table": config.get("csi_mcs_table", "legacy"),
            "residual_freq_hz": config.get("residual_freq_hz", 0.0),
            "scs_khz": config.get("scs_khz", 30),
        }
        per_sat_served_counts: Dict[int, int] = {}
        for si in cand:
            ue_idx = np.flatnonzero(serving == si)
            if ue_idx.size == 0:
                continue
            per_sat_served_counts[si] = int(ue_idx.size)
            # Use full UE arrays; apply scheduling mask to restrict served UEs for this satellite
            cap = cap_s[si]
            snr_lin = snr_lin_s[si]
            cap_wb = np.log2(1.0 + np.maximum(snr_wb_s[si], 1e-12))
            snr_wb = snr_wb_s[si]
            mask = np.zeros(N_UE, dtype=bool)
            mask[ue_idx] = True
            ue_mask = mask[None, :]

            # One-TTI Baseline-Default: contiguous-block PF with baseline per-PRB metric
            # Fixed: use CQI quantization to form baseline metric
            se_base_prb = snr_to_se_sched(
                snr_lin, config.get("use_mcs", False), mcs_params,
                enable_cqi_quant=True,
                cqi_table=config.get("csi_mcs_table", "nr_256qam")
            )
            # Prepare per-satellite HARQ managers if enabled
            harq_base = harq_base_by_sat.get(si)
            harq_rm = harq_rm_by_sat.get(si)
            if harq_base is None and bool(config.get("enable_harq_full", False)):
                harq_base = HarqManagerFull(
                    num_ue=N_UE,
                    num_procs=int(config.get("harq_max_procs", 16)),
                    ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                    config=config,
                )
                harq_base_by_sat[si] = harq_base
            if harq_rm is None and bool(config.get("enable_harq_full", False)):
                harq_rm = HarqManagerFull(
                    num_ue=N_UE,
                    num_procs=int(config.get("harq_max_procs", 16)),
                    ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                    config=config,
                )
                harq_rm_by_sat[si] = harq_rm

            # Disable tail flush in streaming mode (we call per TTI)
            _cfg_base = dict(config); _cfg_base['harq_flush_tail'] = False
            base = pf_schedule_radiomap_blocks(
                cap, 1, beta=config["pf_beta"],
                snr_lin=snr_lin,
                overhead_eff=config.get("overhead_eff", 1.0),
                use_mcs=config.get("use_mcs", False),
                power_split=config.get("power_split", False),
                se_metric_override=None,
                max_prbs_per_ue=prb_cap_unified,
                mcs_params=mcs_params,
                se_metric_time=se_base_prb[None, ...],
                snr_lin_time=None,
                eesm_beta_db=float(config.get("baseline_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
                require_contiguous=bool(config.get("sched_require_contiguous", True)),
                rng=rng,
                ue_mask_time=ue_mask,
                harq_mgr=harq_base,
                dl_power_model=dlpm,
                P_tot_dbm=Ptot,
                P_ref_dbm=config.get("P_tx_dbm"),
                p_min_dbm=pmin,
                p_max_dbm=pmax,
                record_assignments=False,
                assignments_out=None,
                record_ue_thr=False,
                ue_thr_out=None,
                config=_cfg_base,
            )
            _cfg_rm = dict(config); _cfg_rm['harq_flush_tail'] = False
            rm = pf_schedule_radiomap_blocks(
                cap, 1, beta=config["pf_beta"],
                snr_lin=snr_lin,
                overhead_eff=config.get("overhead_eff", 1.0),
                use_mcs=config.get("use_mcs", False),
                power_split=config.get("power_split", False),
                se_metric_override=None,
                max_prbs_per_ue=prb_cap_unified,
                mcs_params=mcs_params,
                se_metric_time=None,
                snr_lin_time=None,
                eesm_beta_db=float(config.get("rm_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
                require_contiguous=bool(config.get("sched_require_contiguous", True)),
                rng=rng,
                ue_mask_time=ue_mask,
                harq_mgr=harq_rm,
                dl_power_model=dlpm,
                P_tot_dbm=Ptot,
                P_ref_dbm=config.get("P_tx_dbm"),
                p_min_dbm=pmin,
                p_max_dbm=pmax,
                record_assignments=False,
                assignments_out=None,
                record_ue_thr=False,
                ue_thr_out=None,
                config=_cfg_rm,
            )
            sum_rate_base_def += base * Z
            sum_rate_rm += rm * Z
            # Update per-satellite KPI
            k = kpi_per_sat.get(si)
            if k is None:
                k = {
                    "name": getattr(orbit.sats[si], 'name', f"SAT-{int(si)}"),
                    "ttis_active": 0,
                    "served_ue_sum": 0,
                    "served_ue_max": 0,
                    "sum_se_base_def": 0.0,
                    "sum_se_rm": 0.0,
                }
                kpi_per_sat[si] = k
            k["ttis_active"] += 1
            k["served_ue_sum"] += int(ue_idx.size)
            k["served_ue_max"] = max(int(k["served_ue_max"]), int(ue_idx.size))
            k["sum_se_base_def"] += float(base * Z)
            k["sum_se_rm"] += float(rm * Z)

        # Initialize kpi dict if first time
        # (Declared before loop to satisfy type checker.)
        # Per-satellite no-UE case: not counted as active.

    # Average across T and PRBs (per original convention): divide by T and Z
    avg_se_base_def = sum_rate_base_def / max(1, T) / max(1, Z)
    avg_se_rm = sum_rate_rm / max(1, T) / max(1, Z)

    # Summarize per-satellite KPI
    per_sat_summary = []
    for si, k in sorted(kpi_per_sat.items(), key=lambda x: x[0]):
        tt = int(k["ttis_active"]) or 1
        per_sat_summary.append({
            "sat_index": int(si),
            "name": str(k["name"]),
            "ttis_active": int(k["ttis_active"]),
            "avg_served_ue": float(k["served_ue_sum"]) / float(tt),
            "max_served_ue": int(k["served_ue_max"]),
            "avg_se_base_default_per_prb": float(k["sum_se_base_def"]) / float(tt * max(1, Z)),
            "avg_se_rm_per_prb": float(k["sum_se_rm"]) / float(tt * max(1, Z)),
        })

    imp_pct = (avg_se_rm - avg_se_base_def) / max(1e-9, avg_se_base_def) * 100.0 if avg_se_base_def > 0 else float('inf')

    # Aggregate HARQ stats across satellites (concise summary)
    def _aggregate_harq(hdict: Dict[int, HarqManagerFull]):
        if not hdict:
            return None
        total_started = 0
        total_acked = 0
        total_dropped = 0
        total_init_ack = 0
        total_retx_weighted = 0.0
        for si, mgr in hdict.items():
            if mgr is None or not hasattr(mgr, 'get_stats'):
                continue
            hs = mgr.get_stats()
            tb_started = int(hs.get('tb_started', hs.get('initial_ack_count', 0) + hs.get('initial_nack_count', 0)))
            tb_acked = int(hs.get('tb_acked', hs.get('ack_count', 0)))
            tb_dropped = int(hs.get('tb_dropped', 0))
            init_ack = int(hs.get('initial_ack_count', 0))
            avg_retx = float(hs.get('avg_retx_per_acked', 0.0))
            total_started += tb_started
            total_acked += tb_acked
            total_dropped += tb_dropped
            total_init_ack += init_ack
            # Weighted by acked TBs
            total_retx_weighted += avg_retx * max(0, tb_acked)
        if total_started <= 0:
            return None
        ack_rate = total_acked / float(total_started)
        first_try = total_init_ack / float(total_started)
        avg_retx = (total_retx_weighted / float(total_acked)) if total_acked > 0 else 0.0
        return {
            'tb_started': int(total_started),
            'tb_acked': int(total_acked),
            'tb_dropped': int(total_dropped),
            'ack_rate': float(ack_rate),
            'first_try_ack_rate': float(first_try),
            'avg_retx_per_acked': float(avg_retx),
        }

    harq_stats_base = _aggregate_harq(harq_base_by_sat)
    harq_stats_map = _aggregate_harq(harq_rm_by_sat)

    report = {
        "avg_se_baseline_default": avg_se_base_def,
        "avg_se_radiomap": avg_se_rm,
        "improvement_vs_default_pct": imp_pct,
        "R_xyz_dbm": R_xyz_dbm,
        "ue_pos": ue_pos,
        "T": int(T),
        "Z": int(Z),
        "N_UE": int(N_UE),
        # Bandwidth/throughput metrics
        "prb_bw_hz": float(prb_bw_hz),
        "system_bandwidth_hz": float(prb_bw_hz) * float(Z),
        "total_throughput_baseline_bps": float(avg_se_base_def * prb_bw_hz * Z),
        "total_throughput_radiomap_bps": float(avg_se_rm * prb_bw_hz * Z),
        "avg_ue_throughput_baseline_bps": float((avg_se_base_def * prb_bw_hz * Z) / max(1, int(N_UE))),
        "avg_ue_throughput_radiomap_bps": float((avg_se_rm * prb_bw_hz * Z) / max(1, int(N_UE))),
        "ho_events_per_ue": ho_events,
        "handover_count_per_ue": [int(sum(1 for e in ho_events[i] if e.get("type") == "handover")) for i in range(N_UE)],
        "outage_ttis_per_ue": outage_ttis.tolist(),
        "per_sat_kpis": per_sat_summary,
        "sat_index_to_name": {int(i): getattr(orbit.sats[i], 'name', f"SAT-{int(i)}") for i in range(len(orbit.sats))},
        "harq_stats_base": harq_stats_base,
        "harq_stats_map": harq_stats_map,
    }
    if include_trace and serving_trace is not None:
        report["serving_trace"] = np.stack(serving_trace, axis=0)

    # Optional JSON report
    try:
        if bool(config.get("write_json_report", False)):
            out_dir = config.get("plot_dir", "output")
            os.makedirs(out_dir, exist_ok=True)
            name = str(config.get("report_basename", "constellation_summary"))
            path = os.path.join(out_dir, f"{name}.json")
            def serialize(obj):
                import numpy as _np
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                raise TypeError
            with open(path, 'w') as f:
                import json as _json
                _json.dump(report, f, default=serialize)
    except Exception as e:
        print(f"[WARN] Constellation JSON report failed: {e}")

    logger.debug(
        "run_constellation complete: baseline=%.4f, radiomap=%.4f, improvement=%+.2f%%",
        float(report.get("avg_se_baseline_default", float("nan"))),
        float(report.get("avg_se_radiomap", float("nan"))),
        float(report.get("improvement_vs_default_pct", float("nan"))),
    )
    return report

# CONFIG is provided by code/config.py

if __name__ == '__main__':
    # -----------------------
    # Run constellation or single-satellite experiment
    # -----------------------
    if bool(CONFIG.get("enable_constellation", False)):
        out = run_constellation(CONFIG)
        print("Constellation-run results (independent scheduling, no inter-sat interference)")
        print(f"  Baseline-Default avg SE (bits/s/Hz): {out['avg_se_baseline_default']:.3f}")
        print(f"  RadioMap         avg SE (bits/s/Hz): {out['avg_se_radiomap']:.3f}")
        try:
            print(f"  Gain vs Default (%): {out['improvement_vs_default_pct']:.2f}")
        except Exception:
            pass
        # Optional concise HARQ summary (constellation aggregate)
        if bool(CONFIG.get("print_harq_summary", True)):
            def _print_harq_const(label: str, hs: dict) -> None:
                if not hs:
                    print(f"\n[HARQ] {label}: no HARQ stats available.")
                    return
                tb_started = int(hs.get('tb_started', 0))
                tb_acked = int(hs.get('tb_acked', 0))
                tb_dropped = int(hs.get('tb_dropped', 0))
                ack_rate = float(hs.get('ack_rate', 0.0)) * 100.0
                first_try = float(hs.get('first_try_ack_rate', 0.0)) * 100.0
                avg_retx = float(hs.get('avg_retx_per_acked', 0.0))
                print(f"\n[HARQ] {label} (aggregate):")
                print(f"  TB started/ACKed/dropped: {tb_started}/{tb_acked}/{tb_dropped}  (ACK rate={ack_rate:.1f}%, first-try ACK={first_try:.1f}%)")
                print(f"  Avg retransmissions per ACKed TB: {avg_retx:.2f}")
            try:
                _print_harq_const("Baseline-Default", out.get("harq_stats_base"))
                _print_harq_const("RadioMap", out.get("harq_stats_map"))
            except Exception:
                pass
    else:
        single = run_once(CONFIG)
        print("Single-run results")
        print(f"  Baseline-Default avg SE (bits/s/Hz): {single['avg_se_baseline_default']:.3f}")
        print(f"  RadioMap        avg SE (bits/s/Hz): {single['avg_se_radiomap']:.3f}")
        print(f"  Gain vs Default (%): {single['improvement_vs_default_pct']:.2f}")
        try:
            bw_mhz = single.get('system_bandwidth_hz', 0.0) / 1e6
            th_base = single.get('total_throughput_baseline_bps', None)
            th_map = single.get('total_throughput_radiomap_bps', None)
            if th_base is not None and th_map is not None:
                print(f"  System Bandwidth: {bw_mhz:.3f} MHz")
                print(f"  Baseline-Default total throughput: {th_base/1e6:.3f} Mbps")
                print(f"  RadioMap        total throughput: {th_map/1e6:.3f} Mbps")
                if 'avg_ue_throughput_baseline_bps' in single and 'avg_ue_throughput_radiomap_bps' in single:
                    print(f"  Avg UE throughput (Baseline): {single['avg_ue_throughput_baseline_bps']/1e6:.3f} Mbps/UE")
                    print(f"  Avg UE throughput (RadioMap): {single['avg_ue_throughput_radiomap_bps']/1e6:.3f} Mbps/UE")
        except Exception:
            pass

        # Optional concise HARQ summary
        if bool(CONFIG.get("print_harq_summary", True)):
            def _print_harq(label: str, hs: dict, per_ue_key: str) -> None:
                if not hs:
                    print(f"\n[HARQ] {label}: no HARQ stats available.")
                    return
                tb_started = int(hs.get('tb_started', hs.get('initial_ack_count', 0) + hs.get('initial_nack_count', 0)))
                tb_acked = int(hs.get('tb_acked', hs.get('ack_count', 0)))
                tb_dropped = int(hs.get('tb_dropped', 0))
                init_ack = int(hs.get('initial_ack_count', 0))
                avg_retx = float(hs.get('avg_retx_per_acked', 0.0))
                olla_hist = hs.get('olla_offset_avg', []) or []
                olla_last = float(olla_hist[-1]) if len(olla_hist) > 0 else float(np.mean(hs.get('olla_last_per_ue', []) or [0.0]))
                print(f"\n[HARQ] {label}:")
                if tb_started > 0:
                    ack_rate = 100.0 * tb_acked / float(tb_started)
                    init_ack_rate = 100.0 * init_ack / float(tb_started)
                    print(f"  TB started/ACKed/dropped: {tb_started}/{tb_acked}/{tb_dropped}  (ACK rate={ack_rate:.1f}%, first-try ACK={init_ack_rate:.1f}%)")
                else:
                    print(f"  TB started/ACKed/dropped: {tb_started}/{tb_acked}/{tb_dropped}")
                print(f"  Avg retransmissions per ACKed TB: {avg_retx:.2f}")
                print(f"  OLLA avg offset (last): {olla_last:+.2f} dB")
                # Per-UE goodput (SE per PRB) if available
                per_ue = single.get(per_ue_key)
                if per_ue:
                    arr = np.asarray(per_ue, dtype=float)
                    print(f"  Per-UE avg SE: mean={arr.mean():.3f}, min={arr.min():.3f}, max={arr.max():.3f}")

            _print_harq("Baseline-Default", single.get("harq_stats_base"), "per_ue_avg_se_base")
            _print_harq("RadioMap", single.get("harq_stats_map"), "per_ue_avg_se_map")

        # -----------------------
        # Run multiple seeds to show robustness
        # -----------------------
        seeds = np.arange(1, 21)
        multi = run_many(CONFIG, seeds)
        print("\nMulti-seed summary (N=20)")
        print(f"  Baseline-Default avg SE: {multi['baseline_default'].mean():.3f} ± {multi['baseline_default'].std():.3f}")
        print(f"  RadioMap         avg SE: {multi['radiomap'].mean():.3f} ± {multi['radiomap'].std():.3f}")
        print(f"  Gain vs Default median: {np.median(multi['improvement_vs_default_pct']):.2f}% (min={multi['improvement_vs_default_pct'].min():.2f}%, max={multi['improvement_vs_default_pct'].max():.2f}%)")

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
        plt.hist(multi["improvement_vs_default_pct"], bins=10, edgecolor='black')
        plt.title("Radio Map–aware gain vs Default baseline")
        plt.xlabel("Gain vs. Default baseline (%)")
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

        # 3) Example per-UE wideband vs best-PRB capacity (first 10 UEs)
        ue = np.arange(min(10, CONFIG["N_UE"]))
        best_prb = single["cap"][ue].max(axis=1)
        wb = single["cap_wb"][ue]
        x = np.arange(ue.size)
        plt.figure(figsize=(6,4))
        plt.bar(x - 0.2, wb, width=0.4, label='Wideband (baseline)')
        plt.bar(x + 0.2, best_prb, width=0.4, label='Best PRB (RadioMap)')
        plt.xticks(x, [f"UE{int(i)}" for i in ue])
        plt.ylabel("Spectral efficiency (bits/s/Hz)")
        plt.title("Per-UE: wideband vs best PRB opportunity")
        plt.legend()
        plt.tight_layout()
        maybe_finalize("per_ue_wb_vs_best_prb.png")
