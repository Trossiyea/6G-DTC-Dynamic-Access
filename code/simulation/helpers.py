# -*- coding: utf-8 -*-
"""
Simulation Helper Functions.

Extracted from main.py for modularity and testability.
These functions handle:
- UE position generation
- Noise and bandwidth resolution
- Power control
- Capacity/SNR computation
- Time-varying dynamics
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

# Core utilities
from core.units import dbm_to_mw, mw_to_dbm, thermal_noise_dbm, blur1d
from core.capacity import se_from_snr
from ntn_channel import sample_3gpp_ntn_fading


def generate_ue_positions(N_UE: int, X: int, Y: int, rng: np.random.Generator) -> np.ndarray:
    """Generate uniform random UE grid indices.

    Args:
        N_UE: Number of UEs
        X: Grid X dimension
        Y: Grid Y dimension
        rng: NumPy random generator

    Returns:
        UE positions array of shape [N_UE, 2] (x_idx, y_idx)
    """
    return np.stack([rng.integers(0, X, size=N_UE), rng.integers(0, Y, size=N_UE)], axis=1)


def resolve_noise_and_prb_bw(config: Dict) -> Tuple[float, float]:
    """Resolve thermal noise level (dBm) from SCS -> PRB bandwidth.

    Args:
        config: Configuration dict with 'scs_khz' and optionally 'noise_temp_K'

    Returns:
        Tuple of (noise_dbm, prb_bw_hz)

    Raises:
        ValueError: If scs_khz is not set
    """
    if "scs_khz" not in config or config["scs_khz"] is None:
        raise ValueError("scs_khz must be set to compute PRB bandwidth for noise")
    prb_bw_hz = float(config["scs_khz"]) * 1e3 * 12.0
    noise_dbm = thermal_noise_dbm(prb_bw_hz, temp_K=config.get("noise_temp_K", 290.0))
    return noise_dbm, prb_bw_hz


def apply_open_loop_power_control(
    config: Dict,
    L_fs_per_ue: Union[np.ndarray, float],
    G_rx_per_ue: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:
    """DL-only simplification: return configured DL per-PRB EIRP.

    This function remains for interface compatibility with potential
    future UL support.

    Args:
        config: Configuration dict with 'P_tx_dbm'
        L_fs_per_ue: Free-space path loss (unused in DL)
        G_rx_per_ue: Receiver gain (unused in DL)

    Returns:
        P_tx_dbm from config
    """
    return config["P_tx_dbm"]


def compute_caps(
    R_xyz_dbm: np.ndarray,
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
    channel_profile: str = "s_band_handheld_urban"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-UE per-PRB spectral efficiency based on the Radio Map.

    Args:
        R_xyz_dbm: Radio Map [X, Y, Z] (dBm)
        ue_pos_xy: UE positions [N_UE, 2] (grid indices)
        P_tx_dbm: Transmit power per PRB (dBm), scalar or [N_UE]
        L_fs_db: Free-space path loss (dB), scalar or [N_UE]
        G_rx_db: Receiver beam gain (dB), scalar or [N_UE]
        shadow_db_std: Shadow fading std dev (dB), for lognormal model
        N0_dbm: Thermal noise per PRB (dBm)
        rx_nf_db: Receiver noise figure (dB)
        impl_loss_db: Implementation loss (dB)
        seed: Random seed for channel sampling
        elevation_deg: Elevation angle per UE (degrees), [N_UE] or scalar
        channel_model: Channel model type ('3gpp_ntn' or 'lognormal')
        channel_params: Override parameters for channel model
        channel_profile: NTN channel profile name

    Returns:
        Tuple of:
          - cap: Shannon capacity [N_UE, Z] (bits/s/Hz)
          - cap_wb: Wideband capacity [N_UE] (bits/s/Hz)
          - P_rx_dbm: Received power [N_UE] (dBm)
          - I_total_dbm: Interference+noise [N_UE, Z] (dBm)
          - snr_lin: Linear SNR [N_UE, Z]
          - snr_lin_wb: Wideband linear SNR [N_UE]
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
    gamma_db = P_rx_prb_dbm - I_total_dbm  # [UE,Z]
    snr_lin = 10.0 ** (gamma_db / 10.0)
    cap = np.log2(1.0 + snr_lin)  # [UE,Z]

    # Wideband (3GPP-like) interference. Use mean across subbands for a more
    # conservative and realistic CQI statistic vs median.
    I_wb_mw = np.mean(I_total_mw, axis=1)  # [UE]
    I_wb_dbm = mw_to_dbm(I_wb_mw)
    P_rx_mw = dbm_to_mw(P_rx_dbm)
    mean_fading = np.mean(fading_lin, axis=1)
    snr_lin_wb = (P_rx_mw * mean_fading) / np.maximum(I_wb_mw, 1e-30)
    snr_lin_wb = np.maximum(snr_lin_wb, 1e-12)
    cap_wb = np.log2(1.0 + snr_lin_wb)  # [UE]

    return cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb


def compute_metric_override_static_if_needed(
    config: Dict,
    R_xyz_dbm: np.ndarray,
    ue_pos: np.ndarray,
    L_fs_per_ue: Union[np.ndarray, float],
    G_rx_per_ue: Union[np.ndarray, float],
    noise_dbm: float,
    elev_deg_per_ue: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Compute static predicted per-PRB SE metric for RadioMap scheduler.

    If estimation error/blur configured, compute a predicted metric to
    override instantaneous metric in RadioMap scheduler.

    Args:
        config: Configuration dict
        R_xyz_dbm: Radio Map [X, Y, Z]
        ue_pos: UE positions [N_UE, 2]
        L_fs_per_ue: Path loss per UE
        G_rx_per_ue: Beam gain per UE
        noise_dbm: Thermal noise (dBm)
        elev_deg_per_ue: Elevation per UE

    Returns:
        Predicted SE metric [N_UE, Z] or None if no error/blur configured
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
        P_tx_dbm=config["P_tx_dbm"],
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

    if snr_lin_pred is not None:
        return se_from_snr(snr_lin_pred, config.get("use_mcs", False), mcs_params=mcs_params)
    return cap_pred


def build_time_variation_if_enabled(
    config: Dict,
    R_xyz_dbm: np.ndarray,
    ue_pos: np.ndarray,
    L_fs_per_ue: Union[np.ndarray, float],
    G_rx_per_ue: Union[np.ndarray, float],
    elev_deg_per_ue: Optional[np.ndarray],
    P_tx_per_ue_dbm: Union[np.ndarray, float],
    noise_dbm: float,
    rng: np.random.Generator,
    metric_override: Optional[np.ndarray],
    orbit_model: Optional[object] = None
) -> Optional[Dict[str, np.ndarray]]:
    """Build time series of per-PRB and wideband metrics if time variation enabled.

    Args:
        config: Configuration dict
        R_xyz_dbm: Radio Map [X, Y, Z]
        ue_pos: UE positions [N_UE, 2]
        L_fs_per_ue: Path loss per UE
        G_rx_per_ue: Beam gain per UE
        elev_deg_per_ue: Elevation per UE
        P_tx_per_ue_dbm: Transmit power per UE
        noise_dbm: Thermal noise (dBm)
        rng: NumPy random generator
        metric_override: Static predicted metric (unused if time-varying)
        orbit_model: OrbitModel instance for dynamic geometry

    Returns:
        Dict with keys 'se_time_rm', 'se_time_wb', 'snr_time', 'snr_wb_time',
        'tau_time', 'fd_time', or None if time variation disabled
    """
    if not config.get("enable_time_varying", False):
        return None

    # Lazy import to avoid circular dependency
    from orbit import OrbitModel

    T = config["T"]
    N_UE = int(ue_pos.shape[0])
    X, Y, Z = R_xyz_dbm.shape

    vx, vy = config.get("rm_drift_px", (0, 0))
    flicker = float(config.get("rm_flicker_db_std", 0.0))

    se_time_rm: list = []
    se_time_wb: list = []
    snr_time: list = []
    snr_wb_time: list = []
    tau_time: list = []
    fd_time: list = []
    R_t = R_xyz_dbm.copy()

    if orbit_model is None and config.get("enable_orbit_dynamics", False):
        orbit_model = OrbitModel(config, X, Y)

    for t in range(T):
        if t > 0:
            if vx or vy:
                R_t = np.roll(R_t, shift=(int(vx), int(vy), 0), axis=(0, 1, 2))
            if flicker > 0.0:
                R_t = R_t + rng.normal(0.0, flicker, size=R_t.shape)

        # Predictive schedule metric under imperfect map (optional)
        cap_pred_t = cap_wb_pred_t = snr_lin_pred_t = snr_lin_wb_pred_t = None
        if (metric_override is None) and (
            config.get("radiomap_est_error_db", 0.0) > 0.0 or
            config.get("radiomap_blur_sigma", 0.0) > 0.0
        ):
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
        if (orbit_model is not None) and (dop_frac > 0.0) and len(fd_time) > 0:
            eps_f = np.abs(fd_time[-1]) * dop_frac
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
        "tau_time": None if len(tau_time) == 0 else np.stack(tau_time, axis=0),
        "fd_time": None if len(fd_time) == 0 else np.stack(fd_time, axis=0),
    }


def delay_series(arr: np.ndarray, d: int) -> np.ndarray:
    """Apply CSI delay to a time series array.

    Args:
        arr: Array with time axis 0, shape [T, ...]
        d: Delay in TTIs

    Returns:
        Delayed array of same shape
    """
    if d <= 0:
        return arr
    T0 = arr.shape[0]
    out = np.empty_like(arr)
    for t in range(T0):
        src = max(0, t - d)
        out[t] = arr[src]
    return out


def hold_series(arr: np.ndarray, period: int, offset: int = 0) -> np.ndarray:
    """Hold-last across time axis 0 given a reporting period/offset.

    Args:
        arr: Array with time axis 0, shape [T, ...]
        period: Reporting period in TTIs
        offset: Reporting offset in TTIs

    Returns:
        Held array of same shape
    """
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


def compute_jain_fairness(values) -> Optional[float]:
    """Compute Jain's fairness index.

    Args:
        values: List or array of per-UE values

    Returns:
        Fairness index in [0, 1], or None if empty
    """
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    s = np.sum(arr)
    s2 = np.sum(arr * arr)
    n = arr.size
    return float((s * s) / max(1e-12, n * s2)) if s2 > 0 else 0.0
