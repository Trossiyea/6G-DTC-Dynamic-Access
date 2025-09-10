"""
Central configuration for the NR-NTN simulation.
Edit CONFIG below to explore different regimes.
"""

CONFIG = {
    "X": 60,              # map width
    "Y": 60,              # map height
    "Z": 64,              # number of PRBs/subbands
    "N_UE": 40,           # number of UEs in footprint
    "T": 200,             # number of TTIs
    "K_interferers": 7,   # number of terrestrial interference lobes
    # Noise configuration: either specify fixed "noise_dbm" OR define PRB bandwidth below
    "noise_dbm": -121.45, # fallback if prb_bw_hz is not set (e.g., 180 kHz)
    # Option A: Set PRB bandwidth directly for kTB noise
    # "prb_bw_hz": 180000.0,  # e.g., 180 kHz (LTE 15 kHz SCS)
    # Option B: Set SCS (kHz); PRB bandwidth = 12 * SCS
    # "scs_khz": 30,          # 30 kHz => PRB 360 kHz
    # Thermal noise temperature (K)
    # "noise_temp_K": 290.0,
    "P_tx_dbm": 23.0,     # UE max EIRP (dBm). If multiple PRBs, power may split across them.
    "L_fs_db": 154.0,     # fixed free-space loss (dB) if geometry disabled
    "G_rx_db": 32.0,      # satellite RX boresight gain (dBi)
    "shadow_std_db": 5.0, # UE shadowing (dB, std. dev.)
    "pf_beta": 0.1,       # PF averaging factor
    # Calibration additions
    "rx_nf_db": 5.0,      # receiver noise figure (dB)
    "impl_loss_db": 2.0,  # implementation loss modeled as noise rise (dB)
    "overhead_eff": 0.8,  # PHY/MAC overhead efficiency factor
    "use_mcs": True,      # use MCS table instead of Shannon
    "csi_mcs_table": "nr_64qam",  # use NR CQI Table 1-equivalent SE (same as legacy values)
    "csi_olla_offset_db": 0.0,  # OLLA offset (dB), 0 keeps baseline
    "csi_delay_ttis": 0,  # CSI report delay in TTIs (applies to metric only for now)
    "power_split": True,  # per-UE PRB power splitting (P/k per PRB)
    "max_prbs_per_ue": 4,  # optional cap per UE per TTI (int)
    # Uplink open-loop power control (optional)
    "enable_power_control": False,
    "P_max_dbm": 23.0,
    "pc_P0_dbm": -90.0,
    "pc_alpha": 0.8,
    "pc_M_ref": 1,
    # Geometry + beam (optional realism)
    "enable_geometry": False,
    "sat_altitude_km": 600.0,
    "carrier_freq_GHz": 2.0,
    "beam_center_xy": None,   # default: map center if None
    "beam_half_bw_deg": 4.0,
    "beam_edge_drop_db": 3.0,
    "cell_size_km": 5.0,      # ground resolution per pixel
    # Orbit dynamics (optional realism)
    "enable_orbit_dynamics": False,
    "tti_ms": 1.0,
    "sat_ground_speed_kms": 7.5,
    "sat_heading_deg": 0.0,   # 0: +x direction in map grid
    "doppler_residual_fraction": 0.0,  # fraction of Doppler left after precompensation (0..1)
    # Time-varying Radio Map (optional realism)
    "enable_time_varying": False,
    "rm_drift_px": (0, 0),
    "rm_flicker_db_std": 0.0,
    # Radio Map estimation imperfections for scheduler metric
    "radiomap_est_error_db": 0.0,  # std dev of map error in dB
    "radiomap_blur_sigma": 0.0,    # simple box blur radius (pixels)
    # External Radio Map (optional): if provided, overrides X/Y/Z
    "radio_map_mat_path": "radio_map/Data_moderate.mat",     # use moderate-interference Data
    "radio_map_mat_var": "X_true",                 # will auto-fallback to first data var if absent
    "radio_map_units": "mW",                        # Data.mat values are mW (per PRB external interference)
    # SCS / PRB bandwidth for noise
    "scs_khz": 30,  # PRB BW=12*30kHz=360 kHz used in kTB noise
    # Residual impairments (frequency offset for ICI penalty)
    "residual_freq_hz": 0.0,
    "seed": 1             # random seed
}

