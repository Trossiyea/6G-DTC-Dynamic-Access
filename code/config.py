"""
Central configuration for the NR-NTN simulation.

This preset aligns parameters closer to a real smartphone UL for NR/NTN using
3GPP-inspired values:
- UE power class 3 (P_max ≈ 23 dBm) per 38.101/38.104.
- Fractional open‑loop power control (38.213): alpha≈0.8, P0 around −85 dBm.
- SCS = 30 kHz in FR1 (38.211); PRB BW = 12×SCS = 360 kHz for kTB noise.
- MCS mapping based on CQI table (approx. NR Table 1 / 64QAM).
- Satellite geometry enabled (LEO ~600 km, S‑band ~2 GHz) with a reasonable
  Rx array gain and modest lognormal shadowing.
"""

CONFIG = {
    # Grid and traffic
    "X": 60,                  # map width (matches radio_map grid)
    "Y": 60,                  # map height
    "Z": 64,                  # subbands in radio_map (not strict NR PRBs)
    "N_UE": 50,               # active UEs in footprint (smartphone-like load)
    "T": 200,                 # number of TTIs
    "K_interferers": 7,       # used only when synthetic map is generated

    # Noise: prefer SCS -> PRB BW for kTB; fallback to fixed noise_dbm
    # 3GPP FR1 SCS: 15/30 kHz typical; keep 30 kHz for NTN robustness
    "scs_khz": 30,            # PRB BW = 12 * 30 kHz = 360 kHz
    "noise_dbm": -121.45,     # fallback if PRB BW not set
    "noise_temp_K": 290.0,    # thermal noise temperature
    # Alternatively you can set: "prb_bw_hz": 360000.0

    # UE transmit power and link budget
    "P_tx_dbm": 23.0,         # UE EIRP when PC disabled (dBm), class-3 maximum
    # If geometry disabled, use fixed path loss + gain (kept for backward compat)
    "L_fs_db": 154.0,         # ~600 km @ 2 GHz FSPL (dB)
    "G_rx_db": 35.0,          # satellite RX boresight gain (dBi), modest phased array
    "shadow_std_db": 7.0,     # lognormal shadowing std (dB), closer to 3GPP UMa
    "rx_nf_db": 5.0,          # receiver noise figure (dB)
    "impl_loss_db": 2.0,      # implementation loss as noise rise (dB)
    "overhead_eff": 0.8,      # PHY/MAC overhead efficiency factor (0.75–0.85 typical)
    "pf_beta": 0.1,           # PF averaging factor

    # Scheduler realism
    "use_mcs": True,          # use MCS table (vs Shannon)
    "csi_mcs_table": "nr_64qam",   # NR Table 1 (64QAM-like) approximation
    "csi_olla_offset_db": 0.0,      # OLLA offset (dB)
    "csi_delay_ttis": 0,            # CSI report delay (TTIs)
    "power_split": True,            # per-UE PRB power split
    "max_prbs_per_ue": 12,          # cap per UE per TTI (keeps allocations phone-like)

    # Uplink fractional open-loop power control (TS 38.213)
    "enable_power_control": True,
    "P_max_dbm": 23.0,        # UE P_cmax (class 3)
    "pc_P0_dbm": -85.0,       # nominal P0 (combined) target
    "pc_alpha": 0.8,          # fractional pathloss compensation
    "pc_M_ref": 1,            # reference PRB count for open-loop term

    # Geometry + beam (NR-NTN smartphone to LEO S-band)
    "enable_geometry": True,  # enable per-UE FSPL and beam gain
    "sat_altitude_km": 600.0,
    "carrier_freq_GHz": 2.0,  # S-band NTN (e.g., n256 vicinity)
    "beam_center_xy": None,   # default: map center
    "beam_half_bw_deg": 20.0,
    "beam_edge_drop_db": 3.0,
    "cell_size_km": 5.0,      # ground resolution per pixel

    # Orbit dynamics (optional; keep static for baseline comparisons)
    "enable_orbit_dynamics": False,
    "tti_ms": 1.0,
    "sat_ground_speed_kms": 7.5,
    "sat_heading_deg": 0.0,   # 0: +x direction
    "doppler_residual_fraction": 0.0,  # fraction of Doppler left after precompensation (0..1)

    # Time-varying Radio Map (optional realism)
    "enable_time_varying": False,
    "rm_drift_px": (0, 0),
    "rm_flicker_db_std": 0.0,

    # Radio Map estimation imperfections for scheduler metric
    "radiomap_est_error_db": 0.0,  # std dev of map error in dB
    "radiomap_blur_sigma": 0.0,    # simple box blur radius (pixels)

    # External Radio Map (overrides X/Y/Z if provided)
    "radio_map_mat_path": "radio_map/Data_moderate.mat",  # moderate-interference dataset
    "radio_map_mat_var": "X_true",                         # variable in .mat
    "radio_map_units": "mW",                               # dataset values are mW per PRB

    # Residual impairments (frequency offset for ICI penalty)
    "residual_freq_hz": 200.0,  # small residual CFO (~0.1 ppm @ 2 GHz)

    "seed": 1                   # random seed
}
