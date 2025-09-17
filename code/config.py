"""
Central configuration for the NR-NTN downlink (DL) simulation.

This repo is DL‑only. All UL‑specific features (UL power control, TA, UL
Doppler pre‑compensation) and their configs are removed. The Radio Map models
terrestrial interference at the UE receiver per PRB.
"""

import os


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


CONFIG = {
    # Grid and traffic
    "X": 60,                  # map width (matches radio_map grid)
    "Y": 60,                  # map height
    # For 20 MHz @ 30 kHz SCS, NR has 51 PRBs. Align Radio Map Z to PRB count.
    "Z": 51,                  # frequency subbands (align to NR PRBs = 51 for 20 MHz @ 30 kHz)
    "N_UE": 40,               # active UEs in footprint (per-beam slice)
    "T": 1000,                 # number of TTIs per measurement window
    "K_interferers": 7,       # used only when synthetic map is generated
    "seed": 101,              # master RNG seed for reproducibility

    'radio_map_mat_path': 'radio_map/Data_strong_51.mat',
    'radio_map_mat_var': 'X_true',
    'radio_map_units': 'mW',
    
    # Noise: prefer SCS -> PRB BW for kTB; fallback to fixed noise_dbm
    "scs_khz": 30,            # PRB BW = 12 * 30 kHz = 360 kHz (20 MHz channel => 51 PRBs)
    "cp_type": "normal",      # cyclic prefix type (FR1 normal)
    "noise_dbm": -121.45,     # fallback if PRB BW not set
    "noise_temp_K": 290.0,    # thermal noise temperature
    # Alternatively you can set: "prb_bw_hz": 360000.0

    # Channel model
    "channel_model": "3gpp_ntn",
    "ntn_channel_profile": "s_band_handheld_urban",
    # Starlink-like overrides: milder losses and stronger LoS K-factor than handheld
    "channel_params": {
        "additional_loss_db": {"slos": 7.0, "nlos": 16.0},
        "shadow_sigma_db": {"slos": 4.0, "nlos": 6.0},
        "k_factor_db": {"los": 14.0, "slos": 8.0},
    },

    # DL transmit power and link budget
    "P_tx_dbm": 33.0,         # DL per‑PRB EIRP (dBm) baseline in equal-power mode
    # If geometry disabled, use fixed path loss + composite gain
    "L_fs_db": 154.0,         # ~600 km @ 2 GHz FSPL (dB)
    "G_rx_db": 42.0,          # composite gain term (e.g., TX boresight + UT) in dB
    "shadow_std_db": 7.0,     # lognormal shadowing std (dB)
    "rx_nf_db": 5.0,          # UE receiver noise figure (dB)
    "impl_loss_db": 1.0,      # implementation loss as noise rise (dB)
    "overhead_eff": 0.85,     # PHY/MAC overhead efficiency factor (Starlink-like)
    "pf_beta": 0.1,           # PF averaging factor

    # Scheduler realism
    "use_mcs": True,          # use MCS table (vs Shannon)
    "csi_mcs_table": "nr_64qam",   # NR Table 1 (CQI->SE from 38.214)
    "csi_olla_offset_db": 0.0,      # OLLA offset (dB)
    "csi_delay_ttis": 10,           # legacy knob (kept for compatibility)
    "baseline_csi_delay_ttis": 8,   # baseline CSI delay (ms slots) per 3GPP regen assumptions (reduced)
    "rm_csi_delay_ttis": 1,         # RadioMap CSI delay (near real-time on-board, tighter)
    "power_split": False,           # DL default: no per-UE power split penalty
    "max_prbs_per_ue": 12,

    # CSI periodicity / estimation error (baseline vs Radio Map)
    "enable_cqi_periodicity_base": True,
    "enable_cqi_periodicity_rm": False,
    "cqi_period_ttis": 5,
    "cqi_offset_ttis": 0,
    "radiomap_est_error_db": 1.5,
    "radiomap_blur_sigma": 1.0,

    # Geometry + beam (NR-NTN LEO S-band)
    "enable_geometry": True,  # enable per-UE FSPL and beam gain
    "sat_altitude_km": 600.0,
    # Center frequency 2.000 GHz (1990–2010 MHz band edges, 20 MHz BW)
    "carrier_freq_GHz": 2.000,  # FR1 S-band center (1990–2010 MHz)
    "beam_center_xy": None,   # default: map center
    "beam_half_bw_deg": 8.0,
    "beam_edge_drop_db": 3.0,
    "cell_size_km": 5.0,      # ground resolution per pixel

    # Orbit dynamics (optional; keep static for baseline comparisons)
    "enable_orbit_dynamics": True,
    "tti_ms": 1.0,
    "sat_ground_speed_kms": 7.5,
    "sat_heading_deg": 0.0,   # 0: +x direction

    # Radio Map dynamics (interference drift & flicker)
    "enable_time_varying": True,
    "rm_flicker_db_std": 1.5,
    "rm_drift_px": (1, 0),

    # DL power allocation model
    # equal_prb: constant per‑PRB EIRP = P_tx_dbm
    # waterfill: total power P_tot_dbm allocated per PRB via water‑filling
    "dl_power_model": "equal_prb",
    "P_tot_dbm": None,        # set (e.g., 50.0) to enable water‑filling
    "p_min_dbm": None,
    "p_max_dbm": None,
}

# Scheduler
CONFIG.update({
    "sched_require_contiguous": True,  # one contiguous block per UE per TTI
    # Default EESM beta (fallback)
    "sched_eesm_beta_db": 2.5,
    # Per-path overrides (slightly higher to reduce freq-selective penalty)
    # Baseline tuned a bit more conservative than RM to stabilize ACKs
    "baseline_sched_eesm_beta_db": 2.7,
    "rm_sched_eesm_beta_db": 3.5,
})

# Access gating/HO removed in minimal DL-only preset

# HARQ/BLER/OLLA (optional; generic to DL)
CONFIG.update({
    "enable_harq_deferral": True,      # off by default
    "harq_max_procs": 24,
    "harq_ack_delay_ttis": 6,
    "enable_harq_full": True,          # keep off by default
    "harq_target_bler": 0.1,
    "harq_max_retx": 4,
    "mcs_table_kind": "3gpp_table_1",
    # PDSCH DMRS/overhead for N_RE computation
    "pdsch_dmrs_sym_per_slot": 1,
    "dmrs_re_per_sym_per_prb": 6,
    "oh_prb": 0,
    # BLER curve parameters (AWGN-like sigmoid)
    "bler_slope_db": 1.0,
    "bler_margin_db": 1.5,
    # Optional external BLER curves JSON (per table/MCS idx). If set, overrides AWGN model.
    "bler_curve_path": None,
    # OLLA steps (slightly stronger initial conservatism & steps)
    "olla_step_up_db": 0.06,
    "olla_step_down_db": 0.12,
    "olla_init_offset_db": -1.5,
    "olla_min_db": -3.0,
    "olla_max_db": 6.0,
    # Retransmission scheduling priority boost in PF metric (additive)
    "harq_retx_priority_bonus": 0.5,
    # Tail-ACK flush after last TTI to remove boundary loss
    "harq_flush_tail": True,
    # Optional path to 3GPP MCS tables (JSON). If set, you can choose '3gpp_table_1/2/3'.
    "mcs_3gpp_table_path": os.path.join(_BASE_DIR, "..", "docs", "mcs_tables_38_214.json"),
    # 256QAM (Table 2) now enabled per your test
    "csi_mcs_table": "nr_256qam",
    "mcs_table_kind": "3gpp_table_2",
    "enable_cqi_periodicity": False,    # legacy switch (applies to both if specific flags not set)
    # New: decouple CQI periodicity per path
    "enable_cqi_periodicity_base": True,   # apply periodic CQI to baseline metrics
    "enable_cqi_periodicity_rm": False,    # apply periodic CQI to RM metrics
    "cqi_period_ttis": 5,
    "cqi_offset_ttis": 0,
})

# Multi-beam and external orbit models removed in minimal preset

# A3/HO parameters removed

# Optional Skyfield/TLE-driven orbit (disabled by default)
CONFIG.update({
    "enable_skyfield_orbit": True,   # enable TLE-driven orbit by default per request
    # Provide either two-line TLE via 'tle_lines' (list[str,str]) or a file path via 'tle_path'
    # "tle_lines": [
    #     "1 58705C 24002A   25256.77687500  .00004774  00000+0  39334-4 0  2566",
    #     "2 58705  53.1569  66.1378 0000804  91.2245 181.2078 15.69698397    13",
    # ],
    "tle_lines": [
        "1 59421C 24065A   25259.10395833  .00000071  00000+0  58945-6 0  2596",
        "2 59421  53.1566 234.6672 0001137  93.5671 155.3063 15.69664283    16",
    ],    
    "tle_path": None,
    # "tle_name": "STARLINK-11072 [DTC]",
    "tle_name": "STARLINK-11087 [DTC]",
    # Orbit start time for t=0 (ISO8601).
    "orbit_start_datetime": "2025-09-13T18:38:42Z",
    # Mapping the simulation grid (x,y) to Earth surface around a reference lat/lon (degrees).
    # Each pixel corresponds to 'cell_size_km' in local ENU, with an optional rotation.
    # By default, auto-center the grid to the sub-satellite point at t=0 to ensure overpass.
    "auto_ref_from_tle": True,
    "ref_lat_deg": 31.2,   # fallback if auto_ref_from_tle is False
    "ref_lon_deg": 121.5,  # fallback if auto_ref_from_tle is False
    "map_rotation_deg": 0.0,
})

# Printing helpers
CONFIG.update({
    # Print concise HARQ summary after single run
    "print_harq_summary": True,
    # RadioMap-specific DL power allocation: enable water-filling with total-power constraint
    "rm_dl_power_model": "waterfill",
    # Power allocation knobs (≈ 33 dBm/PRB * 51 PRBs ≈ 50 dBm total)
    "rm_P_tot_dbm": 50.0,
    # Allow more concentration on high-SINR PRBs while avoiding starvation
    "rm_p_min_dbm": 28.0,
    "rm_p_max_dbm": 36.0,
    # Baseline now uses moderated water-filling (tighter bounds vs RM)
    "baseline_dl_power_model": "waterfill",
    # Match equal-power budget to 33 dBm/PRB * 51 PRBs ≈ 50 dBm
    "baseline_P_tot_dbm": 50.0,
    # Allow slightly wider per-PRB box constraints for better fit
    "baseline_p_min_dbm": 27.0,
    "baseline_p_max_dbm": 33.0,
    # Split per-path PRB cap to widen RM advantage
    "baseline_max_prbs_per_ue": 10,
    "rm_max_prbs_per_ue": 20,
})
