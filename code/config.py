"""
Central configuration for the NR-NTN downlink (DL) simulation.

This repo is DL‑only. All UL‑specific features (UL power control, TA, UL
Doppler pre‑compensation) and their configs are removed. The Radio Map models
terrestrial interference at the UE receiver per PRB.
"""

CONFIG = {
    # Grid and traffic
    "X": 60,                  # map width (matches radio_map grid)
    "Y": 60,                  # map height
    "Z": 64,                  # subbands in radio_map (not strict NR PRBs)
    "N_UE": 50,               # active UEs in footprint
    "T": 200,                 # number of TTIs
    "K_interferers": 7,       # used only when synthetic map is generated

    # Noise: prefer SCS -> PRB BW for kTB; fallback to fixed noise_dbm
    "scs_khz": 30,            # PRB BW = 12 * 30 kHz = 360 kHz
    "cp_type": "normal",      # cyclic prefix type (FR1 normal)
    "noise_dbm": -121.45,     # fallback if PRB BW not set
    "noise_temp_K": 290.0,    # thermal noise temperature
    # Alternatively you can set: "prb_bw_hz": 360000.0

    # DL transmit power and link budget
    "P_tx_dbm": 30.0,         # DL per‑PRB EIRP (dBm) baseline in equal-power mode
    # If geometry disabled, use fixed path loss + composite gain
    "L_fs_db": 154.0,         # ~600 km @ 2 GHz FSPL (dB)
    "G_rx_db": 38.0,          # composite gain term (e.g., TX beam boresight) in dB
    "shadow_std_db": 7.0,     # lognormal shadowing std (dB)
    "rx_nf_db": 7.0,          # UE receiver noise figure (dB)
    "impl_loss_db": 1.5,      # implementation loss as noise rise (dB)
    "overhead_eff": 0.8,      # PHY/MAC overhead efficiency factor (0.75–0.85 typical)
    "pf_beta": 0.1,           # PF averaging factor

    # Scheduler realism
    "use_mcs": True,          # use MCS table (vs Shannon)
    "csi_mcs_table": "nr_64qam",   # NR Table 1 (64QAM-like) approximation
    "csi_olla_offset_db": 0.0,      # OLLA offset (dB)
    "csi_delay_ttis": 10,           # default CSI report delay (TTIs)
    "baseline_csi_delay_ttis": 10,  # baseline scheduler metric delay
    "rm_csi_delay_ttis": 0,         # RadioMap scheduler metric delay
    "power_split": False,           # DL default: no per-UE power split penalty
    "max_prbs_per_ue": 6,

    # Geometry + beam (NR-NTN LEO S-band)
    "enable_geometry": True,  # enable per-UE FSPL and beam gain
    "sat_altitude_km": 600.0,
    "carrier_freq_GHz": 1.995,  # FR1 n255 S-band
    "beam_center_xy": None,   # default: map center
    "beam_half_bw_deg": 8.0,
    "beam_edge_drop_db": 3.0,
    "cell_size_km": 5.0,      # ground resolution per pixel

    # Orbit dynamics (optional; keep static for baseline comparisons)
    "enable_orbit_dynamics": True,
    "tti_ms": 1.0,
    "sat_ground_speed_kms": 7.5,
    "sat_heading_deg": 0.0,   # 0: +x direction

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
    "sched_eesm_beta_db": 1.0,
})

# Access gating/HO removed in minimal DL-only preset

# HARQ/BLER/OLLA (optional; generic to DL)
CONFIG.update({
    "enable_harq_deferral": False,      # off by default
    "harq_max_procs": 16,               # typical NR max processes
    "harq_ack_delay_ttis": 10,          # example ACK delay (TTIs)
    "enable_harq_full": False,          # keep off by default
    "harq_target_bler": 0.1,
    "harq_max_retx": 4,
    "mcs_table_kind": "table_1_64qam",
    # PDSCH DMRS/overhead for N_RE computation
    "pdsch_dmrs_sym_per_slot": 1,
    "dmrs_re_per_sym_per_prb": 6,
    "oh_prb": 0,
    # BLER curve parameters (AWGN-like sigmoid)
    "bler_slope_db": 1.0,
    "bler_margin_db": 1.5,
    # Optional external BLER curves JSON (per table/MCS idx). If set, overrides AWGN model.
    "bler_curve_path": None,
    # OLLA steps
    "olla_step_up_db": 0.1,
    "olla_step_down_db": 0.1,
    "olla_init_offset_db": 0.0,
    # Retransmission scheduling priority boost in PF metric (additive)
    "harq_retx_priority_bonus": 0.0,
    # Optional path to 3GPP MCS tables (JSON). If set, you can choose '3gpp_table_1/2/3'.
    "mcs_3gpp_table_path": None,
    "enable_cqi_periodicity": False,    # hold-last CQI reporting with period/offset
})

# Multi-beam and external orbit models removed in minimal preset

# A3/HO parameters removed
