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
    "cp_type": "normal",      # cyclic prefix type (FR1 normal)
    "noise_dbm": -121.45,     # fallback if PRB BW not set
    "noise_temp_K": 290.0,    # thermal noise temperature
    # Alternatively you can set: "prb_bw_hz": 360000.0

    # UE transmit power and link budget
    "P_tx_dbm": 23.0,         # UE EIRP when PC disabled (dBm), class-3 maximum
    # If geometry disabled, use fixed path loss + gain (kept for backward compat)
    "L_fs_db": 154.0,         # ~600 km @ 2 GHz FSPL (dB)
    "G_rx_db": 38.0,          # satellite RX boresight gain (dBi), stronger phased array
    "shadow_std_db": 7.0,     # lognormal shadowing std (dB), closer to 3GPP UMa
    "rx_nf_db": 4.0,          # receiver noise figure (dB)
    "impl_loss_db": 1.5,      # implementation loss as noise rise (dB)
    "overhead_eff": 0.8,      # PHY/MAC overhead efficiency factor (0.75–0.85 typical)
    "pf_beta": 0.1,           # PF averaging factor

    # Scheduler realism
    "use_mcs": True,          # use MCS table (vs Shannon)
    "csi_mcs_table": "nr_64qam",   # NR Table 1 (64QAM-like) approximation
    "csi_olla_offset_db": 0.0,      # OLLA offset (dB)
    "csi_delay_ttis": 10,           # default CSI report delay (TTIs)
    "baseline_csi_delay_ttis": 10,  # baseline scheduler metric delay
    "rm_csi_delay_ttis": 0,         # RadioMap scheduler metric delay (predictive map can be faster)
    "power_split": True,            # per-UE PRB power split
    "max_prbs_per_ue": 6,           # tighter cap per UE to reduce power-split loss

    # Uplink fractional open-loop power control (TS 38.213)
    "enable_power_control": True,
    "P_max_dbm": 23.0,        # UE P_cmax (class 3)
    "pc_P0_dbm": -85.0,       # nominal P0 (combined) target
    "pc_alpha": 0.8,          # fractional pathloss compensation
    "pc_M_ref": 1,            # reference PRB count for open-loop term

    # Geometry + beam (NR-NTN smartphone to LEO S-band)
    "enable_geometry": True,  # enable per-UE FSPL and beam gain
    "sat_altitude_km": 600.0,
    "carrier_freq_GHz": 1.995,  # FR1 n255 uplink mid-band (~1980–2010 MHz)
    "beam_center_xy": None,   # default: map center
    "beam_half_bw_deg": 8.0,
    "beam_edge_drop_db": 3.0,
    "cell_size_km": 5.0,      # ground resolution per pixel

    # Orbit dynamics (optional; keep static for baseline comparisons)
    "enable_orbit_dynamics": True,
    "tti_ms": 1.0,
    "sat_ground_speed_kms": 7.5,
    "sat_heading_deg": 0.0,   # 0: +x direction
    "doppler_residual_fraction": 0.0,  # using explicit precomp model instead

    # NTN uplink frequency pre-compensation (Doppler prediction model)
    # Disabled by default to preserve baseline behavior.
    # When enabled, the simulator will update a predicted Doppler per UE with
    # a configurable periodicity and latency, and apply an ICI penalty based on
    # the residual frequency error eps_f = |f_d(t) - f_pred(t-lat)| plus random/quantization error.
    "enable_ntn_freq_precomp": True,
    "freq_precomp_update_ttis": 1,     # update every TTI
    "freq_precomp_latency_ttis": 1,    # 1 TTI latency
    "freq_precomp_error_std_hz": 50.0,
    "freq_precomp_quant_hz": 10.0,
    # LO/CFO budget and PTRS/DMRS tracking capability (Hz)
    "ue_lo_ppm": 0.05,
    "gnb_lo_ppm": 0.02,
    "lo_mismatch_ppm": None,          # if set, overrides ue/gnb ppm difference
    "ptrs_cfo_track_hz": 500.0,       # stronger PTRS/DMRS CFO tracking

    # NTN uplink Timing Advance (TA) model
    # Disabled by default; when enabled, TA is periodically commanded with latency
    # and granularity, and misalignment beyond CP-margins causes strong degradation.
    "enable_ta_model": True,
    # CP and TA will be derived from SCS when possible; the following allow overrides
    "cp_us": None,                  # if None, derived from SCS and cp_type
    "ta_granularity_us": None,     # if None, derived from SCS (ΔTA=16*Ts*2^μ)
    "ta_update_ttis": 10,
    "ta_latency_ttis": 1,
    "ta_margin_us": 0.2,
    "ta_drop_if_exceed": True,
    "ta_penalty_exponent": 2.0,

    # Time-varying Radio Map (optional realism)
    "enable_time_varying": True,
    "rm_drift_px": (1, 0),
    "rm_flicker_db_std": 1.5,

    # Radio Map estimation imperfections for scheduler metric
    "radiomap_est_error_db": 1.0,  # std dev of map error in dB (reduced for moderate RM advantage)
    "radiomap_blur_sigma": 0.0,    # simple box blur radius (pixels)

    # External Radio Map (overrides X/Y/Z if provided)
    "radio_map_mat_path": "radio_map/Data_moderate.mat",  # moderate-interference dataset
    "radio_map_mat_var": "X_true",                         # variable in .mat
    "radio_map_units": "mW",                               # dataset values are mW per PRB

    # Residual impairments (frequency offset for ICI penalty)
    "residual_freq_hz": 200.0,  # small constant CFO (e.g., uncompensated LO drift)

    # CSI/CQI quantization for scheduler metric (optional standard-like)
    "enable_cqi_quantization": True,
    "cqi_period_ttis": 8,
    "cqi_offset_ttis": 0,

    "seed": 1                   # random seed
}

# Scheduler enhancements (P1–P3)
# - Block scheduling with contiguous RBs and EESM-based single-MCS evaluation
# - Power-aware greedy allocation (marginal SE with power split)
# - Robust metric and light exploration for map uncertainty
CONFIG.update({
    "sched_block_mode": True,          # enable enhanced block-based scheduler
    "sched_require_contiguous": True,  # one contiguous block per UE per TTI
    "sched_eesm_beta_db": 1.0,         # EESM beta (dB)
    "sched_robust_kappa_db": 1.5,      # robustness factor (subtract kappa*sigma_dB from SINR)
    "baseline_block_mode": True,       # strong baseline: contiguous-block PF using per-PRB metric
    # New: strict wideband baseline + subband baseline
    "baseline_force_wideband_throughput": True,  # when baseline_block_mode=False, force WB-only throughput
    "enable_baseline_subband": True,             # enable second baseline based on subband CQI
    "baseline_subband_groups": 4,                # split Z PRBs into this many contiguous subbands
    "baseline_eesm_beta_db": 1.0,                # EESM beta for subband CQI
    "baseline_max_groups_per_ue": 1,             # cap groups per UE (one group per UE)
})
