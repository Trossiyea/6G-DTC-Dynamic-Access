"""
Configuration schema definitions using dataclasses.

This module defines type-safe configuration models with validation,
documentation, and JSON Schema generation support.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, get_type_hints


# =============================================================================
# Base Configuration Class
# =============================================================================

@dataclass
class ConfigGroup:
    """Base class for configuration groups with common utilities."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigGroup":
        """Create instance from dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def update(self, data: Dict[str, Any]) -> None:
        """Update fields from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def get_field_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get field metadata for documentation/UI generation."""
        result = {}
        hints = get_type_hints(cls) if hasattr(cls, "__annotations__") else {}
        for f in fields(cls):
            info = {
                "type": str(hints.get(f.name, f.type)),
                "default": f.default if f.default is not field(default=None).default else None,
                "description": f.metadata.get("description", ""),
                "units": f.metadata.get("units", ""),
                "min": f.metadata.get("min"),
                "max": f.metadata.get("max"),
                "options": f.metadata.get("options"),
            }
            # Handle default_factory
            if f.default_factory is not field(default=None).default_factory:
                try:
                    info["default"] = f.default_factory()
                except Exception:
                    pass
            result[f.name] = info
        return result


# =============================================================================
# Configuration Groups
# =============================================================================

@dataclass
class SimulationConfig(ConfigGroup):
    """Core simulation parameters."""

    Z: int = field(default=51, metadata={
        "description": "Number of PRBs (must match Radio Map Z dimension)",
        "min": 1, "max": 275
    })
    N_UE: int = field(default=100, metadata={
        "description": "Number of active UEs per run",
        "min": 1, "max": 10000
    })
    T: int = field(default=2000, metadata={
        "description": "Number of TTIs (subframes) per run",
        "min": 1
    })
    seed: int = field(default=101, metadata={
        "description": "Random number generator seed for reproducibility"
    })


@dataclass
class RadioMapConfig(ConfigGroup):
    """Radio Map input configuration."""

    radio_map_mat_path: str = field(
        default="radio_map/Toronto/RadioMap/RM_toronto125_dBm.mat",
        metadata={"description": "Path to Radio Map file (MAT or HDF5 format)"}
    )
    radio_map_mat_var: str = field(
        default="XdB_recon_tensor",
        metadata={"description": "Variable name in MAT file containing the radio map"}
    )
    radio_map_units: str = field(
        default="dBm",
        metadata={
            "description": "Units of Radio Map values",
            "options": ["dBm", "mW", "W"]
        }
    )
    radiomap_est_error_db: float = field(default=1.5, metadata={
        "description": "Radio Map estimation error standard deviation",
        "units": "dB", "min": 0.0
    })
    radiomap_blur_sigma: float = field(default=1.0, metadata={
        "description": "Gaussian blur sigma for Radio Map spatial smoothing",
        "min": 0.0
    })


@dataclass
class NumerologyConfig(ConfigGroup):
    """Noise and numerology configuration."""

    scs_khz: int = field(default=30, metadata={
        "description": "Subcarrier spacing",
        "units": "kHz",
        "options": [15, 30, 60, 120]
    })
    cp_type: str = field(default="normal", metadata={
        "description": "Cyclic prefix type",
        "options": ["normal", "extended"]
    })
    noise_temp_K: float = field(default=290.0, metadata={
        "description": "Noise temperature for thermal noise calculation",
        "units": "K", "min": 0.0
    })


@dataclass
class ChannelConfig(ConfigGroup):
    """Channel model configuration."""

    channel_model: str = field(default="3gpp_ntn", metadata={
        "description": "Channel model type",
        "options": ["3gpp_ntn", "lognormal", "legacy"]
    })
    ntn_channel_profile: str = field(default="s_band_handheld_urban", metadata={
        "description": "NTN channel profile name"
    })
    channel_params: Dict[str, Any] = field(default_factory=lambda: {
        "additional_loss_db": {"slos": 7.0, "nlos": 16.0},
        "shadow_sigma_db": {"slos": 4.0, "nlos": 6.0},
        "k_factor_db": {"los": 14.0, "slos": 8.0},
    }, metadata={"description": "Channel model parameter overrides"})


@dataclass
class LinkBudgetConfig(ConfigGroup):
    """Link budget parameters."""

    P_tx_dbm: float = field(default=30.0, metadata={
        "description": "DL per-PRB EIRP baseline (equal-power)",
        "units": "dBm"
    })
    G_rx_db: float = field(default=38.0, metadata={
        "description": "Beam boresight gain",
        "units": "dB"
    })
    shadow_std_db: float = field(default=7.0, metadata={
        "description": "Lognormal shadowing standard deviation",
        "units": "dB", "min": 0.0
    })
    rx_nf_db: float = field(default=7.0, metadata={
        "description": "UE receiver noise figure",
        "units": "dB", "min": 0.0
    })
    impl_loss_db: float = field(default=1.0, metadata={
        "description": "Implementation loss modeled as noise rise",
        "units": "dB", "min": 0.0
    })
    overhead_eff: float = field(default=0.85, metadata={
        "description": "PHY/MAC overhead efficiency factor",
        "min": 0.0, "max": 1.0
    })


@dataclass
class GeometryConfig(ConfigGroup):
    """Satellite geometry and beam configuration."""

    sat_altitude_km: float = field(default=600.0, metadata={
        "description": "Satellite altitude",
        "units": "km", "min": 100.0
    })
    carrier_freq_GHz: float = field(default=2.0, metadata={
        "description": "Carrier frequency",
        "units": "GHz", "min": 0.1
    })
    beam_center_xy: Optional[Tuple[float, float]] = field(default=None, metadata={
        "description": "Beam center coordinates (None = map center)"
    })
    beam_half_bw_deg: float = field(default=8.0, metadata={
        "description": "Beam half-power beamwidth",
        "units": "degrees", "min": 0.1
    })
    beam_edge_drop_db: float = field(default=3.0, metadata={
        "description": "Beam edge gain drop",
        "units": "dB", "min": 0.0
    })
    cell_size_km: float = field(default=0.125, metadata={
        "description": "Ground resolution per pixel",
        "units": "km", "min": 0.001
    })


@dataclass
class OrbitConfig(ConfigGroup):
    """Orbit dynamics configuration."""

    enable_orbit_dynamics: bool = field(default=True, metadata={
        "description": "Enable time-varying orbit geometry"
    })
    tti_ms: float = field(default=1.0, metadata={
        "description": "TTI duration",
        "units": "ms", "min": 0.1
    })
    sat_ground_speed_kms: float = field(default=7.5, metadata={
        "description": "Satellite ground track speed",
        "units": "km/s"
    })
    sat_heading_deg: float = field(default=0.0, metadata={
        "description": "Satellite heading angle",
        "units": "degrees"
    })

    # TLE parameters
    tle_name: str = field(default="STARLINK-11090 [DTC]", metadata={
        "description": "TLE satellite name"
    })
    tle_lines: Optional[List[str]] = field(default_factory=lambda: [
        "1 59422C 24065B   25266.77548611  .00029064  00000+0  23954-3 0  2662",
        "2 59422  53.1572 196.6800 0001379  82.5466  65.8632 15.69667376    15",
    ], metadata={"description": "Two-line TLE elements"})
    tle_path: Optional[str] = field(default=None, metadata={
        "description": "Path to TLE file (alternative to tle_lines)"
    })
    orbit_start_datetime: str = field(
        default="2025-10-07T21:38:31.574982+00:00",
        metadata={"description": "Orbit start time (ISO8601 format)"}
    )

    # Reference location
    auto_ref_from_tle: bool = field(default=False, metadata={
        "description": "Auto-derive reference location from TLE"
    })
    ref_lat_deg: float = field(default=43.65108, metadata={
        "description": "Reference latitude",
        "units": "degrees", "min": -90.0, "max": 90.0
    })
    ref_lon_deg: float = field(default=-79.34702, metadata={
        "description": "Reference longitude",
        "units": "degrees", "min": -180.0, "max": 180.0
    })
    map_rotation_deg: float = field(default=0.0, metadata={
        "description": "Map rotation angle",
        "units": "degrees"
    })


@dataclass
class TimeVaryingConfig(ConfigGroup):
    """Time-varying Radio Map dynamics."""

    enable_time_varying: bool = field(default=True, metadata={
        "description": "Enable time-varying Radio Map dynamics"
    })
    rm_flicker_db_std: float = field(default=0.5, metadata={
        "description": "Radio Map temporal flicker standard deviation",
        "units": "dB", "min": 0.0
    })
    rm_drift_px: Tuple[int, int] = field(default=(0, 0), metadata={
        "description": "Radio Map spatial drift per TTI (x, y pixels)"
    })


@dataclass
class CSIConfig(ConfigGroup):
    """CSI feedback and periodicity configuration."""

    csi_olla_offset_db: float = field(default=0.0, metadata={
        "description": "OLLA offset for CSI",
        "units": "dB"
    })
    baseline_csi_delay_ttis: int = field(default=12, metadata={
        "description": "Baseline CSI feedback delay",
        "units": "TTIs", "min": 0
    })
    rm_csi_delay_ttis: int = field(default=0, metadata={
        "description": "RadioMap CSI feedback delay",
        "units": "TTIs", "min": 0
    })
    enable_cqi_periodicity_base: bool = field(default=True, metadata={
        "description": "Enable CQI reporting periodicity for baseline"
    })
    enable_cqi_periodicity_rm: bool = field(default=False, metadata={
        "description": "Enable CQI reporting periodicity for RadioMap"
    })
    cqi_period_ttis: int = field(default=5, metadata={
        "description": "CQI reporting period",
        "units": "TTIs", "min": 1
    })
    cqi_offset_ttis: int = field(default=0, metadata={
        "description": "CQI reporting offset",
        "units": "TTIs", "min": 0
    })


@dataclass
class SchedulerConfig(ConfigGroup):
    """Scheduler configuration."""

    pf_beta: float = field(default=0.1, metadata={
        "description": "Proportional fair averaging factor",
        "min": 0.0, "max": 1.0
    })
    use_mcs: bool = field(default=True, metadata={
        "description": "Map SNR to SE via MCS table"
    })
    power_split: bool = field(default=False, metadata={
        "description": "Enable per-UE power split penalty"
    })
    sched_require_contiguous: bool = field(default=True, metadata={
        "description": "Require contiguous PRB blocks per UE per TTI"
    })
    sched_eesm_beta_db: float = field(default=2.5, metadata={
        "description": "Default EESM beta (fallback)",
        "units": "dB"
    })
    baseline_sched_eesm_beta_db: float = field(default=2.7, metadata={
        "description": "EESM beta for baseline scheduler",
        "units": "dB"
    })
    rm_sched_eesm_beta_db: float = field(default=3.5, metadata={
        "description": "EESM beta for RadioMap scheduler",
        "units": "dB"
    })


@dataclass
class PowerAllocationConfig(ConfigGroup):
    """DL power allocation configuration."""

    # Baseline path
    baseline_dl_power_model: str = field(default="waterfill", metadata={
        "description": "Baseline DL power allocation model",
        "options": ["equal_prb", "waterfill"]
    })
    baseline_P_tot_dbm: float = field(default=50.0, metadata={
        "description": "Baseline total DL power budget",
        "units": "dBm"
    })
    baseline_p_min_dbm: float = field(default=27.0, metadata={
        "description": "Baseline minimum per-PRB power",
        "units": "dBm"
    })
    baseline_p_max_dbm: float = field(default=33.0, metadata={
        "description": "Baseline maximum per-PRB power",
        "units": "dBm"
    })
    baseline_max_prbs_per_ue: int = field(default=20, metadata={
        "description": "Baseline max PRBs per UE per TTI",
        "min": 1
    })

    # RadioMap path
    rm_dl_power_model: str = field(default="waterfill", metadata={
        "description": "RadioMap DL power allocation model",
        "options": ["equal_prb", "waterfill"]
    })
    rm_P_tot_dbm: float = field(default=50.0, metadata={
        "description": "RadioMap total DL power budget",
        "units": "dBm"
    })
    rm_p_min_dbm: float = field(default=28.0, metadata={
        "description": "RadioMap minimum per-PRB power",
        "units": "dBm"
    })
    rm_p_max_dbm: float = field(default=36.0, metadata={
        "description": "RadioMap maximum per-PRB power",
        "units": "dBm"
    })
    rm_max_prbs_per_ue: int = field(default=20, metadata={
        "description": "RadioMap max PRBs per UE per TTI",
        "min": 1
    })


@dataclass
class HarqConfig(ConfigGroup):
    """HARQ/OLLA/BLER configuration."""

    enable_harq_full: bool = field(default=True, metadata={
        "description": "Enable full HARQ with soft combining"
    })
    enable_harq_deferral: bool = field(default=False, metadata={
        "description": "Enable HARQ deferral mode"
    })
    harq_max_procs: int = field(default=24, metadata={
        "description": "Maximum HARQ processes per UE",
        "min": 1, "max": 32
    })
    harq_ack_delay_ttis: int = field(default=6, metadata={
        "description": "HARQ ACK feedback delay",
        "units": "TTIs", "min": 1
    })
    harq_target_bler: float = field(default=0.1, metadata={
        "description": "Target BLER for initial transmission",
        "min": 0.0, "max": 1.0
    })
    harq_max_retx: int = field(default=4, metadata={
        "description": "Maximum retransmissions per TB",
        "min": 0
    })
    harq_retx_priority_bonus: float = field(default=0.5, metadata={
        "description": "Retransmission scheduling priority boost"
    })
    harq_flush_tail: bool = field(default=True, metadata={
        "description": "Flush pending HARQ at simulation end"
    })

    # BLER curve parameters
    bler_slope_db: float = field(default=1.0, metadata={
        "description": "BLER curve slope",
        "units": "dB"
    })
    bler_margin_db: float = field(default=1.5, metadata={
        "description": "BLER margin",
        "units": "dB"
    })
    bler_curve_path: Optional[str] = field(default=None, metadata={
        "description": "Path to external BLER curves JSON"
    })

    # OLLA parameters
    olla_step_up_db: float = field(default=0.06, metadata={
        "description": "OLLA step up on ACK",
        "units": "dB"
    })
    olla_step_down_db: float = field(default=0.12, metadata={
        "description": "OLLA step down on NACK",
        "units": "dB"
    })
    olla_init_offset_db: float = field(default=-1.5, metadata={
        "description": "OLLA initial offset",
        "units": "dB"
    })
    olla_min_db: float = field(default=-3.0, metadata={
        "description": "OLLA minimum offset",
        "units": "dB"
    })
    olla_max_db: float = field(default=6.0, metadata={
        "description": "OLLA maximum offset",
        "units": "dB"
    })


@dataclass
class MCSConfig(ConfigGroup):
    """MCS table configuration."""

    mcs_table_kind: str = field(default="3gpp_table_2", metadata={
        "description": "MCS table for data transmission",
        "options": ["3gpp_table_1", "3gpp_table_2", "3gpp_table_3", "legacy"]
    })
    csi_mcs_table: str = field(default="3gpp_table_1", metadata={
        "description": "MCS table for CSI/CQI computation",
        "options": ["3gpp_table_1", "3gpp_table_2", "3gpp_table_3", "nr_256qam", "legacy"]
    })
    mcs_3gpp_table_path: Optional[str] = field(default=None, metadata={
        "description": "Path to 3GPP MCS tables JSON file"
    })

    # PDSCH overhead
    pdsch_dmrs_sym_per_slot: int = field(default=1, metadata={
        "description": "DMRS symbols per slot"
    })
    dmrs_re_per_sym_per_prb: int = field(default=6, metadata={
        "description": "DMRS REs per symbol per PRB"
    })
    oh_prb: int = field(default=0, metadata={
        "description": "Additional overhead per PRB"
    })


@dataclass
class ConstellationConfig(ConfigGroup):
    """Multi-satellite constellation configuration."""

    enable_constellation: bool = field(default=False, metadata={
        "description": "Enable constellation (multi-satellite) mode"
    })
    tle_catalog_path: Optional[str] = field(default=None, metadata={
        "description": "Path to TLE catalog file for constellation"
    })
    constellation_max_ground_radius_km: float = field(default=1200.0, metadata={
        "description": "Maximum ground radius for candidate satellites",
        "units": "km"
    })
    constellation_max_sats_per_tti: int = field(default=6, metadata={
        "description": "Maximum satellites to consider per TTI",
        "min": 1
    })
    min_elev_deg: float = field(default=20.0, metadata={
        "description": "Minimum UE elevation for visibility",
        "units": "degrees", "min": 0.0, "max": 90.0
    })
    association_metric: str = field(default="snr_wb", metadata={
        "description": "UE-satellite association metric",
        "options": ["snr_wb", "prx_dbm"]
    })

    # Handover
    ho_enabled: bool = field(default=True, metadata={
        "description": "Enable handover between satellites"
    })
    ho_hyst_db: float = field(default=2.0, metadata={
        "description": "Handover hysteresis",
        "units": "dB", "min": 0.0
    })
    ho_ttt_ttis: int = field(default=20, metadata={
        "description": "Time-to-trigger for handover",
        "units": "TTIs", "min": 0
    })
    constellation_prb_cap: int = field(default=20, metadata={
        "description": "Unified PRB cap per UE in constellation mode",
        "min": 1
    })


@dataclass
class OutputConfig(ConfigGroup):
    """Output and reporting configuration."""

    write_json_report: bool = field(default=True, metadata={
        "description": "Write JSON summary report"
    })
    report_basename: str = field(default="summary", metadata={
        "description": "Base name for output report files"
    })
    plot_dir: str = field(default="output", metadata={
        "description": "Directory for output plots and reports"
    })
    save_plots: bool = field(default=True, metadata={
        "description": "Save generated plots to files"
    })
    show_plots: bool = field(default=False, metadata={
        "description": "Display plots interactively"
    })
    print_harq_summary: bool = field(default=True, metadata={
        "description": "Print HARQ statistics summary"
    })
    show_progress: bool = field(default=True, metadata={
        "description": "Show progress bar during simulation"
    })
    include_serving_trace: bool = field(default=False, metadata={
        "description": "Include full serving timeline in constellation report"
    })

    # Debug/recording options
    record_assignments: bool = field(default=False, metadata={
        "description": "Record PRB assignment timeline"
    })
    record_assignments_target: str = field(default="rm", metadata={
        "description": "Which scheduler to record assignments for",
        "options": ["base", "rm", "both", "all"]
    })
    record_ue_thr: bool = field(default=False, metadata={
        "description": "Record per-UE throughput timeline"
    })


@dataclass
class TrafficConfig(ConfigGroup):
    """Traffic model configuration."""

    traffic_model: str = field(default="full_buffer", metadata={
        "description": "Traffic arrival model",
        "options": ["full_buffer", "poisson", "ftp3", "video", "voip", "mixed"]
    })

    # Poisson parameters
    poisson_arrival_rate_hz: float = field(default=100.0, metadata={
        "description": "Poisson arrival rate per UE",
        "units": "packets/s", "min": 0.0
    })
    poisson_packet_size_bytes: int = field(default=1500, metadata={
        "description": "Mean packet size for Poisson traffic",
        "units": "bytes", "min": 1
    })
    poisson_packet_size_std_bytes: int = field(default=0, metadata={
        "description": "Packet size std dev (0 = fixed size)",
        "units": "bytes", "min": 0
    })
    poisson_qci: int = field(default=9, metadata={
        "description": "QCI for Poisson traffic",
        "min": 1, "max": 85
    })

    # FTP Model 3 parameters
    ftp3_file_size_bytes: int = field(default=524288, metadata={
        "description": "FTP Model 3 file size (default 512 KB)",
        "units": "bytes", "min": 1
    })
    ftp3_reading_time_ms: float = field(default=180.0, metadata={
        "description": "Mean reading time between files",
        "units": "ms", "min": 0.0
    })
    ftp3_qci: int = field(default=9, metadata={
        "description": "QCI for FTP traffic",
        "min": 1, "max": 85
    })

    # Video streaming parameters
    video_frame_rate_fps: float = field(default=30.0, metadata={
        "description": "Video frame rate",
        "units": "fps", "min": 1.0
    })
    video_i_frame_size_bytes: int = field(default=50000, metadata={
        "description": "I-frame (keyframe) size",
        "units": "bytes", "min": 1
    })
    video_p_frame_size_bytes: int = field(default=10000, metadata={
        "description": "P-frame (predicted) size",
        "units": "bytes", "min": 1
    })
    video_gop_size: int = field(default=15, metadata={
        "description": "GOP length (frames per I-frame)",
        "min": 1
    })
    video_qci: int = field(default=4, metadata={
        "description": "QCI for video traffic",
        "min": 1, "max": 85
    })

    # VoIP parameters
    voip_codec: str = field(default="AMR-WB", metadata={
        "description": "VoIP codec",
        "options": ["AMR-WB", "G.711", "EVS"]
    })
    voip_activity_factor: float = field(default=0.5, metadata={
        "description": "Voice activity factor",
        "min": 0.0, "max": 1.0
    })
    voip_packet_interval_ms: float = field(default=20.0, metadata={
        "description": "Packet interval during talk spurts",
        "units": "ms", "min": 1.0
    })
    voip_qci: int = field(default=1, metadata={
        "description": "QCI for VoIP traffic",
        "min": 1, "max": 85
    })

    # Mixed traffic composition (percentages must sum to 100)
    mixed_embb_pct: float = field(default=70.0, metadata={
        "description": "Percentage of UEs with eMBB traffic",
        "min": 0.0, "max": 100.0
    })
    mixed_urllc_pct: float = field(default=20.0, metadata={
        "description": "Percentage of UEs with URLLC traffic",
        "min": 0.0, "max": 100.0
    })
    mixed_voip_pct: float = field(default=10.0, metadata={
        "description": "Percentage of UEs with VoIP traffic",
        "min": 0.0, "max": 100.0
    })


@dataclass
class QoSConfig(ConfigGroup):
    """QoS and scheduling configuration."""

    enable_qos: bool = field(default=False, metadata={
        "description": "Enable multi-QoS differentiation"
    })

    scheduler_algorithm: str = field(default="pf", metadata={
        "description": "Scheduling algorithm",
        "options": ["pf", "m-lwdf", "exp-pf", "round_robin", "max_rate"]
    })

    # M-LWDF parameters
    mlwdf_alpha: float = field(default=0.01, metadata={
        "description": "M-LWDF delay weight factor (unused, computed from delta)",
        "min": 0.0
    })
    mlwdf_delta: float = field(default=0.01, metadata={
        "description": "M-LWDF target delay violation probability",
        "min": 0.0, "max": 1.0
    })

    # EXP-PF parameters
    exppf_beta: float = field(default=1.0, metadata={
        "description": "EXP-PF exponential weight",
        "min": 0.0
    })

    # URLLC prioritization
    urllc_preemption: bool = field(default=True, metadata={
        "description": "Allow URLLC to preempt eMBB transmissions"
    })
    urllc_mini_slot: bool = field(default=False, metadata={
        "description": "Enable mini-slot scheduling for URLLC"
    })

    # NTN-specific delay compensation
    ntn_pdb_extension_factor: float = field(default=1.0, metadata={
        "description": "Factor to extend PDB for NTN propagation delay",
        "min": 1.0
    })
    compensate_rtt_in_pdb: bool = field(default=True, metadata={
        "description": "Subtract estimated RTT from PDB budget"
    })


@dataclass
class LatencyKPIConfig(ConfigGroup):
    """Latency and packet loss KPI configuration."""

    record_packet_latency: bool = field(default=True, metadata={
        "description": "Record per-packet latency for CDF computation"
    })
    latency_percentiles: List[float] = field(
        default_factory=lambda: [50.0, 90.0, 95.0, 99.0, 99.9],
        metadata={"description": "Percentiles to compute for latency CDF"}
    )
    record_per_qos_stats: bool = field(default=True, metadata={
        "description": "Track latency/loss separately per QoS class"
    })
    max_stored_packets: int = field(default=1_000_000, metadata={
        "description": "Maximum packets to store for statistics",
        "min": 1000
    })


@dataclass
class MACConfig(ConfigGroup):
    """MAC layer configuration (Phase 9).

    Includes BSR (Buffer Status Report), DRX (Discontinuous Reception),
    and NTN-specific timing parameters.
    """

    # BSR Configuration
    enable_bsr: bool = field(default=True, metadata={
        "description": "Enable Buffer Status Report mechanism"
    })
    bsr_table_bits: int = field(default=8, metadata={
        "description": "BSR table size (5 or 8 bits)",
        "options": [5, 8]
    })
    bsr_periodic_timer_ms: float = field(default=20.0, metadata={
        "description": "Periodic BSR timer period",
        "units": "ms", "min": 1.0
    })
    bsr_retx_timer_ms: float = field(default=10.0, metadata={
        "description": "BSR retransmission timer period",
        "units": "ms", "min": 1.0
    })

    # DRX Configuration (Phase 9.2 - placeholder)
    enable_drx: bool = field(default=False, metadata={
        "description": "Enable Discontinuous Reception"
    })
    drx_on_duration_ms: float = field(default=10.0, metadata={
        "description": "DRX On Duration timer",
        "units": "ms", "min": 1.0
    })
    drx_inactivity_timer_ms: float = field(default=100.0, metadata={
        "description": "DRX Inactivity timer",
        "units": "ms", "min": 1.0
    })
    drx_short_cycle_ms: float = field(default=20.0, metadata={
        "description": "DRX short cycle duration",
        "units": "ms", "min": 2.0
    })
    drx_long_cycle_ms: float = field(default=320.0, metadata={
        "description": "DRX long cycle duration",
        "units": "ms", "min": 10.0
    })
    drx_short_cycle_timer: int = field(default=2, metadata={
        "description": "Number of short cycles before long cycle",
        "min": 1
    })

    # NTN Timing Configuration (Phase 9.3 - placeholder)
    ntn_timing_adaptation: bool = field(default=True, metadata={
        "description": "Enable NTN-specific timing adaptation"
    })
    k1_slots_base: int = field(default=4, metadata={
        "description": "Base K1 value (PDSCH-to-HARQ-ACK slots)",
        "min": 0, "max": 15
    })
    k1_ntn_extension_factor: float = field(default=1.0, metadata={
        "description": "K1 extension factor for NTN propagation delay",
        "min": 1.0, "max": 10.0
    })
    harq_rtt_scaling: bool = field(default=True, metadata={
        "description": "Scale HARQ RTT based on propagation delay"
    })


@dataclass
class OALSConfig(ConfigGroup):
    """OALS (Orbit-Aware Lookahead Scheduling) configuration (Phase 11 - Patent).

    This configuration group contains parameters for the OALS algorithm,
    which leverages satellite orbit predictability for scheduling optimization.

    Core innovations:
    1. Lookahead factor (Φ) computation from predicted satellite geometry
    2. Metric correction function f(Φ, urgency) for scheduling time optimization
    3. Predictive handover scheduling strategy
    4. HARQ lookahead MCS selection
    """

    # Enable OALS
    enable_oals: bool = field(default=False, metadata={
        "description": "Enable Orbit-Aware Lookahead Scheduling"
    })

    # Lookahead window parameters
    lookahead_horizon_ttis: int = field(default=200, metadata={
        "description": "Lookahead prediction window length",
        "units": "TTIs", "min": 10, "max": 2000
    })
    lookahead_sample_interval: int = field(default=5, metadata={
        "description": "Sparse sampling interval for lookahead computation",
        "units": "TTIs", "min": 1, "max": 50
    })
    lookahead_update_interval: int = field(default=10, metadata={
        "description": "Interval between lookahead cache updates",
        "units": "TTIs", "min": 1, "max": 100
    })

    # Metric correction parameters
    alpha_urgent: float = field(default=0.8, metadata={
        "description": "Urgency threshold for immediate scheduling (紧急阈值)",
        "min": 0.5, "max": 1.0
    })
    beta_wait: float = field(default=0.3, metadata={
        "description": "Urgency threshold below which waiting is allowed (可等待阈值)",
        "min": 0.0, "max": 0.5
    })
    theta_lookahead: float = field(default=0.7, metadata={
        "description": "Lookahead factor threshold to trigger waiting (前瞻触发阈值)",
        "min": 0.5, "max": 1.0
    })
    gamma_decay: float = field(default=2.0, metadata={
        "description": "Decay exponent for wait penalty (衰减指数)",
        "min": 1.0, "max": 5.0
    })
    boost_factor: float = field(default=2.0, metadata={
        "description": "Priority boost factor for urgent traffic (紧急提升因子)",
        "min": 1.0, "max": 5.0
    })

    # Trend correction
    enable_trend_correction: bool = field(default=True, metadata={
        "description": "Enable channel trend-based correction"
    })
    trend_threshold_db: float = field(default=0.5, metadata={
        "description": "Threshold for trend detection",
        "units": "dB/TTI", "min": 0.1, "max": 2.0
    })
    trend_epsilon: float = field(default=0.1, metadata={
        "description": "Trend correction amplitude",
        "min": 0.01, "max": 0.3
    })

    # Predictive handover
    enable_predictive_ho: bool = field(default=True, metadata={
        "description": "Enable predictive handover optimization"
    })
    handover_prep_ttis: int = field(default=100, metadata={
        "description": "Handover preparation window length",
        "units": "TTIs", "min": 10, "max": 500
    })
    handover_recovery_ttis: int = field(default=50, metadata={
        "description": "Handover recovery window length",
        "units": "TTIs", "min": 5, "max": 200
    })

    # HARQ lookahead MCS
    enable_harq_lookahead: bool = field(default=True, metadata={
        "description": "Enable HARQ lookahead MCS selection"
    })
    harq_sinr_aggressive_db: float = field(default=1.5, metadata={
        "description": "SINR boost for aggressive MCS when future channel is better",
        "units": "dB", "min": 0.0, "max": 5.0
    })
    harq_sinr_conservative_db: float = field(default=1.5, metadata={
        "description": "SINR margin for conservative MCS when future channel is worse",
        "units": "dB", "min": 0.0, "max": 5.0
    })
    harq_delta_threshold_db: float = field(default=3.0, metadata={
        "description": "SINR change threshold for MCS strategy selection",
        "units": "dB", "min": 1.0, "max": 10.0
    })

    # Early retransmission
    early_retx_trend_threshold: float = field(default=-1.0, metadata={
        "description": "Trend threshold for triggering early retransmission",
        "units": "dB/TTI", "max": 0.0
    })
    early_retx_max_rv: int = field(default=2, metadata={
        "description": "Maximum RV index for early retransmission",
        "min": 0, "max": 3
    })


# =============================================================================
# Main Configuration Class
# =============================================================================

@dataclass
class NTNSimConfig(ConfigGroup):
    """
    Complete NTN-NR Downlink Simulation Configuration.

    This class aggregates all configuration groups and provides
    a unified interface for accessing and managing simulation parameters.
    """

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    radio_map: RadioMapConfig = field(default_factory=RadioMapConfig)
    numerology: NumerologyConfig = field(default_factory=NumerologyConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    link_budget: LinkBudgetConfig = field(default_factory=LinkBudgetConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    orbit: OrbitConfig = field(default_factory=OrbitConfig)
    time_varying: TimeVaryingConfig = field(default_factory=TimeVaryingConfig)
    csi: CSIConfig = field(default_factory=CSIConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    power_allocation: PowerAllocationConfig = field(default_factory=PowerAllocationConfig)
    harq: HarqConfig = field(default_factory=HarqConfig)
    mcs: MCSConfig = field(default_factory=MCSConfig)
    constellation: ConstellationConfig = field(default_factory=ConstellationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    qos: QoSConfig = field(default_factory=QoSConfig)
    latency_kpi: LatencyKPIConfig = field(default_factory=LatencyKPIConfig)
    mac: MACConfig = field(default_factory=MACConfig)
    oals: OALSConfig = field(default_factory=OALSConfig)

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary format compatible with legacy code.

        This method flattens all nested configuration groups into a single
        dictionary, maintaining backward compatibility with existing code.
        """
        result = {}

        # Simulation
        result.update({
            "Z": self.simulation.Z,
            "N_UE": self.simulation.N_UE,
            "T": self.simulation.T,
            "seed": self.simulation.seed,
        })

        # Radio Map
        result.update({
            "radio_map_mat_path": self.radio_map.radio_map_mat_path,
            "radio_map_mat_var": self.radio_map.radio_map_mat_var,
            "radio_map_units": self.radio_map.radio_map_units,
            "radiomap_est_error_db": self.radio_map.radiomap_est_error_db,
            "radiomap_blur_sigma": self.radio_map.radiomap_blur_sigma,
        })

        # Numerology
        result.update({
            "scs_khz": self.numerology.scs_khz,
            "cp_type": self.numerology.cp_type,
            "noise_temp_K": self.numerology.noise_temp_K,
        })

        # Channel
        result.update({
            "channel_model": self.channel.channel_model,
            "ntn_channel_profile": self.channel.ntn_channel_profile,
            "channel_params": self.channel.channel_params,
        })

        # Link Budget
        result.update({
            "P_tx_dbm": self.link_budget.P_tx_dbm,
            "G_rx_db": self.link_budget.G_rx_db,
            "shadow_std_db": self.link_budget.shadow_std_db,
            "rx_nf_db": self.link_budget.rx_nf_db,
            "impl_loss_db": self.link_budget.impl_loss_db,
            "overhead_eff": self.link_budget.overhead_eff,
        })

        # Geometry
        result.update({
            "sat_altitude_km": self.geometry.sat_altitude_km,
            "carrier_freq_GHz": self.geometry.carrier_freq_GHz,
            "beam_center_xy": self.geometry.beam_center_xy,
            "beam_half_bw_deg": self.geometry.beam_half_bw_deg,
            "beam_edge_drop_db": self.geometry.beam_edge_drop_db,
            "cell_size_km": self.geometry.cell_size_km,
        })

        # Orbit
        result.update({
            "enable_orbit_dynamics": self.orbit.enable_orbit_dynamics,
            "tti_ms": self.orbit.tti_ms,
            "sat_ground_speed_kms": self.orbit.sat_ground_speed_kms,
            "sat_heading_deg": self.orbit.sat_heading_deg,
            "tle_name": self.orbit.tle_name,
            "tle_lines": self.orbit.tle_lines,
            "tle_path": self.orbit.tle_path,
            "orbit_start_datetime": self.orbit.orbit_start_datetime,
            "auto_ref_from_tle": self.orbit.auto_ref_from_tle,
            "ref_lat_deg": self.orbit.ref_lat_deg,
            "ref_lon_deg": self.orbit.ref_lon_deg,
            "map_rotation_deg": self.orbit.map_rotation_deg,
        })

        # Time Varying
        result.update({
            "enable_time_varying": self.time_varying.enable_time_varying,
            "rm_flicker_db_std": self.time_varying.rm_flicker_db_std,
            "rm_drift_px": self.time_varying.rm_drift_px,
        })

        # CSI
        result.update({
            "csi_olla_offset_db": self.csi.csi_olla_offset_db,
            "baseline_csi_delay_ttis": self.csi.baseline_csi_delay_ttis,
            "rm_csi_delay_ttis": self.csi.rm_csi_delay_ttis,
            "enable_cqi_periodicity_base": self.csi.enable_cqi_periodicity_base,
            "enable_cqi_periodicity_rm": self.csi.enable_cqi_periodicity_rm,
            "cqi_period_ttis": self.csi.cqi_period_ttis,
            "cqi_offset_ttis": self.csi.cqi_offset_ttis,
        })

        # Scheduler
        result.update({
            "pf_beta": self.scheduler.pf_beta,
            "use_mcs": self.scheduler.use_mcs,
            "power_split": self.scheduler.power_split,
            "sched_require_contiguous": self.scheduler.sched_require_contiguous,
            "sched_eesm_beta_db": self.scheduler.sched_eesm_beta_db,
            "baseline_sched_eesm_beta_db": self.scheduler.baseline_sched_eesm_beta_db,
            "rm_sched_eesm_beta_db": self.scheduler.rm_sched_eesm_beta_db,
        })

        # Power Allocation
        result.update({
            "baseline_dl_power_model": self.power_allocation.baseline_dl_power_model,
            "baseline_P_tot_dbm": self.power_allocation.baseline_P_tot_dbm,
            "baseline_p_min_dbm": self.power_allocation.baseline_p_min_dbm,
            "baseline_p_max_dbm": self.power_allocation.baseline_p_max_dbm,
            "baseline_max_prbs_per_ue": self.power_allocation.baseline_max_prbs_per_ue,
            "rm_dl_power_model": self.power_allocation.rm_dl_power_model,
            "rm_P_tot_dbm": self.power_allocation.rm_P_tot_dbm,
            "rm_p_min_dbm": self.power_allocation.rm_p_min_dbm,
            "rm_p_max_dbm": self.power_allocation.rm_p_max_dbm,
            "rm_max_prbs_per_ue": self.power_allocation.rm_max_prbs_per_ue,
        })

        # HARQ
        result.update({
            "enable_harq_full": self.harq.enable_harq_full,
            "enable_harq_deferral": self.harq.enable_harq_deferral,
            "harq_max_procs": self.harq.harq_max_procs,
            "harq_ack_delay_ttis": self.harq.harq_ack_delay_ttis,
            "harq_target_bler": self.harq.harq_target_bler,
            "harq_max_retx": self.harq.harq_max_retx,
            "harq_retx_priority_bonus": self.harq.harq_retx_priority_bonus,
            "harq_flush_tail": self.harq.harq_flush_tail,
            "bler_slope_db": self.harq.bler_slope_db,
            "bler_margin_db": self.harq.bler_margin_db,
            "bler_curve_path": self.harq.bler_curve_path,
            "olla_step_up_db": self.harq.olla_step_up_db,
            "olla_step_down_db": self.harq.olla_step_down_db,
            "olla_init_offset_db": self.harq.olla_init_offset_db,
            "olla_min_db": self.harq.olla_min_db,
            "olla_max_db": self.harq.olla_max_db,
        })

        # MCS
        result.update({
            "mcs_table_kind": self.mcs.mcs_table_kind,
            "csi_mcs_table": self.mcs.csi_mcs_table,
            "mcs_3gpp_table_path": self.mcs.mcs_3gpp_table_path,
            "pdsch_dmrs_sym_per_slot": self.mcs.pdsch_dmrs_sym_per_slot,
            "dmrs_re_per_sym_per_prb": self.mcs.dmrs_re_per_sym_per_prb,
            "oh_prb": self.mcs.oh_prb,
        })

        # Constellation
        result.update({
            "enable_constellation": self.constellation.enable_constellation,
            "tle_catalog_path": self.constellation.tle_catalog_path,
            "constellation_max_ground_radius_km": self.constellation.constellation_max_ground_radius_km,
            "constellation_max_sats_per_tti": self.constellation.constellation_max_sats_per_tti,
            "min_elev_deg": self.constellation.min_elev_deg,
            "association_metric": self.constellation.association_metric,
            "ho_enabled": self.constellation.ho_enabled,
            "ho_hyst_db": self.constellation.ho_hyst_db,
            "ho_ttt_ttis": self.constellation.ho_ttt_ttis,
            "constellation_prb_cap": self.constellation.constellation_prb_cap,
        })

        # Output
        result.update({
            "write_json_report": self.output.write_json_report,
            "report_basename": self.output.report_basename,
            "plot_dir": self.output.plot_dir,
            "save_plots": self.output.save_plots,
            "show_plots": self.output.show_plots,
            "print_harq_summary": self.output.print_harq_summary,
            "show_progress": self.output.show_progress,
            "include_serving_trace": self.output.include_serving_trace,
            "record_assignments": self.output.record_assignments,
            "record_assignments_target": self.output.record_assignments_target,
            "record_ue_thr": self.output.record_ue_thr,
        })

        # Traffic
        result.update({
            "traffic_model": self.traffic.traffic_model,
            "poisson_arrival_rate_hz": self.traffic.poisson_arrival_rate_hz,
            "poisson_packet_size_bytes": self.traffic.poisson_packet_size_bytes,
            "poisson_packet_size_std_bytes": self.traffic.poisson_packet_size_std_bytes,
            "poisson_qci": self.traffic.poisson_qci,
            "ftp3_file_size_bytes": self.traffic.ftp3_file_size_bytes,
            "ftp3_reading_time_ms": self.traffic.ftp3_reading_time_ms,
            "ftp3_qci": self.traffic.ftp3_qci,
            "video_frame_rate_fps": self.traffic.video_frame_rate_fps,
            "video_i_frame_size_bytes": self.traffic.video_i_frame_size_bytes,
            "video_p_frame_size_bytes": self.traffic.video_p_frame_size_bytes,
            "video_gop_size": self.traffic.video_gop_size,
            "video_qci": self.traffic.video_qci,
            "voip_codec": self.traffic.voip_codec,
            "voip_activity_factor": self.traffic.voip_activity_factor,
            "voip_packet_interval_ms": self.traffic.voip_packet_interval_ms,
            "voip_qci": self.traffic.voip_qci,
            "mixed_embb_pct": self.traffic.mixed_embb_pct,
            "mixed_urllc_pct": self.traffic.mixed_urllc_pct,
            "mixed_voip_pct": self.traffic.mixed_voip_pct,
        })

        # QoS
        result.update({
            "enable_qos": self.qos.enable_qos,
            "scheduler_algorithm": self.qos.scheduler_algorithm,
            "mlwdf_alpha": self.qos.mlwdf_alpha,
            "mlwdf_delta": self.qos.mlwdf_delta,
            "exppf_beta": self.qos.exppf_beta,
            "urllc_preemption": self.qos.urllc_preemption,
            "urllc_mini_slot": self.qos.urllc_mini_slot,
            "ntn_pdb_extension_factor": self.qos.ntn_pdb_extension_factor,
            "compensate_rtt_in_pdb": self.qos.compensate_rtt_in_pdb,
        })

        # Latency KPI
        result.update({
            "record_packet_latency": self.latency_kpi.record_packet_latency,
            "latency_percentiles": self.latency_kpi.latency_percentiles,
            "record_per_qos_stats": self.latency_kpi.record_per_qos_stats,
            "max_stored_packets": self.latency_kpi.max_stored_packets,
        })

        # MAC (Phase 9)
        result.update({
            "enable_bsr": self.mac.enable_bsr,
            "bsr_table_bits": self.mac.bsr_table_bits,
            "bsr_periodic_timer_ms": self.mac.bsr_periodic_timer_ms,
            "bsr_retx_timer_ms": self.mac.bsr_retx_timer_ms,
            "enable_drx": self.mac.enable_drx,
            "drx_on_duration_ms": self.mac.drx_on_duration_ms,
            "drx_inactivity_timer_ms": self.mac.drx_inactivity_timer_ms,
            "drx_short_cycle_ms": self.mac.drx_short_cycle_ms,
            "drx_long_cycle_ms": self.mac.drx_long_cycle_ms,
            "drx_short_cycle_timer": self.mac.drx_short_cycle_timer,
            "ntn_timing_adaptation": self.mac.ntn_timing_adaptation,
            "k1_slots_base": self.mac.k1_slots_base,
            "k1_ntn_extension_factor": self.mac.k1_ntn_extension_factor,
            "harq_rtt_scaling": self.mac.harq_rtt_scaling,
        })

        # OALS (Phase 11 - Patent)
        result.update({
            "enable_oals": self.oals.enable_oals,
            "lookahead_horizon_ttis": self.oals.lookahead_horizon_ttis,
            "lookahead_sample_interval": self.oals.lookahead_sample_interval,
            "lookahead_update_interval": self.oals.lookahead_update_interval,
            "alpha_urgent": self.oals.alpha_urgent,
            "beta_wait": self.oals.beta_wait,
            "theta_lookahead": self.oals.theta_lookahead,
            "gamma_decay": self.oals.gamma_decay,
            "boost_factor": self.oals.boost_factor,
            "enable_trend_correction": self.oals.enable_trend_correction,
            "trend_threshold_db": self.oals.trend_threshold_db,
            "trend_epsilon": self.oals.trend_epsilon,
            "enable_predictive_ho": self.oals.enable_predictive_ho,
            "handover_prep_ttis": self.oals.handover_prep_ttis,
            "handover_recovery_ttis": self.oals.handover_recovery_ttis,
            "enable_harq_lookahead": self.oals.enable_harq_lookahead,
            "harq_sinr_aggressive_db": self.oals.harq_sinr_aggressive_db,
            "harq_sinr_conservative_db": self.oals.harq_sinr_conservative_db,
            "harq_delta_threshold_db": self.oals.harq_delta_threshold_db,
            "early_retx_trend_threshold": self.oals.early_retx_trend_threshold,
            "early_retx_max_rv": self.oals.early_retx_max_rv,
        })

        return result

    @classmethod
    def from_flat_dict(cls, data: Dict[str, Any]) -> "NTNSimConfig":
        """
        Create configuration from flat dictionary (legacy format).

        This method maps flat dictionary keys to the appropriate
        nested configuration groups.
        """
        config = cls()

        # Simulation
        if "Z" in data:
            config.simulation.Z = data["Z"]
        if "N_UE" in data:
            config.simulation.N_UE = data["N_UE"]
        if "T" in data:
            config.simulation.T = data["T"]
        if "seed" in data:
            config.simulation.seed = data["seed"]

        # Radio Map
        if "radio_map_mat_path" in data:
            config.radio_map.radio_map_mat_path = data["radio_map_mat_path"]
        if "radio_map_mat_var" in data:
            config.radio_map.radio_map_mat_var = data["radio_map_mat_var"]
        if "radio_map_units" in data:
            config.radio_map.radio_map_units = data["radio_map_units"]
        if "radiomap_est_error_db" in data:
            config.radio_map.radiomap_est_error_db = data["radiomap_est_error_db"]
        if "radiomap_blur_sigma" in data:
            config.radio_map.radiomap_blur_sigma = data["radiomap_blur_sigma"]

        # Numerology
        if "scs_khz" in data:
            config.numerology.scs_khz = data["scs_khz"]
        if "cp_type" in data:
            config.numerology.cp_type = data["cp_type"]
        if "noise_temp_K" in data:
            config.numerology.noise_temp_K = data["noise_temp_K"]

        # Channel
        if "channel_model" in data:
            config.channel.channel_model = data["channel_model"]
        if "ntn_channel_profile" in data:
            config.channel.ntn_channel_profile = data["ntn_channel_profile"]
        if "channel_params" in data:
            config.channel.channel_params = data["channel_params"]

        # Link Budget
        if "P_tx_dbm" in data:
            config.link_budget.P_tx_dbm = data["P_tx_dbm"]
        if "G_rx_db" in data:
            config.link_budget.G_rx_db = data["G_rx_db"]
        if "shadow_std_db" in data:
            config.link_budget.shadow_std_db = data["shadow_std_db"]
        if "rx_nf_db" in data:
            config.link_budget.rx_nf_db = data["rx_nf_db"]
        if "impl_loss_db" in data:
            config.link_budget.impl_loss_db = data["impl_loss_db"]
        if "overhead_eff" in data:
            config.link_budget.overhead_eff = data["overhead_eff"]

        # Geometry
        if "sat_altitude_km" in data:
            config.geometry.sat_altitude_km = data["sat_altitude_km"]
        if "carrier_freq_GHz" in data:
            config.geometry.carrier_freq_GHz = data["carrier_freq_GHz"]
        if "beam_center_xy" in data:
            config.geometry.beam_center_xy = data["beam_center_xy"]
        if "beam_half_bw_deg" in data:
            config.geometry.beam_half_bw_deg = data["beam_half_bw_deg"]
        if "beam_edge_drop_db" in data:
            config.geometry.beam_edge_drop_db = data["beam_edge_drop_db"]
        if "cell_size_km" in data:
            config.geometry.cell_size_km = data["cell_size_km"]

        # Orbit
        if "enable_orbit_dynamics" in data:
            config.orbit.enable_orbit_dynamics = data["enable_orbit_dynamics"]
        if "tti_ms" in data:
            config.orbit.tti_ms = data["tti_ms"]
        if "sat_ground_speed_kms" in data:
            config.orbit.sat_ground_speed_kms = data["sat_ground_speed_kms"]
        if "sat_heading_deg" in data:
            config.orbit.sat_heading_deg = data["sat_heading_deg"]
        if "tle_name" in data:
            config.orbit.tle_name = data["tle_name"]
        if "tle_lines" in data:
            config.orbit.tle_lines = data["tle_lines"]
        if "tle_path" in data:
            config.orbit.tle_path = data["tle_path"]
        if "orbit_start_datetime" in data:
            config.orbit.orbit_start_datetime = data["orbit_start_datetime"]
        if "auto_ref_from_tle" in data:
            config.orbit.auto_ref_from_tle = data["auto_ref_from_tle"]
        if "ref_lat_deg" in data:
            config.orbit.ref_lat_deg = data["ref_lat_deg"]
        if "ref_lon_deg" in data:
            config.orbit.ref_lon_deg = data["ref_lon_deg"]
        if "map_rotation_deg" in data:
            config.orbit.map_rotation_deg = data["map_rotation_deg"]

        # Time Varying
        if "enable_time_varying" in data:
            config.time_varying.enable_time_varying = data["enable_time_varying"]
        if "rm_flicker_db_std" in data:
            config.time_varying.rm_flicker_db_std = data["rm_flicker_db_std"]
        if "rm_drift_px" in data:
            config.time_varying.rm_drift_px = data["rm_drift_px"]

        # CSI
        if "csi_olla_offset_db" in data:
            config.csi.csi_olla_offset_db = data["csi_olla_offset_db"]
        if "baseline_csi_delay_ttis" in data:
            config.csi.baseline_csi_delay_ttis = data["baseline_csi_delay_ttis"]
        if "rm_csi_delay_ttis" in data:
            config.csi.rm_csi_delay_ttis = data["rm_csi_delay_ttis"]
        if "enable_cqi_periodicity_base" in data:
            config.csi.enable_cqi_periodicity_base = data["enable_cqi_periodicity_base"]
        if "enable_cqi_periodicity_rm" in data:
            config.csi.enable_cqi_periodicity_rm = data["enable_cqi_periodicity_rm"]
        if "cqi_period_ttis" in data:
            config.csi.cqi_period_ttis = data["cqi_period_ttis"]
        if "cqi_offset_ttis" in data:
            config.csi.cqi_offset_ttis = data["cqi_offset_ttis"]

        # Scheduler
        if "pf_beta" in data:
            config.scheduler.pf_beta = data["pf_beta"]
        if "use_mcs" in data:
            config.scheduler.use_mcs = data["use_mcs"]
        if "power_split" in data:
            config.scheduler.power_split = data["power_split"]
        if "sched_require_contiguous" in data:
            config.scheduler.sched_require_contiguous = data["sched_require_contiguous"]
        if "sched_eesm_beta_db" in data:
            config.scheduler.sched_eesm_beta_db = data["sched_eesm_beta_db"]
        if "baseline_sched_eesm_beta_db" in data:
            config.scheduler.baseline_sched_eesm_beta_db = data["baseline_sched_eesm_beta_db"]
        if "rm_sched_eesm_beta_db" in data:
            config.scheduler.rm_sched_eesm_beta_db = data["rm_sched_eesm_beta_db"]

        # Power Allocation
        if "baseline_dl_power_model" in data:
            config.power_allocation.baseline_dl_power_model = data["baseline_dl_power_model"]
        if "baseline_P_tot_dbm" in data:
            config.power_allocation.baseline_P_tot_dbm = data["baseline_P_tot_dbm"]
        if "baseline_p_min_dbm" in data:
            config.power_allocation.baseline_p_min_dbm = data["baseline_p_min_dbm"]
        if "baseline_p_max_dbm" in data:
            config.power_allocation.baseline_p_max_dbm = data["baseline_p_max_dbm"]
        if "baseline_max_prbs_per_ue" in data:
            config.power_allocation.baseline_max_prbs_per_ue = data["baseline_max_prbs_per_ue"]
        if "rm_dl_power_model" in data:
            config.power_allocation.rm_dl_power_model = data["rm_dl_power_model"]
        if "rm_P_tot_dbm" in data:
            config.power_allocation.rm_P_tot_dbm = data["rm_P_tot_dbm"]
        if "rm_p_min_dbm" in data:
            config.power_allocation.rm_p_min_dbm = data["rm_p_min_dbm"]
        if "rm_p_max_dbm" in data:
            config.power_allocation.rm_p_max_dbm = data["rm_p_max_dbm"]
        if "rm_max_prbs_per_ue" in data:
            config.power_allocation.rm_max_prbs_per_ue = data["rm_max_prbs_per_ue"]

        # HARQ
        if "enable_harq_full" in data:
            config.harq.enable_harq_full = data["enable_harq_full"]
        if "enable_harq_deferral" in data:
            config.harq.enable_harq_deferral = data["enable_harq_deferral"]
        if "harq_max_procs" in data:
            config.harq.harq_max_procs = data["harq_max_procs"]
        if "harq_ack_delay_ttis" in data:
            config.harq.harq_ack_delay_ttis = data["harq_ack_delay_ttis"]
        if "harq_target_bler" in data:
            config.harq.harq_target_bler = data["harq_target_bler"]
        if "harq_max_retx" in data:
            config.harq.harq_max_retx = data["harq_max_retx"]
        if "harq_retx_priority_bonus" in data:
            config.harq.harq_retx_priority_bonus = data["harq_retx_priority_bonus"]
        if "harq_flush_tail" in data:
            config.harq.harq_flush_tail = data["harq_flush_tail"]
        if "bler_slope_db" in data:
            config.harq.bler_slope_db = data["bler_slope_db"]
        if "bler_margin_db" in data:
            config.harq.bler_margin_db = data["bler_margin_db"]
        if "bler_curve_path" in data:
            config.harq.bler_curve_path = data["bler_curve_path"]
        if "olla_step_up_db" in data:
            config.harq.olla_step_up_db = data["olla_step_up_db"]
        if "olla_step_down_db" in data:
            config.harq.olla_step_down_db = data["olla_step_down_db"]
        if "olla_init_offset_db" in data:
            config.harq.olla_init_offset_db = data["olla_init_offset_db"]
        if "olla_min_db" in data:
            config.harq.olla_min_db = data["olla_min_db"]
        if "olla_max_db" in data:
            config.harq.olla_max_db = data["olla_max_db"]

        # MCS
        if "mcs_table_kind" in data:
            config.mcs.mcs_table_kind = data["mcs_table_kind"]
        if "csi_mcs_table" in data:
            config.mcs.csi_mcs_table = data["csi_mcs_table"]
        if "mcs_3gpp_table_path" in data:
            config.mcs.mcs_3gpp_table_path = data["mcs_3gpp_table_path"]
        if "pdsch_dmrs_sym_per_slot" in data:
            config.mcs.pdsch_dmrs_sym_per_slot = data["pdsch_dmrs_sym_per_slot"]
        if "dmrs_re_per_sym_per_prb" in data:
            config.mcs.dmrs_re_per_sym_per_prb = data["dmrs_re_per_sym_per_prb"]
        if "oh_prb" in data:
            config.mcs.oh_prb = data["oh_prb"]

        # Constellation
        if "enable_constellation" in data:
            config.constellation.enable_constellation = data["enable_constellation"]
        if "tle_catalog_path" in data:
            config.constellation.tle_catalog_path = data["tle_catalog_path"]
        if "constellation_max_ground_radius_km" in data:
            config.constellation.constellation_max_ground_radius_km = data["constellation_max_ground_radius_km"]
        if "constellation_max_sats_per_tti" in data:
            config.constellation.constellation_max_sats_per_tti = data["constellation_max_sats_per_tti"]
        if "min_elev_deg" in data:
            config.constellation.min_elev_deg = data["min_elev_deg"]
        if "association_metric" in data:
            config.constellation.association_metric = data["association_metric"]
        if "ho_enabled" in data:
            config.constellation.ho_enabled = data["ho_enabled"]
        if "ho_hyst_db" in data:
            config.constellation.ho_hyst_db = data["ho_hyst_db"]
        if "ho_ttt_ttis" in data:
            config.constellation.ho_ttt_ttis = data["ho_ttt_ttis"]
        if "constellation_prb_cap" in data:
            config.constellation.constellation_prb_cap = data["constellation_prb_cap"]

        # Output
        if "write_json_report" in data:
            config.output.write_json_report = data["write_json_report"]
        if "report_basename" in data:
            config.output.report_basename = data["report_basename"]
        if "plot_dir" in data:
            config.output.plot_dir = data["plot_dir"]
        if "save_plots" in data:
            config.output.save_plots = data["save_plots"]
        if "show_plots" in data:
            config.output.show_plots = data["show_plots"]
        if "print_harq_summary" in data:
            config.output.print_harq_summary = data["print_harq_summary"]
        if "show_progress" in data:
            config.output.show_progress = data["show_progress"]
        if "include_serving_trace" in data:
            config.output.include_serving_trace = data["include_serving_trace"]
        if "record_assignments" in data:
            config.output.record_assignments = data["record_assignments"]
        if "record_assignments_target" in data:
            config.output.record_assignments_target = data["record_assignments_target"]
        if "record_ue_thr" in data:
            config.output.record_ue_thr = data["record_ue_thr"]

        # Traffic
        if "traffic_model" in data:
            config.traffic.traffic_model = data["traffic_model"]
        if "poisson_arrival_rate_hz" in data:
            config.traffic.poisson_arrival_rate_hz = data["poisson_arrival_rate_hz"]
        if "poisson_packet_size_bytes" in data:
            config.traffic.poisson_packet_size_bytes = data["poisson_packet_size_bytes"]
        if "poisson_packet_size_std_bytes" in data:
            config.traffic.poisson_packet_size_std_bytes = data["poisson_packet_size_std_bytes"]
        if "poisson_qci" in data:
            config.traffic.poisson_qci = data["poisson_qci"]
        if "ftp3_file_size_bytes" in data:
            config.traffic.ftp3_file_size_bytes = data["ftp3_file_size_bytes"]
        if "ftp3_reading_time_ms" in data:
            config.traffic.ftp3_reading_time_ms = data["ftp3_reading_time_ms"]
        if "ftp3_qci" in data:
            config.traffic.ftp3_qci = data["ftp3_qci"]
        if "video_frame_rate_fps" in data:
            config.traffic.video_frame_rate_fps = data["video_frame_rate_fps"]
        if "video_i_frame_size_bytes" in data:
            config.traffic.video_i_frame_size_bytes = data["video_i_frame_size_bytes"]
        if "video_p_frame_size_bytes" in data:
            config.traffic.video_p_frame_size_bytes = data["video_p_frame_size_bytes"]
        if "video_gop_size" in data:
            config.traffic.video_gop_size = data["video_gop_size"]
        if "video_qci" in data:
            config.traffic.video_qci = data["video_qci"]
        if "voip_codec" in data:
            config.traffic.voip_codec = data["voip_codec"]
        if "voip_activity_factor" in data:
            config.traffic.voip_activity_factor = data["voip_activity_factor"]
        if "voip_packet_interval_ms" in data:
            config.traffic.voip_packet_interval_ms = data["voip_packet_interval_ms"]
        if "voip_qci" in data:
            config.traffic.voip_qci = data["voip_qci"]
        if "mixed_embb_pct" in data:
            config.traffic.mixed_embb_pct = data["mixed_embb_pct"]
        if "mixed_urllc_pct" in data:
            config.traffic.mixed_urllc_pct = data["mixed_urllc_pct"]
        if "mixed_voip_pct" in data:
            config.traffic.mixed_voip_pct = data["mixed_voip_pct"]

        # QoS
        if "enable_qos" in data:
            config.qos.enable_qos = data["enable_qos"]
        if "scheduler_algorithm" in data:
            config.qos.scheduler_algorithm = data["scheduler_algorithm"]
        if "mlwdf_alpha" in data:
            config.qos.mlwdf_alpha = data["mlwdf_alpha"]
        if "mlwdf_delta" in data:
            config.qos.mlwdf_delta = data["mlwdf_delta"]
        if "exppf_beta" in data:
            config.qos.exppf_beta = data["exppf_beta"]
        if "urllc_preemption" in data:
            config.qos.urllc_preemption = data["urllc_preemption"]
        if "urllc_mini_slot" in data:
            config.qos.urllc_mini_slot = data["urllc_mini_slot"]
        if "ntn_pdb_extension_factor" in data:
            config.qos.ntn_pdb_extension_factor = data["ntn_pdb_extension_factor"]
        if "compensate_rtt_in_pdb" in data:
            config.qos.compensate_rtt_in_pdb = data["compensate_rtt_in_pdb"]

        # Latency KPI
        if "record_packet_latency" in data:
            config.latency_kpi.record_packet_latency = data["record_packet_latency"]
        if "latency_percentiles" in data:
            config.latency_kpi.latency_percentiles = data["latency_percentiles"]
        if "record_per_qos_stats" in data:
            config.latency_kpi.record_per_qos_stats = data["record_per_qos_stats"]
        if "max_stored_packets" in data:
            config.latency_kpi.max_stored_packets = data["max_stored_packets"]

        # MAC (Phase 9)
        if "enable_bsr" in data:
            config.mac.enable_bsr = data["enable_bsr"]
        if "bsr_table_bits" in data:
            config.mac.bsr_table_bits = data["bsr_table_bits"]
        if "bsr_periodic_timer_ms" in data:
            config.mac.bsr_periodic_timer_ms = data["bsr_periodic_timer_ms"]
        if "bsr_retx_timer_ms" in data:
            config.mac.bsr_retx_timer_ms = data["bsr_retx_timer_ms"]
        if "enable_drx" in data:
            config.mac.enable_drx = data["enable_drx"]
        if "drx_on_duration_ms" in data:
            config.mac.drx_on_duration_ms = data["drx_on_duration_ms"]
        if "drx_inactivity_timer_ms" in data:
            config.mac.drx_inactivity_timer_ms = data["drx_inactivity_timer_ms"]
        if "drx_short_cycle_ms" in data:
            config.mac.drx_short_cycle_ms = data["drx_short_cycle_ms"]
        if "drx_long_cycle_ms" in data:
            config.mac.drx_long_cycle_ms = data["drx_long_cycle_ms"]
        if "drx_short_cycle_timer" in data:
            config.mac.drx_short_cycle_timer = data["drx_short_cycle_timer"]
        if "ntn_timing_adaptation" in data:
            config.mac.ntn_timing_adaptation = data["ntn_timing_adaptation"]
        if "k1_slots_base" in data:
            config.mac.k1_slots_base = data["k1_slots_base"]
        if "k1_ntn_extension_factor" in data:
            config.mac.k1_ntn_extension_factor = data["k1_ntn_extension_factor"]
        if "harq_rtt_scaling" in data:
            config.mac.harq_rtt_scaling = data["harq_rtt_scaling"]

        # OALS (Phase 11 - Patent)
        if "enable_oals" in data:
            config.oals.enable_oals = data["enable_oals"]
        if "lookahead_horizon_ttis" in data:
            config.oals.lookahead_horizon_ttis = data["lookahead_horizon_ttis"]
        if "lookahead_sample_interval" in data:
            config.oals.lookahead_sample_interval = data["lookahead_sample_interval"]
        if "lookahead_update_interval" in data:
            config.oals.lookahead_update_interval = data["lookahead_update_interval"]
        if "alpha_urgent" in data:
            config.oals.alpha_urgent = data["alpha_urgent"]
        if "beta_wait" in data:
            config.oals.beta_wait = data["beta_wait"]
        if "theta_lookahead" in data:
            config.oals.theta_lookahead = data["theta_lookahead"]
        if "gamma_decay" in data:
            config.oals.gamma_decay = data["gamma_decay"]
        if "boost_factor" in data:
            config.oals.boost_factor = data["boost_factor"]
        if "enable_trend_correction" in data:
            config.oals.enable_trend_correction = data["enable_trend_correction"]
        if "trend_threshold_db" in data:
            config.oals.trend_threshold_db = data["trend_threshold_db"]
        if "trend_epsilon" in data:
            config.oals.trend_epsilon = data["trend_epsilon"]
        if "enable_predictive_ho" in data:
            config.oals.enable_predictive_ho = data["enable_predictive_ho"]
        if "handover_prep_ttis" in data:
            config.oals.handover_prep_ttis = data["handover_prep_ttis"]
        if "handover_recovery_ttis" in data:
            config.oals.handover_recovery_ttis = data["handover_recovery_ttis"]
        if "enable_harq_lookahead" in data:
            config.oals.enable_harq_lookahead = data["enable_harq_lookahead"]
        if "harq_sinr_aggressive_db" in data:
            config.oals.harq_sinr_aggressive_db = data["harq_sinr_aggressive_db"]
        if "harq_sinr_conservative_db" in data:
            config.oals.harq_sinr_conservative_db = data["harq_sinr_conservative_db"]
        if "harq_delta_threshold_db" in data:
            config.oals.harq_delta_threshold_db = data["harq_delta_threshold_db"]
        if "early_retx_trend_threshold" in data:
            config.oals.early_retx_trend_threshold = data["early_retx_trend_threshold"]
        if "early_retx_max_rv" in data:
            config.oals.early_retx_max_rv = data["early_retx_max_rv"]

        return config

    def to_json_schema(self) -> Dict[str, Any]:
        """
        Generate JSON Schema for the configuration.

        This can be used by web UIs to generate dynamic forms.
        """
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "NTN-NR Downlink Simulation Configuration",
            "type": "object",
            "properties": {},
            "definitions": {},
        }

        # Add each group as a property
        group_map = {
            "simulation": (self.simulation, SimulationConfig),
            "radio_map": (self.radio_map, RadioMapConfig),
            "numerology": (self.numerology, NumerologyConfig),
            "channel": (self.channel, ChannelConfig),
            "link_budget": (self.link_budget, LinkBudgetConfig),
            "geometry": (self.geometry, GeometryConfig),
            "orbit": (self.orbit, OrbitConfig),
            "time_varying": (self.time_varying, TimeVaryingConfig),
            "csi": (self.csi, CSIConfig),
            "scheduler": (self.scheduler, SchedulerConfig),
            "power_allocation": (self.power_allocation, PowerAllocationConfig),
            "harq": (self.harq, HarqConfig),
            "mcs": (self.mcs, MCSConfig),
            "constellation": (self.constellation, ConstellationConfig),
            "output": (self.output, OutputConfig),
            "traffic": (self.traffic, TrafficConfig),
            "qos": (self.qos, QoSConfig),
            "latency_kpi": (self.latency_kpi, LatencyKPIConfig),
            "mac": (self.mac, MACConfig),
            "oals": (self.oals, OALSConfig),
        }

        for group_name, (_, group_cls) in group_map.items():
            group_schema = _dataclass_to_json_schema(group_cls)
            schema["properties"][group_name] = group_schema

        return schema


def _python_type_to_json_type(py_type: str) -> Dict[str, Any]:
    """Convert Python type annotation to JSON Schema type."""
    py_type_lower = py_type.lower()

    if "int" in py_type_lower and "float" not in py_type_lower:
        return {"type": "integer"}
    elif "float" in py_type_lower or "number" in py_type_lower:
        return {"type": "number"}
    elif "bool" in py_type_lower:
        return {"type": "boolean"}
    elif "str" in py_type_lower:
        return {"type": "string"}
    elif "list" in py_type_lower:
        return {"type": "array"}
    elif "dict" in py_type_lower:
        return {"type": "object"}
    elif "tuple" in py_type_lower:
        return {"type": "array"}
    elif "none" in py_type_lower or "optional" in py_type_lower:
        # Handle Optional types
        if "str" in py_type_lower:
            return {"type": ["string", "null"]}
        elif "int" in py_type_lower:
            return {"type": ["integer", "null"]}
        elif "float" in py_type_lower:
            return {"type": ["number", "null"]}
        return {"type": ["string", "null"]}  # Default for unknown Optional
    else:
        return {"type": "string"}  # Default fallback


def _dataclass_to_json_schema(cls) -> Dict[str, Any]:
    """Convert a dataclass to JSON Schema."""
    schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    field_info = cls.get_field_info()

    for name, info in field_info.items():
        prop = _python_type_to_json_type(info["type"])

        if info.get("description"):
            prop["description"] = info["description"]
        if info.get("default") is not None:
            prop["default"] = info["default"]
        if info.get("min") is not None:
            prop["minimum"] = info["min"]
        if info.get("max") is not None:
            prop["maximum"] = info["max"]
        if info.get("options"):
            prop["enum"] = info["options"]
        if info.get("units"):
            prop["x-units"] = info["units"]  # Custom extension for units

        schema["properties"][name] = prop

    return schema
