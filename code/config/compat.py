"""
Backward-compatible configuration wrapper.

This module provides a dict-like wrapper around NTNSimConfig that allows
legacy code to continue using dictionary-style access patterns.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Dict, Iterator, Optional

from .schema import NTNSimConfig


# =============================================================================
# Key Mapping: flat key -> (group_name, field_name)
# =============================================================================

_KEY_TO_GROUP: Dict[str, str] = {
    # Simulation
    "Z": "simulation",
    "N_UE": "simulation",
    "T": "simulation",
    "seed": "simulation",

    # Radio Map
    "radio_map_mat_path": "radio_map",
    "radio_map_mat_var": "radio_map",
    "radio_map_units": "radio_map",
    "radiomap_est_error_db": "radio_map",
    "radiomap_blur_sigma": "radio_map",

    # Numerology
    "scs_khz": "numerology",
    "cp_type": "numerology",
    "noise_temp_K": "numerology",

    # Channel
    "channel_model": "channel",
    "ntn_channel_profile": "channel",
    "channel_params": "channel",

    # Link Budget
    "P_tx_dbm": "link_budget",
    "G_rx_db": "link_budget",
    "shadow_std_db": "link_budget",
    "rx_nf_db": "link_budget",
    "impl_loss_db": "link_budget",
    "overhead_eff": "link_budget",

    # Geometry
    "sat_altitude_km": "geometry",
    "carrier_freq_GHz": "geometry",
    "beam_center_xy": "geometry",
    "beam_half_bw_deg": "geometry",
    "beam_edge_drop_db": "geometry",
    "cell_size_km": "geometry",

    # Orbit
    "enable_orbit_dynamics": "orbit",
    "tti_ms": "orbit",
    "sat_ground_speed_kms": "orbit",
    "sat_heading_deg": "orbit",
    "tle_name": "orbit",
    "tle_lines": "orbit",
    "tle_path": "orbit",
    "orbit_start_datetime": "orbit",
    "auto_ref_from_tle": "orbit",
    "ref_lat_deg": "orbit",
    "ref_lon_deg": "orbit",
    "map_rotation_deg": "orbit",
    "auto_orbit_start_for_visibility": "orbit",
    "auto_orbit_start_search_hours": "orbit",
    "auto_orbit_start_min_elev_deg": "orbit",

    # Time Varying
    "enable_time_varying": "time_varying",
    "rm_flicker_db_std": "time_varying",
    "rm_drift_px": "time_varying",

    # CSI
    "csi_olla_offset_db": "csi",
    "baseline_csi_delay_ttis": "csi",
    "rm_csi_delay_ttis": "csi",
    "enable_cqi_periodicity_base": "csi",
    "enable_cqi_periodicity_rm": "csi",
    "cqi_period_ttis": "csi",
    "cqi_offset_ttis": "csi",

    # Scheduler
    "pf_beta": "scheduler",
    "use_mcs": "scheduler",
    "power_split": "scheduler",
    "sched_require_contiguous": "scheduler",
    "sched_eesm_beta_db": "scheduler",
    "baseline_sched_eesm_beta_db": "scheduler",
    "rm_sched_eesm_beta_db": "scheduler",

    # Power Allocation
    "baseline_dl_power_model": "power_allocation",
    "baseline_P_tot_dbm": "power_allocation",
    "baseline_p_min_dbm": "power_allocation",
    "baseline_p_max_dbm": "power_allocation",
    "baseline_max_prbs_per_ue": "power_allocation",
    "rm_dl_power_model": "power_allocation",
    "rm_P_tot_dbm": "power_allocation",
    "rm_p_min_dbm": "power_allocation",
    "rm_p_max_dbm": "power_allocation",
    "rm_max_prbs_per_ue": "power_allocation",

    # HARQ
    "enable_harq_full": "harq",
    "enable_harq_deferral": "harq",
    "harq_max_procs": "harq",
    "harq_ack_delay_ttis": "harq",
    "harq_target_bler": "harq",
    "harq_max_retx": "harq",
    "harq_retx_priority_bonus": "harq",
    "harq_flush_tail": "harq",
    "bler_slope_db": "harq",
    "bler_margin_db": "harq",
    "bler_curve_path": "harq",
    "olla_step_up_db": "harq",
    "olla_step_down_db": "harq",
    "olla_init_offset_db": "harq",
    "olla_min_db": "harq",
    "olla_max_db": "harq",

    # MCS
    "mcs_table_kind": "mcs",
    "csi_mcs_table": "mcs",
    "mcs_3gpp_table_path": "mcs",
    "pdsch_dmrs_sym_per_slot": "mcs",
    "dmrs_re_per_sym_per_prb": "mcs",
    "oh_prb": "mcs",

    # Constellation
    "enable_constellation": "constellation",
    "tle_catalog_path": "constellation",
    "constellation_max_ground_radius_km": "constellation",
    "constellation_max_sats_per_tti": "constellation",
    "min_elev_deg": "constellation",
    "association_metric": "constellation",
    "ho_enabled": "constellation",
    "ho_hyst_db": "constellation",
    "ho_ttt_ttis": "constellation",
    "constellation_prb_cap": "constellation",

    # Traffic (Phase 8)
    "traffic_model": "traffic",
    "poisson_arrival_rate_hz": "traffic",
    "poisson_packet_size_bytes": "traffic",
    "poisson_packet_size_std_bytes": "traffic",
    "poisson_qci": "traffic",
    "ftp3_file_size_bytes": "traffic",
    "ftp3_reading_time_ms": "traffic",
    "ftp3_qci": "traffic",
    "video_frame_rate_fps": "traffic",
    "video_gop_size": "traffic",
    "video_i_frame_size_bytes": "traffic",
    "video_p_frame_size_bytes": "traffic",
    "video_qci": "traffic",
    "voip_codec": "traffic",
    "voip_activity_factor": "traffic",
    "voip_packet_interval_ms": "traffic",
    "voip_qci": "traffic",
    "mixed_embb_pct": "traffic",
    "mixed_urllc_pct": "traffic",
    "mixed_voip_pct": "traffic",

    # QoS (Phase 8/10)
    "enable_qos": "qos",
    "scheduler_algorithm": "qos",
    "mlwdf_alpha": "qos",
    "mlwdf_delta": "qos",
    "mlwdf_tau": "qos",
    "exppf_beta": "qos",
    "exppf_c": "qos",
    "urllc_preemption": "qos",
    "urllc_mini_slot": "qos",
    "ntn_pdb_extension_factor": "qos",
    "compensate_rtt_in_pdb": "qos",

    # Latency KPI (Phase 8)
    "record_packet_latency": "latency_kpi",
    "latency_percentiles": "latency_kpi",
    "record_per_qos_stats": "latency_kpi",
    "max_stored_packets": "latency_kpi",

    # MAC (Phase 9)
    "enable_bsr": "mac",
    "bsr_table_type": "mac",
    "bsr_periodic_timer_ms": "mac",
    "bsr_retx_timer_ms": "mac",
    "enable_drx": "mac",
    "drx_on_duration_ms": "mac",
    "drx_inactivity_timer_ms": "mac",
    "drx_short_cycle_ms": "mac",
    "drx_long_cycle_ms": "mac",
    "drx_short_cycle_timer": "mac",
    "ntn_timing_adaptation": "mac",
    "k1_slots_base": "mac",
    "k1_ntn_extension_factor": "mac",
    "harq_rtt_scaling": "mac",

    # OALS (Phase 11 - Patent)
    "enable_oals": "oals",
    "lookahead_horizon_ttis": "oals",
    "lookahead_sample_interval": "oals",
    "lookahead_update_interval": "oals",
    "alpha_urgent": "oals",
    "beta_wait": "oals",
    "theta_lookahead": "oals",
    "gamma_decay": "oals",
    "boost_factor": "oals",
    "enable_trend_correction": "oals",
    "trend_threshold_db": "oals",
    "trend_epsilon": "oals",
    "enable_predictive_ho": "oals",
    "handover_prep_ttis": "oals",
    "handover_recovery_ttis": "oals",
    "enable_harq_lookahead": "oals",
    "harq_sinr_aggressive_db": "oals",
    "harq_sinr_conservative_db": "oals",
    "harq_delta_threshold_db": "oals",
    "early_retx_trend_threshold": "oals",
    "early_retx_max_rv": "oals",

    # Output
    "write_json_report": "output",
    "report_basename": "output",
    "plot_dir": "output",
    "save_plots": "output",
    "show_plots": "output",
    "print_harq_summary": "output",
    "show_progress": "output",
    "include_serving_trace": "output",
    "record_assignments": "output",
    "record_assignments_target": "output",
    "record_ue_thr": "output",
    "record_ue_ack_thr": "output",
    "enable_trace": "output",
    "trace_level": "output",
}


class ConfigDict(MutableMapping):
    """
    A dict-like wrapper around NTNSimConfig for backward compatibility.

    This class allows legacy code to continue using dictionary-style access
    patterns (config["key"], config.get("key"), config.update({})) while
    internally using the type-safe NTNSimConfig dataclass.

    Example:
        >>> config = ConfigDict()
        >>> config["N_UE"] = 50
        >>> print(config["N_UE"])
        50
        >>> config.update({"seed": 42, "T": 100})
        >>> print(config.get("enable_orbit_dynamics", True))
        True
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        config: Optional[NTNSimConfig] = None
    ):
        """
        Initialize ConfigDict.

        Args:
            data: Optional initial data dictionary (flat format)
            config: Optional NTNSimConfig instance to wrap
        """
        if config is not None:
            self._config = config
        elif data is not None:
            self._config = NTNSimConfig.from_flat_dict(data)
        else:
            self._config = NTNSimConfig()

        # Store any extra keys not in the schema
        self._extra: Dict[str, Any] = {}

        # If data was provided, store any extra keys
        if data is not None:
            for key in data:
                if key not in _KEY_TO_GROUP:
                    self._extra[key] = data[key]

    @property
    def typed_config(self) -> NTNSimConfig:
        """Access the underlying typed configuration."""
        return self._config

    def _get_group(self, key: str):
        """Get the configuration group object for a key."""
        group_name = _KEY_TO_GROUP.get(key)
        if group_name is None:
            return None
        return getattr(self._config, group_name, None)

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value by key."""
        # Check extra keys first
        if key in self._extra:
            return self._extra[key]

        # Get from typed config
        group = self._get_group(key)
        if group is None:
            raise KeyError(key)

        if hasattr(group, key):
            return getattr(group, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a configuration value by key."""
        group = self._get_group(key)
        if group is None:
            # Store as extra key
            self._extra[key] = value
            return

        if hasattr(group, key):
            setattr(group, key, value)
        else:
            # Store as extra key
            self._extra[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete a configuration value (only for extra keys)."""
        if key in self._extra:
            del self._extra[key]
        elif key in _KEY_TO_GROUP:
            raise KeyError(f"Cannot delete schema key: {key}")
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over all configuration keys."""
        # Yield all schema keys
        yield from _KEY_TO_GROUP.keys()
        # Yield extra keys
        yield from self._extra.keys()

    def __len__(self) -> int:
        """Return total number of configuration keys."""
        return len(_KEY_TO_GROUP) + len(self._extra)

    def __contains__(self, key: object) -> bool:
        """Check if a key exists in configuration."""
        if not isinstance(key, str):
            return False
        return key in _KEY_TO_GROUP or key in self._extra

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default."""
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, other: Dict[str, Any] = None, **kwargs) -> None:
        """Update configuration from another dict or kwargs."""
        if other:
            for key, value in other.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def copy(self) -> "ConfigDict":
        """Create a shallow copy of the configuration."""
        # Convert to flat dict and create new instance
        data = self.to_flat_dict()
        return ConfigDict(data=data)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary."""
        result = self._config.to_flat_dict()
        result.update(self._extra)
        return result

    def to_nested_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary format."""
        from dataclasses import asdict
        result = asdict(self._config)
        if self._extra:
            result["_extra"] = self._extra.copy()
        return result

    def keys(self):
        """Return configuration keys."""
        return list(self)

    def values(self):
        """Return configuration values."""
        return [self[k] for k in self]

    def items(self):
        """Return configuration key-value pairs."""
        return [(k, self[k]) for k in self]

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigDict({self.to_flat_dict()})"

    # ==========================================================================
    # Convenience accessors for configuration groups
    # ==========================================================================

    @property
    def simulation(self):
        """Access simulation configuration group."""
        return self._config.simulation

    @property
    def radio_map(self):
        """Access radio map configuration group."""
        return self._config.radio_map

    @property
    def numerology(self):
        """Access numerology configuration group."""
        return self._config.numerology

    @property
    def channel(self):
        """Access channel configuration group."""
        return self._config.channel

    @property
    def link_budget(self):
        """Access link budget configuration group."""
        return self._config.link_budget

    @property
    def geometry(self):
        """Access geometry configuration group."""
        return self._config.geometry

    @property
    def orbit(self):
        """Access orbit configuration group."""
        return self._config.orbit

    @property
    def time_varying(self):
        """Access time varying configuration group."""
        return self._config.time_varying

    @property
    def csi(self):
        """Access CSI configuration group."""
        return self._config.csi

    @property
    def scheduler(self):
        """Access scheduler configuration group."""
        return self._config.scheduler

    @property
    def power_allocation(self):
        """Access power allocation configuration group."""
        return self._config.power_allocation

    @property
    def harq(self):
        """Access HARQ configuration group."""
        return self._config.harq

    @property
    def mcs(self):
        """Access MCS configuration group."""
        return self._config.mcs

    @property
    def constellation(self):
        """Access constellation configuration group."""
        return self._config.constellation

    @property
    def output(self):
        """Access output configuration group."""
        return self._config.output

    @property
    def traffic(self):
        """Access traffic configuration group."""
        return self._config.traffic

    @property
    def qos(self):
        """Access QoS configuration group."""
        return self._config.qos

    @property
    def latency_kpi(self):
        """Access latency KPI configuration group."""
        return self._config.latency_kpi

    @property
    def mac(self):
        """Access MAC configuration group."""
        return self._config.mac

    @property
    def oals(self):
        """Access OALS configuration group."""
        return self._config.oals

    # ==========================================================================
    # Schema and documentation methods
    # ==========================================================================

    def to_json_schema(self) -> Dict[str, Any]:
        """Generate JSON Schema for the configuration."""
        return self._config.to_json_schema()

    def get_field_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific field."""
        group = self._get_group(key)
        if group is None:
            return None
        info = group.get_field_info()
        return info.get(key)

    def get_all_field_info(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all fields, organized by group."""
        result = {}
        for group_name in [
            "simulation", "radio_map", "numerology", "channel",
            "link_budget", "geometry", "orbit", "time_varying",
            "csi", "scheduler", "power_allocation", "harq",
            "mcs", "constellation", "output",
            "traffic", "qos", "latency_kpi", "mac", "oals",
        ]:
            group = getattr(self._config, group_name, None)
            if group:
                result[group_name] = group.get_field_info()
        return result


def create_config(base: Optional[Dict[str, Any]] = None) -> ConfigDict:
    """
    Factory function to create a new ConfigDict.

    Args:
        base: Optional base dictionary to initialize with

    Returns:
        New ConfigDict instance
    """
    return ConfigDict(data=base)


def merge_configs(base: ConfigDict, override: Dict[str, Any]) -> ConfigDict:
    """
    Merge override dictionary into a copy of base configuration.

    Args:
        base: Base ConfigDict instance
        override: Dictionary with override values

    Returns:
        New ConfigDict with merged configuration
    """
    result = base.copy()
    result.update(override)
    return result
