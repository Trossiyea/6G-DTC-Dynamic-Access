# -*- coding: utf-8 -*-
"""
Simulation State Data Classes.

Provides serializable state containers for simulation data, enabling:
- Checkpoint/resume functionality
- Web API state transfer
- Unit testing with known states
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np


@dataclass
class GeometryResult:
    """Result of geometry and beam calculations for a set of UEs.

    Attributes:
        L_fs_db: Free-space path loss per UE [N_UE] (dB)
        G_rx_db: Receiver beam gain per UE [N_UE] (dB)
        elev_deg: Elevation angle per UE [N_UE] (degrees)
        tau_s: Propagation delay per UE [N_UE] (seconds), optional
        fd_hz: Doppler shift per UE [N_UE] (Hz), optional
    """
    L_fs_db: np.ndarray
    G_rx_db: np.ndarray
    elev_deg: np.ndarray
    tau_s: Optional[np.ndarray] = None
    fd_hz: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "L_fs_db": self.L_fs_db.tolist() if self.L_fs_db is not None else None,
            "G_rx_db": self.G_rx_db.tolist() if self.G_rx_db is not None else None,
            "elev_deg": self.elev_deg.tolist() if self.elev_deg is not None else None,
            "tau_s": self.tau_s.tolist() if self.tau_s is not None else None,
            "fd_hz": self.fd_hz.tolist() if self.fd_hz is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryResult":
        """Create from dict."""
        return cls(
            L_fs_db=np.array(d["L_fs_db"]) if d.get("L_fs_db") else None,
            G_rx_db=np.array(d["G_rx_db"]) if d.get("G_rx_db") else None,
            elev_deg=np.array(d["elev_deg"]) if d.get("elev_deg") else None,
            tau_s=np.array(d["tau_s"]) if d.get("tau_s") else None,
            fd_hz=np.array(d["fd_hz"]) if d.get("fd_hz") else None,
        )


@dataclass
class SchedulerResult:
    """Result of a scheduler execution.

    Attributes:
        avg_se: Average spectral efficiency (bits/s/Hz)
        total_bits: Total bits scheduled
        assignments: PRB assignment matrix [T, Z] (UE indices), optional
        ue_throughput: Per-UE throughput time series [T, N_UE], optional
        harq_stats: HARQ statistics dict, optional
    """
    avg_se: float
    total_bits: float = 0.0
    assignments: Optional[np.ndarray] = None
    ue_throughput: Optional[np.ndarray] = None
    harq_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "avg_se": float(self.avg_se),
            "total_bits": float(self.total_bits),
            "assignments": self.assignments.tolist() if self.assignments is not None else None,
            "ue_throughput": self.ue_throughput.tolist() if self.ue_throughput is not None else None,
            "harq_stats": self.harq_stats,
        }


@dataclass
class TimeSeriesData:
    """Time-varying simulation data.

    Attributes:
        se_time_rm: Per-PRB SE metric time series [T, UE, Z]
        se_time_wb: Wideband SE metric time series [T, UE]
        snr_time: Per-PRB SNR time series [T, UE, Z]
        snr_wb_time: Wideband SNR time series [T, UE]
        tau_time: Propagation delay time series [T, UE], optional
        fd_time: Doppler shift time series [T, UE], optional
    """
    se_time_rm: np.ndarray
    se_time_wb: np.ndarray
    snr_time: np.ndarray
    snr_wb_time: np.ndarray
    tau_time: Optional[np.ndarray] = None
    fd_time: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict (summary only for large arrays)."""
        return {
            "shape_se_time_rm": list(self.se_time_rm.shape),
            "shape_snr_time": list(self.snr_time.shape),
            "has_tau_time": self.tau_time is not None,
            "has_fd_time": self.fd_time is not None,
        }


@dataclass
class SimulationState:
    """Complete state of a single-satellite simulation.

    This class encapsulates all intermediate data during simulation,
    enabling checkpoint/resume, unit testing, and Web state transfer.

    Attributes:
        config: Simulation configuration dict
        rng: Random number generator state (seed)

        # Radio Map and UE data
        R_xyz_dbm: Radio Map [X, Y, Z] (dBm)
        ue_pos: UE positions [N_UE, 2] (grid indices)
        X, Y, Z: Radio Map dimensions
        N_UE: Number of UEs
        T: Number of TTIs

        # Geometry
        geometry: GeometryResult for static snapshot

        # Capacity/SNR (static snapshot)
        cap: Per-UE per-PRB capacity [N_UE, Z]
        cap_wb: Wideband capacity [N_UE]
        snr_lin: Per-UE per-PRB linear SNR [N_UE, Z]
        snr_lin_wb: Wideband linear SNR [N_UE]
        P_rx_dbm: Received power [N_UE] (dBm)
        I_total_dbm: Interference+noise [N_UE, Z] (dBm)

        # Noise/power
        noise_dbm: Thermal noise level (dBm)
        prb_bw_hz: PRB bandwidth (Hz)

        # Time-varying data (optional)
        time_series: TimeSeriesData if time-varying enabled
        metric_override: Static predicted metric for RadioMap scheduler

        # Execution state
        current_tti: Current TTI index (for streaming)
        phase: Current phase ("uninitialized", "initialized", "running", "complete")
    """
    # Configuration
    config: Dict[str, Any]
    seed: int

    # Dimensions
    X: int = 0
    Y: int = 0
    Z: int = 0
    N_UE: int = 0
    T: int = 0

    # Radio Map and UE positions
    R_xyz_dbm: Optional[np.ndarray] = None
    ue_pos: Optional[np.ndarray] = None

    # Geometry
    geometry: Optional[GeometryResult] = None

    # Capacity/SNR (static snapshot)
    cap: Optional[np.ndarray] = None
    cap_wb: Optional[np.ndarray] = None
    snr_lin: Optional[np.ndarray] = None
    snr_lin_wb: Optional[np.ndarray] = None
    P_rx_dbm: Optional[np.ndarray] = None
    I_total_dbm: Optional[np.ndarray] = None

    # Noise/power
    noise_dbm: float = -121.45
    prb_bw_hz: float = 360000.0
    P_tx_per_ue_dbm: Optional[Union[np.ndarray, float]] = None

    # Time-varying data
    time_series: Optional[TimeSeriesData] = None
    metric_override: Optional[np.ndarray] = None

    # Orbit model reference (not serializable)
    orbit_model: Optional[Any] = field(default=None, repr=False)

    # Execution state
    current_tti: int = 0
    phase: str = "uninitialized"

    # Scheduler results (populated after run)
    result_baseline: Optional[SchedulerResult] = None
    result_radiomap: Optional[SchedulerResult] = None

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Convert to JSON-serializable dict.

        Args:
            include_arrays: If True, include full numpy arrays (large).
                           If False, only include shapes and summary stats.
        """
        d = {
            "seed": self.seed,
            "X": self.X,
            "Y": self.Y,
            "Z": self.Z,
            "N_UE": self.N_UE,
            "T": self.T,
            "noise_dbm": self.noise_dbm,
            "prb_bw_hz": self.prb_bw_hz,
            "current_tti": self.current_tti,
            "phase": self.phase,
        }

        if include_arrays:
            d["R_xyz_dbm"] = self.R_xyz_dbm.tolist() if self.R_xyz_dbm is not None else None
            d["ue_pos"] = self.ue_pos.tolist() if self.ue_pos is not None else None
            d["cap"] = self.cap.tolist() if self.cap is not None else None
            d["snr_lin"] = self.snr_lin.tolist() if self.snr_lin is not None else None
        else:
            # Summary only
            if self.R_xyz_dbm is not None:
                d["R_xyz_dbm_shape"] = list(self.R_xyz_dbm.shape)
                d["R_xyz_dbm_range"] = [float(self.R_xyz_dbm.min()), float(self.R_xyz_dbm.max())]
            if self.cap is not None:
                d["cap_shape"] = list(self.cap.shape)
                d["cap_mean"] = float(self.cap.mean())
            if self.snr_lin is not None:
                d["snr_lin_mean_db"] = float(10 * np.log10(self.snr_lin.mean()))

        if self.geometry is not None:
            d["geometry"] = self.geometry.to_dict()

        if self.time_series is not None:
            d["time_series"] = self.time_series.to_dict()

        if self.result_baseline is not None:
            d["result_baseline"] = self.result_baseline.to_dict()
        if self.result_radiomap is not None:
            d["result_radiomap"] = self.result_radiomap.to_dict()

        return d

    def is_initialized(self) -> bool:
        """Check if state has been initialized."""
        return self.phase != "uninitialized" and self.R_xyz_dbm is not None

    def is_complete(self) -> bool:
        """Check if simulation is complete."""
        return self.phase == "complete"


@dataclass
class ConstellationState:
    """State for multi-satellite constellation simulation.

    Extends SimulationState with constellation-specific data.

    Attributes:
        base_state: Underlying SimulationState

        # Constellation-specific
        num_satellites: Number of satellites in constellation
        serving: Current serving satellite per UE [N_UE]
        ho_timer: Handover timer per UE [N_UE]
        curr_metric_db: Current association metric per UE [N_UE]
        ho_events: Handover event log per UE
        outage_ttis: Outage TTI count per UE [N_UE]

        # Per-satellite caches (cleared each TTI)
        snr_lin_per_sat: Dict[sat_idx, np.ndarray[N_UE, Z]]
        cap_per_sat: Dict[sat_idx, np.ndarray[N_UE, Z]]

        # Per-satellite KPI accumulators
        kpi_per_sat: Dict[sat_idx, KPI dict]

        # HARQ managers per satellite
        harq_base_by_sat: Dict[sat_idx, HarqManager]
        harq_rm_by_sat: Dict[sat_idx, HarqManager]
    """
    config: Dict[str, Any]
    seed: int

    # Dimensions (inherited from RadioMap)
    X: int = 0
    Y: int = 0
    Z: int = 0
    N_UE: int = 0
    T: int = 0

    # Radio Map and UE positions
    R_xyz_dbm: Optional[np.ndarray] = None
    R_t: Optional[np.ndarray] = None  # Time-varying copy
    ue_pos: Optional[np.ndarray] = None

    # Noise/power
    noise_dbm: float = -121.45
    prb_bw_hz: float = 360000.0
    P_tx_dbm: float = 0.0

    # Constellation-specific
    num_satellites: int = 0
    serving: Optional[np.ndarray] = None  # [N_UE], -1 if unassociated
    ho_timer: Optional[np.ndarray] = None  # [N_UE]
    curr_metric_db: Optional[np.ndarray] = None  # [N_UE]
    ho_events: Optional[List[List[Dict]]] = None  # [N_UE][events]
    outage_ttis: Optional[np.ndarray] = None  # [N_UE]
    serving_trace: Optional[List[np.ndarray]] = None  # [T][N_UE]

    # Per-satellite caches (transient, not serialized)
    snr_lin_per_sat: Dict[int, np.ndarray] = field(default_factory=dict)
    snr_wb_per_sat: Dict[int, np.ndarray] = field(default_factory=dict)
    cap_per_sat: Dict[int, np.ndarray] = field(default_factory=dict)
    prx_dbm_per_sat: Dict[int, np.ndarray] = field(default_factory=dict)
    elev_per_sat: Dict[int, np.ndarray] = field(default_factory=dict)

    # KPI accumulators
    sum_rate_rm: float = 0.0
    sum_rate_base_def: float = 0.0
    kpi_per_sat: Dict[int, Dict] = field(default_factory=dict)

    # HARQ managers (not serializable)
    harq_base_by_sat: Dict[int, Any] = field(default_factory=dict, repr=False)
    harq_rm_by_sat: Dict[int, Any] = field(default_factory=dict, repr=False)

    # Orbit model reference (not serializable)
    orbit: Optional[Any] = field(default=None, repr=False)

    # Execution state
    current_tti: int = 0
    phase: str = "uninitialized"

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = {
            "seed": self.seed,
            "X": self.X,
            "Y": self.Y,
            "Z": self.Z,
            "N_UE": self.N_UE,
            "T": self.T,
            "num_satellites": self.num_satellites,
            "noise_dbm": self.noise_dbm,
            "prb_bw_hz": self.prb_bw_hz,
            "current_tti": self.current_tti,
            "phase": self.phase,
            "sum_rate_rm": self.sum_rate_rm,
            "sum_rate_base_def": self.sum_rate_base_def,
        }

        if self.serving is not None:
            d["serving"] = self.serving.tolist()
        if self.outage_ttis is not None:
            d["outage_ttis"] = self.outage_ttis.tolist()
        if self.ho_events is not None:
            d["ho_event_counts"] = [len(events) for events in self.ho_events]

        if include_arrays and self.R_xyz_dbm is not None:
            d["R_xyz_dbm"] = self.R_xyz_dbm.tolist()

        return d

    def clear_per_tti_caches(self):
        """Clear per-TTI satellite caches."""
        self.snr_lin_per_sat.clear()
        self.snr_wb_per_sat.clear()
        self.cap_per_sat.clear()
        self.prx_dbm_per_sat.clear()
        self.elev_per_sat.clear()

    def is_initialized(self) -> bool:
        """Check if state has been initialized."""
        return self.phase != "uninitialized" and self.R_xyz_dbm is not None

    def is_complete(self) -> bool:
        """Check if simulation is complete."""
        return self.phase == "complete"
