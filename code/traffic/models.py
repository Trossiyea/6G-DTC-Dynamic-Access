"""
Traffic generation models for downlink flows.

Implements various 3GPP-compliant traffic models including Poisson arrivals,
FTP Model 3, video streaming, and VoIP with voice activity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from .packets import Packet


class TrafficGenerator(ABC):
    """Abstract base class for traffic generators.

    Each generator produces packet arrivals for a specific traffic model.
    """

    @abstractmethod
    def generate_arrivals(
        self,
        tti: int,
        ue_id: int,
        rng: np.random.Generator
    ) -> List[Packet]:
        """Generate packet arrivals for this TTI and UE.

        Args:
            tti: Current TTI index
            ue_id: UE index
            rng: Random number generator

        Returns:
            List of newly arrived packets (may be empty)
        """
        pass

    @abstractmethod
    def get_average_rate_bps(self) -> float:
        """Get average offered load in bits per second."""
        pass

    def is_full_buffer(self) -> bool:
        """Check if this is a full buffer model (no arrivals)."""
        return False


class FullBufferGenerator(TrafficGenerator):
    """Legacy full-buffer model with infinite backlog.

    This model assumes each UE always has data to send. No actual packet
    arrivals are generated; buffers are pre-filled at initialization.
    """

    def __init__(self, infinite_bits: int = 100_000_000):
        """Initialize full buffer generator.

        Args:
            infinite_bits: Initial buffer size per UE (large value)
        """
        self.infinite_bits = infinite_bits

    def generate_arrivals(self, tti: int, ue_id: int, rng: np.random.Generator) -> List[Packet]:
        """No arrivals in full buffer model."""
        return []

    def get_average_rate_bps(self) -> float:
        """Infinite rate (saturated)."""
        return float('inf')

    def is_full_buffer(self) -> bool:
        return True


class PoissonGenerator(TrafficGenerator):
    """Poisson arrival process with configurable packet size distribution.

    Generates independent packet arrivals with exponentially distributed
    inter-arrival times (Poisson process).
    """

    def __init__(
        self,
        arrival_rate_hz: float,
        packet_size_bytes: int,
        packet_size_std_bytes: int = 0,
        tti_duration_ms: float = 1.0,
        qci: int = 9,
        bearer_id: int = 0,
    ):
        """Initialize Poisson traffic generator.

        Args:
            arrival_rate_hz: Lambda (packets per second)
            packet_size_bytes: Mean packet size in bytes
            packet_size_std_bytes: Std dev of packet size (0 = fixed size)
            tti_duration_ms: TTI duration in milliseconds
            qci: QCI/5QI class for generated packets
            bearer_id: Bearer identifier
        """
        self.arrival_rate_hz = arrival_rate_hz
        self.arrival_rate_per_tti = arrival_rate_hz * (tti_duration_ms / 1000.0)
        self.packet_size_bytes = packet_size_bytes
        self.packet_size_std = packet_size_std_bytes
        self.tti_duration_ms = tti_duration_ms
        self.qci = qci
        self.bearer_id = bearer_id
        self._packet_counter = 0

    def generate_arrivals(self, tti: int, ue_id: int, rng: np.random.Generator) -> List[Packet]:
        """Generate Poisson arrivals for this TTI."""
        # Poisson number of arrivals
        n_arrivals = rng.poisson(self.arrival_rate_per_tti)

        packets = []
        for _ in range(n_arrivals):
            size_bytes = self._sample_packet_size(rng)
            pkt = Packet(
                packet_id=self._packet_counter,
                ue_id=ue_id,
                bearer_id=self.bearer_id,
                size_bits=size_bytes * 8,
                arrival_tti=tti,
                deadline_tti=None,  # Set by QoS manager
                priority=0,
                qci=self.qci,
            )
            packets.append(pkt)
            self._packet_counter += 1

        return packets

    def _sample_packet_size(self, rng: np.random.Generator) -> int:
        """Sample packet size from distribution."""
        if self.packet_size_std > 0:
            size = rng.normal(self.packet_size_bytes, self.packet_size_std)
            return max(1, int(size))
        else:
            return self.packet_size_bytes

    def get_average_rate_bps(self) -> float:
        """Average offered load."""
        return self.arrival_rate_hz * self.packet_size_bytes * 8.0


class FTPModel3Generator(TrafficGenerator):
    """3GPP FTP Model 3: File downloads with reading time.

    Models file download sessions with exponentially distributed reading
    times between successive downloads.
    """

    def __init__(
        self,
        file_size_bytes: int = 512 * 1024,
        reading_time_ms: float = 180.0,
        tti_duration_ms: float = 1.0,
        qci: int = 9,
        bearer_id: int = 0,
    ):
        """Initialize FTP Model 3 generator.

        Args:
            file_size_bytes: Mean file size (512 KB default)
            reading_time_ms: Mean inter-arrival time (reading time)
            tti_duration_ms: TTI duration in milliseconds
            qci: QCI/5QI class
            bearer_id: Bearer identifier
        """
        self.file_size_bytes = file_size_bytes
        self.reading_time_ms = reading_time_ms
        self.reading_time_ttis = reading_time_ms / tti_duration_ms
        self.tti_duration_ms = tti_duration_ms
        self.qci = qci
        self.bearer_id = bearer_id

        # Per-UE state: next arrival TTI
        self._next_arrival: Dict[int, int] = {}
        self._packet_counter = 0

    def generate_arrivals(self, tti: int, ue_id: int, rng: np.random.Generator) -> List[Packet]:
        """Generate file arrivals for this TTI."""
        # Initialize first arrival for new UE
        if ue_id not in self._next_arrival:
            self._next_arrival[ue_id] = tti

        # Check if it's time for new file
        if tti >= self._next_arrival[ue_id]:
            # Generate file as single large packet
            pkt = Packet(
                packet_id=self._packet_counter,
                ue_id=ue_id,
                bearer_id=self.bearer_id,
                size_bits=self.file_size_bytes * 8,
                arrival_tti=tti,
                deadline_tti=None,
                priority=0,
                qci=self.qci,
            )
            self._packet_counter += 1

            # Schedule next arrival (exponential reading time)
            delay_ttis = int(rng.exponential(self.reading_time_ttis))
            self._next_arrival[ue_id] = tti + max(1, delay_ttis)

            return [pkt]

        return []

    def get_average_rate_bps(self) -> float:
        """Average offered load."""
        if self.reading_time_ms <= 0:
            return 0.0
        # Rate = file_size / (reading_time + download_time)
        # Approximation: download_time << reading_time
        avg_rate_bps = (self.file_size_bytes * 8.0) / (self.reading_time_ms / 1000.0)
        return avg_rate_bps


class VideoStreamingGenerator(TrafficGenerator):
    """Video streaming with I/P/B frame structure.

    Models periodic video frames with variable sizes following a GOP
    (Group of Pictures) pattern.
    """

    def __init__(
        self,
        frame_rate_fps: float = 30.0,
        i_frame_size_bytes: int = 50000,
        p_frame_size_bytes: int = 10000,
        gop_size: int = 15,
        tti_duration_ms: float = 1.0,
        qci: int = 4,
        bearer_id: int = 0,
    ):
        """Initialize video streaming generator.

        Args:
            frame_rate_fps: Video frame rate (frames per second)
            i_frame_size_bytes: I-frame (keyframe) size
            p_frame_size_bytes: P-frame (predicted) size
            gop_size: GOP length (frames per I-frame)
            tti_duration_ms: TTI duration in milliseconds
            qci: QCI/5QI class (default 4 = non-conversational video)
            bearer_id: Bearer identifier
        """
        self.frame_rate_fps = frame_rate_fps
        self.i_frame_size = i_frame_size_bytes
        self.p_frame_size = p_frame_size_bytes
        self.gop_size = gop_size
        self.tti_duration_ms = tti_duration_ms
        self.qci = qci
        self.bearer_id = bearer_id

        # Frame period in TTIs
        self.frame_period_ms = 1000.0 / frame_rate_fps
        self.frame_period_ttis = self.frame_period_ms / tti_duration_ms

        # Per-UE state
        self._next_frame_tti: Dict[int, int] = {}
        self._frame_counter: Dict[int, int] = {}
        self._packet_counter = 0

    def generate_arrivals(self, tti: int, ue_id: int, rng: np.random.Generator) -> List[Packet]:
        """Generate video frame arrivals."""
        # Initialize first frame for new UE
        if ue_id not in self._next_frame_tti:
            self._next_frame_tti[ue_id] = tti
            self._frame_counter[ue_id] = 0

        # Check if it's time for next frame
        if tti >= self._next_frame_tti[ue_id]:
            frame_num = self._frame_counter[ue_id]

            # Determine frame type (I-frame every gop_size frames)
            is_i_frame = (frame_num % self.gop_size) == 0
            frame_size = self.i_frame_size if is_i_frame else self.p_frame_size

            pkt = Packet(
                packet_id=self._packet_counter,
                ue_id=ue_id,
                bearer_id=self.bearer_id,
                size_bits=frame_size * 8,
                arrival_tti=tti,
                deadline_tti=None,  # Set by QoS manager
                priority=0 if is_i_frame else 1,  # I-frames higher priority
                qci=self.qci,
            )
            self._packet_counter += 1

            # Schedule next frame
            self._frame_counter[ue_id] += 1
            self._next_frame_tti[ue_id] = tti + int(self.frame_period_ttis)

            return [pkt]

        return []

    def get_average_rate_bps(self) -> float:
        """Average offered load."""
        # Average frame size
        avg_frame_size = (
            self.i_frame_size + (self.gop_size - 1) * self.p_frame_size
        ) / self.gop_size
        return avg_frame_size * 8.0 * self.frame_rate_fps


class VoIPGenerator(TrafficGenerator):
    """VoIP traffic with ON/OFF voice activity.

    Models periodic small packets during talk spurts with silence periods
    (ON/OFF model).
    """

    def __init__(
        self,
        codec: str = "AMR-WB",
        activity_factor: float = 0.5,
        packet_interval_ms: float = 20.0,
        tti_duration_ms: float = 1.0,
        qci: int = 1,
        bearer_id: int = 0,
    ):
        """Initialize VoIP generator.

        Args:
            codec: Codec name (AMR-WB, G.711, EVS)
            activity_factor: Voice activity factor (0-1)
            packet_interval_ms: Packet interval during talk spurts
            tti_duration_ms: TTI duration in milliseconds
            qci: QCI/5QI class (default 1 = conversational voice)
            bearer_id: Bearer identifier
        """
        self.codec = codec
        self.activity_factor = activity_factor
        self.packet_interval_ms = packet_interval_ms
        self.packet_interval_ttis = packet_interval_ms / tti_duration_ms
        self.tti_duration_ms = tti_duration_ms
        self.qci = qci
        self.bearer_id = bearer_id

        # Codec-specific parameters
        self.packet_size_bytes = self._get_codec_packet_size(codec)

        # Per-UE state
        self._next_packet_tti: Dict[int, int] = {}
        self._is_active: Dict[int, bool] = {}
        self._state_change_tti: Dict[int, int] = {}
        self._packet_counter = 0

    def _get_codec_packet_size(self, codec: str) -> int:
        """Get typical packet size for codec."""
        codec_sizes = {
            "AMR-WB": 40,  # AMR-WB 12.65 kbps mode
            "G.711": 160,  # G.711 with 20ms packetization
            "EVS": 50,     # EVS SWB mode
        }
        return codec_sizes.get(codec, 40)

    def generate_arrivals(self, tti: int, ue_id: int, rng: np.random.Generator) -> List[Packet]:
        """Generate VoIP packet arrivals with ON/OFF activity."""
        # Initialize state for new UE
        if ue_id not in self._is_active:
            self._is_active[ue_id] = rng.random() < self.activity_factor
            self._next_packet_tti[ue_id] = tti
            self._state_change_tti[ue_id] = tti + self._sample_state_duration(rng, self._is_active[ue_id])

        # Check for state transitions (ON -> OFF or OFF -> ON)
        if tti >= self._state_change_tti[ue_id]:
            self._is_active[ue_id] = not self._is_active[ue_id]
            self._state_change_tti[ue_id] = tti + self._sample_state_duration(rng, self._is_active[ue_id])

        # Generate packet if in active state and it's time
        if self._is_active[ue_id] and tti >= self._next_packet_tti[ue_id]:
            pkt = Packet(
                packet_id=self._packet_counter,
                ue_id=ue_id,
                bearer_id=self.bearer_id,
                size_bits=self.packet_size_bytes * 8,
                arrival_tti=tti,
                deadline_tti=None,  # Set by QoS manager
                priority=0,
                qci=self.qci,
            )
            self._packet_counter += 1

            # Schedule next packet
            self._next_packet_tti[ue_id] = tti + int(self.packet_interval_ttis)

            return [pkt]

        return []

    def _sample_state_duration(self, rng: np.random.Generator, is_active: bool) -> int:
        """Sample duration of ON or OFF state."""
        # Typical values: ON ~2s, OFF ~2s
        mean_duration_ms = 2000.0
        duration_ms = rng.exponential(mean_duration_ms)
        return max(1, int(duration_ms / self.tti_duration_ms))

    def get_average_rate_bps(self) -> float:
        """Average offered load accounting for activity factor."""
        packet_rate_hz = 1000.0 / self.packet_interval_ms
        return self.packet_size_bytes * 8.0 * packet_rate_hz * self.activity_factor


def create_traffic_generator(config: Dict) -> TrafficGenerator:
    """Factory function to create traffic generator from configuration.

    Args:
        config: Configuration dictionary with traffic model parameters

    Returns:
        TrafficGenerator instance

    Raises:
        ValueError: If traffic model is unknown
    """
    model = config.get("traffic_model", "full_buffer")
    tti_ms = config.get("tti_ms", 1.0)

    if model == "full_buffer":
        return FullBufferGenerator()

    elif model == "poisson":
        return PoissonGenerator(
            arrival_rate_hz=config.get("poisson_arrival_rate_hz", 100.0),
            packet_size_bytes=config.get("poisson_packet_size_bytes", 1500),
            packet_size_std_bytes=config.get("poisson_packet_size_std_bytes", 0),
            tti_duration_ms=tti_ms,
            qci=config.get("poisson_qci", 9),
            bearer_id=config.get("poisson_bearer_id", 0),
        )

    elif model == "ftp3":
        return FTPModel3Generator(
            file_size_bytes=config.get("ftp3_file_size_bytes", 512 * 1024),
            reading_time_ms=config.get("ftp3_reading_time_ms", 180.0),
            tti_duration_ms=tti_ms,
            qci=config.get("ftp3_qci", 9),
            bearer_id=config.get("ftp3_bearer_id", 0),
        )

    elif model == "video":
        return VideoStreamingGenerator(
            frame_rate_fps=config.get("video_frame_rate_fps", 30.0),
            i_frame_size_bytes=config.get("video_i_frame_size_bytes", 50000),
            p_frame_size_bytes=config.get("video_p_frame_size_bytes", 10000),
            gop_size=config.get("video_gop_size", 15),
            tti_duration_ms=tti_ms,
            qci=config.get("video_qci", 4),
            bearer_id=config.get("video_bearer_id", 0),
        )

    elif model == "voip":
        return VoIPGenerator(
            codec=config.get("voip_codec", "AMR-WB"),
            activity_factor=config.get("voip_activity_factor", 0.5),
            packet_interval_ms=config.get("voip_packet_interval_ms", 20.0),
            tti_duration_ms=tti_ms,
            qci=config.get("voip_qci", 1),
            bearer_id=config.get("voip_bearer_id", 0),
        )

    else:
        raise ValueError(f"Unknown traffic model: {model}")
