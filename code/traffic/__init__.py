"""
Traffic modeling subsystem for NTN RAN simulator.

This module provides:
- Packet and buffer data structures
- Multiple 3GPP-compliant traffic generators
- QoS class definitions and management
- Traffic statistics tracking and KPI computation

Public API (~25 exports):

Data Structures:
    - Packet: Single packet representation
    - UEBuffer: Per-UE multi-bearer buffer
    - BufferManager: Centralized buffer management
    - QoSClass: 5QI class definition
    - TrafficStatisticsTracker: KPI collection

Traffic Generators:
    - TrafficGenerator: Abstract base class
    - FullBufferGenerator: Legacy infinite backlog model
    - PoissonGenerator: Poisson arrival process
    - FTPModel3Generator: 3GPP FTP Model 3
    - VideoStreamingGenerator: Video streaming with GOP
    - VoIPGenerator: VoIP with ON/OFF activity
    - create_traffic_generator: Factory function

QoS Management:
    - QoSManager: Standard QoS management
    - NTNQoSManager: NTN-aware QoS with delay compensation
    - STANDARD_5QI: 5QI definitions dictionary

Usage:
    from traffic import BufferManager, create_traffic_generator, QoSManager

    # Create components
    buffer_mgr = BufferManager(n_ue=100, config=config)
    traffic_gen = create_traffic_generator(config)
    qos_mgr = QoSManager(config)

    # Per-TTI loop
    for tti in range(T):
        # Generate arrivals
        for ue in range(N_UE):
            arrivals = traffic_gen.generate_arrivals(tti, ue, rng)
            for pkt in arrivals:
                buffer_mgr.enqueue(pkt)

        # ... scheduling ...

        # Dequeue scheduled bits
        completed = buffer_mgr.dequeue_bits(ue, bits, tti)
"""

from .packets import Packet
from .buffer import UEBuffer, BufferManager
from .qos import QoSClass, QoSManager, NTNQoSManager, STANDARD_5QI
from .models import (
    TrafficGenerator,
    FullBufferGenerator,
    PoissonGenerator,
    FTPModel3Generator,
    VideoStreamingGenerator,
    VoIPGenerator,
    create_traffic_generator,
)
from .statistics import TrafficStatisticsTracker

__all__ = [
    # Data structures
    "Packet",
    "UEBuffer",
    "BufferManager",
    "QoSClass",
    "TrafficStatisticsTracker",
    # Traffic generators
    "TrafficGenerator",
    "FullBufferGenerator",
    "PoissonGenerator",
    "FTPModel3Generator",
    "VideoStreamingGenerator",
    "VoIPGenerator",
    "create_traffic_generator",
    # QoS management
    "QoSManager",
    "NTNQoSManager",
    "STANDARD_5QI",
]

__version__ = "1.0.0"
