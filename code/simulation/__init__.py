# -*- coding: utf-8 -*-
"""
Simulation Engine Package for NR-NTN Downlink.

This package provides a modular simulation engine with:
- SimulationState: Serializable state container for simulation data
- SimulationEngine: Single-satellite simulation orchestrator
- ConstellationEngine: Multi-satellite constellation simulation
- Callbacks: Progress reporting and Web integration hooks
- TrafficSimulator: Post-processing traffic layer simulation

Usage:
------
1. Basic usage (backward compatible with run_once):
    >>> from simulation import SimulationEngine
    >>> engine = SimulationEngine(config)
    >>> result = engine.run()

2. With progress callbacks (for Web integration):
    >>> from simulation import SimulationEngine, ProgressCallback
    >>> engine = SimulationEngine(config)
    >>> engine.on_progress(lambda tti, metrics: print(f"TTI {tti}: SE={metrics['se']:.3f}"))
    >>> result = engine.run()

3. Access intermediate state:
    >>> engine = SimulationEngine(config)
    >>> engine.initialize()
    >>> state = engine.state  # Access SimulationState
    >>> result = engine.run()

4. With traffic simulation (latency/goodput KPIs):
    >>> from simulation import SimulationEngine, simulate_traffic_layer
    >>> engine = SimulationEngine(config)
    >>> result = engine.run()
    >>> result_with_traffic = simulate_traffic_layer(config, result)
"""

from .state import (
    SimulationState,
    ConstellationState,
    GeometryResult,
    SchedulerResult,
)
from .callbacks import (
    ProgressCallback,
    TTIMetrics,
    CallbackManager,
)
from .helpers import (
    generate_ue_positions,
    resolve_noise_and_prb_bw,
    apply_open_loop_power_control,
    compute_metric_override_static_if_needed,
    build_time_variation_if_enabled,
    compute_caps,
)
from .engine import SimulationEngine
from .constellation_engine import ConstellationEngine
from .traffic_simulator import TrafficSimulator, simulate_traffic_layer

__all__ = [
    # State classes
    "SimulationState",
    "ConstellationState",
    "GeometryResult",
    "SchedulerResult",
    # Callback system
    "ProgressCallback",
    "TTIMetrics",
    "CallbackManager",
    # Helper functions
    "generate_ue_positions",
    "resolve_noise_and_prb_bw",
    "apply_open_loop_power_control",
    "compute_metric_override_static_if_needed",
    "build_time_variation_if_enabled",
    "compute_caps",
    # Engine classes
    "SimulationEngine",
    "ConstellationEngine",
    # Traffic simulation
    "TrafficSimulator",
    "simulate_traffic_layer",
]
