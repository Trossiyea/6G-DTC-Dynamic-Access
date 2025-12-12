# -*- coding: utf-8 -*-
"""
Callback System for Simulation Progress Reporting.

Provides hooks for:
- Real-time progress updates (Web integration)
- Streaming TTI-level metrics (WebSocket)
- Phase transition notifications
- Custom logging/monitoring
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
from enum import Enum


class SimulationPhase(Enum):
    """Simulation lifecycle phases."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class TTIMetrics:
    """Metrics reported after each TTI.

    Attributes:
        tti: Current TTI index
        progress: Progress fraction (0.0 to 1.0)
        se_baseline: Current baseline SE (bits/s/Hz)
        se_radiomap: Current RadioMap SE (bits/s/Hz)
        cumulative_se_baseline: Cumulative average baseline SE
        cumulative_se_radiomap: Cumulative average RadioMap SE
        active_ues: Number of active UEs this TTI
        served_prbs: Number of PRBs allocated this TTI
        extra: Additional metrics dict
    """
    tti: int
    progress: float
    se_baseline: float = 0.0
    se_radiomap: float = 0.0
    cumulative_se_baseline: float = 0.0
    cumulative_se_radiomap: float = 0.0
    active_ues: int = 0
    served_prbs: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "tti": self.tti,
            "progress": round(self.progress, 4),
            "se_baseline": round(self.se_baseline, 4),
            "se_radiomap": round(self.se_radiomap, 4),
            "cumulative_se_baseline": round(self.cumulative_se_baseline, 4),
            "cumulative_se_radiomap": round(self.cumulative_se_radiomap, 4),
            "active_ues": self.active_ues,
            "served_prbs": self.served_prbs,
            **self.extra,
        }


@dataclass
class PhaseEvent:
    """Event emitted on phase transitions.

    Attributes:
        phase: New phase
        previous_phase: Previous phase
        message: Human-readable message
        data: Additional context data
    """
    phase: SimulationPhase
    previous_phase: Optional[SimulationPhase] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "phase": self.phase.value,
            "previous_phase": self.previous_phase.value if self.previous_phase else None,
            "message": self.message,
            "data": self.data,
        }


class ProgressCallback(ABC):
    """Abstract base class for progress callbacks.

    Implement this to receive simulation progress updates.

    Example:
        class WebSocketCallback(ProgressCallback):
            def __init__(self, websocket):
                self.ws = websocket

            async def on_tti_complete(self, metrics: TTIMetrics):
                await self.ws.send_json(metrics.to_dict())

            async def on_phase_change(self, event: PhaseEvent):
                await self.ws.send_json({"type": "phase", **event.to_dict()})
    """

    @abstractmethod
    def on_tti_complete(self, metrics: TTIMetrics) -> None:
        """Called after each TTI completes.

        Args:
            metrics: TTI-level metrics
        """
        pass

    def on_phase_change(self, event: PhaseEvent) -> None:
        """Called when simulation phase changes.

        Args:
            event: Phase transition event

        Override this for phase-aware callbacks.
        """
        pass

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Called when an error occurs.

        Args:
            error: The exception that occurred
            context: Additional context (tti, phase, etc.)

        Override this for error handling.
        """
        pass


class FunctionCallback(ProgressCallback):
    """Callback wrapper for simple functions.

    Example:
        callback = FunctionCallback(
            on_tti=lambda m: print(f"TTI {m.tti}: {m.progress*100:.1f}%")
        )
        engine.add_callback(callback)
    """

    def __init__(
        self,
        on_tti: Optional[Callable[[TTIMetrics], None]] = None,
        on_phase: Optional[Callable[[PhaseEvent], None]] = None,
        on_error: Optional[Callable[[Exception, Dict], None]] = None,
    ):
        self._on_tti = on_tti
        self._on_phase = on_phase
        self._on_error_fn = on_error

    def on_tti_complete(self, metrics: TTIMetrics) -> None:
        if self._on_tti:
            self._on_tti(metrics)

    def on_phase_change(self, event: PhaseEvent) -> None:
        if self._on_phase:
            self._on_phase(event)

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        if self._on_error_fn:
            self._on_error_fn(error, context)


class CallbackManager:
    """Manages multiple callbacks for a simulation engine.

    Thread-safe callback invocation with error isolation.

    Example:
        manager = CallbackManager()
        manager.add(FunctionCallback(on_tti=print_progress))
        manager.add(WebSocketCallback(ws))

        # In engine:
        manager.notify_tti(metrics)
        manager.notify_phase(event)
    """

    def __init__(self):
        self._callbacks: List[ProgressCallback] = []
        self._error_handler: Optional[Callable[[Exception, str], None]] = None

    def add(self, callback: ProgressCallback) -> "CallbackManager":
        """Add a callback. Returns self for chaining."""
        self._callbacks.append(callback)
        return self

    def remove(self, callback: ProgressCallback) -> bool:
        """Remove a callback. Returns True if found."""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        """Remove all callbacks."""
        self._callbacks.clear()

    def set_error_handler(self, handler: Callable[[Exception, str], None]) -> None:
        """Set global error handler for callback failures."""
        self._error_handler = handler

    def notify_tti(self, metrics: TTIMetrics) -> None:
        """Notify all callbacks of TTI completion."""
        for cb in self._callbacks:
            try:
                cb.on_tti_complete(metrics)
            except Exception as e:
                self._handle_callback_error(e, f"on_tti_complete(tti={metrics.tti})")

    def notify_phase(self, event: PhaseEvent) -> None:
        """Notify all callbacks of phase change."""
        for cb in self._callbacks:
            try:
                cb.on_phase_change(event)
            except Exception as e:
                self._handle_callback_error(e, f"on_phase_change({event.phase.value})")

    def notify_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Notify all callbacks of simulation error."""
        for cb in self._callbacks:
            try:
                cb.on_error(error, context)
            except Exception as e:
                self._handle_callback_error(e, "on_error")

    def _handle_callback_error(self, error: Exception, context: str) -> None:
        """Handle errors in callback execution."""
        if self._error_handler:
            try:
                self._error_handler(error, context)
            except Exception:
                pass  # Avoid infinite recursion


class ProgressAggregator(ProgressCallback):
    """Aggregates metrics across TTIs for summary statistics.

    Useful for batch runs or when you need post-hoc analysis.

    Example:
        aggregator = ProgressAggregator()
        engine.add_callback(aggregator)
        result = engine.run()
        print(f"Avg SE over time: {aggregator.get_summary()}")
    """

    def __init__(self):
        self._metrics: List[TTIMetrics] = []
        self._phases: List[PhaseEvent] = []
        self._errors: List[tuple] = []

    def on_tti_complete(self, metrics: TTIMetrics) -> None:
        self._metrics.append(metrics)

    def on_phase_change(self, event: PhaseEvent) -> None:
        self._phases.append(event)

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        self._errors.append((error, context))

    def get_metrics(self) -> List[TTIMetrics]:
        """Get all recorded TTI metrics."""
        return self._metrics.copy()

    def get_phases(self) -> List[PhaseEvent]:
        """Get all phase transition events."""
        return self._phases.copy()

    def get_errors(self) -> List[tuple]:
        """Get all recorded errors."""
        return self._errors.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._metrics:
            return {"count": 0}

        import numpy as np
        se_base = [m.se_baseline for m in self._metrics]
        se_rm = [m.se_radiomap for m in self._metrics]

        return {
            "count": len(self._metrics),
            "se_baseline_mean": float(np.mean(se_base)),
            "se_baseline_std": float(np.std(se_base)),
            "se_radiomap_mean": float(np.mean(se_rm)),
            "se_radiomap_std": float(np.std(se_rm)),
            "final_progress": self._metrics[-1].progress if self._metrics else 0.0,
            "error_count": len(self._errors),
        }

    def clear(self) -> None:
        """Clear all recorded data."""
        self._metrics.clear()
        self._phases.clear()
        self._errors.clear()
