"""
Public API surface for the NR-NTN downlink simulator.
"""

from .config import CONFIG
from .logging_utils import configure_logging, get_logger
from .main import run_constellation, run_once
from .result_schema import SimulationResult, to_serializable_result

__all__ = [
    "CONFIG",
    "configure_logging",
    "get_logger",
    "run_constellation",
    "run_once",
    "SimulationResult",
    "to_serializable_result",
]
