"""
NTN-NR Downlink Simulation Configuration Module.

This module provides a type-safe, documented configuration system with
backward compatibility for legacy dict-style access patterns.

Usage:
------

1. Legacy dict-style (backward compatible):
    >>> from config import CONFIG
    >>> print(CONFIG["N_UE"])
    100
    >>> CONFIG["seed"] = 42

2. Type-safe with IDE autocomplete:
    >>> from config import CONFIG
    >>> print(CONFIG.simulation.N_UE)
    100
    >>> CONFIG.simulation.seed = 42

3. Loading from files:
    >>> from config import load_config, CONFIG
    >>> scenario_config = load_config("test/config_toronto_single.py", base=CONFIG)

4. Generating JSON Schema for Web UI:
    >>> from config import CONFIG, save_json_schema
    >>> save_json_schema(CONFIG, "docs/config_schema.json")

5. Creating configuration templates:
    >>> from config import generate_config_template
    >>> template = generate_config_template(format="yaml", nested=True)

Configuration Groups:
--------------------
- simulation: Core simulation parameters (N_UE, T, seed, Z)
- radio_map: Radio Map input settings
- numerology: Noise and numerology (SCS, noise temperature)
- channel: Channel model configuration
- link_budget: Link budget parameters
- geometry: Satellite geometry and beam settings
- orbit: Orbit dynamics and TLE parameters
- time_varying: Time-varying Radio Map dynamics
- csi: CSI feedback configuration
- scheduler: Scheduler parameters
- power_allocation: DL power allocation settings
- harq: HARQ/OLLA/BLER configuration
- mcs: MCS table configuration
- constellation: Multi-satellite constellation settings
- output: Output and reporting configuration
"""

import os

from .schema import (
    # Base class
    ConfigGroup,
    # Configuration groups
    SimulationConfig,
    RadioMapConfig,
    NumerologyConfig,
    ChannelConfig,
    LinkBudgetConfig,
    GeometryConfig,
    OrbitConfig,
    TimeVaryingConfig,
    CSIConfig,
    SchedulerConfig,
    PowerAllocationConfig,
    HarqConfig,
    MCSConfig,
    ConstellationConfig,
    OutputConfig,
    # Main config class
    NTNSimConfig,
)

from .compat import (
    ConfigDict,
    create_config,
    merge_configs,
)

from .loader import (
    # File loading
    load_config,
    load_json,
    load_yaml,
    load_python_config,
    # File saving
    save_json,
    save_json_nested,
    save_json_schema,
    save_yaml,
    save_yaml_nested,
    # Path resolution
    ConfigResolver,
    resolve_config_paths,
    # Template generation
    generate_config_template,
)


# =============================================================================
# Default Configuration Instance
# =============================================================================

# Base directory for relative path resolution (project root: DL/)
# __file__ is config/__init__.py, so we go up two levels: config/ -> code/ -> DL/
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_DIR = os.path.dirname(_CODE_DIR)  # DL/

# Create the default CONFIG instance
CONFIG = ConfigDict()

# Set the MCS table path relative to the code directory
CONFIG["mcs_3gpp_table_path"] = os.path.join(_BASE_DIR, "docs", "mcs_tables_38_214.json")

# Set the default TLE catalog path for constellation mode
CONFIG["tle_catalog_path"] = os.path.join(_BASE_DIR, "tles", "Satnet_DTC.txt")


# =============================================================================
# Convenience functions
# =============================================================================

def get_default_config() -> ConfigDict:
    """
    Get a fresh copy of the default configuration.

    Returns:
        New ConfigDict instance with default values
    """
    cfg = ConfigDict()
    cfg["mcs_3gpp_table_path"] = os.path.join(_BASE_DIR, "docs", "mcs_tables_38_214.json")
    cfg["tle_catalog_path"] = os.path.join(_BASE_DIR, "tles", "Satnet_DTC.txt")
    return cfg


def load_scenario_config(scenario_path: str) -> ConfigDict:
    """
    Load a scenario configuration, merging with defaults.

    Args:
        scenario_path: Path to scenario config file (relative to project root)

    Returns:
        ConfigDict with merged configuration
    """
    base = get_default_config()
    return load_config(scenario_path, base=base)


__all__ = [
    # Main config instance
    "CONFIG",
    # Convenience functions
    "get_default_config",
    "load_scenario_config",
    # Schema classes
    "ConfigGroup",
    "SimulationConfig",
    "RadioMapConfig",
    "NumerologyConfig",
    "ChannelConfig",
    "LinkBudgetConfig",
    "GeometryConfig",
    "OrbitConfig",
    "TimeVaryingConfig",
    "CSIConfig",
    "SchedulerConfig",
    "PowerAllocationConfig",
    "HarqConfig",
    "MCSConfig",
    "ConstellationConfig",
    "OutputConfig",
    "NTNSimConfig",
    # Compatibility
    "ConfigDict",
    "create_config",
    "merge_configs",
    # Loaders
    "load_config",
    "load_json",
    "load_yaml",
    "load_python_config",
    # Savers
    "save_json",
    "save_json_nested",
    "save_json_schema",
    "save_yaml",
    "save_yaml_nested",
    # Path resolution
    "ConfigResolver",
    "resolve_config_paths",
    # Template
    "generate_config_template",
]
