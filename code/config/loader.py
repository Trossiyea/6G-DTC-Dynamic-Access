"""
Configuration loaders for various file formats.

Supports loading configuration from:
- JSON files
- YAML files (if PyYAML is installed)
- Python files (for legacy compatibility)
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .compat import ConfigDict, merge_configs
from .schema import NTNSimConfig


# =============================================================================
# JSON Loader
# =============================================================================

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary with configuration values

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(config: ConfigDict, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save configuration to a JSON file.

    Args:
        config: ConfigDict instance
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_flat_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_json_nested(config: ConfigDict, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save configuration to a JSON file in nested format.

    Args:
        config: ConfigDict instance
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_nested_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_json_schema(config: ConfigDict, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save JSON Schema to a file.

    Args:
        config: ConfigDict instance
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema = config.to_json_schema()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=indent, ensure_ascii=False)


# =============================================================================
# YAML Loader (optional dependency)
# =============================================================================

def _yaml_available() -> bool:
    """Check if PyYAML is available."""
    try:
        import yaml
        return True
    except ImportError:
        return False


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with configuration values

    Raises:
        ImportError: If PyYAML is not installed
        FileNotFoundError: If file doesn't exist
    """
    if not _yaml_available():
        raise ImportError(
            "PyYAML is required to load YAML files. "
            "Install it with: pip install pyyaml"
        )

    import yaml
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(config: ConfigDict, path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: ConfigDict instance
        path: Output file path

    Raises:
        ImportError: If PyYAML is not installed
    """
    if not _yaml_available():
        raise ImportError(
            "PyYAML is required to save YAML files. "
            "Install it with: pip install pyyaml"
        )

    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_flat_dict()
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def save_yaml_nested(config: ConfigDict, path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file in nested format.

    Args:
        config: ConfigDict instance
        path: Output file path
    """
    if not _yaml_available():
        raise ImportError(
            "PyYAML is required to save YAML files. "
            "Install it with: pip install pyyaml"
        )

    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_nested_dict()
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# =============================================================================
# Python File Loader (legacy compatibility)
# =============================================================================

def load_python_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a Python file.

    The Python file should define a CONFIG dictionary at module level.
    This is used for compatibility with existing test/config_*.py files.

    Args:
        path: Path to Python file

    Returns:
        CONFIG dictionary from the file

    Raises:
        FileNotFoundError: If file doesn't exist
        AttributeError: If file doesn't define CONFIG
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"Config file does not define CONFIG: {path}")

    return module.CONFIG


# =============================================================================
# Unified Config Loader
# =============================================================================

def load_config(
    path: Union[str, Path],
    base: Optional[ConfigDict] = None,
    format: Optional[str] = None
) -> ConfigDict:
    """
    Load configuration from file, auto-detecting format.

    Supported formats:
    - .json: JSON format
    - .yaml, .yml: YAML format (requires PyYAML)
    - .py: Python file with CONFIG dict

    Args:
        path: Path to configuration file
        base: Optional base ConfigDict to merge with
        format: Optional format override ("json", "yaml", "python")

    Returns:
        ConfigDict with loaded configuration

    Example:
        >>> config = load_config("test/config_toronto_single.py")
        >>> config = load_config("config.json", base=default_config)
    """
    path = Path(path)

    # Detect format
    if format is None:
        suffix = path.suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix in (".yaml", ".yml"):
            format = "yaml"
        elif suffix == ".py":
            format = "python"
        else:
            raise ValueError(f"Unknown config file format: {suffix}")

    # Load data
    if format == "json":
        data = load_json(path)
    elif format == "yaml":
        data = load_yaml(path)
    elif format == "python":
        data = load_python_config(path)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Handle nested vs flat format
    if _is_nested_format(data):
        data = _flatten_nested(data)

    # Create or merge config
    if base is not None:
        return merge_configs(base, data)
    else:
        return ConfigDict(data=data)


def _is_nested_format(data: Dict[str, Any]) -> bool:
    """Check if data is in nested format (has group keys)."""
    group_keys = {
        "simulation", "radio_map", "numerology", "channel",
        "link_budget", "geometry", "orbit", "time_varying",
        "csi", "scheduler", "power_allocation", "harq",
        "mcs", "constellation", "output"
    }
    return any(k in data and isinstance(data[k], dict) for k in group_keys)


def _flatten_nested(data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested format to flat format."""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict) and key != "channel_params":
            # This is a group - flatten its contents
            result.update(value)
        else:
            # This is a direct key
            result[key] = value
    return result


# =============================================================================
# Config Resolver (handles relative paths)
# =============================================================================

class ConfigResolver:
    """
    Resolves relative paths in configuration based on a base directory.

    This is useful for ensuring Radio Map paths and TLE paths work correctly
    regardless of where the script is run from.
    """

    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize resolver with base directory.

        Args:
            base_dir: Base directory for resolving relative paths
        """
        self.base_dir = Path(base_dir).absolute()

    def resolve_paths(self, config: ConfigDict) -> ConfigDict:
        """
        Resolve all relative paths in configuration.

        Paths that are resolved:
        - radio_map_mat_path
        - tle_path
        - tle_catalog_path
        - mcs_3gpp_table_path
        - bler_curve_path
        - plot_dir

        Args:
            config: ConfigDict with potentially relative paths

        Returns:
            ConfigDict with resolved absolute paths
        """
        path_keys = [
            "radio_map_mat_path",
            "tle_path",
            "tle_catalog_path",
            "mcs_3gpp_table_path",
            "bler_curve_path",
        ]

        for key in path_keys:
            value = config.get(key)
            if value and not os.path.isabs(value):
                resolved = self.base_dir / value
                if resolved.exists():
                    config[key] = str(resolved)

        return config


def resolve_config_paths(config: ConfigDict, base_dir: Union[str, Path]) -> ConfigDict:
    """
    Convenience function to resolve paths in configuration.

    Args:
        config: ConfigDict instance
        base_dir: Base directory for path resolution

    Returns:
        ConfigDict with resolved paths
    """
    resolver = ConfigResolver(base_dir)
    return resolver.resolve_paths(config)


# =============================================================================
# Template Generation
# =============================================================================

def generate_config_template(
    format: str = "json",
    path: Optional[Union[str, Path]] = None,
    nested: bool = False
) -> Optional[str]:
    """
    Generate a configuration template file.

    Args:
        format: Output format ("json" or "yaml")
        path: Optional output file path (if None, returns string)
        nested: If True, use nested format with groups

    Returns:
        Template string if path is None, otherwise None
    """
    config = ConfigDict()

    if nested:
        data = config.to_nested_dict()
    else:
        data = config.to_flat_dict()

    if format == "json":
        content = json.dumps(data, indent=2, ensure_ascii=False)
    elif format == "yaml":
        if not _yaml_available():
            raise ImportError("PyYAML required for YAML template")
        import yaml
        content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError(f"Unknown format: {format}")

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return None
    else:
        return content
