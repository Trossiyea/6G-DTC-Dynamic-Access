# -*- coding: utf-8 -*-
"""
Radio Map loading utilities.

Supports loading Radio Map data from various formats:
- MATLAB .mat files (traditional and HDF5/v7.3)
- NumPy .npy/.npz files
- In-memory numpy arrays

The Radio Map represents spatial interference/signal patterns as a 3D tensor
R[x, y, z] where x, y are spatial coordinates and z is frequency (PRB index).
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

# Optional imports for MAT file support
try:
    from scipy.io import loadmat
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    loadmat = None

try:
    import h5py
    _H5PY_OK = True
except ImportError:
    _H5PY_OK = False
    h5py = None


def _convert_units_to_dbm(data: np.ndarray, units: str) -> np.ndarray:
    """
    Convert Radio Map data to dBm units.

    Args:
        data: Input data array
        units: Source units ("dBm", "mW", or "W")

    Returns:
        Data in dBm

    Raises:
        ValueError: If units is not recognized
    """
    units_lower = units.lower()
    if units_lower == "dbm":
        return data.astype(float)
    elif units_lower == "mw":
        return 10.0 * np.log10(np.maximum(data.astype(float), 1e-30))
    elif units_lower == "w":
        return 10.0 * np.log10(np.maximum(data.astype(float) * 1e3, 1e-30))
    else:
        raise ValueError(f"Unsupported units: {units} (use 'mW', 'W', or 'dBm')")


def _load_mat_hdf5(path: str, var_name: str) -> np.ndarray:
    """Load data from HDF5-based MAT file (MATLAB v7.3)."""
    if not _H5PY_OK:
        raise ImportError("h5py is required to load HDF5 MAT files")

    with h5py.File(path, 'r') as f:
        if var_name in f:
            return np.array(f[var_name], dtype=float)
        # Find first dataset if var_name not found
        keys = list(f.keys())
        if not keys:
            raise KeyError(f"No data variables found in {path}")
        pick_key = keys[0]
        print(f"[load_radio_map] '{var_name}' not found in HDF5. Using '{pick_key}' instead.")
        return np.array(f[pick_key], dtype=float)


def _load_mat_scipy(path: str, var_name: str) -> np.ndarray:
    """Load data from traditional MAT file using scipy."""
    if not _SCIPY_OK:
        raise ImportError("scipy is required to load traditional MAT files")

    data = loadmat(path)
    # Filter out meta keys
    keys = [k for k in data.keys() if not k.startswith("__")]
    pick_key = var_name if var_name in data else (keys[0] if keys else None)

    if pick_key is None:
        raise KeyError(f"No data variables found in {path}. Raw keys: {list(data.keys())}")

    if var_name not in data:
        print(f"[load_radio_map] '{var_name}' not found. Using '{pick_key}' instead.")

    return np.array(data[pick_key], dtype=float)


def load_radio_map_from_mat(
    path: str,
    var_name: str = "X_true",
    units: str = "mW",
) -> np.ndarray:
    """
    Load a Radio Map from a MATLAB .mat file and return dBm tensor R[x,y,z].

    Supports both traditional MAT files and HDF5-based MAT files (v7.3).

    Args:
        path: Path to the .mat file
        var_name: Variable name inside MAT file (default: "X_true")
        units: Units of the data in file ("mW", "W", or "dBm")

    Returns:
        3D numpy array of shape [X, Y, Z] in dBm

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be loaded or units are invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Radio Map file not found: {path}")

    # Try HDF5 format first (MATLAB v7.3)
    try:
        X = _load_mat_hdf5(path, var_name)
    except (OSError, ImportError, Exception):
        # Fall back to traditional MAT format
        try:
            X = _load_mat_scipy(path, var_name)
        except Exception as e:
            raise ValueError(f"Failed to load MAT file {path}: {e}")

    return _convert_units_to_dbm(X, units)


def load_radio_map_from_npy(
    path: str,
    units: str = "dBm",
) -> np.ndarray:
    """
    Load a Radio Map from a NumPy .npy or .npz file.

    Args:
        path: Path to the .npy or .npz file
        units: Units of the data ("mW", "W", or "dBm")

    Returns:
        3D numpy array in dBm

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Radio Map file not found: {path}")

    if path.endswith('.npz'):
        data = np.load(path)
        # Get first array in npz
        keys = list(data.keys())
        X = data[keys[0]]
    else:
        X = np.load(path)

    return _convert_units_to_dbm(X, units)


def load_radiomap(
    source: Union[str, Path, np.ndarray],
    var_name: str = "X_true",
    units: str = "dBm",
) -> np.ndarray:
    """
    Unified Radio Map loading interface.

    Automatically detects the source type and loads accordingly:
    - File path (MAT, NPY, NPZ): Load from file
    - numpy array: Use directly (apply unit conversion)

    This is the preferred function for loading Radio Maps as it provides
    a consistent interface for different data sources.

    Args:
        source: File path (str/Path) or numpy array
        var_name: Variable name for MAT files (ignored for other types)
        units: Units of the source data ("mW", "W", or "dBm")

    Returns:
        3D numpy array of shape [X, Y, Z] in dBm

    Raises:
        FileNotFoundError: If file path does not exist
        ValueError: If data cannot be loaded or format is unsupported
    """
    # Handle numpy array input
    if isinstance(source, np.ndarray):
        return _convert_units_to_dbm(source, units)

    # Handle file path
    path = Path(source) if isinstance(source, str) else source
    path_str = str(path)

    if not path.exists():
        raise FileNotFoundError(f"Radio Map file not found: {path_str}")

    suffix = path.suffix.lower()

    if suffix == ".mat":
        return load_radio_map_from_mat(path_str, var_name, units)
    elif suffix in (".npy", ".npz"):
        return load_radio_map_from_npy(path_str, units)
    else:
        raise ValueError(f"Unsupported Radio Map file format: {suffix}")


def select_radio_map(config: Dict) -> Tuple[np.ndarray, int, int, int]:
    """
    Load Radio Map from config and validate dimensions.

    This is a convenience function that reads Radio Map configuration
    from a config dictionary and validates that the Z dimension matches
    the expected PRB count.

    Args:
        config: Configuration dictionary with keys:
            - radio_map_mat_path: Path to Radio Map file
            - radio_map_mat_var: Variable name (default: "X_true")
            - radio_map_units: Units (default: "mW")
            - Z: Expected PRB count (optional, for validation)

    Returns:
        Tuple of (R_xyz_dbm, X, Y, Z):
            - R_xyz_dbm: 3D Radio Map array in dBm
            - X, Y: Spatial dimensions
            - Z: Frequency dimension (PRB count)

    Raises:
        ValueError: If configuration is invalid or dimensions don't match
    """
    expected_Z = int(config["Z"]) if ("Z" in config and config["Z"] is not None) else None
    path = config.get("radio_map_mat_path")

    if not path:
        raise ValueError("radio_map_mat_path must be provided (MAT file with 3D Radio Map)")

    R_xyz_dbm = load_radiomap(
        path,
        var_name=config.get("radio_map_mat_var", "X_true"),
        units=config.get("radio_map_units", "mW"),
    )

    if R_xyz_dbm.ndim != 3:
        raise ValueError(f"Loaded Radio Map must be 3D, got shape {R_xyz_dbm.shape}")

    X, Y, Z = R_xyz_dbm.shape

    # Verify Z matches expected PRB count
    if expected_Z is not None and Z != expected_Z:
        raise ValueError(
            f"Radio Map Z dimension ({Z}) does not match configured PRB count ({expected_Z}). "
            f"Please provide a map with Z={expected_Z} or update config['Z'] to {Z}."
        )

    return R_xyz_dbm, X, Y, Z
