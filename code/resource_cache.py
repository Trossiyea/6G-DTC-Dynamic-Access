"""
Resource caching utilities for IEEE TMC experiment suite.

Provides thread-safe LRU caches for expensive resources:
- RadioMapCache: Caches loaded radio maps (MAT files)
- ModelCache: Caches NS-GBS scorer models
- ConfigCache: Caches loaded configuration dicts
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


class RadioMapCache:
    """
    Thread-safe LRU cache for radio maps.

    Key: (path, var_name, units)
    Value: np.ndarray (R_xyz_dbm)
    """

    _instance: Optional['RadioMapCache'] = None
    _lock = threading.Lock()

    def __init__(self, max_size: int = 4):
        self._cache: OrderedDict[Tuple[str, str, str], np.ndarray] = OrderedDict()
        self._max_size = max_size
        self._cache_lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @classmethod
    def get_instance(cls, max_size: int = 4) -> 'RadioMapCache':
        """Get singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_size)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def get_or_load(
        self,
        path: str,
        var_name: str,
        units: str,
        loader_func: Callable[[str, str, str], np.ndarray]
    ) -> np.ndarray:
        """
        Get radio map from cache or load it.

        Args:
            path: Path to MAT file
            var_name: Variable name in MAT file
            units: Units of the data ("mW", "W", "dBm")
            loader_func: Function to load the data (path, var_name, units) -> ndarray

        Returns:
            Radio map as numpy array
        """
        key = (str(path), str(var_name), str(units))

        with self._cache_lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]

        # Load outside lock to allow concurrent loads of different maps
        data = loader_func(path, var_name=var_name, units=units)

        with self._cache_lock:
            # Check again in case another thread loaded it
            if key in self._cache:
                self._hits += 1
                return self._cache[key]

            # Evict LRU if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = data
            self._misses += 1

        return data

    def clear(self) -> None:
        """Clear the cache."""
        with self._cache_lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class ModelCache:
    """
    Thread-safe cache for NS-GBS scorer models.

    Key: (model_path, device)
    Value: NSGBSScorer instance

    Also caches model metadata (nsgbs_feature_dim, nsgbs_add_z, nsgbs_add_step)
    to ensure proper feature construction on cache hits.
    """

    _instance: Optional['ModelCache'] = None
    _lock = threading.Lock()

    def __init__(self, max_size: int = 8):
        self._cache: OrderedDict[Tuple[str, str], Any] = OrderedDict()
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}  # model_path -> metadata
        self._max_size = max_size
        self._cache_lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @classmethod
    def get_instance(cls, max_size: int = 8) -> 'ModelCache':
        """Get singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_size)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def get_or_load(
        self,
        model_path: Optional[str],
        device: str,
        loader_func: Callable[[Dict], Any],
        cfg: Dict
    ) -> Optional[Any]:
        """
        Get model from cache or load it.

        Args:
            model_path: Path to model file (None returns None)
            device: Device to load model on ("cpu", "cuda", etc.)
            loader_func: Function to load the model (cfg) -> scorer
            cfg: Configuration dict (must contain nsgbs_model_path, nsgbs_device)

        Returns:
            NSGBSScorer instance or None
        """
        if model_path is None:
            return None

        key = (str(model_path), str(device))
        path_key = str(model_path)

        with self._cache_lock:
            if key in self._cache:
                # Cache hit: restore metadata to cfg
                if path_key in self._metadata_cache:
                    for k, v in self._metadata_cache[path_key].items():
                        cfg[k] = v
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]

        # Record nsgbs_ values before loading
        nsgbs_before = {k: v for k, v in cfg.items() if k.startswith("nsgbs_")}

        # Load outside lock
        scorer = loader_func(cfg)

        # Capture metadata changed by loader (new or modified nsgbs_ keys)
        metadata = {k: v for k, v in cfg.items() if k.startswith("nsgbs_") and nsgbs_before.get(k) != v}

        with self._cache_lock:
            if key in self._cache:
                # Another thread loaded it
                if path_key in self._metadata_cache:
                    for k, v in self._metadata_cache[path_key].items():
                        cfg[k] = v
                self._hits += 1
                return self._cache[key]

            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = scorer
            self._metadata_cache[path_key] = metadata
            self._misses += 1

        return scorer

    def clear(self) -> None:
        """Clear the cache."""
        with self._cache_lock:
            self._cache.clear()
            self._metadata_cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class ConfigCache:
    """
    Cache for loaded Python configuration modules.

    Uses module-level caching (not instance-based) since configs are immutable.
    """

    _configs: Dict[str, Dict] = {}
    _lock = threading.Lock()
    _base_config: Optional[Dict] = None

    @classmethod
    def get_base_config(cls, config_dir: Optional[str] = None) -> Dict:
        """
        Get base configuration (cached).

        Args:
            config_dir: Directory containing config.py (optional)

        Returns:
            Copy of base CONFIG dict
        """
        if cls._base_config is None:
            with cls._lock:
                if cls._base_config is None:
                    import importlib.util
                    if config_dir is None:
                        config_dir = Path(__file__).parent
                    config_path = Path(config_dir) / "config.py"
                    spec = importlib.util.spec_from_file_location("base_config", str(config_path))
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    cls._base_config = module.CONFIG.copy()
        return cls._base_config.copy()

    @classmethod
    def get_scenario_config(cls, path: str) -> Dict:
        """
        Get scenario configuration (cached).

        Args:
            path: Path to scenario config file

        Returns:
            Copy of scenario CONFIG dict
        """
        path_str = str(path)
        if path_str not in cls._configs:
            with cls._lock:
                if path_str not in cls._configs:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("scenario_config", path_str)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    cls._configs[path_str] = module.CONFIG.copy()
        return cls._configs[path_str].copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all cached configs."""
        with cls._lock:
            cls._configs.clear()
            cls._base_config = None


def get_optimal_workers(memory_per_run_gb: float = 2.0, min_workers: int = 1) -> int:
    """
    Determine optimal worker count based on CPU and available memory.

    Args:
        memory_per_run_gb: Estimated memory per worker (default 2.0 GB)
        min_workers: Minimum number of workers

    Returns:
        Optimal number of parallel workers
    """
    import multiprocessing as mp

    cpu_count = mp.cpu_count()

    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        system_reserve_gb = 2.0
        usable_gb = max(memory_per_run_gb, available_gb - system_reserve_gb)
        memory_limited = int(usable_gb / memory_per_run_gb)
    except ImportError:
        # psutil not available, use conservative estimate
        memory_limited = max(2, cpu_count // 2)

    optimal = min(cpu_count, memory_limited)
    return max(min_workers, optimal)


def print_cache_stats() -> None:
    """Print statistics for all caches."""
    rm_stats = RadioMapCache.get_instance().stats()
    model_stats = ModelCache.get_instance().stats()

    print("\n=== Cache Statistics ===")
    print(f"RadioMap Cache: {rm_stats['hits']} hits, {rm_stats['misses']} misses "
          f"(hit rate: {rm_stats['hit_rate']:.1%})")
    print(f"Model Cache: {model_stats['hits']} hits, {model_stats['misses']} misses "
          f"(hit rate: {model_stats['hit_rate']:.1%})")
