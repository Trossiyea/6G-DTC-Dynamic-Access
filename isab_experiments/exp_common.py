#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for IEEE TMC experiment scripts.

Provides common functionality for parallel execution across all figure experiments.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = SCRIPT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Default model paths
DEFAULT_MLP_MODEL = "output/models/nsgbs_scorer.pt"
DEFAULT_ISAB_MODEL = "output/models/nsgbs_isab_tau0.2.pt"

# Common scenarios
SCENARIOS = {
    "toronto_single": "test/config_toronto_single.py",
    "toronto_constellation": "test/config_toronto_constellation.py",
    "shanghai_single": "test/config_shanghai_single.py",
    "shanghai_constellation": "test/config_shanghai_constellation.py",
}


def load_config_from_file(config_path: Path) -> Dict:
    """Load CONFIG dict from a Python config file."""
    spec = importlib.util.spec_from_file_location("scenario_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def load_base_config() -> Dict:
    """Load base CONFIG from code/config.py."""
    config_path = SCRIPT_DIR / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("base_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def add_parallel_args(parser):
    """Add common parallel execution arguments to an argument parser."""
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Enable parallel execution (default: True)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential execution"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)"
    )
    return parser


def run_parallel_experiments(
    run_specs: List,
    base_cfg: Dict,
    scenario_configs: Dict[str, Dict],
    model_paths: Dict[str, str],
    extra_cfg_overrides: Optional[Dict] = None,
    workers: Optional[int] = None,
    desc: str = "Exp",
    disable_progress: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run experiments in parallel using ParallelExperimentRunner.

    Args:
        run_specs: List of RunSpec objects
        base_cfg: Base configuration dict
        scenario_configs: Dict mapping scenario name to config
        model_paths: Dict with 'mlp' and 'isab' model paths
        extra_cfg_overrides: Optional additional config overrides
        workers: Number of workers (None = auto)
        desc: Progress bar description
        disable_progress: Disable progress bar

    Returns:
        List of result dicts
    """
    from parallel_runner import (
        ParallelExperimentRunner,
        ParallelConfig,
        get_optimal_workers,
    )
    from tqdm import tqdm

    workers = workers or get_optimal_workers()

    pbar = tqdm(
        total=len(run_specs),
        desc=desc,
        disable=disable_progress,
        bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    def progress_callback(completed, total):
        pbar.n = completed
        pbar.refresh()

    config = ParallelConfig(max_workers=workers)
    runner = ParallelExperimentRunner(config)

    try:
        results = runner.run_batch(
            run_specs,
            base_cfg,
            scenario_configs,
            model_paths,
            extra_cfg_overrides=extra_cfg_overrides,
            progress_callback=progress_callback,
        )
    finally:
        pbar.close()

    return results


def print_cache_stats_if_available():
    """Print cache statistics if resource_cache module is available."""
    try:
        from resource_cache import print_cache_stats
        print_cache_stats()
    except ImportError:
        pass
