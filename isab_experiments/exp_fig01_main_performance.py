#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 1: Main Performance Across Scenarios.

Goal: Establish headline gains and consistency across different scenarios.

Compares:
- B1_3GPP: 3GPP-like CSI/CQI scheduler (practical baseline)
- B3_heuristic: RadioMap heuristic (no learning)
- P1_MLP: RadioMap + NS-GBS (MLP)
- P2_ISAB: RadioMap + NS-GBS (ISAB)

Outputs CSV with columns:
  scenario, method, seed, avg_se, avg_se_baseline, gain_pct

Supports parallel execution via --parallel flag (default: enabled).
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = SCRIPT_DIR / "code"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CODE_DIR))

# Scenario configurations
SCENARIOS = {
    "toronto_single": "test/config_toronto_single.py",
    "toronto_constellation": "test/config_toronto_constellation.py",
    "shanghai_single": "test/config_shanghai_single.py",
    "shanghai_constellation": "test/config_shanghai_constellation.py",
}

# Default model paths
DEFAULT_MLP_MODEL = "output/models/nsgbs_scorer.pt"
DEFAULT_ISAB_MODEL = "output/models/nsgbs_isab_tau0.2.pt"


def load_config_from_file(config_path: Path):
    """Load CONFIG dict from a Python config file."""
    spec = importlib.util.spec_from_file_location("scenario_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def load_base_config():
    """Load base CONFIG from code/config.py."""
    config_path = SCRIPT_DIR / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("base_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def run_experiment_sequential(cfg, method, model_mlp, model_isab):
    """
    Configure and run a single experiment (sequential mode).

    Args:
        cfg: Base config dict (will be modified)
        method: One of 'B1_3GPP', 'B3_heuristic', 'P1_MLP', 'P2_ISAB'
        model_mlp: Path to MLP model
        model_isab: Path to ISAB model

    Returns:
        Dict with experiment results
    """
    from main import run_once, run_constellation

    if method == "B1_3GPP":
        cfg["scheduler_kind"] = "heuristic"
        cfg["nsgbs_model_path"] = None
    elif method == "B3_heuristic":
        cfg["scheduler_kind"] = "heuristic"
        cfg["nsgbs_model_path"] = None
    elif method == "P1_MLP":
        cfg["scheduler_kind"] = "nsgbs"
        cfg["nsgbs_model_path"] = str(model_mlp)
    elif method == "P2_ISAB":
        cfg["scheduler_kind"] = "nsgbs"
        cfg["nsgbs_model_path"] = str(model_isab)
    else:
        raise ValueError(f"Unknown method: {method}")

    if cfg.get("enable_constellation", False):
        result = run_constellation(cfg)
    else:
        result = run_once(cfg)

    is_baseline = method == "B1_3GPP"
    return {
        "avg_se": float(result.get("avg_se_baseline_default", 0.0)) if is_baseline else float(result.get("avg_se_radiomap", 0.0)),
        "avg_se_baseline": float(result.get("avg_se_baseline_default", 0.0)),
        "gain_pct": 0.0 if is_baseline else float(result.get("improvement_vs_default_pct", 0.0)),
    }


def process_parallel_results(results):
    """Convert parallel runner results to CSV rows."""
    rows = []
    for r in results:
        if r.get("success", False):
            res = r.get("result", {})
            is_baseline = r["method"] == "B1_3GPP"
            rows.append({
                "scenario": r["scenario"],
                "method": r["method"],
                "seed": r["seed"],
                "avg_se": float(res.get("avg_se_baseline_default", 0.0)) if is_baseline else float(res.get("avg_se_radiomap", 0.0)),
                "avg_se_baseline": float(res.get("avg_se_baseline_default", 0.0)),
                "gain_pct": 0.0 if is_baseline else float(res.get("improvement_vs_default_pct", 0.0)),
            })
        else:
            rows.append({
                "scenario": r["scenario"],
                "method": r["method"],
                "seed": r["seed"],
                "avg_se": float('nan'),
                "avg_se_baseline": float('nan'),
                "gain_pct": float('nan'),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Figure 1: Main Performance Across Scenarios"
    )
    parser.add_argument(
        "-s", "--scenario",
        action="append",
        choices=list(SCENARIOS.keys()),
        help="Scenario(s) to run. Can be specified multiple times."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scenarios"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds per scenario/method (default: 5)"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Starting seed (default: 1)"
    )
    parser.add_argument(
        "--T",
        type=int,
        default=None,
        help="Override number of TTIs (default: use config)"
    )
    parser.add_argument(
        "--N_UE",
        type=int,
        default=None,
        help="Override number of UEs (default: use config)"
    )
    parser.add_argument(
        "--methods",
        default="B1_3GPP,B3_heuristic,P1_MLP,P2_ISAB",
        help="Comma-separated list of methods to compare"
    )
    parser.add_argument(
        "--model-mlp",
        default=DEFAULT_MLP_MODEL,
        help=f"Path to MLP model (default: {DEFAULT_MLP_MODEL})"
    )
    parser.add_argument(
        "--model-isab",
        default=DEFAULT_ISAB_MODEL,
        help=f"Path to ISAB model (default: {DEFAULT_ISAB_MODEL})"
    )
    parser.add_argument(
        "--out",
        default="output/results/fig01_main_performance.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    # Parallel execution arguments
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
    args = parser.parse_args()

    # Determine scenarios to run
    if args.all:
        scenarios = list(SCENARIOS.keys())
    elif args.scenario:
        scenarios = args.scenario
    else:
        parser.print_help()
        print("\nError: Specify --scenario or --all")
        return 1

    # Parse methods
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Validate model paths
    model_mlp = SCRIPT_DIR / args.model_mlp
    model_isab = SCRIPT_DIR / args.model_isab

    if "P1_MLP" in methods and not model_mlp.exists():
        print(f"Error: MLP model not found: {model_mlp}")
        return 1
    if "P2_ISAB" in methods and not model_isab.exists():
        print(f"Error: ISAB model not found: {model_isab}")
        return 1

    # Load base config
    base_cfg = load_base_config()

    # Build extra config overrides
    extra_cfg = {}
    if args.T is not None:
        extra_cfg["T"] = args.T
    if args.N_UE is not None:
        extra_cfg["N_UE"] = args.N_UE

    # Calculate total runs
    seed_range = range(args.seed_start, args.seed_start + args.seeds)
    total_runs = len(scenarios) * len(methods) * args.seeds

    print(f"\nFigure 1: Main Performance Experiment")
    print(f"  Scenarios: {scenarios}")
    print(f"  Methods: {methods}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total runs: {total_runs}")

    use_parallel = args.parallel and not args.sequential

    if use_parallel:
        # Parallel execution
        from parallel_runner import (
            ParallelExperimentRunner,
            ParallelConfig,
            RunSpec,
            get_optimal_workers,
        )

        workers = args.workers or get_optimal_workers()
        print(f"  Mode: PARALLEL ({workers} workers)\n")

        # Load scenario configs
        scenario_configs = {
            name: load_config_from_file(SCRIPT_DIR / path)
            for name, path in SCENARIOS.items()
            if name in scenarios
        }

        # Build run specs
        run_specs = []
        for scenario in scenarios:
            for method in methods:
                for seed in seed_range:
                    run_specs.append(RunSpec(
                        scenario=scenario,
                        method=method,
                        seed=seed,
                        extra_params=extra_cfg.copy(),
                        run_id=f"fig01_{scenario}_{method}_{seed}",
                    ))

        # Setup progress bar
        pbar = tqdm(
            total=total_runs,
            desc="Fig01",
            disable=args.no_progress,
            bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )

        def progress_callback(completed, total):
            pbar.n = completed
            pbar.refresh()

        # Run parallel
        config = ParallelConfig(max_workers=workers)
        runner = ParallelExperimentRunner(config)

        model_paths = {
            "mlp": str(model_mlp),
            "isab": str(model_isab),
        }

        try:
            results = runner.run_batch(
                run_specs,
                base_cfg,
                scenario_configs,
                model_paths,
                extra_cfg_overrides=None,
                progress_callback=progress_callback,
            )
        finally:
            pbar.close()

        rows = process_parallel_results(results)

    else:
        # Sequential execution (original behavior)
        print(f"  Mode: SEQUENTIAL\n")

        rows = []
        pbar = tqdm(
            total=total_runs,
            desc="Fig01",
            disable=args.no_progress,
            bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )

        try:
            for scenario in scenarios:
                cfg_path = SCRIPT_DIR / SCENARIOS[scenario]
                scenario_cfg = load_config_from_file(cfg_path)

                for method in methods:
                    for seed in seed_range:
                        cfg = base_cfg.copy()
                        cfg.update(scenario_cfg)
                        cfg.update(extra_cfg)
                        cfg["seed"] = seed
                        cfg["show_progress"] = False

                        pbar.set_description(f"Fig01 [{scenario[:8]}/{method}/{seed}]")

                        try:
                            result = run_experiment_sequential(cfg, method, model_mlp, model_isab)
                            row = {
                                "scenario": scenario,
                                "method": method,
                                "seed": seed,
                                "avg_se": result["avg_se"],
                                "avg_se_baseline": result["avg_se_baseline"],
                                "gain_pct": result["gain_pct"],
                            }
                            rows.append(row)

                            pbar.set_postfix(
                                se=f"{result['avg_se']:.4f}",
                                gain=f"{result['gain_pct']:+.2f}%"
                            )
                        except Exception as e:
                            print(f"\nError in {scenario}/{method}/seed={seed}: {e}")
                            rows.append({
                                "scenario": scenario,
                                "method": method,
                                "seed": seed,
                                "avg_se": float('nan'),
                                "avg_se_baseline": float('nan'),
                                "gain_pct": float('nan'),
                            })

                        pbar.update(1)
        finally:
            pbar.close()

    # Write results
    out_path = SCRIPT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["scenario", "method", "seed", "avg_se", "avg_se_baseline", "gain_pct"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path}")
    print(f"  Total rows: {len(rows)}")

    # Print cache stats if available
    try:
        from resource_cache import print_cache_stats
        print_cache_stats()
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
