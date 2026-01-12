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


def run_experiment(cfg, method, model_mlp, model_isab):
    """
    Configure and run a single experiment.

    Args:
        cfg: Base config dict (will be modified)
        method: One of 'B1_3GPP', 'B3_heuristic', 'P1_MLP', 'P2_ISAB'
        model_mlp: Path to MLP model
        model_isab: Path to ISAB model

    Returns:
        Dict with experiment results
    """
    from main import run_once

    if method == "B1_3GPP":
        # 3GPP baseline: use baseline scheduler with CSI delay
        cfg["scheduler_kind"] = "heuristic"
        cfg["nsgbs_model_path"] = None
        # Baseline CSI delay is already configured in base config
    elif method == "B3_heuristic":
        # RadioMap heuristic: no learning
        cfg["scheduler_kind"] = "heuristic"
        cfg["nsgbs_model_path"] = None
    elif method == "P1_MLP":
        # RadioMap + MLP
        cfg["scheduler_kind"] = "nsgbs"
        cfg["nsgbs_model_path"] = str(model_mlp)
    elif method == "P2_ISAB":
        # RadioMap + ISAB
        cfg["scheduler_kind"] = "nsgbs"
        cfg["nsgbs_model_path"] = str(model_isab)
    else:
        raise ValueError(f"Unknown method: {method}")

    result = run_once(cfg)

    return {
        "avg_se": float(result.get("avg_se_radiomap", 0.0)),
        "avg_se_baseline": float(result.get("avg_se_baseline_default", 0.0)),
        "gain_pct": float(result.get("improvement_vs_default_pct", 0.0)),
    }


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
    model_mlp = Path(args.model_mlp)
    model_isab = Path(args.model_isab)

    if "P1_MLP" in methods and not model_mlp.exists():
        print(f"Error: MLP model not found: {model_mlp}")
        return 1
    if "P2_ISAB" in methods and not model_isab.exists():
        print(f"Error: ISAB model not found: {model_isab}")
        return 1

    # Load base config
    base_cfg = load_base_config()

    # Calculate total runs
    total_runs = len(scenarios) * len(methods) * args.seeds
    print(f"\nFigure 1: Main Performance Experiment")
    print(f"  Scenarios: {scenarios}")
    print(f"  Methods: {methods}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total runs: {total_runs}\n")

    rows = []
    pbar = tqdm(total=total_runs, desc="Fig01", disable=args.no_progress,
                bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
        for scenario in scenarios:
            # Load scenario config
            cfg_path = SCRIPT_DIR / SCENARIOS[scenario]
            scenario_cfg = load_config_from_file(cfg_path)

            for method in methods:
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    # Build config for this run
                    cfg = base_cfg.copy()
                    cfg.update(scenario_cfg)

                    if args.T is not None:
                        cfg["T"] = args.T
                    if args.N_UE is not None:
                        cfg["N_UE"] = args.N_UE
                    cfg["seed"] = seed
                    cfg["show_progress"] = False

                    pbar.set_description(f"Fig01 [{scenario[:8]}/{method}/{seed}]")

                    try:
                        result = run_experiment(cfg, method, model_mlp, model_isab)
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
                        # Record error row
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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["scenario", "method", "seed", "avg_se", "avg_se_baseline", "gain_pct"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path}")
    print(f"  Total rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
