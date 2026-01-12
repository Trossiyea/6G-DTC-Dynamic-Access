#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 2: User-Level Throughput Distribution & Fairness.

Goal: Show cell-edge improvements and fairness across methods.

Collects per-UE throughput data and fairness metrics (Jain's index).

Outputs two CSV files:
1. fig02_user_fairness_per_ue.csv: Per-UE throughput (long format)
   Columns: method, seed, ue_id, throughput_bps

2. fig02_user_fairness_summary.csv: Aggregate fairness metrics
   Columns: method, seed, jain_index, pctl_5, pctl_10, median, mean
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = SCRIPT_DIR / "code"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CODE_DIR))

# Default scenario for fairness analysis
DEFAULT_SCENARIO = "toronto_single"
DEFAULT_SCENARIO_PATH = "test/config_toronto_single.py"

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


def compute_jain_index(values):
    """
    Compute Jain's fairness index.

    J = (sum(x))^2 / (n * sum(x^2))

    Returns value in [0, 1], where 1 = perfect fairness.
    """
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return 0.0
    sum_x = np.sum(values)
    sum_x2 = np.sum(values ** 2)
    if sum_x2 == 0:
        return 1.0
    return (sum_x ** 2) / (n * sum_x2)


def run_experiment(cfg, method, model_mlp, model_isab):
    """
    Configure and run a single experiment.

    Returns:
        Dict with per-UE throughput and fairness metrics
    """
    from main import run_once

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

    result = run_once(cfg)

    # Extract per-UE throughput
    # For methods using radiomap scheduling, use radiomap throughput
    # For B1_3GPP, use baseline throughput
    if method == "B1_3GPP":
        per_ue_tput = result.get("per_ue_throughput_baseline_bps", [])
        jain = result.get("fairness_jain_base", 0.0)
    else:
        per_ue_tput = result.get("per_ue_throughput_radiomap_bps", [])
        jain = result.get("fairness_jain_map", 0.0)

    per_ue_tput = np.asarray(per_ue_tput)

    # Compute additional fairness metrics if not provided
    if jain == 0.0 and len(per_ue_tput) > 0:
        jain = compute_jain_index(per_ue_tput)

    # Compute percentiles
    if len(per_ue_tput) > 0:
        pctl_5 = np.percentile(per_ue_tput, 5)
        pctl_10 = np.percentile(per_ue_tput, 10)
        median = np.percentile(per_ue_tput, 50)
        mean = np.mean(per_ue_tput)
    else:
        pctl_5 = pctl_10 = median = mean = 0.0

    return {
        "per_ue_throughput": per_ue_tput.tolist(),
        "jain_index": float(jain),
        "pctl_5": float(pctl_5),
        "pctl_10": float(pctl_10),
        "median": float(median),
        "mean": float(mean),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Figure 2: User-Level Throughput Distribution & Fairness"
    )
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help=f"Scenario to run (default: {DEFAULT_SCENARIO})"
    )
    parser.add_argument(
        "--scenario-path",
        default=None,
        help="Path to scenario config file (overrides --scenario)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds (default: 5)"
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
        help="Override number of TTIs"
    )
    parser.add_argument(
        "--N_UE",
        type=int,
        default=100,
        help="Number of UEs (default: 100)"
    )
    parser.add_argument(
        "--methods",
        default="B1_3GPP,B3_heuristic,P1_MLP,P2_ISAB",
        help="Comma-separated list of methods"
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
        "--out-per-ue",
        default="output/results/fig02_user_fairness_per_ue.csv",
        help="Output CSV path for per-UE data"
    )
    parser.add_argument(
        "--out-summary",
        default="output/results/fig02_user_fairness_summary.csv",
        help="Output CSV path for summary data"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    args = parser.parse_args()

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

    # Load configs
    base_cfg = load_base_config()

    if args.scenario_path:
        scenario_cfg = load_config_from_file(Path(args.scenario_path))
    else:
        scenario_cfg = load_config_from_file(SCRIPT_DIR / DEFAULT_SCENARIO_PATH)

    # Calculate total runs
    total_runs = len(methods) * args.seeds
    print(f"\nFigure 2: User Fairness Experiment")
    print(f"  Scenario: {args.scenario}")
    print(f"  Methods: {methods}")
    print(f"  N_UE: {args.N_UE}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total runs: {total_runs}\n")

    per_ue_rows = []
    summary_rows = []

    pbar = tqdm(total=total_runs, desc="Fig02", disable=args.no_progress,
                bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
        for method in methods:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                # Build config
                cfg = base_cfg.copy()
                cfg.update(scenario_cfg)

                if args.T is not None:
                    cfg["T"] = args.T
                cfg["N_UE"] = args.N_UE
                cfg["seed"] = seed
                cfg["show_progress"] = False

                pbar.set_description(f"Fig02 [{method}/{seed}]")

                try:
                    result = run_experiment(cfg, method, model_mlp, model_isab)

                    # Record per-UE data
                    for ue_id, tput in enumerate(result["per_ue_throughput"]):
                        per_ue_rows.append({
                            "method": method,
                            "seed": seed,
                            "ue_id": ue_id,
                            "throughput_bps": tput,
                        })

                    # Record summary
                    summary_rows.append({
                        "method": method,
                        "seed": seed,
                        "jain_index": result["jain_index"],
                        "pctl_5": result["pctl_5"],
                        "pctl_10": result["pctl_10"],
                        "median": result["median"],
                        "mean": result["mean"],
                    })

                    pbar.set_postfix(
                        jain=f"{result['jain_index']:.4f}",
                        p5=f"{result['pctl_5']/1e6:.2f}M"
                    )

                except Exception as e:
                    print(f"\nError in {method}/seed={seed}: {e}")

                pbar.update(1)
    finally:
        pbar.close()

    # Write per-UE CSV
    out_per_ue = Path(args.out_per_ue)
    out_per_ue.parent.mkdir(parents=True, exist_ok=True)

    with open(out_per_ue, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seed", "ue_id", "throughput_bps"])
        writer.writeheader()
        writer.writerows(per_ue_rows)

    print(f"\nSaved per-UE data to {out_per_ue}")
    print(f"  Total rows: {len(per_ue_rows)}")

    # Write summary CSV
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seed", "jain_index", "pctl_5", "pctl_10", "median", "mean"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary data to {out_summary}")
    print(f"  Total rows: {len(summary_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
