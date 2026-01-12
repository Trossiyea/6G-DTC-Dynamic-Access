#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 5: Non-Stationary Interference + Orbit Dynamics.

Goal: Validate performance under fast-changing conditions.

Sweeps:
(a) rm_flicker_db_std: [0, 0.5, 1.0, 2.0, 3.0, 5.0]
(b) doppler_residual_fraction: [0, 0.05, 0.1, 0.2, 0.5, 1.0]

Outputs CSV with columns:
  method, dynamic_type, dynamic_value, seed, avg_se, avg_se_baseline, gain_pct
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

# Default scenario
DEFAULT_SCENARIO_PATH = "test/config_toronto_single.py"

# Default model paths
DEFAULT_MLP_MODEL = "output/models/nsgbs_scorer.pt"
DEFAULT_ISAB_MODEL = "output/models/nsgbs_isab_tau0.2.pt"

# Sweep values
FLICKER_STD_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
DOPPLER_RESIDUAL_VALUES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


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

    Returns:
        Dict with experiment results
    """
    from main import run_once

    if method == "B1_3GPP":
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

    return {
        "avg_se": float(result.get("avg_se_radiomap", 0.0)),
        "avg_se_baseline": float(result.get("avg_se_baseline_default", 0.0)),
        "gain_pct": float(result.get("improvement_vs_default_pct", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Figure 5: Non-Stationary Interference + Orbit Dynamics"
    )
    parser.add_argument(
        "--scenario-path",
        default=DEFAULT_SCENARIO_PATH,
        help=f"Path to scenario config (default: {DEFAULT_SCENARIO_PATH})"
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
        default=None,
        help="Override number of UEs"
    )
    parser.add_argument(
        "--flicker-values",
        default=",".join(map(str, FLICKER_STD_VALUES)),
        help=f"Comma-separated flicker std values (default: {FLICKER_STD_VALUES})"
    )
    parser.add_argument(
        "--doppler-values",
        default=",".join(map(str, DOPPLER_RESIDUAL_VALUES)),
        help=f"Comma-separated Doppler residual values (default: {DOPPLER_RESIDUAL_VALUES})"
    )
    parser.add_argument(
        "--methods",
        default="B1_3GPP,P1_MLP,P2_ISAB",
        help="Comma-separated methods (default: B1_3GPP,P1_MLP,P2_ISAB)"
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
        default="output/results/fig05_dynamics.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    args = parser.parse_args()

    # Parse values
    flicker_values = [float(x) for x in args.flicker_values.split(",")]
    doppler_values = [float(x) for x in args.doppler_values.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Validate model paths
    model_mlp = Path(args.model_mlp)
    model_isab = Path(args.model_isab)

    if "P1_MLP" in methods and not model_mlp.exists():
        print(f"Warning: MLP model not found: {model_mlp}")
    if "P2_ISAB" in methods and not model_isab.exists():
        print(f"Warning: ISAB model not found: {model_isab}")

    # Load configs
    base_cfg = load_base_config()
    scenario_cfg = load_config_from_file(SCRIPT_DIR / args.scenario_path)

    # Calculate total runs
    n_flicker_runs = len(flicker_values) * len(methods) * args.seeds
    n_doppler_runs = len(doppler_values) * len(methods) * args.seeds
    total_runs = n_flicker_runs + n_doppler_runs

    print(f"\nFigure 5: Dynamics Experiment")
    print(f"  Scenario: {args.scenario_path}")
    print(f"  Methods: {methods}")
    print(f"  Flicker std values: {flicker_values}")
    print(f"  Doppler residual values: {doppler_values}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total runs: {total_runs}\n")

    rows = []
    pbar = tqdm(total=total_runs, desc="Fig05", disable=args.no_progress,
                bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
        # Part (a): Flicker std sweep
        for flicker_std in flicker_values:
            for method in methods:
                model_path = model_mlp if method == "P1_MLP" else (model_isab if method == "P2_ISAB" else None)
                if method in ("P1_MLP", "P2_ISAB") and model_path and not model_path.exists():
                    pbar.update(args.seeds)
                    continue

                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    cfg = base_cfg.copy()
                    cfg.update(scenario_cfg)

                    if args.T is not None:
                        cfg["T"] = args.T
                    if args.N_UE is not None:
                        cfg["N_UE"] = args.N_UE
                    cfg["seed"] = seed
                    cfg["show_progress"] = False

                    # Set flicker std
                    cfg["rm_flicker_db_std"] = flicker_std

                    pbar.set_description(f"Fig05 [flicker={flicker_std}/{method[:4]}/s{seed}]")

                    try:
                        result = run_experiment(cfg, method, model_mlp, model_isab)
                        rows.append({
                            "method": method,
                            "dynamic_type": "flicker_db_std",
                            "dynamic_value": flicker_std,
                            "seed": seed,
                            "avg_se": result["avg_se"] if method != "B1_3GPP" else result["avg_se_baseline"],
                            "avg_se_baseline": result["avg_se_baseline"],
                            "gain_pct": result["gain_pct"] if method != "B1_3GPP" else 0.0,
                        })
                        pbar.set_postfix(se=f"{result['avg_se']:.4f}")
                    except Exception as e:
                        print(f"\nError flicker={flicker_std}/{method}/seed={seed}: {e}")

                    pbar.update(1)

        # Part (b): Doppler residual sweep
        for doppler_res in doppler_values:
            for method in methods:
                model_path = model_mlp if method == "P1_MLP" else (model_isab if method == "P2_ISAB" else None)
                if method in ("P1_MLP", "P2_ISAB") and model_path and not model_path.exists():
                    pbar.update(args.seeds)
                    continue

                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    cfg = base_cfg.copy()
                    cfg.update(scenario_cfg)

                    if args.T is not None:
                        cfg["T"] = args.T
                    if args.N_UE is not None:
                        cfg["N_UE"] = args.N_UE
                    cfg["seed"] = seed
                    cfg["show_progress"] = False

                    # Set Doppler residual fraction
                    cfg["doppler_residual_fraction"] = doppler_res

                    pbar.set_description(f"Fig05 [doppler={doppler_res}/{method[:4]}/s{seed}]")

                    try:
                        result = run_experiment(cfg, method, model_mlp, model_isab)
                        rows.append({
                            "method": method,
                            "dynamic_type": "doppler_residual",
                            "dynamic_value": doppler_res,
                            "seed": seed,
                            "avg_se": result["avg_se"] if method != "B1_3GPP" else result["avg_se_baseline"],
                            "avg_se_baseline": result["avg_se_baseline"],
                            "gain_pct": result["gain_pct"] if method != "B1_3GPP" else 0.0,
                        })
                        pbar.set_postfix(se=f"{result['avg_se']:.4f}")
                    except Exception as e:
                        print(f"\nError doppler={doppler_res}/{method}/seed={seed}: {e}")

                    pbar.update(1)

    finally:
        pbar.close()

    # Write results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "dynamic_type", "dynamic_value", "seed", "avg_se", "avg_se_baseline", "gain_pct"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path}")
    print(f"  Total rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
