#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 6: Scalability With UE Load and Bandwidth.

Goal: Show performance under varying load and scheduling hardness.

Sweeps:
(a) N_UE: [20, 50, 100, 150, 200]
(b) Z (PRBs): [25, 51, 75, 100] (optional, depends on RadioMap availability)

Outputs CSV with columns:
  method, N_UE, Z, seed, avg_se, avg_se_baseline, gain_pct, pctl_5_tput, median_tput
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

# Default scenario
DEFAULT_SCENARIO_PATH = "test/config_toronto_single.py"

# Default model paths
DEFAULT_MLP_MODEL = "output/models/nsgbs_scorer.pt"
DEFAULT_ISAB_MODEL = "output/models/nsgbs_isab_tau0.2.pt"

# Sweep values
N_UE_VALUES = [20, 50, 100, 150, 200]
Z_VALUES = [25, 51, 75, 100]  # PRB counts


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
        Dict with experiment results including per-UE statistics
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

    # Get per-UE throughput
    if method == "B1_3GPP":
        per_ue_tput = result.get("per_ue_throughput_baseline_bps", [])
    else:
        per_ue_tput = result.get("per_ue_throughput_radiomap_bps", [])

    per_ue_tput = np.asarray(per_ue_tput)

    # Compute percentiles
    if len(per_ue_tput) > 0:
        pctl_5 = float(np.percentile(per_ue_tput, 5))
        median = float(np.percentile(per_ue_tput, 50))
    else:
        pctl_5 = 0.0
        median = 0.0

    return {
        "avg_se": float(result.get("avg_se_radiomap", 0.0)),
        "avg_se_baseline": float(result.get("avg_se_baseline_default", 0.0)),
        "gain_pct": float(result.get("improvement_vs_default_pct", 0.0)),
        "pctl_5_tput": pctl_5,
        "median_tput": median,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Figure 6: Scalability With UE Load"
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
        "--n-ue-values",
        default=",".join(map(str, N_UE_VALUES)),
        help=f"Comma-separated N_UE values (default: {N_UE_VALUES})"
    )
    parser.add_argument(
        "--sweep-z",
        action="store_true",
        help="Also sweep Z (PRB count) - requires compatible RadioMaps"
    )
    parser.add_argument(
        "--z-values",
        default=",".join(map(str, Z_VALUES)),
        help=f"Comma-separated Z values (default: {Z_VALUES})"
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
        default="output/results/fig06_scalability.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    args = parser.parse_args()

    # Parse values
    n_ue_values = [int(x) for x in args.n_ue_values.split(",")]
    z_values = [int(x) for x in args.z_values.split(",")] if args.sweep_z else []
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

    # Get default Z from config
    default_z = scenario_cfg.get("Z", base_cfg.get("Z", 51))

    # Calculate total runs
    n_nue_runs = len(n_ue_values) * len(methods) * args.seeds
    n_z_runs = len(z_values) * len(methods) * args.seeds if args.sweep_z else 0
    total_runs = n_nue_runs + n_z_runs

    print(f"\nFigure 6: Scalability Experiment")
    print(f"  Scenario: {args.scenario_path}")
    print(f"  Methods: {methods}")
    print(f"  N_UE values: {n_ue_values}")
    print(f"  Z sweep: {z_values if args.sweep_z else 'disabled'}")
    print(f"  Default Z: {default_z}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total runs: {total_runs}\n")

    rows = []
    pbar = tqdm(total=total_runs, desc="Fig06", disable=args.no_progress,
                bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
        # Part (a): N_UE sweep
        for n_ue in n_ue_values:
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
                    cfg["N_UE"] = n_ue
                    cfg["seed"] = seed
                    cfg["show_progress"] = False

                    pbar.set_description(f"Fig06 [N_UE={n_ue}/{method[:4]}/s{seed}]")

                    try:
                        result = run_experiment(cfg, method, model_mlp, model_isab)
                        rows.append({
                            "method": method,
                            "N_UE": n_ue,
                            "Z": default_z,
                            "seed": seed,
                            "avg_se": result["avg_se"] if method != "B1_3GPP" else result["avg_se_baseline"],
                            "avg_se_baseline": result["avg_se_baseline"],
                            "gain_pct": result["gain_pct"] if method != "B1_3GPP" else 0.0,
                            "pctl_5_tput": result["pctl_5_tput"],
                            "median_tput": result["median_tput"],
                        })
                        pbar.set_postfix(
                            se=f"{result['avg_se']:.4f}",
                            p5=f"{result['pctl_5_tput']/1e6:.2f}M"
                        )
                    except Exception as e:
                        print(f"\nError N_UE={n_ue}/{method}/seed={seed}: {e}")

                    pbar.update(1)

        # Part (b): Z sweep (optional)
        if args.sweep_z:
            default_n_ue = scenario_cfg.get("N_UE", base_cfg.get("N_UE", 100))

            for z in z_values:
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
                        cfg["N_UE"] = default_n_ue
                        cfg["Z"] = z
                        cfg["seed"] = seed
                        cfg["show_progress"] = False

                        pbar.set_description(f"Fig06 [Z={z}/{method[:4]}/s{seed}]")

                        try:
                            result = run_experiment(cfg, method, model_mlp, model_isab)
                            rows.append({
                                "method": method,
                                "N_UE": default_n_ue,
                                "Z": z,
                                "seed": seed,
                                "avg_se": result["avg_se"] if method != "B1_3GPP" else result["avg_se_baseline"],
                                "avg_se_baseline": result["avg_se_baseline"],
                                "gain_pct": result["gain_pct"] if method != "B1_3GPP" else 0.0,
                                "pctl_5_tput": result["pctl_5_tput"],
                                "median_tput": result["median_tput"],
                            })
                            pbar.set_postfix(se=f"{result['avg_se']:.4f}")
                        except Exception as e:
                            print(f"\nError Z={z}/{method}/seed={seed}: {e}")

                        pbar.update(1)

    finally:
        pbar.close()

    # Write results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "N_UE", "Z", "seed", "avg_se", "avg_se_baseline", "gain_pct", "pctl_5_tput", "median_tput"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path}")
    print(f"  Total rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
