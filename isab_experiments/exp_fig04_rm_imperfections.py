#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 4: Robustness to RadioMap Imperfections.

Goal: Show graceful degradation under RadioMap estimation error and spatial blur.

Sweeps:
(a) RadioMap estimation error: [0, 0.5, 1.0, 1.5, 2.0, 3.0] dB
(b) RadioMap spatial blur: [0, 0.5, 1.0, 1.5, 2.0] sigma
    + Resolution comparison: 125m vs 150m

Outputs CSV with columns:
  method, imperfection_type, imperfection_value, seed, avg_se, avg_se_baseline, gain_pct
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

# Resolution scenarios for comparison
RESOLUTION_SCENARIOS = {
    "125m": "test/config_toronto_single_125m.py",
    "150m": "test/config_toronto_single_150m.py",
}

# Default model paths
DEFAULT_MLP_MODEL = "output/models/nsgbs_scorer.pt"
DEFAULT_ISAB_MODEL = "output/models/nsgbs_isab_tau0.2.pt"

# Sweep values
EST_ERROR_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
BLUR_SIGMA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]


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
        description="Figure 4: Robustness to RadioMap Imperfections"
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
        "--est-errors",
        default=",".join(map(str, EST_ERROR_VALUES)),
        help=f"Comma-separated estimation error values in dB (default: {EST_ERROR_VALUES})"
    )
    parser.add_argument(
        "--blur-sigmas",
        default=",".join(map(str, BLUR_SIGMA_VALUES)),
        help=f"Comma-separated blur sigma values (default: {BLUR_SIGMA_VALUES})"
    )
    parser.add_argument(
        "--methods",
        default="P1_MLP,P2_ISAB",
        help="Comma-separated methods to test (default: P1_MLP,P2_ISAB)"
    )
    parser.add_argument(
        "--include-resolution",
        action="store_true",
        help="Include resolution comparison (125m vs 150m)"
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
        default="output/results/fig04_rm_imperfections.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    args = parser.parse_args()

    # Parse values
    est_errors = [float(x) for x in args.est_errors.split(",")]
    blur_sigmas = [float(x) for x in args.blur_sigmas.split(",")]
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
    scenario_cfg = load_config_from_file(SCRIPT_DIR / args.scenario_path)

    # Calculate total runs
    n_est_runs = len(est_errors) * len(methods) * args.seeds
    n_blur_runs = len(blur_sigmas) * len(methods) * args.seeds
    n_resolution_runs = 2 * len(methods) * args.seeds if args.include_resolution else 0

    total_runs = n_est_runs + n_blur_runs + n_resolution_runs

    print(f"\nFigure 4: RadioMap Imperfections Experiment")
    print(f"  Scenario: {args.scenario_path}")
    print(f"  Methods: {methods}")
    print(f"  Estimation errors: {est_errors}")
    print(f"  Blur sigmas: {blur_sigmas}")
    print(f"  Include resolution: {args.include_resolution}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total runs: {total_runs}\n")

    rows = []
    pbar = tqdm(total=total_runs, desc="Fig04", disable=args.no_progress,
                bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
        # Part (a): Estimation error sweep
        for est_error in est_errors:
            for method in methods:
                model_path = model_mlp if method == "P1_MLP" else model_isab
                if not model_path.exists():
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

                    # Set estimation error
                    cfg["radiomap_est_error_db"] = est_error

                    pbar.set_description(f"Fig04 [est_err={est_error}/{method[:4]}/s{seed}]")

                    try:
                        result = run_experiment(cfg, method, model_mlp, model_isab)
                        rows.append({
                            "method": method,
                            "imperfection_type": "est_error_db",
                            "imperfection_value": est_error,
                            "seed": seed,
                            "avg_se": result["avg_se"],
                            "avg_se_baseline": result["avg_se_baseline"],
                            "gain_pct": result["gain_pct"],
                        })
                        pbar.set_postfix(se=f"{result['avg_se']:.4f}", gain=f"{result['gain_pct']:+.2f}%")
                    except Exception as e:
                        print(f"\nError est_err={est_error}/{method}/seed={seed}: {e}")

                    pbar.update(1)

        # Part (b): Blur sigma sweep
        for blur_sigma in blur_sigmas:
            for method in methods:
                model_path = model_mlp if method == "P1_MLP" else model_isab
                if not model_path.exists():
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

                    # Set blur sigma
                    cfg["radiomap_blur_sigma"] = blur_sigma

                    pbar.set_description(f"Fig04 [blur={blur_sigma}/{method[:4]}/s{seed}]")

                    try:
                        result = run_experiment(cfg, method, model_mlp, model_isab)
                        rows.append({
                            "method": method,
                            "imperfection_type": "blur_sigma",
                            "imperfection_value": blur_sigma,
                            "seed": seed,
                            "avg_se": result["avg_se"],
                            "avg_se_baseline": result["avg_se_baseline"],
                            "gain_pct": result["gain_pct"],
                        })
                        pbar.set_postfix(se=f"{result['avg_se']:.4f}", gain=f"{result['gain_pct']:+.2f}%")
                    except Exception as e:
                        print(f"\nError blur={blur_sigma}/{method}/seed={seed}: {e}")

                    pbar.update(1)

        # Resolution comparison (optional)
        if args.include_resolution:
            for resolution, res_path in RESOLUTION_SCENARIOS.items():
                res_cfg_path = SCRIPT_DIR / res_path
                if not res_cfg_path.exists():
                    print(f"\nSkipping resolution {resolution}: config not found")
                    pbar.update(len(methods) * args.seeds)
                    continue

                res_cfg = load_config_from_file(res_cfg_path)

                for method in methods:
                    model_path = model_mlp if method == "P1_MLP" else model_isab
                    if not model_path.exists():
                        pbar.update(args.seeds)
                        continue

                    for seed in range(args.seed_start, args.seed_start + args.seeds):
                        cfg = base_cfg.copy()
                        cfg.update(res_cfg)

                        if args.T is not None:
                            cfg["T"] = args.T
                        if args.N_UE is not None:
                            cfg["N_UE"] = args.N_UE
                        cfg["seed"] = seed
                        cfg["show_progress"] = False

                        pbar.set_description(f"Fig04 [res={resolution}/{method[:4]}/s{seed}]")

                        try:
                            result = run_experiment(cfg, method, model_mlp, model_isab)
                            # Store resolution as imperfection value (125 or 150)
                            res_value = float(resolution.replace("m", ""))
                            rows.append({
                                "method": method,
                                "imperfection_type": "resolution_m",
                                "imperfection_value": res_value,
                                "seed": seed,
                                "avg_se": result["avg_se"],
                                "avg_se_baseline": result["avg_se_baseline"],
                                "gain_pct": result["gain_pct"],
                            })
                            pbar.set_postfix(se=f"{result['avg_se']:.4f}", gain=f"{result['gain_pct']:+.2f}%")
                        except Exception as e:
                            print(f"\nError res={resolution}/{method}/seed={seed}: {e}")

                        pbar.update(1)

    finally:
        pbar.close()

    # Write results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "imperfection_type", "imperfection_value", "seed", "avg_se", "avg_se_baseline", "gain_pct"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path}")
    print(f"  Total rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
