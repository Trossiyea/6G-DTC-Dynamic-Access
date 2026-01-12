#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 10: Learning Generalization + Ablations.

Goal: Address overfitting concerns and explain what matters.

Experiments:
(a) Cross-city: Train Toronto → Test Shanghai, and vice versa
(b) ISAB τ sweep: [0.1, 0.15, 0.2, 0.3]

Outputs CSV with columns:
  exp_type, train_city, test_city, model_variant, tau, seed, avg_se, avg_se_baseline, gain_pct
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

# City scenario configs
CITY_SCENARIOS = {
    "toronto": "test/config_toronto_single.py",
    "shanghai": "test/config_shanghai_single.py",
}

# Default model paths (assuming models exist for each city/variant)
MODEL_DIR = "output/models"

# ISAB tau values to sweep
TAU_VALUES = [0.1, 0.15, 0.2, 0.3]


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


def find_model(model_dir: Path, variant: str, tau: float = None, train_city: str = None):
    """
    Find model path based on variant and parameters.

    Looks for patterns like:
    - nsgbs_scorer.pt (MLP)
    - nsgbs_isab_tau0.2.pt (ISAB with specific tau)
    - nsgbs_isab_toronto_tau0.2.pt (city-specific)
    """
    if variant == "MLP":
        # Look for MLP model
        candidates = [
            model_dir / "nsgbs_scorer.pt",
            model_dir / f"nsgbs_mlp_{train_city}.pt" if train_city else None,
        ]
    elif variant == "ISAB":
        tau_str = f"tau{tau}" if tau else "tau0.2"
        candidates = [
            model_dir / f"nsgbs_isab_{tau_str}.pt",
            model_dir / f"nsgbs_isab_{train_city}_{tau_str}.pt" if train_city else None,
            model_dir / f"nsgbs_isab.pt",  # Fallback
        ]
    else:
        return None

    for path in candidates:
        if path and path.exists():
            return path

    return None


def run_experiment(cfg, model_path):
    """
    Configure and run a single experiment.

    Returns:
        Dict with experiment results
    """
    from main import run_once

    cfg["scheduler_kind"] = "nsgbs"
    cfg["nsgbs_model_path"] = str(model_path) if model_path else None

    result = run_once(cfg)

    return {
        "avg_se": float(result.get("avg_se_radiomap", 0.0)),
        "avg_se_baseline": float(result.get("avg_se_baseline_default", 0.0)),
        "gain_pct": float(result.get("improvement_vs_default_pct", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Figure 10: Learning Generalization + Ablations"
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
        "--tau-values",
        default=",".join(map(str, TAU_VALUES)),
        help=f"Comma-separated tau values for ISAB sweep (default: {TAU_VALUES})"
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help=f"Directory containing trained models (default: {MODEL_DIR})"
    )
    parser.add_argument(
        "--skip-cross-city",
        action="store_true",
        help="Skip cross-city generalization experiments"
    )
    parser.add_argument(
        "--skip-tau-sweep",
        action="store_true",
        help="Skip tau sweep experiments"
    )
    parser.add_argument(
        "--out",
        default="output/results/fig10_generalization.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    args = parser.parse_args()

    # Parse tau values
    tau_values = [float(x) for x in args.tau_values.split(",")]
    model_dir = Path(args.model_dir)

    # Load base config
    base_cfg = load_base_config()

    # Calculate experiments
    experiments = []

    # (a) Cross-city generalization
    if not args.skip_cross_city:
        # Train Toronto -> Test Shanghai
        # Train Shanghai -> Test Toronto
        cross_city_pairs = [
            ("toronto", "shanghai"),
            ("shanghai", "toronto"),
            ("toronto", "toronto"),  # Same-city baseline
            ("shanghai", "shanghai"),  # Same-city baseline
        ]
        for train_city, test_city in cross_city_pairs:
            for variant in ["MLP", "ISAB"]:
                experiments.append({
                    "exp_type": "cross_city",
                    "train_city": train_city,
                    "test_city": test_city,
                    "model_variant": variant,
                    "tau": 0.2 if variant == "ISAB" else None,
                })

    # (b) Tau sweep (ISAB only)
    if not args.skip_tau_sweep:
        for tau in tau_values:
            experiments.append({
                "exp_type": "tau_sweep",
                "train_city": "toronto",  # Use Toronto for tau sweep
                "test_city": "toronto",
                "model_variant": "ISAB",
                "tau": tau,
            })

    total_runs = len(experiments) * args.seeds

    print(f"\nFigure 10: Generalization Experiment")
    print(f"  Model dir: {model_dir}")
    print(f"  Cross-city: {not args.skip_cross_city}")
    print(f"  Tau sweep: {not args.skip_tau_sweep}")
    print(f"  Tau values: {tau_values}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Total runs: {total_runs}\n")

    rows = []
    pbar = tqdm(total=total_runs, desc="Fig10", disable=args.no_progress,
                bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
        for exp in experiments:
            exp_type = exp["exp_type"]
            train_city = exp["train_city"]
            test_city = exp["test_city"]
            variant = exp["model_variant"]
            tau = exp["tau"]

            # Find model
            model_path = find_model(model_dir, variant, tau, train_city)

            if model_path is None or not model_path.exists():
                print(f"\nWarning: Model not found for {variant} (tau={tau}, train={train_city})")
                pbar.update(args.seeds)
                continue

            # Load test scenario
            test_scenario_path = SCRIPT_DIR / CITY_SCENARIOS[test_city]
            test_cfg = load_config_from_file(test_scenario_path)

            for seed in range(args.seed_start, args.seed_start + args.seeds):
                cfg = base_cfg.copy()
                cfg.update(test_cfg)

                if args.T is not None:
                    cfg["T"] = args.T
                if args.N_UE is not None:
                    cfg["N_UE"] = args.N_UE
                cfg["seed"] = seed
                cfg["show_progress"] = False

                desc = f"Fig10 [{exp_type[:5]}/{train_city[:3]}->{test_city[:3]}/{variant}/s{seed}]"
                pbar.set_description(desc)

                try:
                    result = run_experiment(cfg, model_path)
                    rows.append({
                        "exp_type": exp_type,
                        "train_city": train_city,
                        "test_city": test_city,
                        "model_variant": variant,
                        "tau": tau if tau else "",
                        "seed": seed,
                        "avg_se": result["avg_se"],
                        "avg_se_baseline": result["avg_se_baseline"],
                        "gain_pct": result["gain_pct"],
                    })
                    pbar.set_postfix(se=f"{result['avg_se']:.4f}", gain=f"{result['gain_pct']:+.2f}%")
                except Exception as e:
                    print(f"\nError {desc}: {e}")

                pbar.update(1)

    finally:
        pbar.close()

    # Write results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["exp_type", "train_city", "test_city", "model_variant", "tau", "seed",
                  "avg_se", "avg_se_baseline", "gain_pct"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path}")
    print(f"  Total rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
