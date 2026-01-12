#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 7: Constellation Mode.

Goal: Show multi-satellite relevance and handover/outage characteristics.

Sweeps:
(a) constellation_max_sats_per_tti: [1, 2, 4, 6, 8]
(a) min_elev_deg: [10, 15, 20, 25, 30]
(b) Collect HO events, outage per method

Outputs CSV with columns:
  method, scenario, max_sats, min_elev, seed, avg_se, avg_se_baseline, gain_pct,
  total_ho_count, avg_ho_per_ue, total_outage_ttis, avg_outage_frac
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

# Constellation scenarios
CONSTELLATION_SCENARIOS = {
    "toronto_constellation": "test/config_toronto_constellation.py",
    "shanghai_constellation": "test/config_shanghai_constellation.py",
}

# Default model paths
DEFAULT_MLP_MODEL = "output/models/nsgbs_scorer.pt"
DEFAULT_ISAB_MODEL = "output/models/nsgbs_isab_tau0.2.pt"

# Sweep values
MAX_SATS_VALUES = [1, 2, 4, 6, 8]
MIN_ELEV_VALUES = [10, 15, 20, 25, 30]


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


def run_constellation_experiment(cfg, method, model_mlp, model_isab):
    """
    Configure and run a constellation experiment.

    Returns:
        Dict with experiment results including HO and outage metrics
    """
    from main import run_once_constellation

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

    result = run_once_constellation(cfg)

    # Extract HO and outage metrics
    ho_counts = result.get("handover_count_per_ue", [])
    outage_ttis = result.get("outage_ttis_per_ue", [])
    T = cfg.get("T", 1000)
    N_UE = len(ho_counts) if ho_counts else cfg.get("N_UE", 100)

    total_ho = sum(ho_counts) if ho_counts else 0
    avg_ho_per_ue = total_ho / N_UE if N_UE > 0 else 0

    total_outage = sum(outage_ttis) if outage_ttis else 0
    avg_outage_frac = total_outage / (N_UE * T) if N_UE > 0 and T > 0 else 0

    return {
        "avg_se": float(result.get("avg_se_radiomap", 0.0)),
        "avg_se_baseline": float(result.get("avg_se_baseline_default", 0.0)),
        "gain_pct": float(result.get("improvement_vs_default_pct", 0.0)),
        "total_ho_count": int(total_ho),
        "avg_ho_per_ue": float(avg_ho_per_ue),
        "total_outage_ttis": int(total_outage),
        "avg_outage_frac": float(avg_outage_frac),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Figure 7: Constellation Mode"
    )
    parser.add_argument(
        "-s", "--scenario",
        action="append",
        choices=list(CONSTELLATION_SCENARIOS.keys()),
        help="Scenario(s) to run. Can be specified multiple times."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all constellation scenarios"
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
        "--max-sats-values",
        default=",".join(map(str, MAX_SATS_VALUES)),
        help=f"Comma-separated max_sats values (default: {MAX_SATS_VALUES})"
    )
    parser.add_argument(
        "--min-elev-values",
        default=",".join(map(str, MIN_ELEV_VALUES)),
        help=f"Comma-separated min_elev values (default: {MIN_ELEV_VALUES})"
    )
    parser.add_argument(
        "--sweep-mode",
        choices=["max_sats", "min_elev", "both"],
        default="both",
        help="Which parameter to sweep (default: both)"
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
        default="output/results/fig07_constellation.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    args = parser.parse_args()

    # Determine scenarios
    if args.all:
        scenarios = list(CONSTELLATION_SCENARIOS.keys())
    elif args.scenario:
        scenarios = args.scenario
    else:
        scenarios = ["toronto_constellation"]

    # Parse values
    max_sats_values = [int(x) for x in args.max_sats_values.split(",")]
    min_elev_values = [int(x) for x in args.min_elev_values.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Validate model paths
    model_mlp = Path(args.model_mlp)
    model_isab = Path(args.model_isab)

    if "P1_MLP" in methods and not model_mlp.exists():
        print(f"Warning: MLP model not found: {model_mlp}")
    if "P2_ISAB" in methods and not model_isab.exists():
        print(f"Warning: ISAB model not found: {model_isab}")

    # Load base config
    base_cfg = load_base_config()

    # Calculate total runs
    n_max_sats_runs = 0
    n_min_elev_runs = 0

    if args.sweep_mode in ("max_sats", "both"):
        n_max_sats_runs = len(scenarios) * len(max_sats_values) * len(methods) * args.seeds
    if args.sweep_mode in ("min_elev", "both"):
        n_min_elev_runs = len(scenarios) * len(min_elev_values) * len(methods) * args.seeds

    total_runs = n_max_sats_runs + n_min_elev_runs

    print(f"\nFigure 7: Constellation Experiment")
    print(f"  Scenarios: {scenarios}")
    print(f"  Methods: {methods}")
    print(f"  Sweep mode: {args.sweep_mode}")
    print(f"  Max sats values: {max_sats_values if args.sweep_mode in ('max_sats', 'both') else 'N/A'}")
    print(f"  Min elev values: {min_elev_values if args.sweep_mode in ('min_elev', 'both') else 'N/A'}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total runs: {total_runs}\n")

    rows = []
    pbar = tqdm(total=total_runs, desc="Fig07", disable=args.no_progress,
                bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
        for scenario in scenarios:
            cfg_path = SCRIPT_DIR / CONSTELLATION_SCENARIOS[scenario]
            scenario_cfg = load_config_from_file(cfg_path)

            # Get default values from scenario config
            default_max_sats = scenario_cfg.get("constellation_max_sats_per_tti", 6)
            default_min_elev = scenario_cfg.get("min_elev_deg", 20)

            # Sweep max_sats
            if args.sweep_mode in ("max_sats", "both"):
                for max_sats in max_sats_values:
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
                            cfg["constellation_max_sats_per_tti"] = max_sats

                            pbar.set_description(f"Fig07 [{scenario[:8]}/sats={max_sats}/{method[:4]}/s{seed}]")

                            try:
                                result = run_constellation_experiment(cfg, method, model_mlp, model_isab)
                                rows.append({
                                    "method": method,
                                    "scenario": scenario,
                                    "max_sats": max_sats,
                                    "min_elev": default_min_elev,
                                    "seed": seed,
                                    "avg_se": result["avg_se"] if method != "B1_3GPP" else result["avg_se_baseline"],
                                    "avg_se_baseline": result["avg_se_baseline"],
                                    "gain_pct": result["gain_pct"] if method != "B1_3GPP" else 0.0,
                                    "total_ho_count": result["total_ho_count"],
                                    "avg_ho_per_ue": result["avg_ho_per_ue"],
                                    "total_outage_ttis": result["total_outage_ttis"],
                                    "avg_outage_frac": result["avg_outage_frac"],
                                })
                                pbar.set_postfix(se=f"{result['avg_se']:.4f}", ho=result["total_ho_count"])
                            except Exception as e:
                                print(f"\nError {scenario}/sats={max_sats}/{method}/seed={seed}: {e}")

                            pbar.update(1)

            # Sweep min_elev
            if args.sweep_mode in ("min_elev", "both"):
                for min_elev in min_elev_values:
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
                            cfg["min_elev_deg"] = min_elev

                            pbar.set_description(f"Fig07 [{scenario[:8]}/elev={min_elev}/{method[:4]}/s{seed}]")

                            try:
                                result = run_constellation_experiment(cfg, method, model_mlp, model_isab)
                                rows.append({
                                    "method": method,
                                    "scenario": scenario,
                                    "max_sats": default_max_sats,
                                    "min_elev": min_elev,
                                    "seed": seed,
                                    "avg_se": result["avg_se"] if method != "B1_3GPP" else result["avg_se_baseline"],
                                    "avg_se_baseline": result["avg_se_baseline"],
                                    "gain_pct": result["gain_pct"] if method != "B1_3GPP" else 0.0,
                                    "total_ho_count": result["total_ho_count"],
                                    "avg_ho_per_ue": result["avg_ho_per_ue"],
                                    "total_outage_ttis": result["total_outage_ttis"],
                                    "avg_outage_frac": result["avg_outage_frac"],
                                })
                                pbar.set_postfix(se=f"{result['avg_se']:.4f}", ho=result["total_ho_count"])
                            except Exception as e:
                                print(f"\nError {scenario}/elev={min_elev}/{method}/seed={seed}: {e}")

                            pbar.update(1)

    finally:
        pbar.close()

    # Write results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "scenario", "max_sats", "min_elev", "seed", "avg_se", "avg_se_baseline",
                  "gain_pct", "total_ho_count", "avg_ho_per_ue", "total_outage_ttis", "avg_outage_frac"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path}")
    print(f"  Total rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
