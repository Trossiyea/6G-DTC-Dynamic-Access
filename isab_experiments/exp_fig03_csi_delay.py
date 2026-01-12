#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 3: CSI Delay vs RadioMap Freshness.

Goal: Justify the information advantage of RadioMap over CSI-based scheduling.

Sweeps:
(a) Baseline CSI delay: [0, 5, 10, 20, 40] TTIs
(b) RadioMap "staleness" / delay: [0, 2, 5, 10, 20] TTIs

Outputs CSV with columns:
  method, delay_type, delay_value, seed, avg_se, avg_se_baseline, gain_pct
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
BASELINE_CSI_DELAYS = [0, 5, 10, 20, 40]
RM_CSI_DELAYS = [0, 2, 5, 10, 20]


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


def run_experiment(cfg, method, model_mlp, model_isab, delay_type, delay_value):
    """
    Configure and run a single experiment with specific delay settings.

    Args:
        cfg: Base config dict (will be modified)
        method: One of 'B1_3GPP', 'B2_Oracle', 'P1_MLP', 'P2_ISAB'
        delay_type: 'baseline_csi' or 'rm_csi'
        delay_value: Delay in TTIs

    Returns:
        Dict with experiment results
    """
    from main import run_once

    # Configure method
    if method in ("B1_3GPP", "B2_Oracle"):
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

    # Configure delay
    if delay_type == "baseline_csi":
        cfg["baseline_csi_delay_ttis"] = delay_value
        # Oracle has no delay
        if method == "B2_Oracle":
            cfg["baseline_csi_delay_ttis"] = 0
    elif delay_type == "rm_csi":
        cfg["rm_csi_delay_ttis"] = delay_value

    result = run_once(cfg)

    return {
        "avg_se": float(result.get("avg_se_radiomap", 0.0)),
        "avg_se_baseline": float(result.get("avg_se_baseline_default", 0.0)),
        "gain_pct": float(result.get("improvement_vs_default_pct", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Figure 3: CSI Delay vs RadioMap Freshness"
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
        "--baseline-delays",
        default=",".join(map(str, BASELINE_CSI_DELAYS)),
        help=f"Comma-separated baseline CSI delays (default: {BASELINE_CSI_DELAYS})"
    )
    parser.add_argument(
        "--rm-delays",
        default=",".join(map(str, RM_CSI_DELAYS)),
        help=f"Comma-separated RM CSI delays (default: {RM_CSI_DELAYS})"
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
        default="output/results/fig03_csi_delay.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    args = parser.parse_args()

    # Parse delay values
    baseline_delays = [int(x) for x in args.baseline_delays.split(",")]
    rm_delays = [int(x) for x in args.rm_delays.split(",")]

    # Validate model paths
    model_mlp = Path(args.model_mlp)
    model_isab = Path(args.model_isab)

    if not model_mlp.exists():
        print(f"Warning: MLP model not found: {model_mlp}")
    if not model_isab.exists():
        print(f"Warning: ISAB model not found: {model_isab}")

    # Load configs
    base_cfg = load_base_config()
    scenario_cfg = load_config_from_file(SCRIPT_DIR / args.scenario_path)

    # Calculate total runs
    # Part (a): B1_3GPP across baseline delays + B2_Oracle (once)
    # Part (b): P1_MLP and P2_ISAB across RM delays
    n_baseline_runs = len(baseline_delays) * args.seeds  # B1_3GPP
    n_oracle_runs = 1 * args.seeds  # B2_Oracle (delay=0 only)
    n_rm_runs = len(rm_delays) * 2 * args.seeds  # P1_MLP + P2_ISAB

    total_runs = n_baseline_runs + n_oracle_runs + n_rm_runs

    print(f"\nFigure 3: CSI Delay Experiment")
    print(f"  Scenario: {args.scenario_path}")
    print(f"  Baseline CSI delays: {baseline_delays}")
    print(f"  RM CSI delays: {rm_delays}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total runs: {total_runs}\n")

    rows = []
    pbar = tqdm(total=total_runs, desc="Fig03", disable=args.no_progress,
                bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
        # Part (a): Baseline CSI delay sweep
        for delay in baseline_delays:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                cfg = base_cfg.copy()
                cfg.update(scenario_cfg)

                if args.T is not None:
                    cfg["T"] = args.T
                if args.N_UE is not None:
                    cfg["N_UE"] = args.N_UE
                cfg["seed"] = seed
                cfg["show_progress"] = False

                pbar.set_description(f"Fig03 [B1/delay={delay}/s{seed}]")

                try:
                    result = run_experiment(cfg, "B1_3GPP", model_mlp, model_isab,
                                           "baseline_csi", delay)
                    rows.append({
                        "method": "B1_3GPP",
                        "delay_type": "baseline_csi",
                        "delay_value": delay,
                        "seed": seed,
                        "avg_se": result["avg_se_baseline"],  # Use baseline SE for B1
                        "avg_se_baseline": result["avg_se_baseline"],
                        "gain_pct": 0.0,  # B1 is the reference
                    })
                    pbar.set_postfix(se=f"{result['avg_se_baseline']:.4f}")
                except Exception as e:
                    print(f"\nError B1/delay={delay}/seed={seed}: {e}")

                pbar.update(1)

        # Oracle CSI (B2): delay=0 only
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            cfg = base_cfg.copy()
            cfg.update(scenario_cfg)

            if args.T is not None:
                cfg["T"] = args.T
            if args.N_UE is not None:
                cfg["N_UE"] = args.N_UE
            cfg["seed"] = seed
            cfg["show_progress"] = False

            pbar.set_description(f"Fig03 [B2_Oracle/s{seed}]")

            try:
                result = run_experiment(cfg, "B2_Oracle", model_mlp, model_isab,
                                       "baseline_csi", 0)
                rows.append({
                    "method": "B2_Oracle",
                    "delay_type": "baseline_csi",
                    "delay_value": 0,
                    "seed": seed,
                    "avg_se": result["avg_se_baseline"],
                    "avg_se_baseline": result["avg_se_baseline"],
                    "gain_pct": 0.0,
                })
                pbar.set_postfix(se=f"{result['avg_se_baseline']:.4f}")
            except Exception as e:
                print(f"\nError B2_Oracle/seed={seed}: {e}")

            pbar.update(1)

        # Part (b): RadioMap delay sweep
        for method in ["P1_MLP", "P2_ISAB"]:
            model_path = model_mlp if method == "P1_MLP" else model_isab
            if not model_path.exists():
                print(f"\nSkipping {method}: model not found")
                pbar.update(len(rm_delays) * args.seeds)
                continue

            for delay in rm_delays:
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    cfg = base_cfg.copy()
                    cfg.update(scenario_cfg)

                    if args.T is not None:
                        cfg["T"] = args.T
                    if args.N_UE is not None:
                        cfg["N_UE"] = args.N_UE
                    cfg["seed"] = seed
                    cfg["show_progress"] = False

                    pbar.set_description(f"Fig03 [{method[:4]}/rm_delay={delay}/s{seed}]")

                    try:
                        result = run_experiment(cfg, method, model_mlp, model_isab,
                                               "rm_csi", delay)
                        rows.append({
                            "method": method,
                            "delay_type": "rm_csi",
                            "delay_value": delay,
                            "seed": seed,
                            "avg_se": result["avg_se"],
                            "avg_se_baseline": result["avg_se_baseline"],
                            "gain_pct": result["gain_pct"],
                        })
                        pbar.set_postfix(
                            se=f"{result['avg_se']:.4f}",
                            gain=f"{result['gain_pct']:+.2f}%"
                        )
                    except Exception as e:
                        print(f"\nError {method}/rm_delay={delay}/seed={seed}: {e}")

                    pbar.update(1)

    finally:
        pbar.close()

    # Write results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "delay_type", "delay_value", "seed", "avg_se", "avg_se_baseline", "gain_pct"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path}")
    print(f"  Total rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
