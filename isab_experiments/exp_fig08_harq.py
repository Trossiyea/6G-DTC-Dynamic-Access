#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 8: Reliability & HARQ Goodput.

Goal: Verify HARQ realism and show that gains aren't from unrealistic BLER/ACK artifacts.

Collects HARQ statistics from standard runs:
- ACK rate, TB drop rate, avg retx per acked TB
- OLLA offset evolution

Outputs CSV with columns:
  method, seed, ack_rate, init_ack_rate, drop_rate, avg_retx, olla_offset_avg
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


def extract_harq_metrics(harq_stats):
    """
    Extract key metrics from HARQ stats dict.

    Args:
        harq_stats: Dict from harq_manager.get_stats()

    Returns:
        Dict with computed metrics
    """
    if harq_stats is None:
        return {
            "ack_rate": 0.0,
            "init_ack_rate": 0.0,
            "drop_rate": 0.0,
            "avg_retx": 0.0,
            "olla_offset_avg": 0.0,
        }

    ack_count = harq_stats.get("ack_count", 0)
    nack_count = harq_stats.get("nack_count", 0)
    total_ack_nack = ack_count + nack_count

    init_ack = harq_stats.get("initial_ack_count", 0)
    init_nack = harq_stats.get("initial_nack_count", 0)
    total_init = init_ack + init_nack

    tb_started = harq_stats.get("tb_started", 0)
    tb_acked = harq_stats.get("tb_acked", 0)
    tb_dropped = harq_stats.get("tb_dropped", 0)

    avg_retx = harq_stats.get("avg_retx_per_acked", 0.0)
    olla_hist = harq_stats.get("olla_offset_avg", [])

    # Compute rates
    ack_rate = ack_count / total_ack_nack if total_ack_nack > 0 else 0.0
    init_ack_rate = init_ack / total_init if total_init > 0 else 0.0
    drop_rate = tb_dropped / tb_started if tb_started > 0 else 0.0

    # Average OLLA offset (last value or mean of history)
    if olla_hist:
        olla_offset_avg = float(np.mean(olla_hist[-10:]))  # Last 10 values
    else:
        olla_last = harq_stats.get("olla_last_per_ue", [])
        olla_offset_avg = float(np.mean(olla_last)) if olla_last else 0.0

    return {
        "ack_rate": float(ack_rate),
        "init_ack_rate": float(init_ack_rate),
        "drop_rate": float(drop_rate),
        "avg_retx": float(avg_retx),
        "olla_offset_avg": float(olla_offset_avg),
        "tb_started": int(tb_started),
        "tb_acked": int(tb_acked),
        "tb_dropped": int(tb_dropped),
    }


def run_experiment(cfg, method, model_mlp, model_isab):
    """
    Configure and run a single experiment.

    Returns:
        Dict with HARQ metrics for baseline and radiomap
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

    # Ensure HARQ is enabled
    cfg["enable_harq_full"] = True

    result = run_once(cfg)

    # Extract HARQ stats
    harq_base = result.get("harq_stats_base")
    harq_map = result.get("harq_stats_map")

    # Use appropriate stats based on method
    if method == "B1_3GPP":
        harq_stats = harq_base
    else:
        harq_stats = harq_map

    metrics = extract_harq_metrics(harq_stats)
    metrics["avg_se"] = float(result.get("avg_se_radiomap", 0.0) if method != "B1_3GPP"
                              else result.get("avg_se_baseline_default", 0.0))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Figure 8: Reliability & HARQ Goodput"
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
        "--methods",
        default="B1_3GPP,B3_heuristic,P1_MLP,P2_ISAB",
        help="Comma-separated methods"
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
        default="output/results/fig08_harq.csv",
        help="Output CSV path"
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
        print(f"Warning: MLP model not found: {model_mlp}")
    if "P2_ISAB" in methods and not model_isab.exists():
        print(f"Warning: ISAB model not found: {model_isab}")

    # Load configs
    base_cfg = load_base_config()
    scenario_cfg = load_config_from_file(SCRIPT_DIR / args.scenario_path)

    # Calculate total runs
    total_runs = len(methods) * args.seeds

    print(f"\nFigure 8: HARQ Reliability Experiment")
    print(f"  Scenario: {args.scenario_path}")
    print(f"  Methods: {methods}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total runs: {total_runs}\n")

    rows = []
    pbar = tqdm(total=total_runs, desc="Fig08", disable=args.no_progress,
                bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
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

                pbar.set_description(f"Fig08 [{method}/{seed}]")

                try:
                    result = run_experiment(cfg, method, model_mlp, model_isab)
                    rows.append({
                        "method": method,
                        "seed": seed,
                        "avg_se": result["avg_se"],
                        "ack_rate": result["ack_rate"],
                        "init_ack_rate": result["init_ack_rate"],
                        "drop_rate": result["drop_rate"],
                        "avg_retx": result["avg_retx"],
                        "olla_offset_avg": result["olla_offset_avg"],
                        "tb_started": result.get("tb_started", 0),
                        "tb_acked": result.get("tb_acked", 0),
                        "tb_dropped": result.get("tb_dropped", 0),
                    })
                    pbar.set_postfix(
                        ack=f"{result['ack_rate']:.3f}",
                        drop=f"{result['drop_rate']:.4f}"
                    )
                except Exception as e:
                    print(f"\nError {method}/seed={seed}: {e}")

                pbar.update(1)

    finally:
        pbar.close()

    # Write results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "seed", "avg_se", "ack_rate", "init_ack_rate", "drop_rate",
                  "avg_retx", "olla_offset_avg", "tb_started", "tb_acked", "tb_dropped"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path}")
    print(f"  Total rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
