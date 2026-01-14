#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 2: User-Level Throughput Distribution & Fairness.

Goal: Show cell-edge improvements and fairness across methods.

Supports parallel execution via --parallel flag (default: enabled).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = SCRIPT_DIR / "code"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CODE_DIR))

DEFAULT_SCENARIO_PATH = "test/config_toronto_single.py"
DEFAULT_MLP_MODEL = "output/models/nsgbs_scorer.pt"
DEFAULT_ISAB_MODEL = "output/models/nsgbs_isab_tau0.2.pt"


def load_config_from_file(config_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("scenario_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def load_base_config():
    import importlib.util
    config_path = SCRIPT_DIR / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("base_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def compute_jain_index(values):
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return 0.0
    sum_x = np.sum(values)
    sum_x2 = np.sum(values ** 2)
    if sum_x2 == 0:
        return 1.0
    return (sum_x ** 2) / (n * sum_x2)


def process_result(result, method):
    """Extract fairness metrics from run result."""
    if method == "B1_3GPP":
        per_ue_tput = result.get("per_ue_throughput_baseline_bps", [])
        jain = result.get("fairness_jain_base", 0.0)
    else:
        per_ue_tput = result.get("per_ue_throughput_radiomap_bps", [])
        jain = result.get("fairness_jain_map", 0.0)

    per_ue_tput = np.asarray(per_ue_tput)
    if jain == 0.0 and len(per_ue_tput) > 0:
        jain = compute_jain_index(per_ue_tput)

    if len(per_ue_tput) > 0:
        pctl_5 = float(np.percentile(per_ue_tput, 5))
        pctl_10 = float(np.percentile(per_ue_tput, 10))
        median = float(np.percentile(per_ue_tput, 50))
        mean = float(np.mean(per_ue_tput))
    else:
        pctl_5 = pctl_10 = median = mean = 0.0

    return {
        "per_ue_throughput": per_ue_tput.tolist(),
        "jain_index": float(jain),
        "pctl_5": pctl_5,
        "pctl_10": pctl_10,
        "median": median,
        "mean": mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Figure 2: User Fairness")
    parser.add_argument("--scenario-path", default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--N_UE", type=int, default=100)
    parser.add_argument("--methods", default="B1_3GPP,B3_heuristic,P1_MLP,P2_ISAB")
    parser.add_argument("--model-mlp", default=DEFAULT_MLP_MODEL)
    parser.add_argument("--model-isab", default=DEFAULT_ISAB_MODEL)
    parser.add_argument("--out-per-ue", default="output/results/fig02_user_fairness_per_ue.csv")
    parser.add_argument("--out-summary", default="output/results/fig02_user_fairness_summary.csv")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    model_mlp = SCRIPT_DIR / args.model_mlp
    model_isab = SCRIPT_DIR / args.model_isab

    base_cfg = load_base_config()
    scenario_cfg = load_config_from_file(SCRIPT_DIR / args.scenario_path)

    extra_cfg = {"N_UE": args.N_UE}
    if args.T is not None:
        extra_cfg["T"] = args.T

    seed_range = range(args.seed_start, args.seed_start + args.seeds)
    total_runs = len(methods) * args.seeds

    print(f"\nFigure 2: User Fairness Experiment")
    print(f"  Methods: {methods}")
    print(f"  N_UE: {args.N_UE}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total runs: {total_runs}")

    use_parallel = args.parallel and not args.sequential
    per_ue_rows = []
    summary_rows = []

    if use_parallel:
        from parallel_runner import ParallelExperimentRunner, ParallelConfig, RunSpec, get_optimal_workers

        workers = args.workers or get_optimal_workers()
        print(f"  Mode: PARALLEL ({workers} workers)\n")

        scenario_configs = {"default": scenario_cfg}
        run_specs = [
            RunSpec(scenario="default", method=m, seed=s, extra_params=extra_cfg.copy())
            for m in methods for s in seed_range
        ]

        pbar = tqdm(total=total_runs, desc="Fig02", disable=args.no_progress,
                    bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        def progress_cb(c, t):
            pbar.n = c
            pbar.refresh()

        runner = ParallelExperimentRunner(ParallelConfig(max_workers=workers))
        try:
            results = runner.run_batch(
                run_specs, base_cfg, scenario_configs,
                {"mlp": str(model_mlp), "isab": str(model_isab)},
                progress_callback=progress_cb
            )
        finally:
            pbar.close()

        for r in results:
            if r.get("success"):
                res = r["result"]
                metrics = process_result(res, r["method"])
                for ue_id, tput in enumerate(metrics["per_ue_throughput"]):
                    per_ue_rows.append({"method": r["method"], "seed": r["seed"], "ue_id": ue_id, "throughput_bps": tput})
                summary_rows.append({
                    "method": r["method"], "seed": r["seed"],
                    "jain_index": metrics["jain_index"], "pctl_5": metrics["pctl_5"],
                    "pctl_10": metrics["pctl_10"], "median": metrics["median"], "mean": metrics["mean"]
                })
    else:
        print(f"  Mode: SEQUENTIAL\n")
        from main import run_once

        pbar = tqdm(total=total_runs, desc="Fig02", disable=args.no_progress,
                    bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        try:
            for method in methods:
                for seed in seed_range:
                    cfg = base_cfg.copy()
                    cfg.update(scenario_cfg)
                    cfg.update(extra_cfg)
                    cfg["seed"] = seed
                    cfg["show_progress"] = False

                    if method in ("B1_3GPP", "B3_heuristic"):
                        cfg["scheduler_kind"] = "heuristic"
                        cfg["nsgbs_model_path"] = None
                    elif method == "P1_MLP":
                        cfg["scheduler_kind"] = "nsgbs"
                        cfg["nsgbs_model_path"] = str(model_mlp)
                    elif method == "P2_ISAB":
                        cfg["scheduler_kind"] = "nsgbs"
                        cfg["nsgbs_model_path"] = str(model_isab)

                    pbar.set_description(f"Fig02 [{method}/{seed}]")
                    try:
                        result = run_once(cfg)
                        metrics = process_result(result, method)
                        for ue_id, tput in enumerate(metrics["per_ue_throughput"]):
                            per_ue_rows.append({"method": method, "seed": seed, "ue_id": ue_id, "throughput_bps": tput})
                        summary_rows.append({
                            "method": method, "seed": seed,
                            "jain_index": metrics["jain_index"], "pctl_5": metrics["pctl_5"],
                            "pctl_10": metrics["pctl_10"], "median": metrics["median"], "mean": metrics["mean"]
                        })
                        pbar.set_postfix(jain=f"{metrics['jain_index']:.4f}")
                    except Exception as e:
                        print(f"\nError {method}/seed={seed}: {e}")
                    pbar.update(1)
        finally:
            pbar.close()

    # Write outputs
    out_per_ue = SCRIPT_DIR / args.out_per_ue
    out_per_ue.parent.mkdir(parents=True, exist_ok=True)
    with open(out_per_ue, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seed", "ue_id", "throughput_bps"])
        writer.writeheader()
        writer.writerows(per_ue_rows)

    out_summary = SCRIPT_DIR / args.out_summary
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seed", "jain_index", "pctl_5", "pctl_10", "median", "mean"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved per-UE data to {out_per_ue} ({len(per_ue_rows)} rows)")
    print(f"Saved summary to {out_summary} ({len(summary_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
