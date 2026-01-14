#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 6: Scalability With UE Load and Bandwidth.

Goal: Show performance under varying load and scheduling hardness.

Supports parallel execution via --parallel flag (default: enabled).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = SCRIPT_DIR / "code"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CODE_DIR))

DEFAULT_SCENARIO_PATH = "test/config_toronto_single.py"
DEFAULT_MLP_MODEL = "output/models/nsgbs_scorer.pt"
DEFAULT_ISAB_MODEL = "output/models/nsgbs_isab_tau0.2.pt"
N_UE_VALUES = [20, 50, 100, 150, 200]


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


def main():
    parser = argparse.ArgumentParser(description="Figure 6: Scalability")
    parser.add_argument("--scenario-path", default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--n-ue-values", default=",".join(map(str, N_UE_VALUES)))
    parser.add_argument("--methods", default="B1_3GPP,P1_MLP,P2_ISAB")
    parser.add_argument("--model-mlp", default=DEFAULT_MLP_MODEL)
    parser.add_argument("--model-isab", default=DEFAULT_ISAB_MODEL)
    parser.add_argument("--out", default="output/results/fig06_scalability.csv")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    n_ue_values = [int(x) for x in args.n_ue_values.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    model_mlp = SCRIPT_DIR / args.model_mlp
    model_isab = SCRIPT_DIR / args.model_isab

    base_cfg = load_base_config()
    scenario_cfg = load_config_from_file(SCRIPT_DIR / args.scenario_path)
    default_z = scenario_cfg.get("Z", base_cfg.get("Z", 51))

    extra_cfg = {}
    if args.T is not None:
        extra_cfg["T"] = args.T

    seed_range = range(args.seed_start, args.seed_start + args.seeds)
    total_runs = len(n_ue_values) * len(methods) * args.seeds

    print(f"\nFigure 6: Scalability Experiment")
    print(f"  N_UE values: {n_ue_values}")
    print(f"  Methods: {methods}")
    print(f"  Total runs: {total_runs}")

    use_parallel = args.parallel and not args.sequential
    rows = []

    if use_parallel:
        from parallel_runner import ParallelExperimentRunner, ParallelConfig, RunSpec, get_optimal_workers

        workers = args.workers or get_optimal_workers()
        print(f"  Mode: PARALLEL ({workers} workers)\n")

        scenario_configs = {"default": scenario_cfg}
        run_specs = []

        for n_ue in n_ue_values:
            for method in methods:
                for seed in seed_range:
                    params = extra_cfg.copy()
                    params["N_UE"] = n_ue
                    params["_n_ue"] = n_ue
                    run_specs.append(RunSpec(scenario="default", method=method, seed=seed, extra_params=params))

        pbar = tqdm(total=len(run_specs), desc="Fig06", disable=args.no_progress,
                    bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        runner = ParallelExperimentRunner(ParallelConfig(max_workers=workers))
        try:
            results = runner.run_batch(
                run_specs, base_cfg, scenario_configs,
                {"mlp": str(model_mlp), "isab": str(model_isab)},
                progress_callback=lambda c, t: (setattr(pbar, 'n', c), pbar.refresh())
            )
        finally:
            pbar.close()

        for i, r in enumerate(results):
            spec = run_specs[i]
            n_ue = spec.extra_params.get("_n_ue", 100)
            if r.get("success"):
                res = r["result"]
                is_baseline = r["method"] == "B1_3GPP"
                per_ue = res.get("per_ue_throughput_baseline_bps" if is_baseline else "per_ue_throughput_radiomap_bps", [])
                per_ue = np.asarray(per_ue)
                pctl_5 = float(np.percentile(per_ue, 5)) if len(per_ue) > 0 else 0.0
                median = float(np.percentile(per_ue, 50)) if len(per_ue) > 0 else 0.0
                rows.append({
                    "method": r["method"], "N_UE": n_ue, "Z": default_z, "seed": r["seed"],
                    "avg_se": res.get("avg_se_baseline_default", 0.0) if is_baseline else res.get("avg_se_radiomap", 0.0),
                    "avg_se_baseline": res.get("avg_se_baseline_default", 0.0),
                    "gain_pct": 0.0 if is_baseline else res.get("improvement_vs_default_pct", 0.0),
                    "pctl_5_tput": pctl_5, "median_tput": median
                })
    else:
        print(f"  Mode: SEQUENTIAL\n")
        from main import run_once

        pbar = tqdm(total=total_runs, desc="Fig06", disable=args.no_progress,
                    bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        try:
            for n_ue in n_ue_values:
                for method in methods:
                    for seed in seed_range:
                        cfg = base_cfg.copy()
                        cfg.update(scenario_cfg)
                        cfg.update(extra_cfg)
                        cfg["N_UE"] = n_ue
                        cfg["seed"] = seed
                        cfg["show_progress"] = False

                        if method == "B1_3GPP":
                            cfg["scheduler_kind"] = "heuristic"
                            cfg["nsgbs_model_path"] = None
                        elif method == "P1_MLP":
                            cfg["scheduler_kind"] = "nsgbs"
                            cfg["nsgbs_model_path"] = str(model_mlp)
                        elif method == "P2_ISAB":
                            cfg["scheduler_kind"] = "nsgbs"
                            cfg["nsgbs_model_path"] = str(model_isab)

                        pbar.set_description(f"Fig06 [N={n_ue}/{method[:4]}/s{seed}]")
                        try:
                            result = run_once(cfg)
                            is_baseline = method == "B1_3GPP"
                            per_ue = result.get("per_ue_throughput_baseline_bps" if is_baseline else "per_ue_throughput_radiomap_bps", [])
                            per_ue = np.asarray(per_ue)
                            pctl_5 = float(np.percentile(per_ue, 5)) if len(per_ue) > 0 else 0.0
                            median = float(np.percentile(per_ue, 50)) if len(per_ue) > 0 else 0.0
                            rows.append({
                                "method": method, "N_UE": n_ue, "Z": default_z, "seed": seed,
                                "avg_se": result.get("avg_se_baseline_default", 0.0) if is_baseline else result.get("avg_se_radiomap", 0.0),
                                "avg_se_baseline": result.get("avg_se_baseline_default", 0.0),
                                "gain_pct": 0.0 if is_baseline else result.get("improvement_vs_default_pct", 0.0),
                                "pctl_5_tput": pctl_5, "median_tput": median
                            })
                        except Exception as e:
                            print(f"\nError N_UE={n_ue}/{method}/seed={seed}: {e}")
                        pbar.update(1)
        finally:
            pbar.close()

    out_path = SCRIPT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "N_UE", "Z", "seed", "avg_se", "avg_se_baseline", "gain_pct", "pctl_5_tput", "median_tput"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
