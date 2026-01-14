#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 4: Robustness to RadioMap Imperfections.

Goal: Show graceful degradation under RadioMap estimation error and spatial blur.

Supports parallel execution via --parallel flag (default: enabled).
"""

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = SCRIPT_DIR / "code"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CODE_DIR))

DEFAULT_SCENARIO_PATH = "test/config_toronto_single.py"
DEFAULT_MLP_MODEL = "output/models/nsgbs_scorer.pt"
DEFAULT_ISAB_MODEL = "output/models/nsgbs_isab_tau0.2.pt"
EST_ERROR_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
BLUR_SIGMA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]


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
    parser = argparse.ArgumentParser(description="Figure 4: RadioMap Imperfections")
    parser.add_argument("--scenario-path", default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--N_UE", type=int, default=None)
    parser.add_argument("--est-errors", default=",".join(map(str, EST_ERROR_VALUES)))
    parser.add_argument("--blur-sigmas", default=",".join(map(str, BLUR_SIGMA_VALUES)))
    parser.add_argument("--methods", default="P1_MLP,P2_ISAB")
    parser.add_argument("--model-mlp", default=DEFAULT_MLP_MODEL)
    parser.add_argument("--model-isab", default=DEFAULT_ISAB_MODEL)
    parser.add_argument("--out", default="output/results/fig04_rm_imperfections.csv")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    est_errors = [float(x) for x in args.est_errors.split(",")]
    blur_sigmas = [float(x) for x in args.blur_sigmas.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    model_mlp = SCRIPT_DIR / args.model_mlp
    model_isab = SCRIPT_DIR / args.model_isab

    base_cfg = load_base_config()
    scenario_cfg = load_config_from_file(SCRIPT_DIR / args.scenario_path)

    extra_cfg = {}
    if args.T is not None:
        extra_cfg["T"] = args.T
    if args.N_UE is not None:
        extra_cfg["N_UE"] = args.N_UE

    seed_range = range(args.seed_start, args.seed_start + args.seeds)
    total_runs = (len(est_errors) + len(blur_sigmas)) * len(methods) * args.seeds

    print(f"\nFigure 4: RadioMap Imperfections Experiment")
    print(f"  Est errors: {est_errors}")
    print(f"  Blur sigmas: {blur_sigmas}")
    print(f"  Total runs: {total_runs}")

    use_parallel = args.parallel and not args.sequential
    rows = []

    if use_parallel:
        from parallel_runner import ParallelExperimentRunner, ParallelConfig, RunSpec, get_optimal_workers

        workers = args.workers or get_optimal_workers()
        print(f"  Mode: PARALLEL ({workers} workers)\n")

        scenario_configs = {"default": scenario_cfg}
        run_specs = []

        for est_err in est_errors:
            for method in methods:
                for seed in seed_range:
                    params = extra_cfg.copy()
                    params["radiomap_est_error_db"] = est_err
                    params["_type"] = "est_error_db"
                    params["_value"] = est_err
                    run_specs.append(RunSpec(scenario="default", method=method, seed=seed, extra_params=params))

        for blur in blur_sigmas:
            for method in methods:
                for seed in seed_range:
                    params = extra_cfg.copy()
                    params["radiomap_blur_sigma"] = blur
                    params["_type"] = "blur_sigma"
                    params["_value"] = blur
                    run_specs.append(RunSpec(scenario="default", method=method, seed=seed, extra_params=params))

        pbar = tqdm(total=len(run_specs), desc="Fig04", disable=args.no_progress,
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
            if r.get("success"):
                res = r["result"]
                rows.append({
                    "method": r["method"],
                    "imperfection_type": spec.extra_params.get("_type", ""),
                    "imperfection_value": spec.extra_params.get("_value", 0),
                    "seed": r["seed"],
                    "avg_se": res.get("avg_se_radiomap", 0.0),
                    "avg_se_baseline": res.get("avg_se_baseline_default", 0.0),
                    "gain_pct": res.get("improvement_vs_default_pct", 0.0),
                })
    else:
        print(f"  Mode: SEQUENTIAL\n")
        from main import run_once

        pbar = tqdm(total=total_runs, desc="Fig04", disable=args.no_progress,
                    bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        try:
            for est_err in est_errors:
                for method in methods:
                    for seed in seed_range:
                        cfg = base_cfg.copy()
                        cfg.update(scenario_cfg)
                        cfg.update(extra_cfg)
                        cfg["seed"] = seed
                        cfg["show_progress"] = False
                        cfg["radiomap_est_error_db"] = est_err
                        cfg["scheduler_kind"] = "nsgbs"
                        cfg["nsgbs_model_path"] = str(model_mlp if method == "P1_MLP" else model_isab)

                        pbar.set_description(f"Fig04 [err={est_err}/{method[:4]}/s{seed}]")
                        try:
                            result = run_once(cfg)
                            rows.append({
                                "method": method, "imperfection_type": "est_error_db",
                                "imperfection_value": est_err, "seed": seed,
                                "avg_se": result.get("avg_se_radiomap", 0.0),
                                "avg_se_baseline": result.get("avg_se_baseline_default", 0.0),
                                "gain_pct": result.get("improvement_vs_default_pct", 0.0)
                            })
                        except Exception as e:
                            print(f"\nError est_err={est_err}/{method}/seed={seed}: {e}")
                        pbar.update(1)

            for blur in blur_sigmas:
                for method in methods:
                    for seed in seed_range:
                        cfg = base_cfg.copy()
                        cfg.update(scenario_cfg)
                        cfg.update(extra_cfg)
                        cfg["seed"] = seed
                        cfg["show_progress"] = False
                        cfg["radiomap_blur_sigma"] = blur
                        cfg["scheduler_kind"] = "nsgbs"
                        cfg["nsgbs_model_path"] = str(model_mlp if method == "P1_MLP" else model_isab)

                        pbar.set_description(f"Fig04 [blur={blur}/{method[:4]}/s{seed}]")
                        try:
                            result = run_once(cfg)
                            rows.append({
                                "method": method, "imperfection_type": "blur_sigma",
                                "imperfection_value": blur, "seed": seed,
                                "avg_se": result.get("avg_se_radiomap", 0.0),
                                "avg_se_baseline": result.get("avg_se_baseline_default", 0.0),
                                "gain_pct": result.get("improvement_vs_default_pct", 0.0)
                            })
                        except Exception as e:
                            print(f"\nError blur={blur}/{method}/seed={seed}: {e}")
                        pbar.update(1)
        finally:
            pbar.close()

    out_path = SCRIPT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "imperfection_type", "imperfection_value", "seed", "avg_se", "avg_se_baseline", "gain_pct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
