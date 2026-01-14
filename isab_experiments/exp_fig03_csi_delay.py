#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 3: CSI Delay vs RadioMap Freshness.

Goal: Justify the information advantage of RadioMap over CSI-based scheduling.

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
BASELINE_CSI_DELAYS = [0, 5, 10, 20, 40]
RM_CSI_DELAYS = [0, 2, 5, 10, 20]


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
    parser = argparse.ArgumentParser(description="Figure 3: CSI Delay vs RadioMap Freshness")
    parser.add_argument("--scenario-path", default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--N_UE", type=int, default=None)
    parser.add_argument("--baseline-delays", default=",".join(map(str, BASELINE_CSI_DELAYS)))
    parser.add_argument("--rm-delays", default=",".join(map(str, RM_CSI_DELAYS)))
    parser.add_argument("--model-mlp", default=DEFAULT_MLP_MODEL)
    parser.add_argument("--model-isab", default=DEFAULT_ISAB_MODEL)
    parser.add_argument("--out", default="output/results/fig03_csi_delay.csv")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    baseline_delays = [int(x) for x in args.baseline_delays.split(",")]
    rm_delays = [int(x) for x in args.rm_delays.split(",")]
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
    n_baseline = len(baseline_delays) * args.seeds
    n_oracle = args.seeds
    n_rm = len(rm_delays) * 2 * args.seeds
    total_runs = n_baseline + n_oracle + n_rm

    print(f"\nFigure 3: CSI Delay Experiment")
    print(f"  Baseline delays: {baseline_delays}")
    print(f"  RM delays: {rm_delays}")
    print(f"  Total runs: {total_runs}")

    use_parallel = args.parallel and not args.sequential
    rows = []

    if use_parallel:
        from parallel_runner import ParallelExperimentRunner, ParallelConfig, RunSpec, get_optimal_workers

        workers = args.workers or get_optimal_workers()
        print(f"  Mode: PARALLEL ({workers} workers)\n")

        scenario_configs = {"default": scenario_cfg}
        run_specs = []

        # B1_3GPP baseline delay sweep
        for delay in baseline_delays:
            for seed in seed_range:
                params = extra_cfg.copy()
                params["baseline_csi_delay_ttis"] = delay
                params["_delay_type"] = "baseline_csi"
                params["_delay_value"] = delay
                run_specs.append(RunSpec(scenario="default", method="B1_3GPP", seed=seed, extra_params=params))

        # B2_Oracle (delay=0)
        for seed in seed_range:
            params = extra_cfg.copy()
            params["baseline_csi_delay_ttis"] = 0
            params["_delay_type"] = "baseline_csi"
            params["_delay_value"] = 0
            run_specs.append(RunSpec(scenario="default", method="B2_Oracle", seed=seed, extra_params=params))

        # RM delay sweep for P1_MLP and P2_ISAB
        for method in ["P1_MLP", "P2_ISAB"]:
            for delay in rm_delays:
                for seed in seed_range:
                    params = extra_cfg.copy()
                    params["rm_csi_delay_ttis"] = delay
                    params["_delay_type"] = "rm_csi"
                    params["_delay_value"] = delay
                    run_specs.append(RunSpec(scenario="default", method=method, seed=seed, extra_params=params))

        pbar = tqdm(total=len(run_specs), desc="Fig03", disable=args.no_progress,
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
            delay_type = spec.extra_params.get("_delay_type", "")
            delay_value = spec.extra_params.get("_delay_value", 0)
            if r.get("success"):
                res = r["result"]
                is_baseline = r["method"] in ("B1_3GPP", "B2_Oracle")
                rows.append({
                    "method": r["method"],
                    "delay_type": delay_type,
                    "delay_value": delay_value,
                    "seed": r["seed"],
                    "avg_se": res.get("avg_se_baseline_default", 0.0) if is_baseline else res.get("avg_se_radiomap", 0.0),
                    "avg_se_baseline": res.get("avg_se_baseline_default", 0.0),
                    "gain_pct": 0.0 if is_baseline else res.get("improvement_vs_default_pct", 0.0),
                })
    else:
        print(f"  Mode: SEQUENTIAL\n")
        from main import run_once

        pbar = tqdm(total=total_runs, desc="Fig03", disable=args.no_progress,
                    bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        try:
            # B1_3GPP baseline delay sweep
            for delay in baseline_delays:
                for seed in seed_range:
                    cfg = base_cfg.copy()
                    cfg.update(scenario_cfg)
                    cfg.update(extra_cfg)
                    cfg["seed"] = seed
                    cfg["show_progress"] = False
                    cfg["scheduler_kind"] = "heuristic"
                    cfg["nsgbs_model_path"] = None
                    cfg["baseline_csi_delay_ttis"] = delay

                    pbar.set_description(f"Fig03 [B1/delay={delay}/s{seed}]")
                    try:
                        result = run_once(cfg)
                        rows.append({
                            "method": "B1_3GPP", "delay_type": "baseline_csi", "delay_value": delay,
                            "seed": seed, "avg_se": result.get("avg_se_baseline_default", 0.0),
                            "avg_se_baseline": result.get("avg_se_baseline_default", 0.0), "gain_pct": 0.0
                        })
                    except Exception as e:
                        print(f"\nError B1/delay={delay}/seed={seed}: {e}")
                    pbar.update(1)

            # B2_Oracle
            for seed in seed_range:
                cfg = base_cfg.copy()
                cfg.update(scenario_cfg)
                cfg.update(extra_cfg)
                cfg["seed"] = seed
                cfg["show_progress"] = False
                cfg["scheduler_kind"] = "heuristic"
                cfg["nsgbs_model_path"] = None
                cfg["baseline_csi_delay_ttis"] = 0

                pbar.set_description(f"Fig03 [B2_Oracle/s{seed}]")
                try:
                    result = run_once(cfg)
                    rows.append({
                        "method": "B2_Oracle", "delay_type": "baseline_csi", "delay_value": 0,
                        "seed": seed, "avg_se": result.get("avg_se_baseline_default", 0.0),
                        "avg_se_baseline": result.get("avg_se_baseline_default", 0.0), "gain_pct": 0.0
                    })
                except Exception as e:
                    print(f"\nError B2_Oracle/seed={seed}: {e}")
                pbar.update(1)

            # RM delay sweep
            for method in ["P1_MLP", "P2_ISAB"]:
                model_path = model_mlp if method == "P1_MLP" else model_isab
                if not model_path.exists():
                    pbar.update(len(rm_delays) * args.seeds)
                    continue
                for delay in rm_delays:
                    for seed in seed_range:
                        cfg = base_cfg.copy()
                        cfg.update(scenario_cfg)
                        cfg.update(extra_cfg)
                        cfg["seed"] = seed
                        cfg["show_progress"] = False
                        cfg["scheduler_kind"] = "nsgbs"
                        cfg["nsgbs_model_path"] = str(model_path)
                        cfg["rm_csi_delay_ttis"] = delay

                        pbar.set_description(f"Fig03 [{method[:4]}/rm={delay}/s{seed}]")
                        try:
                            result = run_once(cfg)
                            rows.append({
                                "method": method, "delay_type": "rm_csi", "delay_value": delay,
                                "seed": seed, "avg_se": result.get("avg_se_radiomap", 0.0),
                                "avg_se_baseline": result.get("avg_se_baseline_default", 0.0),
                                "gain_pct": result.get("improvement_vs_default_pct", 0.0)
                            })
                        except Exception as e:
                            print(f"\nError {method}/rm={delay}/seed={seed}: {e}")
                        pbar.update(1)
        finally:
            pbar.close()

    out_path = SCRIPT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "delay_type", "delay_value", "seed", "avg_se", "avg_se_baseline", "gain_pct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
