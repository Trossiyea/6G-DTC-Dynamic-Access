#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run NS-GBS benchmarks across scenarios/seeds and compare heuristic/MLP/ISAB.
"""

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = SCRIPT_DIR / "code"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


SCENARIOS = {
    "toronto_single": "test/config_toronto_single.py",
    "toronto_constellation": "test/config_toronto_constellation.py",
    "shanghai_single": "test/config_shanghai_single.py",
    "shanghai_constellation": "test/config_shanghai_constellation.py",
    "toronto_125m": "test/config_toronto_single_125m.py",
    "toronto_150m": "test/config_toronto_single_150m.py",
    "shanghai_125m": "test/config_shanghai_single_125m.py",
    "shanghai_150m": "test/config_shanghai_single_150m.py",
}


def load_config_from_file(config_path: Path):
    spec = importlib.util.spec_from_file_location("bench_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def load_base_config():
    config_path = SCRIPT_DIR / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("base_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def parse_modes(modes_str: str):
    modes = [m.strip().lower() for m in modes_str.split(",") if m.strip()]
    allowed = {"heuristic", "mlp", "isab"}
    for m in modes:
        if m not in allowed:
            raise SystemExit(f"Invalid mode: {m}. Allowed: {sorted(allowed)}")
    return modes


def main():
    parser = argparse.ArgumentParser(description="NS-GBS benchmark runner")
    parser.add_argument("-s", "--scenario", action="append", choices=SCENARIOS.keys())
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--N_UE", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--modes", default="heuristic,mlp,isab")
    parser.add_argument("--model-mlp", default="output/models/nsgbs_scorer.pt")
    parser.add_argument("--model-isab", default=None)
    parser.add_argument("--collect-stats", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--out", default="output/nsgbs_bench.csv")
    args = parser.parse_args()

    if args.all:
        scenarios = list(SCENARIOS.keys())
    elif args.scenario:
        scenarios = args.scenario
    else:
        parser.print_help()
        return 1

    modes = parse_modes(args.modes)
    model_mlp = Path(args.model_mlp)
    model_isab = Path(args.model_isab) if args.model_isab else None

    if "mlp" in modes and not model_mlp.exists():
        raise SystemExit(f"MLP model not found: {model_mlp}")
    if "isab" in modes and (model_isab is None or not model_isab.exists()):
        raise SystemExit("ISAB mode requested but --model-isab is missing or invalid.")

    base_cfg = load_base_config()

    from main import run_once

    rows = []

    # Calculate total iterations for outer progress bar
    total_runs = len(scenarios) * len(modes) * args.num_seeds

    # Create outer progress bar (only if not disabled)
    outer_pbar = tqdm(total=total_runs, desc="NS-GBS Benchmark", unit="run",
                      disable=args.no_progress,
                      bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
        for scenario in scenarios:
            cfg_path = SCRIPT_DIR / SCENARIOS[scenario]
            scenario_cfg = load_config_from_file(cfg_path)
            for mode in modes:
                for s in range(args.seed_start, args.seed_start + args.num_seeds):
                    cfg = base_cfg.copy()
                    cfg.update(scenario_cfg)
                    if args.T is not None:
                        cfg["T"] = int(args.T)
                    if args.N_UE is not None:
                        cfg["N_UE"] = int(args.N_UE)
                    cfg["seed"] = int(s)
                    cfg["show_progress"] = not args.no_progress
                    cfg["progress_leave"] = False  # Don't persist inner progress bars

                    if mode == "heuristic":
                        cfg["scheduler_kind"] = "heuristic"
                        cfg["nsgbs_model_path"] = None
                    elif mode == "mlp":
                        cfg["scheduler_kind"] = "nsgbs"
                        cfg["nsgbs_model_path"] = str(model_mlp)
                    elif mode == "isab":
                        cfg["scheduler_kind"] = "nsgbs"
                        cfg["nsgbs_model_path"] = str(model_isab)

                    if args.collect_stats:
                        cfg["nsgbs_collect_stats"] = True
                        cfg["nsgbs_stats_out"] = {}

                    # Update outer progress bar description
                    outer_pbar.set_description(f"NS-GBS [{scenario}/{mode}/seed{s}]")

                    # Print config diagnostics (once per scenario)
                    if mode == modes[0] and s == args.seed_start:
                        print(f"\n[Config] scenario={scenario}")
                        print(f"[Config] T={cfg.get('T')}, N_UE={cfg.get('N_UE')}, seed_range=[{args.seed_start}, {args.seed_start + args.num_seeds - 1}]")
                        print(f"[Config] enable_time_varying={cfg.get('enable_time_varying')}, enable_harq_full={cfg.get('enable_harq_full')}")

                    out = run_once(cfg)
                    stats = out.get("nsgbs_stats") if args.collect_stats else None

                    row = {
                        "scenario": scenario,
                        "mode": mode,
                        "seed": int(s),
                        "avg_se_radiomap": float(out.get("avg_se_radiomap", 0.0)),
                        "avg_se_baseline_default": float(out.get("avg_se_baseline_default", 0.0)),
                        "improvement_vs_default_pct": float(out.get("improvement_vs_default_pct", 0.0)),
                    }
                    if stats:
                        steps = float(stats.get("steps", 0.0))
                        actions_total = float(stats.get("actions_total", 0.0))
                        score_calls = float(stats.get("score_calls", 0.0))
                        score_time = float(stats.get("score_time_sec", 0.0))
                        row.update({
                            "steps": steps,
                            "actions_total": actions_total,
                            "avg_actions_per_step": (actions_total / steps) if steps > 0 else 0.0,
                            "score_calls": score_calls,
                            "score_time_sec": score_time,
                            "avg_score_ms_per_call": (score_time * 1000.0 / score_calls) if score_calls > 0 else 0.0,
                            "avg_score_ms_per_step": (score_time * 1000.0 / steps) if steps > 0 else 0.0,
                        })
                    rows.append(row)

                    # Update outer progress bar
                    outer_pbar.update(1)
                    outer_pbar.set_postfix(se=f"{row['avg_se_radiomap']:.4f}",
                                           gain=f"{row['improvement_vs_default_pct']:+.2f}%")

                    print(
                        f"{scenario} | {mode} | seed={s} "
                        f"avg_se={row['avg_se_radiomap']:.4f} "
                        f"gain={row['improvement_vs_default_pct']:+.2f}%"
                    )
    finally:
        outer_pbar.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
