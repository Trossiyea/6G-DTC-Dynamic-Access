#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master experiment runner for IEEE TMC figures.

Runs all experiment scripts with configurable parameters.
Supports parallel execution at both figure-level and within-figure level.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# Experiment configurations: (script_name, default_args)
EXPERIMENTS = [
    ("exp_fig01_main_performance.py", ["--all", "--seeds", "5", "--parallel"]),
    ("exp_fig02_user_fairness.py", ["--seeds", "5", "--parallel"]),
    ("exp_fig03_csi_delay.py", ["--seeds", "5", "--parallel"]),
    ("exp_fig04_rm_imperfections.py", ["--seeds", "5", "--parallel"]),
    ("exp_fig05_dynamics.py", ["--seeds", "5", "--parallel"]),
    ("exp_fig06_scalability.py", ["--seeds", "5", "--parallel"]),
    ("exp_fig07_constellation.py", ["--all", "--seeds", "3", "--parallel"]),
    ("exp_fig08_harq.py", ["--seeds", "5", "--parallel"]),
    ("exp_fig09_complexity.py", ["--seeds", "3", "--parallel"]),
    ("exp_fig10_generalization.py", ["--seeds", "5", "--parallel"]),
]

# Memory-heavy experiments that should run separately
HEAVY_EXPERIMENTS = {"exp_fig07_constellation.py", "exp_fig06_scalability.py"}


def run_experiment(script_path: Path, args: List[str], timeout: float = 7200) -> Tuple[str, int, str]:
    """Run a single experiment script."""
    cmd = [sys.executable, str(script_path)] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return (script_path.name, result.returncode, result.stderr if result.returncode != 0 else "")
    except subprocess.TimeoutExpired:
        return (script_path.name, -1, "Timeout")
    except Exception as e:
        return (script_path.name, -2, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Run all TMC figure experiments"
    )
    parser.add_argument(
        "--figures",
        type=str,
        default=None,
        help="Comma-separated list of figure numbers to run (e.g., '1,2,5'). Default: all"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override number of seeds for all experiments"
    )
    parser.add_argument(
        "--T",
        type=int,
        default=None,
        help="Override number of TTIs for all experiments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining experiments if one fails"
    )
    parser.add_argument(
        "--parallel-figures",
        type=int,
        default=1,
        help="Number of figures to run in parallel (default: 1, sequential)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential execution of all figures"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of workers for within-figure parallelism"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Filter experiments if specific figures requested
    if args.figures:
        figure_nums = [int(x.strip()) for x in args.figures.split(",")]
        experiments = [(name, default_args) for name, default_args in EXPERIMENTS
                       if int(name.split("_")[1].replace("fig", "")) in figure_nums]
    else:
        experiments = EXPERIMENTS

    if not experiments:
        print("No experiments to run.")
        return 0

    print("=" * 70)
    print("IEEE TMC Figure Experiments Runner")
    print("=" * 70)
    print(f"Experiments to run: {len(experiments)}")
    print(f"Figures: {[name.split('_')[1] for name, _ in experiments]}")

    parallel_figs = args.parallel_figures if not args.sequential else 1
    print(f"Figure-level parallelism: {parallel_figs}")
    print()

    failed = []
    completed_list = []

    # Build commands for all experiments
    experiment_cmds = []
    for script_name, default_args in experiments:
        script_path = script_dir / script_name

        if not script_path.exists():
            print(f"WARNING: {script_name} not found, skipping")
            failed.append(script_name)
            continue

        # Build command
        cmd_args = list(default_args)

        # Apply overrides
        if args.seeds is not None:
            cmd_args = [c for c in cmd_args if not c.startswith("--seeds")]
            cmd_args = [c for j, c in enumerate(cmd_args) if not (j > 0 and cmd_args[j-1] == "--seeds")]
            cmd_args.extend(["--seeds", str(args.seeds)])

        if args.T is not None:
            cmd_args.extend(["--T", str(args.T)])

        if args.workers is not None:
            cmd_args.extend(["--workers", str(args.workers)])

        experiment_cmds.append((script_path, cmd_args))

    if args.dry_run:
        print("\nDry run - commands that would be executed:")
        for script_path, cmd_args in experiment_cmds:
            print(f"  {sys.executable} {script_path} {' '.join(cmd_args)}")
        return 0

    # Separate heavy and light experiments
    light_cmds = [(p, a) for p, a in experiment_cmds if p.name not in HEAVY_EXPERIMENTS]
    heavy_cmds = [(p, a) for p, a in experiment_cmds if p.name in HEAVY_EXPERIMENTS]

    def run_batch(cmds, parallel_count, batch_name=""):
        """Run a batch of experiments with specified parallelism."""
        batch_failed = []
        batch_completed = []

        if parallel_count <= 1:
            # Sequential execution
            for i, (script_path, cmd_args) in enumerate(cmds, 1):
                print(f"\n{'='*70}")
                print(f"[{i}/{len(cmds)}] Running: {script_path.name}")
                print(f"{'='*70}")

                name, code, err = run_experiment(script_path, cmd_args)
                if code == 0:
                    batch_completed.append(name)
                    print(f"  {name} completed successfully")
                else:
                    batch_failed.append(name)
                    print(f"  {name} FAILED: {err}")
                    if not args.continue_on_error:
                        break
        else:
            # Parallel execution
            print(f"\nRunning {len(cmds)} experiments with {parallel_count} parallel workers...")
            with ProcessPoolExecutor(max_workers=parallel_count) as executor:
                futures = {
                    executor.submit(run_experiment, path, cmd_args): path.name
                    for path, cmd_args in cmds
                }
                for future in as_completed(futures):
                    name, code, err = future.result()
                    if code == 0:
                        batch_completed.append(name)
                        print(f"  {name}: OK")
                    else:
                        batch_failed.append(name)
                        print(f"  {name}: FAILED ({err})")

        return batch_completed, batch_failed

    try:
        # Run light experiments (can be parallelized)
        if light_cmds:
            print(f"\n--- Running {len(light_cmds)} standard experiments ---")
            c, f = run_batch(light_cmds, parallel_figs, "standard")
            completed_list.extend(c)
            failed.extend(f)

        # Run heavy experiments (sequential to avoid memory issues)
        if heavy_cmds and (args.continue_on_error or not failed):
            print(f"\n--- Running {len(heavy_cmds)} memory-intensive experiments (sequential) ---")
            c, f = run_batch(heavy_cmds, 1, "heavy")
            completed_list.extend(c)
            failed.extend(f)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Completed: {len(completed_list)}/{len(experiments)}")
    if completed_list:
        print(f"  Completed: {', '.join(completed_list)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print(f"  Failed: {', '.join(failed)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
