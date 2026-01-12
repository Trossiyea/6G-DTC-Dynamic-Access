#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master experiment runner for IEEE TMC figures.

Runs all experiment scripts in sequence with configurable parameters.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Experiment configurations: (script_name, default_args)
EXPERIMENTS = [
    ("exp_fig01_main_performance.py", ["--all", "--seeds", "5"]),
    ("exp_fig02_user_fairness.py", ["--seeds", "5"]),
    ("exp_fig03_csi_delay.py", ["--seeds", "5"]),
    ("exp_fig04_rm_imperfections.py", ["--seeds", "5"]),
    ("exp_fig05_dynamics.py", ["--seeds", "5"]),
    ("exp_fig06_scalability.py", ["--seeds", "5"]),
    ("exp_fig07_constellation.py", ["--all", "--seeds", "3"]),
    ("exp_fig08_harq.py", ["--seeds", "5"]),
    ("exp_fig09_complexity.py", ["--seeds", "3"]),
    ("exp_fig10_generalization.py", ["--seeds", "5"]),
]


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
    print()

    failed = []
    completed = []

    for i, (script_name, default_args) in enumerate(experiments, 1):
        script_path = script_dir / script_name

        if not script_path.exists():
            print(f"[{i}/{len(experiments)}] WARNING: {script_name} not found, skipping")
            failed.append(script_name)
            continue

        # Build command
        cmd = [sys.executable, str(script_path)] + list(default_args)

        # Apply overrides
        if args.seeds is not None:
            # Replace --seeds argument
            cmd = [c for c in cmd if not c.startswith("--seeds")]
            cmd = [c for j, c in enumerate(cmd) if not (j > 0 and cmd[j-1] == "--seeds")]
            cmd.extend(["--seeds", str(args.seeds)])

        if args.T is not None:
            cmd.extend(["--T", str(args.T)])

        print(f"\n{'='*70}")
        print(f"[{i}/{len(experiments)}] Running: {script_name}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(cmd)}")

        if args.dry_run:
            print("(dry run - not executing)")
            continue

        try:
            result = subprocess.run(cmd, check=True)
            completed.append(script_name)
            print(f"✓ {script_name} completed successfully")
        except subprocess.CalledProcessError as e:
            failed.append(script_name)
            print(f"✗ {script_name} failed with exit code {e.returncode}")
            if not args.continue_on_error:
                print("\nStopping due to error. Use --continue-on-error to continue.")
                break
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Completed: {len(completed)}/{len(experiments)}")
    if completed:
        print(f"  ✓ {', '.join(completed)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print(f"  ✗ {', '.join(failed)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
