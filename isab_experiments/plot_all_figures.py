#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master plotting script for IEEE TMC figures.

Generates all figures from CSV results.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Plot scripts in order
PLOT_SCRIPTS = [
    "plot_fig01_main_performance.py",
    "plot_fig02_user_fairness.py",
    "plot_fig03_csi_delay.py",
    "plot_fig04_rm_imperfections.py",
    "plot_fig05_dynamics.py",
    "plot_fig06_scalability.py",
    "plot_fig07_constellation.py",
    "plot_fig08_harq.py",
    "plot_fig09_complexity.py",
    "plot_fig10_generalization.py",
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate all TMC figures from CSV results"
    )
    parser.add_argument(
        "--figures",
        type=str,
        default=None,
        help="Comma-separated list of figure numbers to plot (e.g., '1,2,5'). Default: all"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively instead of saving"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining plots if one fails"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Filter scripts if specific figures requested
    if args.figures:
        figure_nums = [int(x.strip()) for x in args.figures.split(",")]
        scripts = [name for name in PLOT_SCRIPTS
                   if int(name.split("_")[1].replace("fig", "").replace("plot", "")) in figure_nums]
    else:
        scripts = PLOT_SCRIPTS

    if not scripts:
        print("No plots to generate.")
        return 0

    print("=" * 70)
    print("IEEE TMC Figure Generator")
    print("=" * 70)
    print(f"Figures to plot: {len(scripts)}")
    print()

    failed = []
    completed = []

    for i, script_name in enumerate(scripts, 1):
        script_path = script_dir / script_name

        if not script_path.exists():
            print(f"[{i}/{len(scripts)}] WARNING: {script_name} not found, skipping")
            failed.append(script_name)
            continue

        # Build command
        cmd = [sys.executable, str(script_path)]
        if args.show:
            cmd.append("--show")

        print(f"[{i}/{len(scripts)}] Plotting: {script_name}")

        if args.dry_run:
            print(f"  Command: {' '.join(cmd)} (dry run)")
            continue

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            completed.append(script_name)
            # Print saved paths from stdout
            for line in result.stdout.split('\n'):
                if 'Saved:' in line:
                    print(f"  {line.strip()}")
        except subprocess.CalledProcessError as e:
            failed.append(script_name)
            print(f"  ✗ Failed: {e.stderr[:200] if e.stderr else 'Unknown error'}")
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
    print(f"Generated: {len(completed)}/{len(scripts)} figures")
    if completed:
        print(f"  ✓ {', '.join([s.replace('plot_', '').replace('.py', '') for s in completed])}")
    if failed:
        print(f"Failed: {len(failed)}")
        print(f"  ✗ {', '.join(failed)}")

    # List output files
    output_dir = script_dir.parent / "output" / "figures"
    if output_dir.exists() and not args.dry_run:
        pdf_files = sorted(output_dir.glob("fig*.pdf"))
        if pdf_files:
            print(f"\nGenerated figures in {output_dir}:")
            for f in pdf_files:
                png_exists = f.with_suffix('.png').exists()
                print(f"  {f.name}" + (" (+PNG)" if png_exists else ""))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
