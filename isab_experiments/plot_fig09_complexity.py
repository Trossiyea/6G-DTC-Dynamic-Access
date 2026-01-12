#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot for Figure 9: Runtime / Complexity / Deployability.

Generates two subplots:
(a) Bar chart: Scheduler time per TTI by method
(b) Scatter plot: SE vs compute cost (ms per TTI)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_utils import (
    setup_style, COLORS, METHOD_LABELS, METHOD_ORDER,
    ci_95, save_fig, add_subplot_label
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure 9: Complexity"
    )
    parser.add_argument(
        "--csv",
        default="output/results/fig09_complexity.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out",
        default="output/figures/fig09_complexity.pdf",
        help="Output figure path"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure interactively"
    )
    args = parser.parse_args()

    # Setup style
    setup_style()

    # Load data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Get methods in order
    methods = [m for m in METHOD_ORDER if m in df['method'].unique()]
    print(f"Methods: {methods}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # =========================================================================
    # Subplot (a): Bar chart - ms per TTI by method
    # =========================================================================
    ax = axes[0]

    x = np.arange(len(methods))
    bar_width = 0.6

    ms_means = []
    ms_cis = []

    for method in methods:
        method_data = df[df['method'] == method]
        ms_values = method_data['ms_per_tti'].dropna().values

        if len(ms_values) > 0:
            ms_means.append(np.mean(ms_values))
            ms_cis.append(ci_95(ms_values))
        else:
            ms_means.append(0)
            ms_cis.append(0)

    colors_bars = [COLORS.get(m, f'C{i}') for i, m in enumerate(methods)]

    bars = ax.bar(x, ms_means, bar_width, yerr=ms_cis,
                  color=colors_bars, edgecolor='black', linewidth=0.5,
                  capsize=4, error_kw={'linewidth': 1.0})

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, ms_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ms_cis[i] + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=15, ha='right')
    ax.set_xlabel('Method')
    ax.set_ylabel('Time per TTI (ms)')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis='y')
    add_subplot_label(ax, '(a)')

    # =========================================================================
    # Subplot (b): Scatter - SE vs compute cost
    # =========================================================================
    ax = axes[1]

    markers = {'B3_heuristic': '^', 'P1_MLP': 'D', 'P2_ISAB': 'v'}

    for method in methods:
        method_data = df[df['method'] == method]

        ms_values = method_data['ms_per_tti'].dropna().values
        se_values = method_data['avg_se'].dropna().values

        if len(ms_values) > 0 and len(se_values) > 0:
            ms_mean = np.mean(ms_values)
            ms_std = np.std(ms_values) if len(ms_values) > 1 else 0
            se_mean = np.mean(se_values)
            se_std = np.std(se_values) if len(se_values) > 1 else 0

            color = COLORS.get(method, 'gray')
            marker = markers.get(method, 'o')
            label = METHOD_LABELS.get(method, method)

            # Plot point with error bars
            ax.errorbar(ms_mean, se_mean, xerr=ms_std, yerr=se_std,
                       marker=marker, color=color, markersize=12,
                       capsize=4, capthick=1.5, linewidth=1.5,
                       label=label, markeredgecolor='black', markeredgewidth=0.5)

            # Add label next to point
            ax.annotate(label, (ms_mean, se_mean), textcoords="offset points",
                       xytext=(8, 5), fontsize=8, color=color)

    ax.set_xlabel('Time per TTI (ms)')
    ax.set_ylabel('Average SE (bits/s/Hz)')
    ax.grid(True, alpha=0.3)

    # Add efficiency frontier line (optional - connect points)
    if len(methods) >= 2:
        points = []
        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                ms_mean = method_data['ms_per_tti'].mean()
                se_mean = method_data['avg_se'].mean()
                points.append((ms_mean, se_mean, method))

        # Sort by ms (x-axis)
        points.sort(key=lambda p: p[0])
        ms_sorted = [p[0] for p in points]
        se_sorted = [p[1] for p in points]
        ax.plot(ms_sorted, se_sorted, 'k--', alpha=0.3, linewidth=0.8)

    add_subplot_label(ax, '(b)')

    # =========================================================================
    # Finalize
    # =========================================================================
    plt.tight_layout()

    if args.show:
        plt.show()
    else:
        save_fig(fig, args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
