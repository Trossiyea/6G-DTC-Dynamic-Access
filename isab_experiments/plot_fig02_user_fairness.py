#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot for Figure 2: User-Level Throughput Distribution & Fairness.

Generates two subplots:
(a) CDF of per-UE throughput (one line per method)
(b) Jain fairness index and 5%-tile throughput by method
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


def plot_cdf(ax, data, label, color, linestyle='-'):
    """Plot empirical CDF."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cdf, linestyle, color=color, label=label, linewidth=1.5)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure 2: User-Level Throughput Distribution & Fairness"
    )
    parser.add_argument(
        "--csv-per-ue",
        default="output/results/fig02_user_fairness_per_ue.csv",
        help="Input CSV path for per-UE data"
    )
    parser.add_argument(
        "--csv-summary",
        default="output/results/fig02_user_fairness_summary.csv",
        help="Input CSV path for summary data"
    )
    parser.add_argument(
        "--out",
        default="output/figures/fig02_user_fairness.pdf",
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
    per_ue_path = Path(args.csv_per_ue)
    summary_path = Path(args.csv_summary)

    if not per_ue_path.exists():
        print(f"Error: Per-UE CSV not found: {per_ue_path}")
        return 1
    if not summary_path.exists():
        print(f"Error: Summary CSV not found: {summary_path}")
        return 1

    df_per_ue = pd.read_csv(per_ue_path)
    df_summary = pd.read_csv(summary_path)

    print(f"Loaded {len(df_per_ue)} per-UE rows, {len(df_summary)} summary rows")

    # Get methods in consistent order
    methods = [m for m in METHOD_ORDER if m in df_per_ue['method'].unique()]
    print(f"Methods: {methods}")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # =========================================================================
    # Subplot (a): CDF of per-UE throughput
    # =========================================================================
    ax = axes[0]

    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    for i, method in enumerate(methods):
        # Aggregate throughput across all seeds for this method
        subset = df_per_ue[df_per_ue['method'] == method]
        throughput = subset['throughput_bps'].values / 1e6  # Convert to Mbps

        if len(throughput) > 0:
            color = COLORS.get(method, f'C{i}')
            label = METHOD_LABELS.get(method, method)
            linestyle = linestyles[i % len(linestyles)]
            plot_cdf(ax, throughput, label, color, linestyle)

    ax.set_xlabel('Per-UE Throughput (Mbps)')
    ax.set_ylabel('CDF')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add vertical lines for key percentiles (optional)
    ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.10, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.02, 0.05, '5%', fontsize=8, va='bottom', color='gray')
    ax.text(ax.get_xlim()[1] * 0.02, 0.10, '10%', fontsize=8, va='bottom', color='gray')

    add_subplot_label(ax, '(a)')

    # =========================================================================
    # Subplot (b): Jain index and 5%-tile throughput
    # =========================================================================
    ax = axes[1]

    x = np.arange(len(methods))
    bar_width = 0.35

    # Compute mean and CI for each method
    jain_means = []
    jain_cis = []
    pctl5_means = []
    pctl5_cis = []

    for method in methods:
        subset = df_summary[df_summary['method'] == method]
        jain_values = subset['jain_index'].dropna().values
        pctl5_values = subset['pctl_5'].dropna().values / 1e6  # Convert to Mbps

        if len(jain_values) > 0:
            jain_means.append(np.mean(jain_values))
            jain_cis.append(ci_95(jain_values))
        else:
            jain_means.append(0)
            jain_cis.append(0)

        if len(pctl5_values) > 0:
            pctl5_means.append(np.mean(pctl5_values))
            pctl5_cis.append(ci_95(pctl5_values))
        else:
            pctl5_means.append(0)
            pctl5_cis.append(0)

    # Plot Jain index on left y-axis
    colors_bars = [COLORS.get(m, f'C{i}') for i, m in enumerate(methods)]
    bars1 = ax.bar(x - bar_width/2, jain_means, bar_width, yerr=jain_cis,
                   color=colors_bars, edgecolor='black', linewidth=0.5,
                   capsize=3, error_kw={'linewidth': 0.8}, alpha=0.8,
                   label='Jain Index')

    ax.set_ylabel('Jain Fairness Index', color='black')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='y', labelcolor='black')

    # Plot 5%-tile throughput on right y-axis
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + bar_width/2, pctl5_means, bar_width, yerr=pctl5_cis,
                    color=colors_bars, edgecolor='black', linewidth=0.5,
                    capsize=3, error_kw={'linewidth': 0.8}, alpha=0.4,
                    hatch='//', label='5%-tile Throughput')

    ax2.set_ylabel('5%-tile Throughput (Mbps)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=15, ha='right')
    ax.set_xlabel('Method')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend([bars1, bars2], ['Jain Index', '5%-tile Throughput'],
              loc='upper right', frameon=False, fontsize=9)

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
