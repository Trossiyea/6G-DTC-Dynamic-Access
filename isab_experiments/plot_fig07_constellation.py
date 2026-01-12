#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot for Figure 7: Constellation Mode.

Generates two subplots:
(a) SE vs max_sats and/or min_elev
(b) Bar chart: HO events and outage per method
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
        description="Plot Figure 7: Constellation Mode"
    )
    parser.add_argument(
        "--csv",
        default="output/results/fig07_constellation.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out",
        default="output/figures/fig07_constellation.pdf",
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

    markers = {'B1_3GPP': 'o', 'P1_MLP': 'D', 'P2_ISAB': 'v'}
    linestyles = {'B1_3GPP': ':', 'P1_MLP': '-', 'P2_ISAB': '--'}

    # =========================================================================
    # Subplot (a): SE vs max_sats (or min_elev)
    # =========================================================================
    ax = axes[0]

    # Check which parameter has been swept
    max_sats_values = df['max_sats'].unique()
    min_elev_values = df['min_elev'].unique()

    has_max_sats_sweep = len(max_sats_values) > 1
    has_min_elev_sweep = len(min_elev_values) > 1

    if has_max_sats_sweep:
        # Plot SE vs max_sats
        default_min_elev = df['min_elev'].mode().values[0]
        df_sats = df[df['min_elev'] == default_min_elev]

        for method in methods:
            method_data = df_sats[df_sats['method'] == method]
            if len(method_data) == 0:
                continue

            values = sorted(method_data['max_sats'].unique())
            means = []
            cis = []

            for val in values:
                subset = method_data[method_data['max_sats'] == val]
                se_values = subset['avg_se'].dropna().values
                if len(se_values) > 0:
                    means.append(np.mean(se_values))
                    cis.append(ci_95(se_values))
                else:
                    means.append(np.nan)
                    cis.append(0)

            means = np.array(means)
            cis = np.array(cis)
            color = COLORS.get(method, 'gray')
            marker = markers.get(method, 'o')
            linestyle = linestyles.get(method, '-')
            label = METHOD_LABELS.get(method, method)

            ax.plot(values, means, marker=marker, linestyle=linestyle, color=color,
                    label=label, linewidth=1.8, markersize=6)
            ax.fill_between(values, means - cis, means + cis, color=color, alpha=0.2)

        ax.set_xlabel('Max Satellites per TTI')
        ax.set_ylabel('Average SE (bits/s/Hz)')

    elif has_min_elev_sweep:
        # Plot SE vs min_elev
        default_max_sats = df['max_sats'].mode().values[0]
        df_elev = df[df['max_sats'] == default_max_sats]

        for method in methods:
            method_data = df_elev[df_elev['method'] == method]
            if len(method_data) == 0:
                continue

            values = sorted(method_data['min_elev'].unique())
            means = []
            cis = []

            for val in values:
                subset = method_data[method_data['min_elev'] == val]
                se_values = subset['avg_se'].dropna().values
                if len(se_values) > 0:
                    means.append(np.mean(se_values))
                    cis.append(ci_95(se_values))
                else:
                    means.append(np.nan)
                    cis.append(0)

            means = np.array(means)
            cis = np.array(cis)
            color = COLORS.get(method, 'gray')
            marker = markers.get(method, 'o')
            linestyle = linestyles.get(method, '-')
            label = METHOD_LABELS.get(method, method)

            ax.plot(values, means, marker=marker, linestyle=linestyle, color=color,
                    label=label, linewidth=1.8, markersize=6)
            ax.fill_between(values, means - cis, means + cis, color=color, alpha=0.2)

        ax.set_xlabel('Minimum Elevation (deg)')
        ax.set_ylabel('Average SE (bits/s/Hz)')
    else:
        # Fallback: bar chart by scenario
        scenarios = df['scenario'].unique()
        x = np.arange(len(scenarios))
        bar_width = 0.25
        n_methods = len(methods)
        offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

        for i, method in enumerate(methods):
            means = []
            cis = []
            for scenario in scenarios:
                subset = df[(df['scenario'] == scenario) & (df['method'] == method)]
                se_values = subset['avg_se'].dropna().values
                if len(se_values) > 0:
                    means.append(np.mean(se_values))
                    cis.append(ci_95(se_values))
                else:
                    means.append(0)
                    cis.append(0)

            color = COLORS.get(method, f'C{i}')
            label = METHOD_LABELS.get(method, method)
            ax.bar(x + offsets[i], means, bar_width, yerr=cis, label=label, color=color,
                   edgecolor='black', linewidth=0.5, capsize=2, error_kw={'linewidth': 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_constellation', '') for s in scenarios])
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Average SE (bits/s/Hz)')

    ax.legend(loc='best', frameon=False)
    ax.grid(True, alpha=0.3)
    add_subplot_label(ax, '(a)')

    # =========================================================================
    # Subplot (b): HO events and outage per method
    # =========================================================================
    ax = axes[1]

    x = np.arange(len(methods))
    bar_width = 0.35

    # Calculate mean HO and outage per method
    ho_means = []
    ho_cis = []
    outage_means = []
    outage_cis = []

    for method in methods:
        method_data = df[df['method'] == method]
        ho_values = method_data['avg_ho_per_ue'].dropna().values
        outage_values = method_data['avg_outage_frac'].dropna().values * 100  # Convert to %

        if len(ho_values) > 0:
            ho_means.append(np.mean(ho_values))
            ho_cis.append(ci_95(ho_values))
        else:
            ho_means.append(0)
            ho_cis.append(0)

        if len(outage_values) > 0:
            outage_means.append(np.mean(outage_values))
            outage_cis.append(ci_95(outage_values))
        else:
            outage_means.append(0)
            outage_cis.append(0)

    colors_bars = [COLORS.get(m, f'C{i}') for i, m in enumerate(methods)]

    # Plot HO on left axis
    bars1 = ax.bar(x - bar_width/2, ho_means, bar_width, yerr=ho_cis,
                   color=colors_bars, edgecolor='black', linewidth=0.5,
                   capsize=3, error_kw={'linewidth': 0.8}, alpha=0.8,
                   label='Avg HO/UE')

    ax.set_ylabel('Avg Handovers per UE')
    ax.set_ylim(bottom=0)

    # Plot outage on right axis
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + bar_width/2, outage_means, bar_width, yerr=outage_cis,
                    color=colors_bars, edgecolor='black', linewidth=0.5,
                    capsize=3, error_kw={'linewidth': 0.8}, alpha=0.4,
                    hatch='//', label='Outage (%)')

    ax2.set_ylabel('Outage Fraction (%)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(bottom=0)

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=15, ha='right')
    ax.set_xlabel('Method')

    # Legend
    ax.legend([bars1, bars2], ['Avg HO/UE', 'Outage (%)'], loc='upper right', frameon=False)

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
