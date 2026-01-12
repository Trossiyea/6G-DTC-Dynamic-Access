#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot for Figure 4: Robustness to RadioMap Imperfections.

Generates two subplots:
(a) SE vs RadioMap estimation error (dB)
(b) SE vs RadioMap blur sigma (+ optional resolution bar chart)
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
    setup_style, COLORS, METHOD_LABELS,
    ci_95, save_fig, add_subplot_label
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure 4: Robustness to RadioMap Imperfections"
    )
    parser.add_argument(
        "--csv",
        default="output/results/fig04_rm_imperfections.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out",
        default="output/figures/fig04_rm_imperfections.pdf",
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

    # Get available methods
    methods = [m for m in ['P1_MLP', 'P2_ISAB'] if m in df['method'].unique()]
    print(f"Methods: {methods}")

    # Check if resolution data is available
    has_resolution = 'resolution_m' in df['imperfection_type'].unique()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    markers = {'P1_MLP': 'D', 'P2_ISAB': 'v'}
    linestyles = {'P1_MLP': '-', 'P2_ISAB': '--'}

    # =========================================================================
    # Subplot (a): SE vs Estimation Error
    # =========================================================================
    ax = axes[0]

    df_est = df[df['imperfection_type'] == 'est_error_db']

    for method in methods:
        method_data = df_est[df_est['method'] == method]
        if len(method_data) == 0:
            continue

        values = sorted(method_data['imperfection_value'].unique())
        means = []
        cis = []

        for val in values:
            subset = method_data[method_data['imperfection_value'] == val]
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

    ax.set_xlabel('RadioMap Estimation Error (dB)')
    ax.set_ylabel('Average SE (bits/s/Hz)')
    ax.legend(loc='best', frameon=False)
    ax.grid(True, alpha=0.3)
    add_subplot_label(ax, '(a)')

    # =========================================================================
    # Subplot (b): SE vs Blur Sigma (with optional resolution inset)
    # =========================================================================
    ax = axes[1]

    df_blur = df[df['imperfection_type'] == 'blur_sigma']

    for method in methods:
        method_data = df_blur[df_blur['method'] == method]
        if len(method_data) == 0:
            continue

        values = sorted(method_data['imperfection_value'].unique())
        means = []
        cis = []

        for val in values:
            subset = method_data[method_data['imperfection_value'] == val]
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

    ax.set_xlabel('RadioMap Spatial Blur (σ)')
    ax.set_ylabel('Average SE (bits/s/Hz)')
    ax.legend(loc='best', frameon=False)
    ax.grid(True, alpha=0.3)
    add_subplot_label(ax, '(b)')

    # Add resolution comparison as inset bar chart if data available
    if has_resolution:
        df_res = df[df['imperfection_type'] == 'resolution_m']
        if len(df_res) > 0:
            # Create inset axes
            inset_ax = ax.inset_axes([0.6, 0.55, 0.35, 0.4])

            resolutions = sorted(df_res['imperfection_value'].unique())
            x = np.arange(len(resolutions))
            bar_width = 0.35

            for i, method in enumerate(methods):
                method_data = df_res[df_res['method'] == method]
                means = []
                cis = []

                for res in resolutions:
                    subset = method_data[method_data['imperfection_value'] == res]
                    se_values = subset['avg_se'].dropna().values
                    if len(se_values) > 0:
                        means.append(np.mean(se_values))
                        cis.append(ci_95(se_values))
                    else:
                        means.append(0)
                        cis.append(0)

                color = COLORS.get(method, f'C{i}')
                offset = (i - 0.5) * bar_width
                inset_ax.bar(x + offset, means, bar_width, yerr=cis,
                            color=color, edgecolor='black', linewidth=0.5,
                            capsize=2, error_kw={'linewidth': 0.6})

            inset_ax.set_xticks(x)
            inset_ax.set_xticklabels([f'{int(r)}m' for r in resolutions], fontsize=8)
            inset_ax.set_ylabel('SE', fontsize=8)
            inset_ax.set_title('Resolution', fontsize=9)
            inset_ax.tick_params(labelsize=7)

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
