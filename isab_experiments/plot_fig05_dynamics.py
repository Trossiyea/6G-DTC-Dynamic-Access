#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot for Figure 5: Non-Stationary Interference + Orbit Dynamics.

Generates two subplots:
(a) SE vs interference flicker std (dB)
(b) SE vs residual Doppler fraction
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
        description="Plot Figure 5: Dynamics (Flicker + Doppler)"
    )
    parser.add_argument(
        "--csv",
        default="output/results/fig05_dynamics.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out",
        default="output/figures/fig05_dynamics.pdf",
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
    # Subplot (a): SE vs Flicker Std
    # =========================================================================
    ax = axes[0]

    df_flicker = df[df['dynamic_type'] == 'flicker_db_std']

    for method in methods:
        method_data = df_flicker[df_flicker['method'] == method]
        if len(method_data) == 0:
            continue

        values = sorted(method_data['dynamic_value'].unique())
        means = []
        cis = []

        for val in values:
            subset = method_data[method_data['dynamic_value'] == val]
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

    ax.set_xlabel('Interference Flicker Std (dB)')
    ax.set_ylabel('Average SE (bits/s/Hz)')
    ax.legend(loc='best', frameon=False)
    ax.grid(True, alpha=0.3)
    add_subplot_label(ax, '(a)')

    # =========================================================================
    # Subplot (b): SE vs Doppler Residual
    # =========================================================================
    ax = axes[1]

    df_doppler = df[df['dynamic_type'] == 'doppler_residual']

    for method in methods:
        method_data = df_doppler[df_doppler['method'] == method]
        if len(method_data) == 0:
            continue

        values = sorted(method_data['dynamic_value'].unique())
        means = []
        cis = []

        for val in values:
            subset = method_data[method_data['dynamic_value'] == val]
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

    ax.set_xlabel('Residual Doppler Fraction')
    ax.set_ylabel('Average SE (bits/s/Hz)')
    ax.legend(loc='best', frameon=False)
    ax.grid(True, alpha=0.3)
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
