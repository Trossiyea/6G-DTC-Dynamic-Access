#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot for Figure 3: CSI Delay vs RadioMap Freshness.

Generates two subplots:
(a) SE vs baseline CSI delay (with Oracle as upper bound)
(b) SE vs RadioMap staleness/delay
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
        description="Plot Figure 3: CSI Delay vs RadioMap Freshness"
    )
    parser.add_argument(
        "--csv",
        default="output/results/fig03_csi_delay.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out",
        default="output/figures/fig03_csi_delay.pdf",
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

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # =========================================================================
    # Subplot (a): SE vs Baseline CSI Delay
    # =========================================================================
    ax = axes[0]

    # Filter baseline CSI data
    df_baseline = df[df['delay_type'] == 'baseline_csi']

    # Plot B1_3GPP (practical baseline)
    b1_data = df_baseline[df_baseline['method'] == 'B1_3GPP']
    delays = sorted(b1_data['delay_value'].unique())

    means_b1 = []
    cis_b1 = []
    for delay in delays:
        subset = b1_data[b1_data['delay_value'] == delay]
        se_values = subset['avg_se'].dropna().values
        if len(se_values) > 0:
            means_b1.append(np.mean(se_values))
            cis_b1.append(ci_95(se_values))
        else:
            means_b1.append(np.nan)
            cis_b1.append(0)

    means_b1 = np.array(means_b1)
    cis_b1 = np.array(cis_b1)

    ax.plot(delays, means_b1, 'o-', color=COLORS['B1_3GPP'],
            label=METHOD_LABELS['B1_3GPP'], linewidth=1.8, markersize=6)
    ax.fill_between(delays, means_b1 - cis_b1, means_b1 + cis_b1,
                    color=COLORS['B1_3GPP'], alpha=0.2)

    # Plot B2_Oracle as horizontal line (upper bound)
    b2_data = df_baseline[df_baseline['method'] == 'B2_Oracle']
    if len(b2_data) > 0:
        oracle_se = b2_data['avg_se'].mean()
        oracle_ci = ci_95(b2_data['avg_se'].values)
        ax.axhline(y=oracle_se, color=COLORS['B2_Oracle'], linestyle='--',
                   linewidth=1.5, label=f'{METHOD_LABELS["B2_Oracle"]} (delay=0)')
        ax.axhspan(oracle_se - oracle_ci, oracle_se + oracle_ci,
                   color=COLORS['B2_Oracle'], alpha=0.1)

    ax.set_xlabel('Baseline CSI Delay (TTIs)')
    ax.set_ylabel('Average SE (bits/s/Hz)')
    ax.legend(loc='best', frameon=False)
    ax.set_xlim(delays[0] - 1, delays[-1] + 1)
    ax.grid(True, alpha=0.3)
    add_subplot_label(ax, '(a)')

    # =========================================================================
    # Subplot (b): SE vs RadioMap Staleness
    # =========================================================================
    ax = axes[1]

    # Filter RM CSI data
    df_rm = df[df['delay_type'] == 'rm_csi']

    markers = {'P1_MLP': 'D', 'P2_ISAB': 'v'}
    linestyles = {'P1_MLP': '-', 'P2_ISAB': '--'}

    for method in ['P1_MLP', 'P2_ISAB']:
        method_data = df_rm[df_rm['method'] == method]
        if len(method_data) == 0:
            continue

        delays = sorted(method_data['delay_value'].unique())
        means = []
        cis = []

        for delay in delays:
            subset = method_data[method_data['delay_value'] == delay]
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

        ax.plot(delays, means, marker=marker, linestyle=linestyle, color=color,
                label=label, linewidth=1.8, markersize=6)
        ax.fill_between(delays, means - cis, means + cis, color=color, alpha=0.2)

    # Add reference line for B1_3GPP with default delay (e.g., 12 TTIs)
    # Use the mean SE from B1 with highest delay as comparison
    if len(means_b1) > 0:
        b1_typical = means_b1[delays.index(10)] if 10 in delays else means_b1[-1]
        ax.axhline(y=b1_typical, color=COLORS['B1_3GPP'], linestyle=':',
                   linewidth=1.2, alpha=0.7, label='3GPP (delay=10)')

    ax.set_xlabel('RadioMap Delay/Staleness (TTIs)')
    ax.set_ylabel('Average SE (bits/s/Hz)')
    ax.legend(loc='best', frameon=False)

    rm_delays = sorted(df_rm['delay_value'].unique())
    if len(rm_delays) > 0:
        ax.set_xlim(rm_delays[0] - 0.5, rm_delays[-1] + 0.5)
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
