#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot for Figure 8: Reliability & HARQ Goodput.

Generates two subplots:
(a) Grouped bar: ACK rate and TB drop rate by method
(b) Grouped bar: Avg retx per acked TB + OLLA offset
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
        description="Plot Figure 8: HARQ Reliability"
    )
    parser.add_argument(
        "--csv",
        default="output/results/fig08_harq.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out",
        default="output/figures/fig08_harq.pdf",
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

    x = np.arange(len(methods))
    bar_width = 0.35

    # =========================================================================
    # Subplot (a): ACK rate and Drop rate
    # =========================================================================
    ax = axes[0]

    ack_means = []
    ack_cis = []
    drop_means = []
    drop_cis = []

    for method in methods:
        method_data = df[df['method'] == method]

        ack_values = method_data['ack_rate'].dropna().values * 100  # Convert to %
        drop_values = method_data['drop_rate'].dropna().values * 100  # Convert to %

        if len(ack_values) > 0:
            ack_means.append(np.mean(ack_values))
            ack_cis.append(ci_95(ack_values))
        else:
            ack_means.append(0)
            ack_cis.append(0)

        if len(drop_values) > 0:
            drop_means.append(np.mean(drop_values))
            drop_cis.append(ci_95(drop_values))
        else:
            drop_means.append(0)
            drop_cis.append(0)

    colors_bars = [COLORS.get(m, f'C{i}') for i, m in enumerate(methods)]

    # Plot ACK rate
    bars1 = ax.bar(x - bar_width/2, ack_means, bar_width, yerr=ack_cis,
                   color=colors_bars, edgecolor='black', linewidth=0.5,
                   capsize=3, error_kw={'linewidth': 0.8}, alpha=0.8,
                   label='ACK Rate (%)')

    ax.set_ylabel('ACK Rate (%)')
    ax.set_ylim(80, 100)  # Focus on high ACK rates

    # Plot Drop rate on secondary axis
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + bar_width/2, drop_means, bar_width, yerr=drop_cis,
                    color=colors_bars, edgecolor='black', linewidth=0.5,
                    capsize=3, error_kw={'linewidth': 0.8}, alpha=0.4,
                    hatch='//', label='Drop Rate (%)')

    ax2.set_ylabel('TB Drop Rate (%)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, max(drop_means) * 2 if max(drop_means) > 0 else 1)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=15, ha='right')
    ax.set_xlabel('Method')

    # Legend
    ax.legend([bars1, bars2], ['ACK Rate (%)', 'Drop Rate (%)'], loc='lower right', frameon=False)

    add_subplot_label(ax, '(a)')

    # =========================================================================
    # Subplot (b): Avg retx and OLLA offset
    # =========================================================================
    ax = axes[1]

    retx_means = []
    retx_cis = []
    olla_means = []
    olla_cis = []

    for method in methods:
        method_data = df[df['method'] == method]

        retx_values = method_data['avg_retx'].dropna().values
        olla_values = method_data['olla_offset_avg'].dropna().values

        if len(retx_values) > 0:
            retx_means.append(np.mean(retx_values))
            retx_cis.append(ci_95(retx_values))
        else:
            retx_means.append(0)
            retx_cis.append(0)

        if len(olla_values) > 0:
            olla_means.append(np.mean(olla_values))
            olla_cis.append(ci_95(olla_values))
        else:
            olla_means.append(0)
            olla_cis.append(0)

    # Plot avg retx
    bars1 = ax.bar(x - bar_width/2, retx_means, bar_width, yerr=retx_cis,
                   color=colors_bars, edgecolor='black', linewidth=0.5,
                   capsize=3, error_kw={'linewidth': 0.8}, alpha=0.8,
                   label='Avg Retx/TB')

    ax.set_ylabel('Avg Retransmissions per Acked TB')
    ax.set_ylim(0, max(retx_means) * 1.5 if max(retx_means) > 0 else 1)

    # Plot OLLA offset on secondary axis
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + bar_width/2, olla_means, bar_width, yerr=olla_cis,
                    color=colors_bars, edgecolor='black', linewidth=0.5,
                    capsize=3, error_kw={'linewidth': 0.8}, alpha=0.4,
                    hatch='\\\\', label='OLLA Offset (dB)')

    ax2.set_ylabel('OLLA Offset (dB)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=15, ha='right')
    ax.set_xlabel('Method')

    # Legend
    ax.legend([bars1, bars2], ['Avg Retx/TB', 'OLLA Offset (dB)'], loc='upper right', frameon=False)

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
