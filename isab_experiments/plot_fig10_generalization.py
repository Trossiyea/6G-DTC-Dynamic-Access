#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot for Figure 10: Learning Generalization + Ablations.

Generates two subplots:
(a) Grouped bar: Gain by train-test city pair
(b) Line plot: SE vs τ (ISAB temperature parameter)
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
        description="Plot Figure 10: Generalization"
    )
    parser.add_argument(
        "--csv",
        default="output/results/fig10_generalization.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out",
        default="output/figures/fig10_generalization.pdf",
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

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Define colors for variants
    variant_colors = {
        'MLP': COLORS.get('P1_MLP', '#F58518'),
        'ISAB': COLORS.get('P2_ISAB', '#E45756'),
    }

    # =========================================================================
    # Subplot (a): Cross-city generalization
    # =========================================================================
    ax = axes[0]

    df_cross = df[df['exp_type'] == 'cross_city']

    if len(df_cross) > 0:
        # Create city pair labels
        df_cross = df_cross.copy()
        df_cross['city_pair'] = df_cross['train_city'] + '→' + df_cross['test_city']

        city_pairs = df_cross['city_pair'].unique().tolist()
        variants = df_cross['model_variant'].unique().tolist()

        x = np.arange(len(city_pairs))
        bar_width = 0.35
        n_variants = len(variants)
        offsets = np.linspace(-(n_variants - 1) / 2, (n_variants - 1) / 2, n_variants) * bar_width

        for i, variant in enumerate(variants):
            means = []
            cis = []

            for pair in city_pairs:
                subset = df_cross[(df_cross['city_pair'] == pair) &
                                  (df_cross['model_variant'] == variant)]
                gain_values = subset['gain_pct'].dropna().values

                if len(gain_values) > 0:
                    means.append(np.mean(gain_values))
                    cis.append(ci_95(gain_values))
                else:
                    means.append(0)
                    cis.append(0)

            color = variant_colors.get(variant, f'C{i}')

            ax.bar(x + offsets[i], means, bar_width, yerr=cis,
                   label=f'RM + {variant}', color=color, edgecolor='black', linewidth=0.5,
                   capsize=3, error_kw={'linewidth': 0.8})

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(city_pairs, rotation=20, ha='right')
        ax.set_xlabel('Train → Test City')
        ax.set_ylabel('Gain vs 3GPP Baseline (%)')
        ax.legend(loc='best', frameon=False)
    else:
        ax.text(0.5, 0.5, 'No cross-city data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Train → Test City')
        ax.set_ylabel('Gain vs 3GPP Baseline (%)')

    ax.grid(True, alpha=0.3, axis='y')
    add_subplot_label(ax, '(a)')

    # =========================================================================
    # Subplot (b): Tau sweep
    # =========================================================================
    ax = axes[1]

    df_tau = df[df['exp_type'] == 'tau_sweep']

    if len(df_tau) > 0:
        tau_values = sorted(df_tau['tau'].unique())
        means = []
        cis = []

        for tau in tau_values:
            subset = df_tau[df_tau['tau'] == tau]
            se_values = subset['avg_se'].dropna().values

            if len(se_values) > 0:
                means.append(np.mean(se_values))
                cis.append(ci_95(se_values))
            else:
                means.append(np.nan)
                cis.append(0)

        means = np.array(means)
        cis = np.array(cis)

        ax.plot(tau_values, means, 'o-', color=COLORS.get('P2_ISAB', '#E45756'),
                linewidth=1.8, markersize=8, label='ISAB')
        ax.fill_between(tau_values, means - cis, means + cis,
                        color=COLORS.get('P2_ISAB', '#E45756'), alpha=0.2)

        # Mark best tau
        if len(means) > 0:
            best_idx = np.nanargmax(means)
            best_tau = tau_values[best_idx]
            best_se = means[best_idx]
            ax.scatter([best_tau], [best_se], s=150, c='gold', marker='*',
                       edgecolors='black', linewidths=0.5, zorder=5)
            ax.annotate(f'Best τ={best_tau}', (best_tau, best_se),
                        textcoords="offset points", xytext=(10, 5), fontsize=9)

        ax.set_xlabel('ISAB Temperature (τ)')
        ax.set_ylabel('Average SE (bits/s/Hz)')
        ax.legend(loc='best', frameon=False)
    else:
        ax.text(0.5, 0.5, 'No tau sweep data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('ISAB Temperature (τ)')
        ax.set_ylabel('Average SE (bits/s/Hz)')

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
