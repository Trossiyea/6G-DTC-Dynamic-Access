#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common plotting utilities for IEEE TMC experiment figures.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Color palette for different methods
COLORS = {
    'B1_3GPP': '#4C78A8',       # Blue
    'B2_Oracle': '#9ECAE1',     # Light blue
    'B3_heuristic': '#54A24B',  # Green
    'P1_MLP': '#F58518',        # Orange
    'P2_ISAB': '#E45756',       # Red
}

# Method display labels
METHOD_LABELS = {
    'B1_3GPP': '3GPP CSI/CQI',
    'B2_Oracle': 'Oracle CSI',
    'B3_heuristic': 'RM Heuristic',
    'P1_MLP': 'RM + MLP',
    'P2_ISAB': 'RM + ISAB',
}

# Method order for consistent plotting
METHOD_ORDER = ['B1_3GPP', 'B2_Oracle', 'B3_heuristic', 'P1_MLP', 'P2_ISAB']


def setup_style():
    """Setup matplotlib style for publication-quality figures."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass

    plt.rcParams.update({
        'figure.figsize': (7, 4),
        'figure.dpi': 100,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 1.8,
        'lines.markersize': 5,
    })


def ci_95(data):
    """
    Compute 95% confidence interval half-width using t-distribution.

    Args:
        data: Array-like of values

    Returns:
        Half-width of 95% CI (use as ±error)
    """
    data = np.asarray(data)
    n = len(data)
    if n < 2:
        return 0.0
    se = np.std(data, ddof=1) / np.sqrt(n)
    # For n >= 30, use 1.96; otherwise approximate with t-distribution
    if n >= 30:
        t_val = 1.96
    else:
        # Simplified t-value approximation for common sample sizes
        t_vals = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45,
                  8: 2.37, 9: 2.31, 10: 2.26, 15: 2.14, 20: 2.09, 25: 2.06}
        t_val = t_vals.get(n, 2.0)
    return t_val * se


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95, stat_func=np.mean):
    """
    Compute confidence interval using bootstrap resampling.

    Args:
        data: Array-like of values
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 0.95)
        stat_func: Statistic function (default np.mean)

    Returns:
        Tuple of (lower, upper) bounds
    """
    data = np.asarray(data)
    n = len(data)
    if n < 2:
        val = stat_func(data)
        return val, val

    rng = np.random.default_rng(42)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(sample))

    alpha = 1 - ci
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lower, upper


def save_fig(fig, path, dpi=300, formats=('pdf', 'png')):
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure object
        path: Output path (with or without extension)
        dpi: DPI for raster formats (default 300 for PDF, 150 for PNG)
        formats: Tuple of formats to save (default: pdf and png)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove extension if present
    stem = path.stem
    parent = path.parent

    for fmt in formats:
        out_path = parent / f"{stem}.{fmt}"
        fig_dpi = dpi if fmt == 'pdf' else 150
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {out_path}")

    plt.close(fig)


def add_subplot_label(ax, label, x=-0.12, y=1.05):
    """
    Add subplot label (a), (b), etc. to axis.

    Args:
        ax: Matplotlib axis
        label: Label text (e.g., "(a)")
        x, y: Position in axis coordinates
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom', ha='left')


def grouped_bar_plot(ax, data, groups, methods, colors=None, ylabel='',
                     show_ci=True, bar_width=0.15):
    """
    Create grouped bar plot with error bars.

    Args:
        ax: Matplotlib axis
        data: Dict of {method: {group: (mean, ci)}}
        groups: List of group names (x-axis)
        methods: List of method names
        colors: Dict of method -> color
        ylabel: Y-axis label
        show_ci: Whether to show error bars
        bar_width: Width of each bar
    """
    if colors is None:
        colors = COLORS

    x = np.arange(len(groups))
    n_methods = len(methods)
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

    for i, method in enumerate(methods):
        means = []
        cis = []
        for group in groups:
            if method in data and group in data[method]:
                m, c = data[method][group]
                means.append(m)
                cis.append(c if show_ci else 0)
            else:
                means.append(0)
                cis.append(0)

        color = colors.get(method, f'C{i}')
        label = METHOD_LABELS.get(method, method)

        ax.bar(x + offsets[i], means, bar_width, yerr=cis if show_ci else None,
               label=label, color=color, edgecolor='black', linewidth=0.5,
               capsize=2, error_kw={'linewidth': 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', frameon=False)


def line_plot_with_ci(ax, x_values, data, methods, colors=None, ylabel='',
                      xlabel='', markers=None):
    """
    Create line plot with shaded confidence intervals.

    Args:
        ax: Matplotlib axis
        x_values: X-axis values
        data: Dict of {method: {'mean': [...], 'ci': [...]}}
        methods: List of method names
        colors: Dict of method -> color
        ylabel: Y-axis label
        xlabel: X-axis label
        markers: Dict of method -> marker style
    """
    if colors is None:
        colors = COLORS
    if markers is None:
        markers = {'B1_3GPP': 'o', 'B2_Oracle': 's', 'B3_heuristic': '^',
                   'P1_MLP': 'D', 'P2_ISAB': 'v'}

    for method in methods:
        if method not in data:
            continue

        means = np.array(data[method]['mean'])
        cis = np.array(data[method].get('ci', np.zeros_like(means)))

        color = colors.get(method, 'gray')
        marker = markers.get(method, 'o')
        label = METHOD_LABELS.get(method, method)

        ax.plot(x_values, means, marker=marker, color=color, label=label)
        ax.fill_between(x_values, means - cis, means + cis,
                        color=color, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', frameon=False)


def format_scenario_name(scenario):
    """Format scenario name for display."""
    mapping = {
        'toronto_single': 'Toronto (Single)',
        'toronto_constellation': 'Toronto (Const.)',
        'shanghai_single': 'Shanghai (Single)',
        'shanghai_constellation': 'Shanghai (Const.)',
        'toronto_125m': 'Toronto 125m',
        'toronto_150m': 'Toronto 150m',
        'shanghai_125m': 'Shanghai 125m',
        'shanghai_150m': 'Shanghai 150m',
    }
    return mapping.get(scenario, scenario)
