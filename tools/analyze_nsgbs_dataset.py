#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze NS-GBS dataset to diagnose training ceiling.

Outputs:
1. Delta gap distribution (Top-1 vs Top-2)
2. Action type distribution (seed vs grow)
3. Feature-label correlation analysis
4. Low-discriminability sample ratio
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_dataset(path: str) -> dict:
    """Load npz dataset and return arrays."""
    data = np.load(path, allow_pickle=True)
    return {
        "features": data["features"],
        "deltas": data["deltas"],
        "actions": data["actions"] if "actions" in data else None,
        "steps": data["step"] if "step" in data else None,
        "labels": data["labels"] if "labels" in data else None,
    }


def load_meta(path: str) -> dict:
    """Load metadata json if exists."""
    meta_path = Path(str(path) + ".json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def analyze_delta_gaps(deltas: np.ndarray) -> dict:
    """Analyze delta gap distribution."""
    gaps_1_2 = []
    gaps_1_last = []
    delta_ranges = []
    num_candidates = []

    for d in deltas:
        d = np.asarray(d, dtype=float)
        if d.size < 2:
            continue
        sorted_d = np.sort(d)[::-1]
        gaps_1_2.append(sorted_d[0] - sorted_d[1])
        gaps_1_last.append(sorted_d[0] - sorted_d[-1])
        delta_ranges.append(sorted_d.max() - sorted_d.min())
        num_candidates.append(len(d))

    gaps_1_2 = np.array(gaps_1_2)
    gaps_1_last = np.array(gaps_1_last)
    delta_ranges = np.array(delta_ranges)
    num_candidates = np.array(num_candidates)

    return {
        "gap_1_2_mean": float(np.mean(gaps_1_2)),
        "gap_1_2_median": float(np.median(gaps_1_2)),
        "gap_1_2_std": float(np.std(gaps_1_2)),
        "gap_1_2_p10": float(np.percentile(gaps_1_2, 10)),
        "gap_1_2_p25": float(np.percentile(gaps_1_2, 25)),
        "gap_1_2_p75": float(np.percentile(gaps_1_2, 75)),
        "gap_1_2_p90": float(np.percentile(gaps_1_2, 90)),
        "gap_1_last_mean": float(np.mean(gaps_1_last)),
        "gap_1_last_median": float(np.median(gaps_1_last)),
        "delta_range_mean": float(np.mean(delta_ranges)),
        "delta_range_median": float(np.median(delta_ranges)),
        "num_candidates_mean": float(np.mean(num_candidates)),
        "num_candidates_min": int(np.min(num_candidates)),
        "num_candidates_max": int(np.max(num_candidates)),
        "gaps_1_2": gaps_1_2,  # for histogram
    }


def analyze_action_types(actions: np.ndarray) -> dict:
    """Analyze action type distribution."""
    # actions shape: [num_samples, num_actions, 3] where [..., 0] is action type
    # action types: 0=seed, 1=grow_left, 2=grow_right, 3=fallback
    type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total = 0

    for a in actions:
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[1] >= 1:
            for action_type in a[:, 0]:
                type_counts[int(action_type)] = type_counts.get(int(action_type), 0) + 1
                total += 1

    labels = {0: "seed", 1: "grow_left", 2: "grow_right", 3: "fallback"}
    return {
        "counts": {labels.get(k, f"type_{k}"): v for k, v in type_counts.items()},
        "ratios": {labels.get(k, f"type_{k}"): v / max(1, total) for k, v in type_counts.items()},
        "total": total,
    }


def analyze_feature_label_correlation(features: np.ndarray, deltas: np.ndarray, window_size: int = 3) -> dict:
    """Analyze correlation between window center feature and delta."""
    if not HAS_SCIPY:
        return {"error": "scipy not available"}

    correlations_pearson = []
    correlations_spearman = []
    window_center_idx = window_size // 2

    for feat, d in zip(features, deltas):
        feat = np.asarray(feat, dtype=float)
        d = np.asarray(d, dtype=float)
        if feat.ndim != 2 or feat.shape[0] < 2:
            continue
        if d.shape[0] != feat.shape[0]:
            continue

        # Window center feature (predicted SE at target PRB)
        if feat.shape[1] > window_center_idx:
            window_center = feat[:, window_center_idx]
            try:
                r_pearson, _ = pearsonr(window_center, d)
                r_spearman, _ = spearmanr(window_center, d)
                if np.isfinite(r_pearson):
                    correlations_pearson.append(r_pearson)
                if np.isfinite(r_spearman):
                    correlations_spearman.append(r_spearman)
            except Exception:
                pass

    return {
        "pearson_mean": float(np.mean(correlations_pearson)) if correlations_pearson else None,
        "pearson_median": float(np.median(correlations_pearson)) if correlations_pearson else None,
        "pearson_std": float(np.std(correlations_pearson)) if correlations_pearson else None,
        "spearman_mean": float(np.mean(correlations_spearman)) if correlations_spearman else None,
        "spearman_median": float(np.median(correlations_spearman)) if correlations_spearman else None,
        "num_valid_samples": len(correlations_pearson),
    }


def analyze_low_discriminability(deltas: np.ndarray, threshold: float = 0.1) -> dict:
    """Analyze ratio of low-discriminability samples."""
    low_disc_count = 0
    total = 0

    for d in deltas:
        d = np.asarray(d, dtype=float)
        if d.size < 2:
            continue
        sorted_d = np.sort(d)[::-1]
        gap = sorted_d[0] - sorted_d[1]
        if gap < threshold:
            low_disc_count += 1
        total += 1

    return {
        "threshold": threshold,
        "low_disc_count": low_disc_count,
        "total": total,
        "low_disc_ratio": low_disc_count / max(1, total),
    }


def print_report(stats: dict, meta: dict) -> None:
    """Print analysis report."""
    print("=" * 60)
    print("NS-GBS Dataset Analysis Report")
    print("=" * 60)

    if meta:
        print(f"\nDataset: {meta.get('out', 'N/A')}")
        print(f"Samples: {meta.get('num_samples', 'N/A')}")
        print(f"Feature dim: {meta.get('feature_dim', 'N/A')}")
        print(f"TopB: {meta.get('topB', 'N/A')}, Window: {meta.get('window', 'N/A')}")

    print("\n" + "-" * 60)
    print("1. DELTA GAP DISTRIBUTION (Top-1 vs Top-2)")
    print("-" * 60)
    gap_stats = stats.get("delta_gaps", {})
    print(f"  Mean gap:     {gap_stats.get('gap_1_2_mean', 'N/A'):.4f}")
    print(f"  Median gap:   {gap_stats.get('gap_1_2_median', 'N/A'):.4f}")
    print(f"  Std gap:      {gap_stats.get('gap_1_2_std', 'N/A'):.4f}")
    print(f"  P10 / P90:    {gap_stats.get('gap_1_2_p10', 'N/A'):.4f} / {gap_stats.get('gap_1_2_p90', 'N/A'):.4f}")
    print(f"  Full range:   {gap_stats.get('delta_range_mean', 'N/A'):.4f} (mean)")
    print(f"  Candidates:   {gap_stats.get('num_candidates_mean', 'N/A'):.1f} avg "
          f"[{gap_stats.get('num_candidates_min', 'N/A')}-{gap_stats.get('num_candidates_max', 'N/A')}]")

    print("\n" + "-" * 60)
    print("2. ACTION TYPE DISTRIBUTION")
    print("-" * 60)
    action_stats = stats.get("action_types", {})
    for name, ratio in action_stats.get("ratios", {}).items():
        count = action_stats.get("counts", {}).get(name, 0)
        print(f"  {name:12s}: {count:8d} ({ratio*100:5.1f}%)")

    print("\n" + "-" * 60)
    print("3. FEATURE-LABEL CORRELATION")
    print("-" * 60)
    corr_stats = stats.get("correlation", {})
    if corr_stats.get("error"):
        print(f"  Error: {corr_stats['error']}")
    else:
        print(f"  Pearson (mean):   {corr_stats.get('pearson_mean', 'N/A'):.4f}" if corr_stats.get('pearson_mean') else "  Pearson: N/A")
        print(f"  Pearson (median): {corr_stats.get('pearson_median', 'N/A'):.4f}" if corr_stats.get('pearson_median') else "")
        print(f"  Spearman (mean):  {corr_stats.get('spearman_mean', 'N/A'):.4f}" if corr_stats.get('spearman_mean') else "  Spearman: N/A")
        print(f"  Valid samples:    {corr_stats.get('num_valid_samples', 'N/A')}")

    print("\n" + "-" * 60)
    print("4. LOW DISCRIMINABILITY SAMPLES")
    print("-" * 60)
    for thresh_stats in stats.get("low_discriminability", []):
        thresh = thresh_stats.get("threshold", 0)
        ratio = thresh_stats.get("low_disc_ratio", 0)
        count = thresh_stats.get("low_disc_count", 0)
        total = thresh_stats.get("total", 0)
        print(f"  Gap < {thresh:.2f}: {count:6d} / {total} ({ratio*100:5.1f}%)")

    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    # Interpretation
    gap_median = gap_stats.get("gap_1_2_median", 0)
    low_disc_ratio = stats.get("low_discriminability", [{}])[0].get("low_disc_ratio", 0)
    pearson_corr = corr_stats.get("pearson_mean", 0)

    issues = []
    if gap_median < 0.1:
        issues.append(f"- LOW GAP: Median gap ({gap_median:.4f}) < 0.1 indicates hard-to-distinguish candidates")
    if low_disc_ratio > 0.5:
        issues.append(f"- HIGH NOISE: {low_disc_ratio*100:.1f}% samples have gap < 0.1 (>50% is problematic)")
    if pearson_corr is not None and pearson_corr < 0.5:
        issues.append(f"- WEAK CORRELATION: Pearson r={pearson_corr:.3f} < 0.5 indicates feature-label mismatch")

    if issues:
        print("ISSUES DETECTED:")
        for issue in issues:
            print(issue)
    else:
        print("No major issues detected. Consider model capacity or longer training.")

    print("\nRECOMMENDATIONS:")
    if gap_median < 0.1 or low_disc_ratio > 0.3:
        print("- Filter low-discriminability samples (gap < threshold)")
        print("- Or use weighted loss to down-weight ambiguous samples")
    if pearson_corr is not None and pearson_corr < 0.5:
        print("- Feature-label mismatch: consider using true SE in features (offline)")
        print("- Or increase window size to capture more context")
    print("- Try tau sweep: [0.1, 0.15, 0.2, 0.3, 0.5]")


def main():
    parser = argparse.ArgumentParser(description="Analyze NS-GBS dataset.")
    parser.add_argument("--data", required=True, help="Path to dataset npz")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5],
                        help="Gap thresholds for low-discriminability analysis")
    parser.add_argument("--window", type=int, default=3, help="Window size for correlation analysis")
    parser.add_argument("--output", type=str, default=None, help="Output json path (optional)")
    args = parser.parse_args()

    print(f"Loading dataset: {args.data}")
    data = load_dataset(args.data)
    meta = load_meta(args.data)

    features = data["features"]
    deltas = data["deltas"]
    actions = data["actions"]

    print(f"Samples: {len(deltas)}")

    # Analyze
    stats = {}

    print("Analyzing delta gaps...")
    stats["delta_gaps"] = analyze_delta_gaps(deltas)

    if actions is not None:
        print("Analyzing action types...")
        stats["action_types"] = analyze_action_types(actions)

    print("Analyzing feature-label correlation...")
    stats["correlation"] = analyze_feature_label_correlation(
        features, deltas, window_size=meta.get("window", args.window)
    )

    print("Analyzing low discriminability...")
    stats["low_discriminability"] = [
        analyze_low_discriminability(deltas, threshold=t)
        for t in sorted(args.thresholds)
    ]

    # Print report
    print_report(stats, meta)

    # Save json if requested
    if args.output:
        # Remove numpy arrays for json serialization
        stats_json = {k: v for k, v in stats.items()}
        if "delta_gaps" in stats_json and "gaps_1_2" in stats_json["delta_gaps"]:
            del stats_json["delta_gaps"]["gaps_1_2"]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats_json, f, ensure_ascii=False, indent=2)
        print(f"\nSaved analysis to: {args.output}")


if __name__ == "__main__":
    main()
