import argparse
import csv
import os
import sys

import numpy as np
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute difference statistics per z-slice against a reference slice in a 3D array inside a .mat file."
        )
    )
    parser.add_argument("input_mat", type=str, help="Path to input .mat file")
    parser.add_argument(
        "--varname",
        type=str,
        default="X_true",
        help="Variable name inside .mat (default: X_true)",
    )
    parser.add_argument(
        "--ref_z",
        type=int,
        default=0,
        help="Reference z index to compare against (default: 0)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional output CSV path (default: alongside .mat with _diff_stats.csv)",
    )
    return parser.parse_args()


def compute_stats(diff_flat: np.ndarray) -> dict:
    # diff_flat is 1D float64 array
    stats = {
        "mean": float(np.nanmean(diff_flat)),
        "std": float(np.nanstd(diff_flat)),
        "min": float(np.nanmin(diff_flat)),
        "p1": float(np.nanpercentile(diff_flat, 1)),
        "p5": float(np.nanpercentile(diff_flat, 5)),
        "p25": float(np.nanpercentile(diff_flat, 25)),
        "p50": float(np.nanpercentile(diff_flat, 50)),
        "p75": float(np.nanpercentile(diff_flat, 75)),
        "p95": float(np.nanpercentile(diff_flat, 95)),
        "p99": float(np.nanpercentile(diff_flat, 99)),
        "max": float(np.nanmax(diff_flat)),
        "mae": float(np.nanmean(np.abs(diff_flat))),
        "rmse": float(np.sqrt(np.nanmean(diff_flat ** 2))),
    }
    return stats


def main():
    args = parse_args()

    if not os.path.isfile(args.input_mat):
        print(f"[ERROR] File not found: {args.input_mat}")
        sys.exit(1)

    try:
        mdict = loadmat(args.input_mat)
    except Exception as e:
        print(f"[ERROR] Failed to load .mat: {e}")
        sys.exit(1)

    if args.varname not in mdict:
        keys = [k for k in mdict.keys() if not k.startswith('__')]
        print(f"[ERROR] Variable '{args.varname}' not in file. Available: {keys}")
        sys.exit(1)

    arr = mdict[args.varname]
    if not isinstance(arr, np.ndarray) or arr.ndim != 3:
        print(f"[ERROR] Variable '{args.varname}' must be a 3D ndarray. Got shape {getattr(arr, 'shape', None)}")
        sys.exit(1)

    h, w, c = arr.shape
    if not (0 <= args.ref_z < c):
        print(f"[ERROR] ref_z {args.ref_z} out of range [0, {c-1}]")
        sys.exit(1)

    ref = arr[:, :, args.ref_z].astype(np.float64, copy=False)

    if args.out_csv is None:
        base = os.path.splitext(os.path.basename(args.input_mat))[0]
        args.out_csv = os.path.join(os.path.dirname(args.input_mat), f"{base}_{args.varname}_diff_stats_ref{args.ref_z}.csv")

    fieldnames = [
        "z", "mean", "std", "min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max", "mae", "rmse",
    ]

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for z in range(c):
            sl = arr[:, :, z].astype(np.float64, copy=False)
            diff = (sl - ref).reshape(-1)
            stats = compute_stats(diff)
            row = {"z": z}
            row.update(stats)
            writer.writerow(row)

    print(f"Saved stats CSV: {args.out_csv}")


if __name__ == "__main__":
    main()




