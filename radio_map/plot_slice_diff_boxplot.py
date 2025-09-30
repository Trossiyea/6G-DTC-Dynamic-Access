import argparse
import os
import sys

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot boxplots of per-pixel differences for each z-slice against a reference slice in a 3D array inside a .mat file.\n"
            "This version converts mW to dBm first, then computes differences in dBm."
        )
    )
    parser.add_argument("input_mat", type=str, help="Path to input .mat file")
    parser.add_argument("--varname", type=str, default="X_true", help="Variable name (default: X_true)")
    parser.add_argument("--ref_z", type=int, default=0, help="Reference z index (default: 0)")
    parser.add_argument("--every", type=int, default=1, help="Label every Nth slice on x-axis (default: 1)")
    parser.add_argument("--showfliers", action="store_true", help="Show outliers in boxplot")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output image path (default: alongside .mat with _diff_boxplot_refZ.png)",
    )
    return parser.parse_args()


def mw_to_dbm(power_mw: np.ndarray) -> np.ndarray:
    # Convert mW to dBm safely; non-positive -> NaN
    with np.errstate(divide="ignore"):
        dbm = np.where(power_mw > 0, 10.0 * np.log10(power_mw), np.nan)
    return dbm


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
        print(f"[ERROR] Variable '{args.varname}' not found. Available: {keys}")
        sys.exit(1)

    arr = mdict[args.varname]
    if not isinstance(arr, np.ndarray) or arr.ndim != 3:
        print(f"[ERROR] Variable '{args.varname}' must be 3D ndarray. Got {getattr(arr, 'shape', None)}")
        sys.exit(1)

    h, w, c = arr.shape
    if not (0 <= args.ref_z < c):
        print(f"[ERROR] ref_z {args.ref_z} out of range [0, {c-1}]")
        sys.exit(1)

    # Convert to dBm before differencing
    ref_mw = arr[:, :, args.ref_z].astype(np.float64, copy=False)
    ref_dbm = mw_to_dbm(ref_mw)

    diffs = []
    labels = []
    for z in range(c):
        sl_mw = arr[:, :, z].astype(np.float64, copy=False)
        sl_dbm = mw_to_dbm(sl_mw)
        diff = (sl_dbm - ref_dbm).reshape(-1)
        diffs.append(diff)
        labels.append(str(z))

    if args.save is None:
        base = os.path.splitext(os.path.basename(args.input_mat))[0]
        out_path = os.path.join(
            os.path.dirname(args.input_mat), f"{base}_{args.varname}_diff_dbm_boxplot_ref{args.ref_z}.png"
        )
    else:
        out_path = args.save

    fig, ax = plt.subplots(figsize=(max(8, c * 0.25), 6), dpi=120)
    bp = ax.boxplot(
        diffs,
        showfliers=args.showfliers,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"facecolor": "#4c78a8"},
        whiskerprops={"color": "#666666"},
        capprops={"color": "#666666"},
    )

    ax.set_title(
        f"dBm diff boxplot per z-slice vs ref z={args.ref_z}\n{os.path.basename(args.input_mat)} | {args.varname}"
    )
    ax.set_ylabel("Difference (dBm)")
    ax.set_xlabel("z index")

    # Reduce x tick clutter
    xticks = np.arange(1, c + 1)
    ax.set_xticks(xticks[:: args.every])
    ax.set_xticklabels([str(i) for i in range(c)][:: args.every], rotation=0)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved boxplot: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()


