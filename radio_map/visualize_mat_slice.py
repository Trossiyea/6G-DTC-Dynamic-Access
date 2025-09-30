import argparse
import os
import sys

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a z-axis slice from a 3D matrix stored in a .mat file. "
            "Assumes the 3D array layout is (H, W, C) and values are in mW. "
            "The script converts to dBm for visualization."
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
        "--z",
        type=int,
        default=0,
        help="Z index (0-based) to visualize (default: 0)",
    )
    parser.add_argument(
        "--vmin_dbm",
        type=float,
        default=None,
        help="Color scale min in dBm (default: auto)",
    )
    parser.add_argument(
        "--vmax_dbm",
        type=float,
        default=None,
        help="Color scale max in dBm (default: auto)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name (default: viridis)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the figure (default: <mat_name>_z<idx>.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively (in addition to saving)",
    )
    return parser.parse_args()


def mw_to_dbm(power_mw: np.ndarray) -> np.ndarray:
    # Avoid log of non-positive values; map <=0 to NaN
    with np.errstate(divide="ignore"):
        dbm = np.where(power_mw > 0, 10.0 * np.log10(power_mw), np.nan)
    return dbm


def main():
    args = parse_args()

    if not os.path.isfile(args.input_mat):
        print(f"[ERROR] Input .mat not found: {args.input_mat}")
        sys.exit(1)

    try:
        mdict = loadmat(args.input_mat)
    except Exception as e:
        print(f"[ERROR] Failed to load .mat: {e}")
        sys.exit(1)

    if args.varname not in mdict:
        print(f"[ERROR] Variable '{args.varname}' not found in {args.input_mat}")
        # List candidates
        keys = [k for k in mdict.keys() if not k.startswith('__')]
        print(f" Available variables: {keys}")
        sys.exit(1)

    arr = mdict[args.varname]
    if arr.ndim != 3:
        print(f"[ERROR] Expected 3D array for '{args.varname}', got shape {arr.shape}")
        sys.exit(1)

    h, w, c = arr.shape
    if not (0 <= args.z < c):
        print(f"[ERROR] z index {args.z} out of range [0, {c-1}]")
        sys.exit(1)

    slice_mw = arr[:, :, args.z]
    slice_dbm = mw_to_dbm(slice_mw.astype(np.float64, copy=False))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    im = ax.imshow(
        slice_dbm,
        origin="lower",
        cmap=args.cmap,
        vmin=args.vmin_dbm,
        vmax=args.vmax_dbm,
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power (dBm)")
    ax.set_title(f"{os.path.basename(args.input_mat)} | {args.varname} z={args.z} (dBm)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.grid(False)

    if args.save is None:
        base = os.path.splitext(os.path.basename(args.input_mat))[0]
        save_path = os.path.join(os.path.dirname(args.input_mat), f"{base}_{args.varname}_z{args.z}_dbm.png")
    else:
        save_path = args.save

    try:
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save figure: {e}")
        # Continue to optionally show

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()


