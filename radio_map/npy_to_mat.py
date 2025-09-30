import argparse
import os
import sys

import numpy as np
from scipy.io import savemat


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a .npy file to .mat and reorder axes from (51, 1000, 1000) to (1000, 1000, 51)."
        )
    )
    parser.add_argument(
        "input_npy",
        type=str,
        help="Path to input .npy file",
    )
    parser.add_argument(
        "output_mat",
        type=str,
        nargs="?",
        default=None,
        help="Optional path to output .mat file (default: same name as input with .mat)",
    )
    parser.add_argument(
        "--varname",
        type=str,
        default="X_true",
        help="Variable name to store in .mat file (default: X_true)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input_npy
    if args.output_mat is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".mat"
    else:
        output_path = args.output_mat

    if not os.path.isfile(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    try:
        data = np.load(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to load npy file: {e}")
        sys.exit(1)

    if data.ndim != 3:
        print(f"[ERROR] Expected 3D array, got shape {data.shape}")
        sys.exit(1)

    # Expect (51, 1000, 1000) -> (1000, 1000, 51)
    # Perform a generic axes reorder: (C, H, W) -> (H, W, C)
    reordered = np.transpose(data, (1, 2, 0))

    # Convert units: W -> mW by multiplying 1000
    # Save as float64 to align with typical MATLAB double precision
    reordered_mw = (reordered * 1000.0).astype(np.float64, copy=False)

    try:
        savemat(output_path, {args.varname: reordered_mw})
    except Exception as e:
        print(f"[ERROR] Failed to save mat file: {e}")
        sys.exit(1)

    print(f"Converted: {input_path}")
    print(f" Original shape: {tuple(data.shape)}")
    print(f" Reordered shape: {tuple(reordered.shape)}")
    print(" Units converted: W -> mW (x1000)")
    print(f"Saved .mat to: {output_path} (variable: {args.varname})")


if __name__ == "__main__":
    main()


