import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np

# Prefer SciPy for legacy MAT files; fall back to h5py for v7.3
try:
    from scipy.io import loadmat  # type: ignore
except Exception:  # pragma: no cover
    loadmat = None  # type: ignore

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def _list_numeric_arrays(mdict: dict) -> dict:
    return {
        k: v
        for k, v in mdict.items()
        if (not k.startswith("__")) and isinstance(v, np.ndarray) and v.size > 0
    }


def load_mat_variable(filepath: str, varname: Optional[str] = None) -> Tuple[np.ndarray, str]:
    """
    Load a variable from a .mat file, trying SciPy first (legacy MAT) then h5py (v7.3).

    If varname is None, auto-select the largest 3D numeric array.

    Returns (array, used_varname).
    """

    # Try SciPy first
    mdict = None
    scipy_error = None
    if loadmat is not None:
        try:
            mdict = loadmat(filepath)
            mdict = {k: v for k, v in mdict.items() if not k.startswith("__")}
        except Exception as e:  # pragma: no cover
            scipy_error = e

    if mdict is None and h5py is not None:
        try:
            with h5py.File(filepath, "r") as f:
                mdict = {}
                for k in f.keys():
                    obj = f[k]
                    if isinstance(obj, h5py.Dataset):
                        mdict[k] = np.array(obj)
        except Exception as e:  # pragma: no cover
            if scipy_error:
                raise RuntimeError(
                    f"Failed to load MAT via SciPy ({scipy_error}) and h5py ({e})."
                )
            raise

    if mdict is None:
        raise RuntimeError(
            "No MAT loader available. Please install scipy and/or h5py."
        )

    arrays = _list_numeric_arrays(mdict)
    if not arrays:
        raise KeyError("No numeric arrays found in MAT file.")

    chosen_name = None
    if varname is not None:
        if varname in arrays:
            chosen_name = varname
        else:
            # Graceful fallback to auto-detect if requested var not found
            print(
                f"[WARN] Variable '{varname}' not found. Auto-selecting a 3D array...",
                file=sys.stderr,
            )
    if chosen_name is None:
        # Pick the largest 3D array if possible, otherwise the largest array
        candidates = [(k, v) for k, v in arrays.items() if v.ndim == 3]
        if candidates:
            chosen_name = max(candidates, key=lambda kv: kv[1].size)[0]
        else:
            chosen_name = max(arrays.items(), key=lambda kv: kv[1].size)[0]

    arr = arrays[chosen_name]
    return arr, chosen_name


def maybe_downsample(arr: np.ndarray, max_voxels: int) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Downsample the array by simple striding along each axis until total voxel
    count <= max_voxels. Returns (downsampled_array, strides).
    """
    H, W, C = arr.shape
    stride_h = stride_w = stride_c = 1
    total = H * W * C
    if total <= max_voxels:
        return arr, (1, 1, 1)

    # Greedy striding to reduce voxel count while preserving proportions
    while (H // stride_h) * (W // stride_w) * (C // stride_c) > max_voxels:
        # Choose axis with largest current dimension to stride further
        dims = [H // stride_h, W // stride_w, C // stride_c]
        axis = int(np.argmax(dims))
        if axis == 0:
            stride_h += 1
        elif axis == 1:
            stride_w += 1
        else:
            stride_c += 1
        # Safety break to avoid infinite loop
        if stride_h > H and stride_w > W and stride_c > C:
            break

    ds = arr[::stride_h, ::stride_w, ::stride_c]
    return ds, (stride_h, stride_w, stride_c)


def build_voxel_colors(values: np.ndarray, vmin: Optional[float], vmax: Optional[float], cmap_name: str, alpha: float) -> Tuple[np.ndarray, Normalize, ScalarMappable]:
    """
    Map dBm values to RGBA colors using a colormap.
    Returns (colors_rgba, norm, mappable_for_colorbar).
    """
    if vmin is None:
        vmin = np.nanpercentile(values, 2)
    if vmax is None:
        vmax = np.nanpercentile(values, 98)
    if vmin >= vmax:
        vmax = vmin + 1e-6

    cmap = matplotlib.cm.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(values))
    rgba[..., 3] = alpha
    sm = ScalarMappable(norm=norm, cmap=cmap)
    return rgba, norm, sm


def plot_voxels_cube(
    arr_dbm: np.ndarray,
    title: str,
    cmap: str = "jet",
    alpha: float = 0.6,
    vmin_dbm: Optional[float] = None,
    vmax_dbm: Optional[float] = None,
    max_voxels: int = 150_000,
    save_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 200,
):
    """
    Render a 3D voxel cube where each cell color encodes interference power (dBm).
    X, Y: grid coordinates; Z: PRB index.
    """
    if arr_dbm.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {arr_dbm.shape}")

    # Downsample if needed for performance
    arr_used, strides = maybe_downsample(arr_dbm, max_voxels=max_voxels)
    if strides != (1, 1, 1):
        print(f"[INFO] Downsampled with strides (H, W, C) = {strides}, new shape = {arr_used.shape}")

    # Normalize to colors
    colors, norm, sm = build_voxel_colors(arr_used, vmin_dbm, vmax_dbm, cmap, alpha)

    # Every cell is filled
    filled = np.ones(arr_used.shape, dtype=bool)

    fig = plt.figure(figsize=(9, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    ax.voxels(filled, facecolors=colors, edgecolor=None)

    # Axis labels and ticks
    ax.set_xlabel("X (grid)")
    ax.set_ylabel("Y (grid)")
    ax.set_zlabel("PRB index (Z)")
    ax.set_title(title, pad=10)

    # Colorbar using the same colormap + norm
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.05)
    cbar.set_label("Interference Power (dBm)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved 3D voxel heatmap: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_slices_grid(
    arr_dbm: np.ndarray,
    title: str,
    num_slices: int = 6,
    cmap: str = "jet",
    vmin_dbm: Optional[float] = None,
    vmax_dbm: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 200,
    title_pad: float = 10.0,
    xlabel_pad: float = 8.0,
):
    """
    Faster alternative: show multiple Z-slices as 2D heatmaps in a grid.
    """
    if arr_dbm.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {arr_dbm.shape}")
    H, W, C = arr_dbm.shape
    if vmin_dbm is None:
        vmin_dbm = np.nanpercentile(arr_dbm, 2)
    if vmax_dbm is None:
        vmax_dbm = np.nanpercentile(arr_dbm, 98)
    if vmin_dbm >= vmax_dbm:
        vmax_dbm = vmin_dbm + 1e-6

    zs = np.linspace(0, C - 1, num=min(num_slices, C), dtype=int)
    cols = min(3, len(zs))
    rows = int(np.ceil(len(zs) / cols))

    fig = plt.figure(figsize=(4 * cols + 0.8, 3.8 * rows), dpi=dpi)
    gs = fig.add_gridspec(
        rows,
        cols + 1,
        width_ratios=[1] * cols + [0.06],
        wspace=0.18,
        hspace=0.55,
    )

    axes = []
    im = None
    for i, z in enumerate(zs):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(
            arr_dbm[:, :, z],
            origin="lower",
            cmap=cmap,
            vmin=vmin_dbm,
            vmax=vmax_dbm,
            interpolation="nearest",
        )
        ax.set_title(f"Z = {z}", pad=title_pad)
        ax.set_xlabel("X (grid)")
        ax.xaxis.labelpad = xlabel_pad
        ax.set_ylabel("Y (grid)")
        axes.append(ax)

    # Dedicated colorbar axis on the right to avoid overlap
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Interference Power (dBm)")

    fig.suptitle(title, y=0.98)
    # Use tight_layout for margins without collapsing inter-row spacing
    plt.tight_layout(rect=[0, 0.02, 0.98, 0.96])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved slices grid: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render a 3D interference cube (X,Y grid; Z=PRB index) from a MAT file.\n"
            "Colors encode power in dBm (red=strong, blue=weak).\n"
            "Supports full 3D voxels (may be slow) or a fast multi-slice layout."
        )
    )
    p.add_argument("input_mat", type=str, help="Path to input .mat file")
    p.add_argument(
        "--varname",
        type=str,
        default="XdB_recon_tensor",
        help=(
            "Variable name inside .mat. If missing, auto-select the largest 3D array."
        ),
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["voxels", "slices"],
        default="voxels",
        help="Visualization mode: 'voxels' for 3D cube, 'slices' for multi-Z heatmaps",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="jet",
        help="Colormap (default: jet; red=high, blue=low)",
    )
    p.add_argument("--alpha", type=float, default=0.6, help="Voxel face alpha [0-1]")
    p.add_argument("--vmin_dbm", type=float, default=None, help="Color scale min (dBm)")
    p.add_argument("--vmax_dbm", type=float, default=None, help="Color scale max (dBm)")
    p.add_argument(
        "--max_voxels",
        type=int,
        default=150_000,
        help="Max voxels to render (auto downsample if exceeded)",
    )
    p.add_argument("--num_slices", type=int, default=6, help="Slices count for 'slices' mode")
    p.add_argument("--dpi", type=int, default=200, help="Figure DPI (resolution)")
    p.add_argument("--save", type=str, default=None, help="Output image path")
    p.add_argument("--show", action="store_true", help="Display the figure")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input_mat):
        print(f"[ERROR] MAT not found: {args.input_mat}")
        sys.exit(1)

    try:
        # If the provided varname isn't there, auto-detect
        arr, used_name = load_mat_variable(
            args.input_mat, varname=args.varname if args.varname else None
        )
    except Exception as e:
        print(f"[ERROR] Failed to load MAT: {e}")
        sys.exit(1)

    if arr.ndim != 3:
        print(f"[ERROR] Expected a 3D array, got shape {arr.shape} for '{used_name}'")
        sys.exit(1)

    # Ensure float
    arr = np.asarray(arr, dtype=float)

    base = os.path.splitext(os.path.basename(args.input_mat))[0]
    # title = f"{base} | {used_name} (dBm)"
    title = ""

    if args.save is None:
        out = os.path.join(
            os.path.dirname(args.input_mat),
            f"{base}_{used_name}_{args.mode}.png",
        )
    else:
        out = args.save

    if args.mode == "voxels":
        plot_voxels_cube(
            arr,
            title=title,
            cmap=args.cmap,
            alpha=args.alpha,
            vmin_dbm=args.vmin_dbm,
            vmax_dbm=args.vmax_dbm,
            max_voxels=args.max_voxels,
            save_path=out,
            show=args.show,
            dpi=args.dpi,
        )
    else:
        plot_slices_grid(
            arr,
            title=title,
            num_slices=args.num_slices,
            cmap=args.cmap,
            vmin_dbm=args.vmin_dbm,
            vmax_dbm=args.vmax_dbm,
            save_path=out,
            show=args.show,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
# Global style: Times New Roman fonts (fallbacks included)
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    }
)
