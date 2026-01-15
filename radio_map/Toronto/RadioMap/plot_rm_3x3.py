#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties


def pick_indices(n_layers: int, count: int) -> np.ndarray:
    if n_layers < count:
        raise ValueError(f"Need at least {count} layers, got {n_layers}.")
    indices = np.linspace(0, n_layers - 1, count)
    indices = np.round(indices).astype(int)
    if np.unique(indices).size != count:
        # Fallback if rounding collapses indices (only likely when n_layers is small).
        indices = np.linspace(0, n_layers - 1, count, endpoint=True).astype(int)
    return indices


def resolve_cmap(name: str):
    if name.lower() in {"rgb", "red-green-blue"}:
        return LinearSegmentedColormap.from_list(
            "rgb",
            ["#ff0000", "#00ff00", "#0000ff"],
            N=256,
        )
    return plt.get_cmap(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a 3x3 heatmap grid of evenly sampled height slices."
    )
    parser.add_argument(
        "--input",
        default="RM_toronto125_dBm.npy",
        help="Path to the radio map .npy file.",
    )
    parser.add_argument(
        "--output",
        default="radio_map_3x3.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--cmap",
        default="rgb",
        help="Matplotlib colormap name or 'rgb' for red-green-blue.",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Preferred Chinese font family name (e.g., 'SimHei').",
    )
    parser.add_argument(
        "--font-path",
        default=None,
        help="Path to a .ttf/.otf font file to load.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window.",
    )
    args = parser.parse_args()

    font_set = False
    if args.font_path:
        font_path = Path(args.font_path)
        if not font_path.is_file():
            raise FileNotFoundError(f"Font path not found: {font_path}")
        font_manager.fontManager.addfont(str(font_path))
        font_prop = FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = font_prop.get_name()
        font_set = True
    elif args.font:
        try:
            font_manager.findfont(
                FontProperties(family=args.font), fallback_to_default=False
            )
            plt.rcParams["font.family"] = args.font
            font_set = True
        except ValueError:
            pass

    if not font_set:
        # Try a small, common CJK font fallback list.
        fallback_fonts = [
            "SimHei",
            "Microsoft YaHei",
            "PingFang SC",
            "STHeiti",
            "Heiti SC",
            "WenQuanYi Micro Hei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "AR PL UMing CN",
        ]
        for name in fallback_fonts:
            try:
                font_manager.findfont(
                    FontProperties(family=name), fallback_to_default=False
                )
                plt.rcParams["font.family"] = name
                font_set = True
                break
            except ValueError:
                continue

    # Ensure minus signs render when using CJK fonts.
    plt.rcParams["axes.unicode_minus"] = False

    data = np.load(args.input)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {data.shape}.")

    x_size, y_size, n_layers = data.shape
    indices = pick_indices(n_layers, 9)

    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5), constrained_layout=True)
    axes = axes.ravel()

    last_im = None
    for ax, idx in zip(axes, indices):
        slice_ = data[:, :, idx]
        last_im = ax.imshow(
            slice_,
            origin="lower",
            cmap=resolve_cmap(args.cmap),
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"PRB切片 {idx + 1} / {n_layers}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("干扰强度 (dBm)")

    output_path = Path(args.output)
    fig.savefig(output_path, dpi=300)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
