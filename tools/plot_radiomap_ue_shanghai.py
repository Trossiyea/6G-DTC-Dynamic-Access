#!/usr/bin/env python3
"""
Plot the Radio Map (median over frequency) and UE positions over Shanghai.

Features
- Uses CONFIG and helpers from code/main.py to load/generate the Radio Map and UE positions
- Maps the simulation grid to geographic coordinates around Shanghai (lat/lon)
- High-quality visualization with Cartopy (if available), graceful fallback otherwise

Usage
  python tools/plot_radiomap_ue_shanghai.py --show
  python tools/plot_radiomap_ue_shanghai.py --outfile output/scene_shanghai.png

Dependencies
- Required: numpy, matplotlib
- Optional (for nicer map): cartopy

Notes
- The map uses a small-angle approximation: 1 deg lat ≈ 111 km, 1 deg lon ≈ 111 km * cos(lat0)
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot Radio Map and UE locations over Shanghai.')
    p.add_argument('--outfile', type=str, default='output/scene_shanghai.png', help='Output PNG path')
    p.add_argument('--show', action='store_true', help='Show interactive window')
    p.add_argument('--sh-lat', type=float, default=31.2304, help='Shanghai latitude (deg)')
    p.add_argument('--sh-lon', type=float, default=121.4737, help='Shanghai longitude (deg)')
    # External Radio Map overrides
    p.add_argument('--mat-path', type=str, default=None, help='Path to external Radio Map .mat file')
    p.add_argument('--mat-var', type=str, default='X_true', help='Variable name inside MAT (default: X_true)')
    p.add_argument('--mat-units', type=str, default='mW', choices=['mW', 'W', 'dBm'], help='Units of the Radio Map variable')
    return p.parse_args()


def grid_extent_latlon(lat0: float, lon0: float, X: int, Y: int, cell_km: float) -> Tuple[float, float, float, float]:
    deg_per_km_lat = 1.0 / 111.0
    deg_per_km_lon = 1.0 / (111.0 * max(1e-6, np.cos(np.radians(lat0))))
    half_w_km = 0.5 * X * cell_km
    half_h_km = 0.5 * Y * cell_km
    lat_min = lat0 - half_h_km * deg_per_km_lat
    lat_max = lat0 + half_h_km * deg_per_km_lat
    lon_min = lon0 - half_w_km * deg_per_km_lon
    lon_max = lon0 + half_w_km * deg_per_km_lon
    return lon_min, lon_max, lat_min, lat_max


def idx_to_latlon(ix: np.ndarray, iy: np.ndarray, lat0: float, lon0: float, X: int, Y: int, cell_km: float) -> Tuple[np.ndarray, np.ndarray]:
    deg_per_km_lat = 1.0 / 111.0
    deg_per_km_lon = 1.0 / (111.0 * max(1e-6, np.cos(np.radians(lat0))))
    # Pixel centers: offset from grid center
    dx_km = (ix.astype(float) - (X / 2.0) + 0.5) * cell_km
    dy_km = (iy.astype(float) - (Y / 2.0) + 0.5) * cell_km
    lat = lat0 + dy_km * deg_per_km_lat
    lon = lon0 + dx_km * deg_per_km_lon
    # Wrap longitude to [-180, 180)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def main():
    args = parse_args()

    # Import simulation config and helpers
    import sys
    sys.path.append('code')
    import main as sim_main  # type: ignore
    from config import CONFIG  # type: ignore

    cfg = dict(CONFIG)
    # We only need a static snapshot to illustrate the scene
    cfg.update({
        'enable_time_varying': False,
        'enable_orbit_dynamics': False,
        'auto_ref_from_tle': False,
        'ref_lat_deg': float(args.sh_lat),
        'ref_lon_deg': float(args.sh_lon),
        'map_rotation_deg': 0.0,
    })

    # External Radio Map overrides if provided
    if args.mat_path:
        cfg.update({
            'radio_map_mat_path': args.mat_path,
            'radio_map_mat_var': args.mat_var,
            'radio_map_units': args.mat_units,
        })

    # Load/generate Radio Map and UEs
    try:
        R_xyz_dbm, X, Y, Z = sim_main.select_radio_map(cfg)
    except Exception as e:
        raise SystemExit(f"Failed to load Radio Map: {e}")
    rng = np.random.default_rng(cfg.get('seed', 1))
    ue_pos = sim_main.generate_ue_positions(cfg.get('N_UE', 40), X, Y, rng)

    # Reduce Radio Map over frequency for visualization
    R_med = np.median(R_xyz_dbm, axis=2)
    # Robust display range
    vmin = float(np.percentile(R_med, 5))
    vmax = float(np.percentile(R_med, 95))

    lat0 = float(args.sh_lat)
    lon0 = float(args.sh_lon)
    cell_km = float(cfg.get('cell_size_km', 5.0))
    lon_min, lon_max, lat_min, lat_max = grid_extent_latlon(lat0, lon0, X, Y, cell_km)

    # UE scatter positions in lat/lon
    lat_u, lon_u = idx_to_latlon(ue_pos[:, 0], ue_pos[:, 1], lat0, lon0, X, Y, cell_km)

    # Prepare figure
    plt.rcParams.update({
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })
    fig = plt.figure(figsize=(8.8, 7.0))

    # Try Cartopy for a nice basemap
    used_cartopy = False
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
        used_cartopy = True
        proj = ccrs.PlateCarree()
        ax = plt.axes(projection=proj)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        ax.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor='#e9f2fb')
        ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor='#f7f5ef')
        # ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.6, edgecolor='#444444')
        ax.gridlines(draw_labels=False, linewidth=0.3, color='#666666', alpha=0.4, linestyle='--')

        # Draw Radio Map as an image with geographic extent
        im = ax.imshow(
            R_med.T,  # note transpose to match x->lon, y->lat orientation
            origin='lower',
            extent=[lon_min, lon_max, lat_min, lat_max],
            transform=proj,
            cmap='inferno',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
            zorder=1,
        )

        # Overlay UE positions
        ax.scatter(lon_u, lat_u, s=18, marker='o', facecolor='#00acc1', edgecolor='white', linewidth=0.5, transform=proj, zorder=3, label='UE')
        ax.set_title('Radio Map (median over frequency) and UEs — Shanghai')
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.015, fraction=0.046)
        cbar.set_label('Interference power (dBm)')
        ax.legend(loc='upper right', frameon=True)

    except Exception:
        # Fallback without Cartopy
        ax = plt.gca()
        ax.set_facecolor('#e9f2fb')
        ax.fill_between([lon_min, lon_max], lat_min, lat_max, color='#f7f5ef', zorder=0)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.grid(color='#666666', linestyle='--', linewidth=0.4, alpha=0.5)
        im = ax.imshow(
            R_med.T,
            origin='lower',
            extent=[lon_min, lon_max, lat_min, lat_max],
            cmap='inferno',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
            zorder=1,
        )
        ax.scatter(lon_u, lat_u, s=18, marker='o', facecolor='#00acc1', edgecolor='white', linewidth=0.5, zorder=3, label='UE')
        ax.set_title('Radio Map (median over frequency) and UEs — Shanghai')
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01, fraction=0.04)
        cbar.set_label('Interference power (dBm)')
        ax.legend(loc='upper right', frameon=True)

    # Save/show
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=160, bbox_inches='tight')
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved: {args.outfile} (cartopy={'yes' if used_cartopy else 'no'})")


if __name__ == '__main__':
    main()
