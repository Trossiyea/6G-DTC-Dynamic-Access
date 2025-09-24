#!/usr/bin/env python3
"""
Plot DTC constellation ground tracks from a TLE catalog and overlay the
simulation ground area from code/config.py.

Usage examples:
  python tools/plot_constellation.py                         # defaults from config
  python tools/plot_constellation.py --hours 2 --step-sec 60 --limit-sats 60 --show
  python tools/plot_constellation.py --tle-catalog docs/DTC_tle.txt --outfile output/constellation_tracks.png

Outputs:
  - PNG saved to output/constellation_tracks.png (default; change with --outfile)
  - Optionally shows window with --show

Cartopy is used if available for coastlines; otherwise falls back to plain
lat/lon axes with clean styling.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_config():
    import sys
    sys.path.append('code')
    from config import CONFIG  # type: ignore
    return CONFIG


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot constellation ground tracks and simulation area.')
    p.add_argument('--tle-catalog', type=str, default=None, help='Path to TLE catalog (name+2-line or bare 2-line blocks). Defaults to CONFIG[tle_catalog_path].')
    p.add_argument('--start', type=str, default=None, help='Start time ISO8601 (e.g., 2025-09-13T18:38:42Z). Defaults to CONFIG or now(UTC).')
    p.add_argument('--hours', type=float, default=3.0, help='Duration (hours) to propagate.')
    p.add_argument('--step-sec', type=float, default=60.0, help='Sampling step in seconds.')
    p.add_argument('--limit-sats', type=int, default=80, help='Maximum satellites to plot (for readability).')
    p.add_argument('--outfile', type=str, default='output/constellation_tracks.png', help='Output PNG path.')
    p.add_argument('--show', action='store_true', help='Show plot window.')
    return p.parse_args()


def _parse_start(start: str | None, cfg) -> datetime:
    if start:
        try:
            return datetime.fromisoformat(start.replace('Z', '+00:00')).astimezone(timezone.utc)
        except Exception:
            pass
    s = cfg.get('orbit_start_datetime')
    if s:
        try:
            return datetime.fromisoformat(str(s).replace('Z', '+00:00')).astimezone(timezone.utc)
        except Exception:
            pass
    return datetime.now(tz=timezone.utc)


def _wrap_lon(lon_deg: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_deg)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon


def _load_catalog(path: str):
    import sys
    sys.path.append('code')
    from constellation import parse_tle_catalog  # type: ignore
    return parse_tle_catalog(path)


def _propagate_many(sats, t0: datetime, hours: float, step_sec: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    from skyfield.api import load, EarthSatellite, wgs84  # type: ignore
    ts = load.timescale()
    n_steps = int(max(2, round(hours * 3600.0 / max(1.0, step_sec))))
    times = [t0 + timedelta(seconds=i * step_sec) for i in range(n_steps)]
    t_sf = ts.utc([t.year for t in times], [t.month for t in times], [t.day for t in times],
                  [t.hour for t in times], [t.minute for t in times], [t.second + t.microsecond * 1e-6 for t in times])
    lats_list: List[np.ndarray] = []
    lons_list: List[np.ndarray] = []
    for s in sats:
        try:
            sat = EarthSatellite(s.l1, s.l2, s.name)
            sp = sat.at(t_sf)
            subs = wgs84.subpoint_of(sp)
            lats = np.asarray(subs.latitude.degrees)
            lons = np.asarray(subs.longitude.degrees)
            lons = _wrap_lon(lons)
            lats_list.append(lats)
            lons_list.append(lons)
        except Exception:
            continue
    return lats_list, lons_list


def _plot_world_base(ax, hours: float):
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
        data_crs = ccrs.PlateCarree()
        ax.set_global()
        ax.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor='#e9f2fb')
        ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor='#f7f5ef')
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.6, edgecolor='#444444')
        ax.gridlines(draw_labels=False, linewidth=0.3, color='#666666', alpha=0.5, linestyle='--')
        return data_crs
    except Exception:
        try:
            plt.style.use('seaborn-v0_8')
        except Exception:
            pass
        ax.set_facecolor('#e9f2fb')
        ax.fill_between([-180, 180], -90, 90, color='#f7f5ef', zorder=0)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 30))
        ax.set_yticks(np.arange(-90, 91, 15))
        ax.grid(color='#666666', linestyle='--', linewidth=0.4, alpha=0.5)
        return None


def _plot_tracks(ax, lats_list, lons_list, hours: float, data_crs=None, alpha=0.55):
    # Color each track with a light line; thin points to avoid dateline artifacts
    for lats, lons in zip(lats_list, lons_list):
        t_rel = np.linspace(0.0, 1.0, lats.size)
        if data_crs is not None:
            ax.plot(lons, lats, transform=data_crs, color='#2a6f97', linewidth=0.6, alpha=alpha, zorder=1)
            ax.scatter(lons, lats, c=t_rel, cmap='viridis', s=6, transform=data_crs, edgecolor='none', alpha=0.6, zorder=2)
        else:
            ax.plot(lons, lats, color='#2a6f97', linewidth=0.6, alpha=alpha, zorder=1)
            sc = ax.scatter(lons, lats, c=t_rel, cmap='viridis', s=6, edgecolor='none', alpha=0.6, zorder=2)


def _plot_ground_area(ax, cfg, data_crs=None):
    # Compute 4 corners of the simulation grid (rotated ENU rectangle around ref lat/lon)
    X = int(cfg.get('X', 50))
    Y = int(cfg.get('Y', 50))
    cell_km = float(cfg.get('cell_size_km', 1.0))
    lat0 = float(cfg.get('ref_lat_deg', 0.0))
    lon0 = float(cfg.get('ref_lon_deg', 0.0))
    phi = np.radians(float(cfg.get('map_rotation_deg', 0.0)))
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    # Corners in grid indices relative to center
    half_x = X / 2.0
    half_y = Y / 2.0
    corners_xy = np.array([
        [-half_x, -half_y],
        [ half_x, -half_y],
        [ half_x,  half_y],
        [-half_x,  half_y],
        [-half_x, -half_y],  # close
    ], dtype=float)
    # Convert to km offsets
    off_km = corners_xy * cell_km
    # Rotate to ENU
    east_km = off_km[:, 0] * cos_p - off_km[:, 1] * sin_p
    north_km = off_km[:, 0] * sin_p + off_km[:, 1] * cos_p
    deg_per_km_lat = 1.0 / 111.0
    deg_per_km_lon = 1.0 / (111.0 * max(1e-6, np.cos(np.radians(lat0))))
    lat_poly = lat0 + north_km * deg_per_km_lat
    lon_poly = lon0 + east_km * deg_per_km_lon
    lon_poly = _wrap_lon(lon_poly)

    if data_crs is not None:
        ax.plot(lon_poly, lat_poly, transform=data_crs, color='#d1495b', linewidth=1.2, zorder=3)
    else:
        ax.plot(lon_poly, lat_poly, color='#d1495b', linewidth=1.2, zorder=3)


def main():
    cfg = _load_config()
    args = _parse_args()
    tle_catalog = args.tle_catalog or cfg.get('tle_catalog_path')
    if not tle_catalog or not os.path.exists(tle_catalog):
        raise SystemExit(f"TLE catalog not found: {tle_catalog}")

    # Load TLEs
    try:
        sats_all = _load_catalog(tle_catalog)
    except Exception as e:
        raise SystemExit(f"Failed to parse TLE catalog: {e}")
    if not sats_all:
        raise SystemExit("No satellites in catalog.")
    sats = sats_all[: max(1, int(args.limit_sats))]

    # Start time
    t0 = _parse_start(args.start, cfg)

    # Propagate
    try:
        lats_list, lons_list = _propagate_many(sats, t0, float(args.hours), float(args.step_sec))
    except Exception as e:
        raise SystemExit(f"Failed to propagate satellites. Ensure 'skyfield' and 'sgp4' are installed. Error: {e}")

    # Plot
    fig = plt.figure(figsize=(12, 6.8))
    try:
        import cartopy.crs as ccrs  # type: ignore
        ax = plt.axes(projection=ccrs.Robinson())
        data_crs = _plot_world_base(ax, float(args.hours))
    except Exception:
        ax = plt.gca()
        data_crs = _plot_world_base(ax, float(args.hours))

    _plot_tracks(ax, lats_list, lons_list, float(args.hours), data_crs=data_crs)
    _plot_ground_area(ax, cfg, data_crs=data_crs)

    lat0 = cfg.get('ref_lat_deg', None)
    lon0 = cfg.get('ref_lon_deg', None)
    title = f"DTC Constellation tracks (N={len(lats_list)})\nStart: {t0.isoformat()}Z, Duration: {float(args.hours):.1f} h, Step: {float(args.step_sec):.0f} s"
    if isinstance(lat0, (int, float)) and isinstance(lon0, (int, float)):
        title += f"\nGround area centered at ({float(lat0):.4f}, {float(lon0):.4f})"
    plt.title(title, fontsize=12)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=160, bbox_inches='tight')
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved: {args.outfile}")


if __name__ == '__main__':
    main()

