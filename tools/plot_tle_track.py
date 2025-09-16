#!/usr/bin/env python3
"""
Plot a TLE satellite ground track with a clean, publication-style look.

Usage examples:
  python tools/plot_tle_track.py                       # use TLE from code/config.py
  python tools/plot_tle_track.py --hours 2 --step-sec 15 --show
  python tools/plot_tle_track.py --tle-path my.tle --start 2025-09-13T18:38:42Z

Outputs:
  - Saves PNG to output/tle_ground_track.png by default (use --outfile to change)
  - Optionally shows window if --show is given

Cartopy is used if available for coastlines and projection. If not installed,
the script falls back to a simple latitude/longitude plot with nice styling.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def load_tle_from_config() -> Tuple[str, str, str]:
    """Load TLE lines and name from code/config.py CONFIG, or raise if missing."""
    import sys
    sys.path.append('code')
    from config import CONFIG  # type: ignore
    tle_lines = CONFIG.get('tle_lines')
    tle_path = CONFIG.get('tle_path')
    tle_name = str(CONFIG.get('tle_name', 'SAT'))
    if (isinstance(tle_lines, (list, tuple)) and len(tle_lines) >= 2
            and isinstance(tle_lines[0], str) and isinstance(tle_lines[1], str)):
        return tle_name, tle_lines[0].strip(), tle_lines[1].strip()
    if tle_path and os.path.exists(tle_path):
        with open(tle_path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(lines) >= 2:
            # Accept with or without name line at top
            return tle_name, lines[-2], lines[-1]
    raise RuntimeError('No TLE found in CONFIG (tle_lines or tle_path).')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot TLE satellite ground track.')
    p.add_argument('--tle-path', type=str, default=None, help='Path to TLE file (2-line). Overrides config.')
    p.add_argument('--start', type=str, default=None, help='Start time ISO8601 (e.g., 2025-09-13T18:38:42Z). Defaults to CONFIG or now(UTC).')
    p.add_argument('--hours', type=float, default=2.0, help='Duration (hours) to propagate.')
    p.add_argument('--step-sec', type=float, default=15.0, help='Sampling step in seconds.')
    p.add_argument('--outfile', type=str, default='output/tle_ground_track.png', help='Output PNG path.')
    p.add_argument('--show', action='store_true', help='Show the plot window.')
    return p.parse_args()


def _load_tle(tle_path: Optional[str]) -> Tuple[str, str, str]:
    if tle_path:
        with open(tle_path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(lines) < 2:
            raise RuntimeError('TLE file must contain at least two lines.')
        name = os.path.basename(tle_path)
        return name, lines[-2], lines[-1]
    return load_tle_from_config()


def _parse_start(start: Optional[str]) -> datetime:
    if start:
        try:
            return datetime.fromisoformat(start.replace('Z', '+00:00')).astimezone(timezone.utc)
        except Exception:
            pass
    # Try CONFIG
    try:
        import sys
        sys.path.append('code')
        from config import CONFIG  # type: ignore
        s = CONFIG.get('orbit_start_datetime')
        if s:
            return datetime.fromisoformat(str(s).replace('Z', '+00:00')).astimezone(timezone.utc)
    except Exception:
        pass
    return datetime.now(tz=timezone.utc)


def propagate_ground_track(name: str, l1: str, l2: str, t0: datetime, hours: float, step_sec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from skyfield.api import load, EarthSatellite, wgs84  # type: ignore
    ts = load.timescale()
    n_steps = int(max(2, round(hours * 3600.0 / max(1.0, step_sec))))
    times = [t0 + timedelta(seconds=i * step_sec) for i in range(n_steps)]
    t_sf = ts.utc([t.year for t in times], [t.month for t in times], [t.day for t in times],
                  [t.hour for t in times], [t.minute for t in times], [t.second + t.microsecond * 1e-6 for t in times])
    sat = EarthSatellite(l1, l2, name)
    sp = sat.at(t_sf)
    subs = wgs84.subpoint_of(sp)
    lats = subs.latitude.degrees
    lons = subs.longitude.degrees
    el = subs.elevation.km
    return np.asarray(lats), np.asarray(lons), np.asarray(el)


def _wrap_lon(lon_deg: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_deg)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon


def _plot_cartopy(ax, lats: np.ndarray, lons: np.ndarray, title: str, t0: datetime, hours: float):
    import cartopy.crs as ccrs  # type: ignore
    import cartopy.feature as cfeature  # type: ignore
    data_crs = ccrs.PlateCarree()
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor='#e9f2fb')
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor='#f7f5ef')
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.6, edgecolor='#444444')
    ax.gridlines(draw_labels=False, linewidth=0.3, color='#666666', alpha=0.5, linestyle='--')

    t_rel = np.linspace(0.0, 1.0, lats.size)
    sc = ax.scatter(lons, lats, c=t_rel, s=8, cmap='viridis', transform=data_crs, zorder=3)
    ax.plot(lons, lats, transform=data_crs, color='white', linewidth=0.6, alpha=0.6, zorder=2)

    # Start/end markers
    ax.scatter([lons[0]], [lats[0]], marker='*', s=90, color='#e4572e', edgecolor='white', linewidth=0.6, transform=data_crs, zorder=4)
    ax.scatter([lons[-1]], [lats[-1]], marker='X', s=60, color='#005f73', edgecolor='white', linewidth=0.6, transform=data_crs, zorder=4)

    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.05, fraction=0.05)
    cbar.set_label(f'Time progression (0 → {hours:.1f} h)')
    ax.set_title(title, fontsize=12)


def _plot_plain(ax, lats: np.ndarray, lons: np.ndarray, title: str, t0: datetime, hours: float):
    # Minimal yet clean fallback without cartopy
    try:
        plt.style.use('seaborn-v0_8')  # if available
    except Exception:
        pass
    ax.set_facecolor('#e9f2fb')
    ax.fill_between([-180, 180], -90, 90, color='#f7f5ef', zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 15))
    ax.grid(color='#666666', linestyle='--', linewidth=0.4, alpha=0.5)

    # Use scatter with a nice colormap to avoid ugly dateline crossings
    t_rel = np.linspace(0.0, 1.0, lats.size)
    sc = ax.scatter(lons, lats, c=t_rel, s=10, cmap='viridis', edgecolor='none', zorder=2)
    ax.plot(lons, lats, color='white', linewidth=0.6, alpha=0.5, zorder=1)
    ax.scatter([lons[0]], [lats[0]], marker='*', s=90, color='#e4572e', edgecolor='white', linewidth=0.6, zorder=3)
    ax.scatter([lons[-1]], [lats[-1]], marker='X', s=60, color='#005f73', edgecolor='white', linewidth=0.6, zorder=3)
    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title(title, fontsize=12)
    cbar = plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label(f'Time progression (0 → {hours:.1f} h)')


def main():
    args = parse_args()
    name, l1, l2 = _load_tle(args.tle_path)
    t0 = _parse_start(args.start)

    # Propagate
    try:
        lats, lons, elev_km = propagate_ground_track(name, l1, l2, t0, args.hours, args.step_sec)
    except Exception as e:
        raise SystemExit(f"Failed to propagate TLE. Ensure 'skyfield' and 'sgp4' are installed. Error: {e}")

    lons = _wrap_lon(lons)
    title = f"{name} ground track\nStart: {t0.isoformat()}Z, Duration: {args.hours:.1f} h, Step: {args.step_sec:.0f} s"

    # Plot with cartopy if available
    fig = plt.figure(figsize=(11, 6))
    try:
        import cartopy.crs as ccrs  # type: ignore
        ax = plt.axes(projection=ccrs.Robinson())
        _plot_cartopy(ax, lats, lons, title, t0, args.hours)
    except Exception:
        ax = plt.gca()
        _plot_plain(ax, lats, lons, title, t0, args.hours)

    # Save and maybe show
    out_path = args.outfile
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()

