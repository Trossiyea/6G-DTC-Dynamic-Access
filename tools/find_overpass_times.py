#!/usr/bin/env python3
"""
Find satellite overpass times relative to a ground location using Skyfield.

Two criteria are supported:
  - elevation: report passes where elevation exceeds a threshold (default 60°)
  - radius:    legacy mode; report closest-approach events within a radius (km)

Defaults use the repo's code/config.py for TLE/time and the Shanghai center.

Examples:
  # Elevation-based (default, 60°):
  python tools/find_overpass_times.py
  python tools/find_overpass_times.py --elev-deg 50 --hours 12

  # Radius-based:
  python tools/find_overpass_times.py --criterion radius --radius-km 10

Outputs:
  - Elevation mode: each pass with entry/exit times (at threshold), peak time and peak elevation
  - Radius mode:    local minima within radius, or globally closest approach if none
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0  # km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-15, 1.0 - a)))
    return R * c


def _load_tle_from_config() -> Tuple[str, str, str, Optional[str]]:
    import sys
    sys.path.append('code')
    from config import CONFIG  # type: ignore
    tle_lines = CONFIG.get('tle_lines')
    tle_path = CONFIG.get('tle_path')
    tle_name = str(CONFIG.get('tle_name', 'SAT'))
    start = CONFIG.get('orbit_start_datetime')
    if (isinstance(tle_lines, (list, tuple)) and len(tle_lines) >= 2
            and isinstance(tle_lines[0], str) and isinstance(tle_lines[1], str)):
        return tle_name, tle_lines[0].strip(), tle_lines[1].strip(), start
    if tle_path and os.path.exists(tle_path):
        with open(tle_path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(lines) >= 2:
            return tle_name, lines[-2], lines[-1], start
    raise RuntimeError('No TLE found in CONFIG (tle_lines or tle_path).')


def _parse_start(start: Optional[str]) -> datetime:
    if start:
        try:
            return datetime.fromisoformat(start.replace('Z', '+00:00')).astimezone(timezone.utc)
        except Exception:
            pass
    # Fall back to config
    try:
        _, _, _, s = _load_tle_from_config()
        if s:
            return datetime.fromisoformat(str(s).replace('Z', '+00:00')).astimezone(timezone.utc)
    except Exception:
        pass
    return datetime.now(tz=timezone.utc)


def _propagate_subpoints(name: str, l1: str, l2: str, t0: datetime, hours: float, step_sec: float) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
    try:
        from skyfield.api import EarthSatellite, load, wgs84  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("Skyfield/sgp4 not available. Please `pip install skyfield sgp4`." )
    ts = load.timescale()
    n_steps = int(max(2, round(hours * 3600.0 / max(0.1, step_sec))))
    times = [t0 + timedelta(seconds=i * step_sec) for i in range(n_steps)]
    t_sf = ts.utc([t.year for t in times], [t.month for t in times], [t.day for t in times],
                  [t.hour for t in times], [t.minute for t in times], [t.second + t.microsecond * 1e-6 for t in times])
    sat = EarthSatellite(l1, l2, name)
    sp = sat.at(t_sf)
    subs = wgs84.subpoint_of(sp)
    lats = np.asarray(subs.latitude.degrees)
    lons = np.asarray(subs.longitude.degrees)
    # Wrap longitudes to [-180, 180]
    lons = ((lons + 180.0) % 360.0) - 180.0
    return lats, lons, times


def _local_minima(x: np.ndarray) -> np.ndarray:
    # Return indices i where x[i] is a strict local minimum
    if x.size < 3:
        return np.array([], dtype=int)
    left = x[1:-1] < x[:-2]
    right = x[1:-1] < x[2:]
    return np.nonzero(left & right)[0] + 1


def _refine_min_time(name: str, l1: str, l2: str, t_a: datetime, t_b: datetime, target_lat: float, target_lon: float, n_iter: int = 20) -> Tuple[datetime, float, float, float]:
    """Refine closest-approach time in [t_a, t_b] via golden-section search on distance.

    Returns (t_min, d_min_km, lat_min, lon_min).
    """
    try:
        from skyfield.api import EarthSatellite, load, wgs84  # type: ignore
    except Exception:
        # Fallback: midpoint without refinement
        t_mid = t_a + (t_b - t_a) / 2
        return t_mid, float('nan'), float('nan'), float('nan')

    sat = EarthSatellite(l1, l2, name)
    ts = load.timescale()

    def f(t: datetime) -> Tuple[float, float, float]:
        t_sf = ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        sp = sat.at(t_sf)
        sub = wgs84.subpoint_of(sp)
        lat = float(sub.latitude.degrees)
        lon = float(sub.longitude.degrees)
        lon = ((lon + 180.0) % 360.0) - 180.0
        d = _haversine_km(lat, lon, target_lat, target_lon)
        return d, lat, lon

    invphi = (math.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1/phi^2
    a = t_a
    b = t_b
    # Initialize interior points
    h = (b - a).total_seconds()
    if h <= 1e-6:
        d_mid, lat_mid, lon_mid = f(a)
        return a, d_mid, lat_mid, lon_mid
    n = max(1, n_iter)
    c = a + timedelta(seconds=invphi2 * h)
    d = a + timedelta(seconds=invphi * h)
    f_c = f(c)
    f_d = f(d)
    for _ in range(n):
        if f_c[0] > f_d[0]:
            a = c
            c = d
            f_c = f_d
            h = (b - a).total_seconds()
            d = a + timedelta(seconds=invphi * h)
            f_d = f(d)
        else:
            b = d
            d = c
            f_d = f_c
            h = (b - a).total_seconds()
            c = a + timedelta(seconds=invphi2 * h)
            f_c = f(c)
    # Choose best of f_c and f_d
    if f_c[0] < f_d[0]:
        t_min, (dmin, latm, lonm) = c, f_c
    else:
        t_min, (dmin, latm, lonm) = d, f_d
    return t_min, dmin, latm, lonm


def _sat_and_ts(name: str, l1: str, l2: str):
    from skyfield.api import EarthSatellite, load  # type: ignore
    ts = load.timescale()
    sat = EarthSatellite(l1, l2, name)
    return sat, ts


def _elevation_deg_at(sat, ts, t: datetime, lat: float, lon: float) -> float:
    from skyfield.api import wgs84  # type: ignore
    t_sf = ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
    sp = sat.at(t_sf)
    g = wgs84.latlon(lat, lon)
    topoc = sp - g.at(t_sf)
    alt, az, dist = topoc.altaz()
    return float(alt.degrees)


def _propagate_elevation(name: str, l1: str, l2: str, t0: datetime, hours: float, step_sec: float, lat: float, lon: float) -> Tuple[np.ndarray, List[datetime]]:
    try:
        from skyfield.api import EarthSatellite, load, wgs84  # type: ignore
    except Exception as e:
        raise SystemExit("Skyfield/sgp4 not available. Please `pip install skyfield sgp4`.")
    ts = load.timescale()
    n_steps = int(max(2, round(hours * 3600.0 / max(0.1, step_sec))))
    times = [t0 + timedelta(seconds=i * step_sec) for i in range(n_steps)]
    t_sf = ts.utc([t.year for t in times], [t.month for t in times], [t.day for t in times],
                  [t.hour for t in times], [t.minute for t in times], [t.second + t.microsecond * 1e-6 for t in times])
    sat = EarthSatellite(l1, l2, name)
    g = wgs84.latlon(lat, lon)
    sp = sat.at(t_sf)
    topoc = sp - g.at(t_sf)
    alt, az, dist = topoc.altaz()
    alt_deg = np.asarray(alt.degrees, dtype=float)
    return alt_deg, times


def _bisect_crossing_time(sat, ts, ta: datetime, tb: datetime, lat: float, lon: float, target_alt_deg: float, max_iter: int = 30) -> datetime:
    fa = _elevation_deg_at(sat, ts, ta, lat, lon) - target_alt_deg
    fb = _elevation_deg_at(sat, ts, tb, lat, lon) - target_alt_deg
    if fa == 0:
        return ta
    if fb == 0:
        return tb
    if fa * fb > 0:
        return ta + (tb - ta) / 2
    a, b = ta, tb
    for _ in range(max_iter):
        mid = a + (b - a) / 2
        fm = _elevation_deg_at(sat, ts, mid, lat, lon) - target_alt_deg
        if fa * fm <= 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    return a + (b - a) / 2


def _refine_peak_time(sat, ts, ta: datetime, tb: datetime, lat: float, lon: float, n_iter: int = 25) -> Tuple[datetime, float]:
    invphi = (math.sqrt(5) - 1) / 2
    invphi2 = (3 - math.sqrt(5)) / 2
    a = ta
    b = tb
    h = (b - a).total_seconds()
    if h <= 1e-6:
        alt = _elevation_deg_at(sat, ts, a, lat, lon)
        return a, alt
    n = max(1, n_iter)
    c = a + timedelta(seconds=invphi2 * h)
    d = a + timedelta(seconds=invphi * h)
    f_c = _elevation_deg_at(sat, ts, c, lat, lon)
    f_d = _elevation_deg_at(sat, ts, d, lat, lon)
    for _ in range(n):
        if f_c < f_d:
            a = c
            c = d
            f_c = f_d
            h = (b - a).total_seconds()
            d = a + timedelta(seconds=invphi * h)
            f_d = _elevation_deg_at(sat, ts, d, lat, lon)
        else:
            b = d
            d = c
            f_d = f_c
            h = (b - a).total_seconds()
            c = a + timedelta(seconds=invphi2 * h)
            f_c = _elevation_deg_at(sat, ts, c, lat, lon)
    if f_c > f_d:
        return c, f_c
    else:
        return d, f_d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Find TLE overpass times near a ground point.')
    p.add_argument('--lat', type=float, default=31.2304, help='Ground latitude (deg). Default: Shanghai 31.2304')
    p.add_argument('--lon', type=float, default=121.4737, help='Ground longitude (deg). Default: Shanghai 121.4737')
    p.add_argument('--criterion', type=str, default='elevation', choices=['elevation', 'radius'], help='Overpass criterion: elevation threshold or radius threshold')
    p.add_argument('--elev-deg', type=float, default=60.0, help='Elevation threshold (deg) for elevation criterion.')
    p.add_argument('--radius-km', type=float, default=10.0, help='Pass radius threshold (km) for radius criterion.')
    p.add_argument('--tle-path', type=str, default=None, help='Optional path to TLE file (2-line). Overrides config.')
    p.add_argument('--start', type=str, default=None, help='Start time ISO8601 (e.g., 2025-09-13T18:38:42Z). Defaults to CONFIG or now(UTC).')
    p.add_argument('--hours', type=float, default=24.0, help='Duration (hours) to search.')
    p.add_argument('--step-sec', type=float, default=10.0, help='Sampling step in seconds.')
    p.add_argument('--min-sep-min', type=float, default=20.0, help='Minimum time separation (minutes) between distinct minima in radius mode.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Load TLE
    if args.tle_path:
        with open(args.tle_path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(lines) < 2:
            raise SystemExit('TLE file must have at least two lines.')
        name = os.path.basename(args.tle_path)
        l1, l2 = lines[-2], lines[-1]
        t0 = _parse_start(args.start)
    else:
        name, l1, l2, _ = _load_tle_from_config()
        t0 = _parse_start(args.start)

    lat0 = float(args.lat)
    lon0 = float(args.lon)
    hours = float(args.hours)
    step = float(args.step_sec)

    print(f"TLE name: {name}")
    print(f"Search window: start={t0.isoformat()}Z, duration={hours:.2f} h, step={step:.1f} s")
    print(f"Target: lat={lat0:.4f}, lon={lon0:.4f}")

    if args.criterion == 'elevation':
        thr = float(args.elev_deg)
        print(f"Criterion: elevation >= {thr:.1f} deg")
        try:
            alt_deg, times = _propagate_elevation(name, l1, l2, t0, hours, step, lat0, lon0)
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"Failed to propagate elevation. Ensure skyfield/sgp4 installed. Error: {e}")

        above = alt_deg >= thr
        if not np.any(above):
            if alt_deg.size == 0:
                print("No samples generated; check hours/step.")
                return
            i_max = int(np.argmax(alt_deg))
            ta = times[max(0, i_max - 1)]
            tb = times[min(len(times) - 1, i_max + 1)]
            sat, ts = _sat_and_ts(name, l1, l2)
            t_peak, a_peak = _refine_peak_time(sat, ts, ta, tb, lat0, lon0)
            print(f"No pass with elevation >= {thr:.1f} deg in the window.")
            print(f"Highest elevation: {a_peak:.2f} deg at {t_peak.isoformat()}Z")
            return

        # Build contiguous segments
        indices = np.flatnonzero(above)
        segs: List[Tuple[int, int]] = []
        start_idx = int(indices[0])
        prev = int(indices[0])
        for idx in indices[1:]:
            idx = int(idx)
            if idx == prev + 1:
                prev = idx
            else:
                segs.append((start_idx, prev))
                start_idx = idx
                prev = idx
        segs.append((start_idx, prev))

        sat, ts = _sat_and_ts(name, l1, l2)
        results = []
        for (a0, a1) in segs:
            if a0 > 0:
                t_entry = _bisect_crossing_time(sat, ts, times[a0 - 1], times[a0], lat0, lon0, thr)
            else:
                t_entry = times[a0]
            if a1 < len(times) - 1:
                t_exit = _bisect_crossing_time(sat, ts, times[a1], times[a1 + 1], lat0, lon0, thr)
            else:
                t_exit = times[a1]
            i_local_max = int(a0 + np.argmax(alt_deg[a0:a1 + 1]))
            ta = times[max(0, i_local_max - 1)]
            tb = times[min(len(times) - 1, i_local_max + 1)]
            t_peak, a_peak = _refine_peak_time(sat, ts, ta, tb, lat0, lon0)
            results.append((t_entry, t_peak, a_peak, t_exit))

        print(f"Found {len(results)} pass(es) with elevation >= {thr:.1f} deg:")
        for j, (t_entry, t_peak, a_peak, t_exit) in enumerate(results, 1):
            dur = (t_exit - t_entry).total_seconds()
            print(f"  {j:02d}. entry={t_entry.isoformat()}Z, peak={t_peak.isoformat()}Z ({a_peak:.2f} deg), exit={t_exit.isoformat()}Z, duration={dur:.0f} s")
        return

    # Radius criterion
    radius_km = float(args.radius_km)
    min_sep = float(args.min_sep_min) * 60.0
    print(f"Criterion: ground radius <= {radius_km:.1f} km")

    try:
        lats, lons, times = _propagate_subpoints(name, l1, l2, t0, hours, step)
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"Failed to propagate TLE. Ensure skyfield/sgp4 installed. Error: {e}")

    dists = np.array([_haversine_km(float(la), float(lo), lat0, lon0) for la, lo in zip(lats, lons)])
    idx_mins = _local_minima(dists)
    picked: List[int] = []
    for i in idx_mins:
        i = int(i)
        if not picked:
            picked.append(i)
            continue
        if (times[i] - times[picked[-1]]).total_seconds() >= min_sep:
            picked.append(i)
        elif dists[i] < dists[picked[-1]]:
            picked[-1] = i

    events: List[Tuple[datetime, float, float, float]] = []
    for i in picked:
        i0 = max(0, i - 1)
        i1 = min(len(times) - 1, i + 1)
        t_min, d_min, la_min, lo_min = _refine_min_time(name, l1, l2, times[i0], times[i1], lat0, lon0)
        events.append((t_min, d_min, la_min, lo_min))

    within = [e for e in events if e[1] <= radius_km]
    if within:
        print(f"Found {len(within)} pass(es) within {radius_km:.1f} km:")
        for j, (t_min, d_min, la_min, lo_min) in enumerate(within, 1):
            print(f"  {j:02d}. t={t_min.isoformat()}Z, miss={d_min:.2f} km, subpoint=({la_min:.4f}, {lo_min:.4f})")
    else:
        if dists.size == 0:
            print("No samples generated; check hours/step.")
            return
        i_best = int(np.argmin(dists))
        i0 = max(0, i_best - 1)
        i1 = min(len(times) - 1, i_best + 1)
        t_min, d_min, la_min, lo_min = _refine_min_time(name, l1, l2, times[i0], times[i1], lat0, lon0)
        print(f"No overpass within {radius_km:.1f} km in the window.")
        print(f"Closest approach: t={t_min.isoformat()}Z, miss={d_min:.2f} km, subpoint=({la_min:.4f}, {lo_min:.4f})")


if __name__ == '__main__':
    main()
