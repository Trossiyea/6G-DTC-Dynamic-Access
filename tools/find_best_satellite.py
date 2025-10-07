#!/usr/bin/env python3
"""
Find the best satellite from a TLE file for a given ground location.

This script analyzes ALL satellites in a TLE file and finds:
1. Which satellite has the best overpass (highest elevation)
2. The optimal simulation start time

Usage:
    python tools/find_best_satellite.py --tle-path tles/Satnet_DTC.txt --lat 31.2304 --lon 121.4737
    
    # Customize search parameters
    python tools/find_best_satellite.py --tle-path tles/Satnet_DTC.txt \
        --lat 31.2304 --lon 121.4737 --hours 24 --min-elev 60 --top-n 10
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

try:
    import numpy as np
    from skyfield.api import EarthSatellite, load, wgs84
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please run: pip install numpy skyfield sgp4")
    exit(1)


def load_all_satellites_from_tle(tle_path: str) -> List[Tuple[int, str, str]]:
    """Load all satellites from a TLE file.
    
    Returns:
        List of (sat_id, line1, line2) tuples
    """
    with open(tle_path, 'r') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    
    satellites = []
    i = 0
    while i < len(lines) - 1:
        line1 = lines[i].strip()
        line2 = lines[i + 1].strip()
        
        if line1.startswith('1') and line2.startswith('2'):
            # Extract satellite number from line 1
            sat_num = int(line1[2:7].strip())
            satellites.append((sat_num, line1, line2))
            i += 2
        else:
            i += 1
    
    return satellites


def find_best_pass_for_satellite(
    sat_id: int,
    line1: str,
    line2: str,
    lat: float,
    lon: float,
    start_time: datetime,
    hours: float,
    step_sec: float = 10.0
) -> Tuple[Optional[datetime], float]:
    """Find the best pass (highest elevation) for a single satellite.
    
    Returns:
        (peak_time, peak_elevation) or (None, -90.0) if no calculation possible
    """
    try:
        ts = load.timescale()
        sat = EarthSatellite(line1, line2, f"SAT-{sat_id}")
        ground = wgs84.latlon(lat, lon)
        
        # Generate time array
        n_steps = int(max(2, round(hours * 3600.0 / max(0.1, step_sec))))
        times = [start_time + timedelta(seconds=i * step_sec) for i in range(n_steps)]
        
        # Calculate elevations
        elevations = []
        for t in times:
            t_sf = ts.utc(t.year, t.month, t.day, t.hour, t.minute, 
                         t.second + t.microsecond * 1e-6)
            sp = sat.at(t_sf)
            topoc = sp - ground.at(t_sf)
            alt, az, dist = topoc.altaz()
            elevations.append(float(alt.degrees))
        
        # Find peak elevation
        max_idx = int(np.argmax(elevations))
        peak_elev = elevations[max_idx]
        peak_time = times[max_idx]
        
        return peak_time, peak_elev
        
    except Exception as e:
        # If calculation fails for this satellite, return invalid result
        return None, -90.0


def main():
    parser = argparse.ArgumentParser(
        description='Find the best satellite and optimal simulation start time.'
    )
    parser.add_argument('--tle-path', type=str, required=True,
                       help='Path to TLE file with multiple satellites')
    parser.add_argument('--lat', type=float, default=31.2304,
                       help='Ground latitude (deg). Default: Shanghai 31.2304')
    parser.add_argument('--lon', type=float, default=121.4737,
                       help='Ground longitude (deg). Default: Shanghai 121.4737')
    parser.add_argument('--start', type=str, default=None,
                       help='Start time ISO8601. Defaults to now(UTC)')
    parser.add_argument('--hours', type=float, default=24.0,
                       help='Search duration (hours)')
    parser.add_argument('--step-sec', type=float, default=30.0,
                       help='Sampling step in seconds (larger = faster but less accurate)')
    parser.add_argument('--min-elev', type=float, default=60.0,
                       help='Minimum elevation threshold (deg)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Show top N best satellites')
    parser.add_argument('--max-sats', type=int, default=None,
                       help='Limit analysis to first N satellites (for testing)')
    
    args = parser.parse_args()
    
    # Parse start time
    if args.start:
        try:
            start_time = datetime.fromisoformat(args.start.replace('Z', '+00:00'))
            start_time = start_time.astimezone(timezone.utc)
        except Exception:
            print(f"Invalid start time format: {args.start}")
            return
    else:
        start_time = datetime.now(tz=timezone.utc)
    
    print("="*80)
    print("Finding Best Satellite for Simulation")
    print("="*80)
    print(f"TLE File: {args.tle_path}")
    print(f"Ground Location: lat={args.lat:.4f}°, lon={args.lon:.4f}°")
    print(f"Search Window: {start_time.isoformat()}Z + {args.hours:.1f} hours")
    print(f"Minimum Elevation: {args.min_elev:.1f}°")
    print(f"Step: {args.step_sec:.1f} seconds")
    print("="*80)
    
    # Load satellites
    print("\nLoading satellites from TLE file...")
    satellites = load_all_satellites_from_tle(args.tle_path)
    
    if args.max_sats:
        satellites = satellites[:args.max_sats]
        print(f"Limited to first {args.max_sats} satellites for testing.")
    
    print(f"Found {len(satellites)} satellites in TLE file.")
    
    # Analyze each satellite
    print(f"\nAnalyzing satellites (this may take a while)...")
    results = []
    
    for idx, (sat_id, line1, line2) in enumerate(satellites):
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(satellites)} satellites analyzed...")
        
        peak_time, peak_elev = find_best_pass_for_satellite(
            sat_id, line1, line2, args.lat, args.lon,
            start_time, args.hours, args.step_sec
        )
        
        if peak_time is not None and peak_elev >= args.min_elev:
            results.append((sat_id, peak_time, peak_elev, line1, line2))
    
    print(f"\nAnalysis complete!")
    print("="*80)
    
    # Sort by peak elevation (descending)
    results.sort(key=lambda x: x[2], reverse=True)
    
    if not results:
        print(f"\n❌ No satellites found with elevation >= {args.min_elev:.1f}° in the search window.")
        print(f"   Try:")
        print(f"   - Increasing --hours (current: {args.hours:.1f})")
        print(f"   - Decreasing --min-elev (current: {args.min_elev:.1f})")
        return
    
    print(f"\n✅ Found {len(results)} satellite(s) with elevation >= {args.min_elev:.1f}°")
    print(f"\nTop {min(args.top_n, len(results))} Best Satellites:\n")
    
    for rank, (sat_id, peak_time, peak_elev, line1, line2) in enumerate(results[:args.top_n], 1):
        time_offset = (peak_time - start_time).total_seconds() / 3600.0
        print(f"{rank:2d}. Satellite #{sat_id:5d}")
        print(f"    Peak Elevation: {peak_elev:.2f}°")
        print(f"    Peak Time: {peak_time.isoformat()}Z")
        print(f"    Time Offset: +{time_offset:.2f} hours from start")
        print()
    
    # Recommend the best satellite
    print("="*80)
    print("🎯 RECOMMENDATION FOR SIMULATION:")
    print("="*80)
    
    best_sat_id, best_time, best_elev, best_l1, best_l2 = results[0]
    
    print(f"\n✨ Best Satellite: #{best_sat_id}")
    print(f"   Peak Elevation: {best_elev:.2f}°")
    print(f"   Optimal Start Time: {best_time.isoformat()}Z")
    
    # Suggest a start time a bit before the peak for full pass coverage
    suggested_start = best_time - timedelta(minutes=5)
    print(f"   Suggested Sim Start: {suggested_start.isoformat()}Z")
    print(f"                        (5 minutes before peak)")
    
    print(f"\n📋 TLE Lines for Satellite #{best_sat_id}:")
    print(f"   {best_l1}")
    print(f"   {best_l2}")
    
    print("\n" + "="*80)
    print("You can use this satellite and start time in your simulation config!")
    print("="*80)


if __name__ == '__main__':
    main()
