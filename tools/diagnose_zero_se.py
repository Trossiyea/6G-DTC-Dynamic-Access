#!/usr/bin/env python3
"""
Diagnose why spectral efficiency is zero after using find_best_satellite.py

This script checks:
1. TLE consistency between find_best_satellite and config.py
2. Time window validity
3. Satellite visibility at configured reference point
4. Elevation angles during simulation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from datetime import datetime, timedelta, timezone
import numpy as np

try:
    from skyfield.api import EarthSatellite, load, wgs84
    from config import CONFIG
except ImportError as e:
    print(f"ERROR: Missing dependencies: {e}")
    print("Please install: pip install skyfield sgp4")
    sys.exit(1)

def main():
    print("="*80)
    print("🔍 Diagnosing Zero Spectral Efficiency Problem")
    print("="*80)
    
    # Extract config
    tle_name = CONFIG.get('tle_name', 'Unknown')
    tle_lines = CONFIG.get('tle_lines')
    ref_lat = float(CONFIG.get('ref_lat_deg', 0))
    ref_lon = float(CONFIG.get('ref_lon_deg', 0))
    start_time_str = CONFIG.get('orbit_start_datetime')
    min_elev = float(CONFIG.get('min_elev_deg', 10.0))
    T_ttis = int(CONFIG.get('T', 2000))
    tti_ms = float(CONFIG.get('tti_ms', 1.0))
    
    print(f"\n📋 Configuration from config.py:")
    print(f"   Satellite: {tle_name}")
    print(f"   Reference Location: ({ref_lat:.5f}°N, {ref_lon:.5f}°W)")
    print(f"   Start Time: {start_time_str}")
    print(f"   Min Elevation: {min_elev:.1f}°")
    print(f"   Simulation Length: {T_ttis} TTIs ({T_ttis * tti_ms / 1000:.1f} seconds)")
    
    if not tle_lines or len(tle_lines) < 2:
        print("\n❌ ERROR: No TLE lines found in CONFIG['tle_lines']")
        return
    
    line1, line2 = tle_lines[0], tle_lines[1]
    print(f"\n📡 TLE Lines:")
    print(f"   {line1}")
    print(f"   {line2}")
    
    # Parse start time
    try:
        # Handle various ISO formats: clean up multiple timezone indicators
        time_str = start_time_str.strip()
        # Remove all 'Z' and '+00:00', then add single '+00:00' at end
        time_str = time_str.replace('Z', '').replace('+00:00', '')
        # Handle fractional seconds properly
        if '.' in time_str:
            # Has microseconds
            time_str = time_str + '+00:00'
        else:
            time_str = time_str + '+00:00'
        start_time = datetime.fromisoformat(time_str)
        start_time = start_time.astimezone(timezone.utc)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to parse start time: {e}")
        print(f"   Raw string: {start_time_str}")
        return
    
    print(f"\n⏰ Time Window:")
    print(f"   Start: {start_time.isoformat()}")
    sim_duration_sec = T_ttis * tti_ms / 1000.0
    end_time = start_time + timedelta(seconds=sim_duration_sec)
    print(f"   End:   {end_time.isoformat()}")
    print(f"   Duration: {sim_duration_sec:.1f} seconds ({sim_duration_sec/60:.1f} minutes)")
    
    # Initialize Skyfield
    print(f"\n🛰️  Computing satellite visibility...")
    ts = load.timescale()
    sat = EarthSatellite(line1, line2, tle_name)
    ground = wgs84.latlon(ref_lat, ref_lon)
    
    # Sample every 10 seconds
    sample_interval_sec = 10.0
    n_samples = int(sim_duration_sec / sample_interval_sec) + 1
    sample_times = [start_time + timedelta(seconds=i * sample_interval_sec) 
                   for i in range(n_samples)]
    
    elevations = []
    azimuths = []
    distances = []
    
    for t in sample_times:
        t_sf = ts.utc(t.year, t.month, t.day, t.hour, t.minute, 
                     t.second + t.microsecond * 1e-6)
        sp = sat.at(t_sf)
        topoc = sp - ground.at(t_sf)
        alt, az, dist = topoc.altaz()
        elevations.append(float(alt.degrees))
        azimuths.append(float(az.degrees))
        distances.append(float(dist.km))
    
    elevations = np.array(elevations)
    max_elev = np.max(elevations)
    min_elev_val = np.min(elevations)
    avg_elev = np.mean(elevations)
    
    print(f"\n📊 Elevation Statistics:")
    print(f"   Maximum: {max_elev:.2f}°")
    print(f"   Minimum: {min_elev_val:.2f}°")
    print(f"   Average: {avg_elev:.2f}°")
    print(f"   Required (min_elev_deg): {min_elev:.2f}°")
    
    visible_samples = np.sum(elevations >= min_elev)
    visibility_pct = 100.0 * visible_samples / len(elevations)
    
    print(f"\n👁️  Visibility Analysis:")
    print(f"   Visible samples: {visible_samples}/{len(elevations)} ({visibility_pct:.1f}%)")
    
    if visibility_pct < 1.0:
        print(f"\n❌ PROBLEM FOUND: Satellite is almost never visible!")
        print(f"   The satellite elevation is below {min_elev}° for {100-visibility_pct:.1f}% of the time.")
        print(f"\n   Possible causes:")
        print(f"   1. Wrong TLE (satellite not passing over reference location)")
        print(f"   2. Wrong start time (satellite not at optimal position)")
        print(f"   3. Wrong reference coordinates")
    elif visibility_pct < 50.0:
        print(f"\n⚠️  WARNING: Satellite is visible only {visibility_pct:.1f}% of the time")
        print(f"   This will result in low throughput.")
    else:
        print(f"\n✅ Satellite is visible for most of the simulation ({visibility_pct:.1f}%)")
    
    # Find peak elevation time
    peak_idx = int(np.argmax(elevations))
    peak_time = sample_times[peak_idx]
    peak_elev = elevations[peak_idx]
    
    print(f"\n🎯 Peak Elevation:")
    print(f"   Time: {peak_time.isoformat()}")
    print(f"   Elevation: {peak_elev:.2f}°")
    print(f"   Time offset from start: {(peak_time - start_time).total_seconds():.0f} seconds")
    
    # Check if peak is within simulation window
    if peak_idx == 0 or peak_idx == len(elevations) - 1:
        print(f"\n⚠️  Peak elevation is at the edge of simulation window!")
        print(f"   Consider adjusting start time to center the pass.")
    
    # Recommendations
    print(f"\n" + "="*80)
    print(f"💡 Recommendations:")
    print(f"="*80)
    
    if max_elev < min_elev:
        print(f"\n❌ CRITICAL: Maximum elevation ({max_elev:.2f}°) < min_elev_deg ({min_elev:.2f}°)")
        print(f"\n   Solutions:")
        print(f"   1. Re-run find_best_satellite.py with:")
        print(f"      --lat {ref_lat}")
        print(f"      --lon {ref_lon}")
        print(f"      --min-elev {min_elev}")
        print(f"      --hours 48  # Increase search window")
        print(f"")
        print(f"   2. Or lower min_elev_deg in config.py to {max_elev - 5:.0f}°")
    
    elif visibility_pct < 50.0:
        print(f"\n⚠️  Low visibility ({visibility_pct:.1f}%)")
        print(f"\n   Suggestions:")
        print(f"   1. Adjust orbit_start_datetime to {peak_time - timedelta(minutes=2)}")
        print(f"      This centers the peak in your simulation window.")
        print(f"")
        print(f"   2. Or re-run find_best_satellite.py with larger time window:")
        print(f"      --hours 48")
    
    else:
        print(f"\n✅ Configuration looks good!")
        print(f"   Satellite should be visible during simulation.")
        print(f"\n   If you're still getting zero spectral efficiency, check:")
        print(f"   1. Radio Map loading (errors in logs?)")
        print(f"   2. UE positions (are they within valid map bounds?)")
        print(f"   3. Channel model parameters")
        print(f"   4. Run main.py with verbose output to see UE attachment events")
    
    print(f"\n" + "="*80)
    
    # Output elevation profile for plotting
    print(f"\n📈 Elevation Profile (every 10s):")
    print(f"   Time(s)  Elevation(°)")
    for i in range(0, len(sample_times), max(1, len(sample_times)//20)):
        t_offset = (sample_times[i] - start_time).total_seconds()
        print(f"   {t_offset:6.0f}   {elevations[i]:6.2f}")
    
    print(f"\n" + "="*80)


if __name__ == '__main__':
    main()
