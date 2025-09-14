"""
Probe a TLE with Skyfield to print subpoint (lat, lon, alt) at now and
future TTIs, and suggest grid center for OrbitSkyfield mapping.

Usage (inside ns3env):
  python tools/tle_probe.py "<tle1>" "<tle2>" [tti_ms]
"""

import sys
import json
from typing import Optional

try:
    from skyfield.api import EarthSatellite, wgs84, load
except Exception:
    print(json.dumps({"error": "skyfield not available"}))
    sys.exit(0)

def main(tle1: str, tle2: str, tti_ms: Optional[float] = 1.0):
    ts = load.timescale()
    sat = EarthSatellite(tle1, tle2)
    t0 = ts.now()
    out = []
    for k in range(0, 6):
        t = ts.tt_jd(t0.tt + (k * (tti_ms/1000.0)) / 86400.0)
        sp = wgs84.subpoint(sat.at(t))
        out.append({
            "t_idx": k,
            "lat_deg": float(sp.latitude.degrees),
            "lon_deg": float(sp.longitude.degrees),
            "alt_km": float(sp.elevation.km),
        })
    grid_center = {"grid_center_lat_deg": out[0]["lat_deg"], "grid_center_lon_deg": out[0]["lon_deg"]}
    print(json.dumps({"samples": out, "suggest_grid_center": grid_center}, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/tle_probe.py <tle1> <tle2> [tti_ms]", file=sys.stderr)
        sys.exit(1)
    tle1 = sys.argv[1]
    tle2 = sys.argv[2]
    tti_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    main(tle1, tle2, tti_ms)

