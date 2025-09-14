"""
Optional SGP4-based orbit model (Stage-3 scaffold).

If the 'sgp4' package is available and TLE lines are provided in config,
this model computes satellite subpoint and approximated ground-track
velocity, then reuses OrbitModel's geometry and beam math. If not
available, falls back to a simple constant-velocity model.
"""

from typing import Dict, Tuple
import numpy as np
import math

try:
    from sgp4.api import Satrec, jday
    HAS_SGP4 = True
except Exception:
    HAS_SGP4 = False

from orbit import OrbitModel


class OrbitSGP4(OrbitModel):
    def __init__(self, config: Dict, X: int, Y: int):
        super().__init__(config, X, Y)
        self.has = HAS_SGP4 and bool(config.get("tle_line1")) and bool(config.get("tle_line2"))
        if self.has:
            self.sat = Satrec.twoline2rv(config["tle_line1"], config["tle_line2"])
            # Reference epoch (UTC) from config or J2000 now
            self.jy = int(config.get("tle_ref_year", 2024))
            self.jm = int(config.get("tle_ref_month", 1))
            self.jd = int(config.get("tle_ref_day", 1))
            self.jh = int(config.get("tle_ref_hour", 0))
            self.jmin = int(config.get("tle_ref_min", 0))
            self.js = float(config.get("tle_ref_sec", 0.0))
        else:
            # Fallback
            pass

    def beam_center_at(self, t: int) -> Tuple[float, float]:
        if not self.has:
            return super().beam_center_at(t)
        # Convert t*Tti to epoch seconds
        tti_s = self.tti_s
        jd, fr = jday(self.jy, self.jm, self.jd, self.jh, self.jmin, self.js + t*tti_s)
        e, r, v = self.sat.sgp4(jd, fr)
        if e != 0:
            return super().beam_center_at(t)
        # r (km) ECI; project to simple ground-plane pixels by a crude mapping:
        # use x,y as km offsets relative to origin and wrap into map extents.
        # Note: This is a placeholder; proper ECI->ECEF and subpoint calc can be added later.
        cx = (r[0] / self.cell_km) % self.X
        cy = (r[1] / self.cell_km) % self.Y
        return float(cx), float(cy)

