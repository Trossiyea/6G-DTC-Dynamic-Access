import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from config import CONFIG
from main import run_once

TLE1 = "1 58705C 24002A   25256.77687500  .00004774  00000+0  39334-4 0  2566"
TLE2 = "2 58705  53.1569  66.1378 0000804  91.2245 181.2078 15.69698397    13"

def main():
    c = dict(CONFIG)
    c.update({
        "T": int(os.getenv("MEAS_T", "120")),
        "N_UE": int(os.getenv("MEAS_N_UE", "50")),
        "enable_time_varying": True,
        "enable_orbit_dynamics": True,
        "enable_skyfield_orbit": True,
        "tle_ref_use_now": True,
        "auto_grid_center_from_tle": True,
        "tle_line1": TLE1,
        "tle_line2": TLE2,
        "enable_access_gating": True,
        "enable_beam_ho": True,
        "ho_ttt_ttis": 10,
        "ho_interrupt_ttis": 3,
        "enable_rach_gating": True,
        "rach_proc_ttis": 5,
    })
    out = run_once(c)
    ev = out.get("events") or {}
    ho = ev.get("ho") or {}
    rach = ev.get("rach") or {}
    ho_cnt = sum(len(v) for v in (ho.get("ho_start") or []))
    rach_cnt = sum(len(v) for v in (rach.get("rach_start") or []))
    res = {
        "avg_se_baseline_default": out.get("avg_se_baseline_default"),
        "avg_se_radiomap": out.get("avg_se_radiomap"),
        "impr_vs_default_pct": out.get("improvement_vs_default_pct"),
        "ho_events": ho_cnt,
        "rach_events": rach_cnt,
        "grid_center": {
            "lat_deg": c.get("grid_center_lat_deg"),
            "lon_deg": c.get("grid_center_lon_deg"),
        }
    }
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

