import sys, os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from config import CONFIG
from main import run_once

try:
    from orbit_skyfield import OrbitSkyfield
    HAS_SKY = True
except Exception:
    OrbitSkyfield = None
    HAS_SKY = False

TLE1 = "1 58705C 24002A   25256.77687500  .00004774  00000+0  39334-4 0  2566"
TLE2 = "2 58705  53.1569  66.1378 0000804  91.2245 181.2078 15.69698397    13"


def maybe_mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def percentile_across_ue(arr_t_u, q=50):
    return np.percentile(arr_t_u, q, axis=1)


def main():
    t = int(os.getenv('MEAS_T', '600'))
    n = int(os.getenv('MEAS_N_UE', '100'))
    outdir = os.getenv('MEAS_OUT', f"output/skyfield_measure_{int(time.time())}")
    maybe_mkdir(outdir)

    cfg = dict(CONFIG)
    cfg.update({
        'T': t,
        'N_UE': n,
        'enable_time_varying': True,
        'enable_orbit_dynamics': True,
        'enable_skyfield_orbit': True,
        'tle_ref_use_now': True,
        'auto_grid_center_from_tle': True,
        'tle_line1': TLE1,
        'tle_line2': TLE2,
        'enable_access_gating': True,
        'enable_beam_ho': True,
        'ho_ttt_ttis': 10,
        'ho_interrupt_ttis': 3,
        'enable_rach_gating': True,
        'rach_proc_ttis': 5,
    })

    res = run_once(cfg)
    events = res.get('events') or {}
    ho = events.get('ho') or {}
    rach = events.get('rach') or {}
    ho_counts = np.array([len(x) for x in (ho.get('ho_start') or [[]]*cfg['N_UE'])], dtype=int)
    rach_counts = np.array([len(x) for x in (rach.get('rach_start') or [[]]*cfg['N_UE'])], dtype=int)

    summary = {
        'T': t,
        'N_UE': n,
        'avg_se_baseline_default': res.get('avg_se_baseline_default'),
        'avg_se_radiomap': res.get('avg_se_radiomap'),
        'impr_vs_default_pct': res.get('improvement_vs_default_pct'),
        'ho_events_total': int(ho_counts.sum()) if ho_counts.size else 0,
        'rach_events_total': int(rach_counts.sum()) if rach_counts.size else 0,
        'grid_center': {
            'lat_deg': cfg.get('grid_center_lat_deg'),
            'lon_deg': cfg.get('grid_center_lon_deg'),
        },
    }
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(6,4))
    bins_ho = min(30, max(5, int(np.sqrt(max(1, ho_counts.size)))))
    plt.hist(ho_counts, bins=bins_ho, edgecolor='black')
    plt.title('HO events per UE')
    plt.xlabel('HO count')
    plt.ylabel('UEs')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'ho_counts.png'), dpi=140)
    plt.close()

    plt.figure(figsize=(6,4))
    bins_ra = min(30, max(5, int(np.sqrt(max(1, rach_counts.size)))))
    plt.hist(rach_counts, bins=bins_ra, edgecolor='black')
    plt.title('RACH events per UE')
    plt.xlabel('RACH count')
    plt.ylabel('UEs')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rach_counts.png'), dpi=140)
    plt.close()

    tau_time = res.get('tau_time')
    fd_time = res.get('fd_time')
    if isinstance(tau_time, np.ndarray):
        x = np.arange(tau_time.shape[0])
        plt.figure(figsize=(7,4))
        for q in [5, 50, 95]:
            plt.plot(x, percentile_across_ue(tau_time, q=q)*1e6, label=f'{q}th')
        plt.xlabel('TTI')
        plt.ylabel('Propagation delay (µs)')
        plt.title('Tau percentiles over time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'tau_percentiles.png'), dpi=140)
        plt.close()
    if isinstance(fd_time, np.ndarray):
        x = np.arange(fd_time.shape[0])
        plt.figure(figsize=(7,4))
        for q in [5, 50, 95]:
            plt.plot(x, percentile_across_ue(fd_time, q=q), label=f'{q}th')
        plt.xlabel('TTI')
        plt.ylabel('Doppler (Hz)')
        plt.title('Doppler percentiles over time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'doppler_percentiles.png'), dpi=140)
        plt.close()

    R = res.get('R_xyz_dbm')
    if isinstance(R, np.ndarray):
        R_med = np.median(R, axis=2)
        plt.figure(figsize=(5,5))
        plt.imshow(R_med.T, origin='lower', aspect='equal')
        plt.title('Interference Map median (dBm)')
        plt.colorbar(label='dBm')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'radio_map_median.png'), dpi=140)
        plt.close()

    if bool(cfg.get('enable_skyfield_orbit', False)) and HAS_SKY and isinstance(R, np.ndarray):
        try:
            orb = OrbitSkyfield(cfg, R.shape[0], R.shape[1])
            xs, ys = [], []
            for ti in range(min(t, 1000)):
                cx, cy = orb.beam_center_at(ti)
                xs.append(cx)
                ys.append(cy)
            plt.figure(figsize=(5,5))
            plt.plot(xs, ys, '-', lw=1)
            plt.xlim([0, R.shape[0]])
            plt.ylim([0, R.shape[1]])
            plt.title('Beam center track (px)')
            plt.xlabel('x (px)')
            plt.ylabel('y (px)')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'beam_center_track.png'), dpi=140)
            plt.close()
        except Exception:
            pass

    print(json.dumps({'outdir': outdir, **summary}, indent=2))


if __name__ == '__main__':
    main()

