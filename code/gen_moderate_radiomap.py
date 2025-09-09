import numpy as np
from scipy.io import savemat

def generate_moderate_map(X=60, Y=60, K=64, scs_khz=30.0, seed=42,
                          n_lobes=6, peak_db_range=(-5.0, 8.0),
                          ax_rng=(6,14), ay_rng=(6,14), az_rng=(3,8)):
    rng = np.random.default_rng(seed)
    prb_bw_hz = 12.0 * scs_khz * 1e3
    noise_dbm = -174.0 + 10.0*np.log10(prb_bw_hz)
    noise_mw = 10.0**(noise_dbm/10.0)

    interf = np.zeros((X, Y, K), dtype=float)
    xs = np.arange(X).reshape(-1,1,1)
    ys = np.arange(Y).reshape(1,-1,1)
    zs = np.arange(K).reshape(1,1,-1)

    for _ in range(n_lobes):
        cx, cy, cz = rng.uniform(0, X), rng.uniform(0, Y), rng.uniform(0, K)
        ax, ay, az = rng.uniform(*ax_rng), rng.uniform(*ay_rng), rng.uniform(*az_rng)
        peak_rel_db = rng.uniform(*peak_db_range)
        peak_rel = 10.0**(peak_rel_db/10.0)
        w = np.exp(-(((xs-cx)**2)/(2*ax**2) + ((ys-cy)**2)/(2*ay**2) + ((zs-cz)**2)/(2*az**2)))
        interf += noise_mw * peak_rel * w

    interf = np.maximum(interf, 1e-18)
    return interf, noise_dbm

if __name__ == '__main__':
    X, Y, K = 60, 60, 64
    X_true, noise_dbm = generate_moderate_map(X, Y, K, scs_khz=30.0)
    savemat('radio_map/Data_moderate.mat', {'X_true': X_true})
    stats = (10*np.log10(X_true).min(), np.percentile(10*np.log10(X_true),50), np.percentile(10*np.log10(X_true),90), 10*np.log10(X_true).max())
    print(f'Saved radio_map/Data_moderate.mat shape={X_true.shape}, noise_dbm={noise_dbm:.2f}, interf_dbm[min,p50,p90,max]={stats}')

