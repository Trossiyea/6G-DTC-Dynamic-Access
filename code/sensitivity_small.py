import sys
import numpy as np
if 'code' not in sys.path:
    sys.path.append('code')
import main, config

def run():
    seeds = np.arange(1, 6)
    print('Beta sensitivity (N_UE=40, T=80, 5 seeds):')
    for b in [0.5, 1.0, 2.0, 3.0, 4.0]:
        cfg = dict(config.CONFIG)
        cfg.update(dict(T=80, N_UE=40, sched_block_mode=True, sched_eesm_beta_db=b, sched_eesm_beta_by_mcs=False))
        res = main.run_many(cfg, seeds)
        gain = (res['radiomap'].mean() - res['baseline'].mean()) / max(1e-9, res['baseline'].mean()) * 100.0
        print('  beta=%3.1f  radiomap=%.3f  baseline=%.3f  gain=%.1f%%' % (
            b, res['radiomap'].mean(), res['baseline'].mean(), gain))
    cfg = dict(config.CONFIG)
    cfg.update(dict(T=80, N_UE=40, sched_block_mode=True, sched_eesm_beta_db=1.0, sched_eesm_beta_by_mcs=True))
    res = main.run_many(cfg, seeds)
    gain = (res['radiomap'].mean() - res['baseline'].mean()) / max(1e-9, res['baseline'].mean()) * 100.0
    print('  beta-by-MCS on: radiomap=%.3f baseline=%.3f gain=%.1f%%' % (res['radiomap'].mean(), res['baseline'].mean(), gain))

if __name__ == '__main__':
    run()

