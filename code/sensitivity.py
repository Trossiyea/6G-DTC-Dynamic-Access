import sys
import numpy as np

if 'code' not in sys.path:
    sys.path.append('code')

import main, config

def main_eval():
    betas = [0.5, 1.0, 2.0, 3.0, 4.0]
    seeds = np.arange(1, 11)
    print('Beta sensitivity over 10 seeds:')
    for b in betas:
        cfg = dict(config.CONFIG)
        cfg['sched_block_mode'] = True
        cfg['sched_eesm_beta_db'] = b
        cfg['sched_eesm_beta_by_mcs'] = False
        res = main.run_many(cfg, seeds)
        gain = (res['radiomap'].mean() - res['baseline'].mean()) / max(1e-9, res['baseline'].mean()) * 100.0
        print('  beta=%3.1f  radiomap=%.3f  baseline=%.3f  gain=%.1f%%' % (
            b, res['radiomap'].mean(), res['baseline'].mean(), gain))

    # Also test beta-by-MCS toggle
    cfg = dict(config.CONFIG)
    cfg['sched_block_mode'] = True
    cfg['sched_eesm_beta_db'] = 1.0
    cfg['sched_eesm_beta_by_mcs'] = True
    res = main.run_many(cfg, seeds)
    gain = (res['radiomap'].mean() - res['baseline'].mean()) / max(1e-9, res['baseline'].mean()) * 100.0
    print('  beta-by-MCS on: radiomap=%.3f baseline=%.3f gain=%.1f%%' % (res['radiomap'].mean(), res['baseline'].mean(), gain))

if __name__ == '__main__':
    main_eval()

