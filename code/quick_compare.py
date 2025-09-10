import sys
import numpy as np

# Make 'code' modules importable when run from project root
if 'code' not in sys.path:
    sys.path.append('code')

import main, config

def run_compare():
    cfg_block = dict(config.CONFIG)
    cfg_block['seed'] = int(cfg_block.get('seed', 1))
    cfg_block['sched_block_mode'] = True
    out_block = main.run_once(cfg_block)

    cfg_prb = dict(cfg_block)
    cfg_prb['sched_block_mode'] = False
    out_prb = main.run_once(cfg_prb)

    print('[Single] block-mode:   base=%.3f radiomap=%.3f imp=%.2f%%' % (
        out_block['avg_se_baseline'], out_block['avg_se_radiomap'], out_block['improvement_pct']))
    print('[Single] per-PRB orig: base=%.3f radiomap=%.3f imp=%.2f%%' % (
        out_prb['avg_se_baseline'], out_prb['avg_se_radiomap'], out_prb['improvement_pct']))

    # Multi-seed over 10 seeds
    seeds = np.arange(1, 11)
    res_block = main.run_many(cfg_block, seeds)
    res_prb = main.run_many(cfg_prb, seeds)
    print('[Multi-mean] block radiomap=%.3f baseline=%.3f' % (res_block['radiomap'].mean(), res_block['baseline'].mean()))
    print('[Multi-mean]  prb  radiomap=%.3f baseline=%.3f' % (res_prb['radiomap'].mean(), res_prb['baseline'].mean()))

if __name__ == '__main__':
    run_compare()

