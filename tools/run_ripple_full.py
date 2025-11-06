#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run full standard-scale single-satellite experiment (run_once) and print
Baseline / RadioMap / RIPPLE SE with scheduler timing shares.

Usage (conda DtC env):
  conda run -n DtC python tools/run_ripple_full.py
"""
import importlib.util
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    cfg_mod = load_module(CODE / 'config.py', 'base_config')
    main_mod = load_module(CODE / 'main.py', 'main_module')

    # Use global defaults from config.py without overrides (except UI)
    C = dict(cfg_mod.CONFIG)
    C['show_progress'] = False

    out = main_mod.run_once(C)
    print('\n=== Full-scale single run ===')
    print('Baseline-Default avg SE (bits/s/Hz): %.4f' % out['avg_se_baseline_default'])
    print('RadioMap         avg SE (bits/s/Hz): %.4f' % out['avg_se_radiomap'])
    if 'avg_se_ripple' in out:
        print('RIPPLE           avg SE (bits/s/Hz): %.4f' % out['avg_se_ripple'])
        if 'improvement_vs_default_pct_ripple' in out:
            print('Gain vs Default (RIPPLE) (%%): %+.2f' % out['improvement_vs_default_pct_ripple'])
    print('Gain vs Default (RadioMap) (%%): %+.2f' % out['improvement_vs_default_pct'])

    # Optional debug: ripple segment stats if provided
    dbg = out.get('ripple_debug', None)
    if dbg:
        print('\n--- RIPPLE debug ---')
        print('K=%d, se_seg min/mean/median/max = %.4g / %.4g / %.4g / %.4g' % (
            int(dbg.get('K', 0)), float(dbg.get('se_min', 0.0)), float(dbg.get('se_mean', 0.0)), float(dbg.get('se_median', 0.0)), float(dbg.get('se_max', 0.0))
        ))
        print('seg_thresh_db=%.2f, max_len=%d, from=%s' % (
            float(C.get('ripple_seg_thresh_db', 3.0)), int(C.get('ripple_max_seg_len', 0) or 0), str(C.get('ripple_seg_from', 'interference'))
        ))

    # Timing shares
    tt = float(out.get('time_total_s', 0.0) or 0.0)
    tb = out.get('time_sched_baseline_s', None)
    tr = out.get('time_sched_radiomap_s', None)
    tp = out.get('time_sched_ripple_s', None)
    if tt > 0:
        print('\n--- Scheduler timing ---')
        if tb is not None:
            print('Baseline time: %.3f s (%.2f%% of total)' % (tb, 100.0*tb/tt))
        if tr is not None:
            print('RadioMap time: %.3f s (%.2f%% of total)' % (tr, 100.0*tr/tt))
        if tp is not None:
            print('RIPPLE   time: %.3f s (%.2f%% of total)' % (tp, 100.0*tp/tt))

if __name__ == '__main__':
    main()
