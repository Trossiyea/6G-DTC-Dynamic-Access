#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick constellation run (reduced T) to compare Baseline/RM/RIPPLE using defaults.

Usage (conda DtC env):
  conda run -n DtC python tools/run_constellation_quick.py
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

    C = dict(cfg_mod.CONFIG)
    C['enable_constellation'] = True
    # Use catalog for constellation; clear single-sat TLE
    C['tle_catalog_path'] = str(ROOT / 'tles' / 'starlink_DTC_tle.txt')
    C['tle_name'] = None
    C['tle_lines'] = None
    C['tle_path'] = None
    # Reduce T and disable progress for quicker turnaround
    C['T'] = int(os.environ.get('T', 400))
    C['show_progress'] = False

    out = main_mod.run_constellation(C)
    print('\n=== Constellation quick-run ===')
    print('Baseline-Default avg SE (bits/s/Hz): %.4f' % out['avg_se_baseline_default'])
    print('RadioMap         avg SE (bits/s/Hz): %.4f' % out['avg_se_radiomap'])
    if out.get('avg_se_ripple') is not None:
        print('RIPPLE           avg SE (bits/s/Hz): %.4f' % out['avg_se_ripple'])
        if out.get('improvement_vs_default_pct_ripple') is not None:
            print('Gain vs Default (RIPPLE) (%%): %+.2f' % out['improvement_vs_default_pct_ripple'])
    print('Gain vs Default (RadioMap) (%%): %+.2f' % out['improvement_vs_default_pct'])

if __name__ == '__main__':
    main()

