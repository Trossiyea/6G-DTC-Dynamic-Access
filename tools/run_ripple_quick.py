#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick runner to compare Baseline / RadioMap / RIPPLE with a lightweight config.

Usage (in conda DtC env):
  conda run -n DtC python tools/run_ripple_quick.py
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
    CONFIG = dict(cfg_mod.CONFIG)
    # Lightweight overrides
    CONFIG['enable_ripple'] = True
    CONFIG['show_progress'] = False
    CONFIG['T'] = int(os.environ.get('RIPPLE_T', 120))
    CONFIG['N_UE'] = int(os.environ.get('RIPPLE_N_UE', 60))
    # Avoid strict Z check (map Z may vary by dataset)
    CONFIG['Z'] = None
    main_mod = load_module(CODE / 'main.py', 'main_module')
    out = main_mod.run_once(CONFIG)
    print('\n=== RIPPLE quick-run ===')
    print('Baseline-Default avg SE (bits/s/Hz): %.4f' % out['avg_se_baseline_default'])
    print('RadioMap         avg SE (bits/s/Hz): %.4f' % out['avg_se_radiomap'])
    if 'avg_se_ripple' in out:
        print('RIPPLE           avg SE (bits/s/Hz): %.4f' % out['avg_se_ripple'])
        if 'improvement_vs_default_pct_ripple' in out:
            print('Gain vs Default (RIPPLE) (%%): %+.2f' % out['improvement_vs_default_pct_ripple'])
    print('Gain vs Default (RadioMap) (%%): %+.2f' % out['improvement_vs_default_pct'])

if __name__ == '__main__':
    main()

