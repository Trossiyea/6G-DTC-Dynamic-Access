#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid-search small RIPPLE parameter sets and compare Baseline/RM/RIPPLE.

Strategy:
- Use reduced T for faster iteration (default T=400, keep N_UE from base config).
- Disable progress bars. Disable HARQ to isolate scheduler throughput and speed up.
- Keep power model equal for fairness.
- Measure SE for Baseline/RadioMap/RIPPLE and RIPPLE scheduler time share.

Usage (conda DtC env):
  conda run -n DtC python tools/search_ripple_params.py

Env overrides:
  T, N_UE to adjust quick-run scale.
"""
import importlib.util
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_once_with(cfg_base: dict, overrides: dict):
    main_mod = load_module(CODE / 'main.py', 'main_module')
    C = dict(cfg_base)
    C.update(overrides)
    return main_mod.run_once(C)


def main():
    # Ensure import paths include project and code directories
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))

    cfg_mod = load_module(CODE / 'config.py', 'base_config')
    base = dict(cfg_mod.CONFIG)

    # Quick-run scale
    import os
    T = int(os.environ.get('T', 400))
    N_UE = int(os.environ.get('N_UE', base.get('N_UE', 100)))

    # Common overrides for speed/fairness
    common = {
        'T': T,
        'N_UE': N_UE,
        'show_progress': False,
        'enable_ripple': True,
        'enable_harq_full': False,
        'enable_harq_deferral': False,
        'baseline_dl_power_model': 'equal_prb',
        'rm_dl_power_model': 'equal_prb',
        'ripple_dl_power_model': 'equal',
        'ripple_use_mcs': True,
        # keep beta consistent with RM unless explicitly changed
        # 'ripple_eesm_beta_db': base.get('rm_sched_eesm_beta_db', base.get('sched_eesm_beta_db', 1.0)),
    }

    seg_from_list = ['interference', 'snr']
    seg_thresh_list = [2.0, 2.5, 3.0]
    max_len_list = [0, 6, 10]

    combos = list(itertools.product(seg_from_list, seg_thresh_list, max_len_list))

    print('\n=== RIPPLE parameter search (quick) ===')
    print(f'Scale: T={T}, N_UE={N_UE}, power=equal, HARQ=off')
    hdr = f"{'from':>12s}  {'thr':>4s} {'Lmax':>5s} | {'Base':>8s} {'RM':>8s} {'RIPPLE':>8s} | {'R%':>6s} {'S%':>6s}  time(s)"
    print(hdr)
    print('-' * len(hdr))

    best = None  # (se_ripple, combo, row_str)

    for seg_from, thr, Lmax in combos:
        overrides = dict(common)
        overrides.update({
            'ripple_seg_from': seg_from,
            'ripple_seg_thresh_db': thr,
            'ripple_max_seg_len': Lmax,
        })
        try:
            out = run_once_with(base, overrides)
            b = float(out['avg_se_baseline_default'])
            m = float(out['avg_se_radiomap'])
            r = float(out.get('avg_se_ripple', 0.0))
            tt = float(out.get('time_total_s', 0.0) or 0.0)
            tr = float(out.get('time_sched_ripple_s', 0.0) or 0.0)
            r_imp = 100.0 * (r - b) / max(1e-9, b)
            r_share = 100.0 * tr / max(1e-9, tt)
            row = f"{seg_from:>12s}  {thr:4.1f} {Lmax:5d} | {b:8.4f} {m:8.4f} {r:8.4f} | {r_imp:6.2f} {r_share:6.2f}  {tr:6.2f}"
            print(row)
            if (best is None) or (r > best[0]):
                best = (r, (seg_from, thr, Lmax), row)
        except Exception as e:
            print(f"{seg_from:>12s}  {thr:4.1f} {Lmax:5d} | ERROR: {e}")
            continue

    if best is not None:
        print('\nBest RIPPLE SE (quick):')
        print(best[2])
        print('Use these in full-scale: ripple_seg_from=%s, ripple_seg_thresh_db=%.1f, ripple_max_seg_len=%d' % best[1])


if __name__ == '__main__':
    main()
