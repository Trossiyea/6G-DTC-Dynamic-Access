#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark RIPPLE vs RadioMap (scheduler runtime, standard scale).

This script isolates the scheduler loops by:
- Building inputs (cap, snr_lin, I_total_dbm) once from the base config
- Disabling progress bars and HARQ for fair timing
- Disabling water-filling so both schedulers use equal-power inside

Usage (conda DtC env):
  conda run -n DtC python tools/bench_ripple_vs_rm.py

Environment overrides:
  RIPPLE_T, RIPPLE_N_UE  (optional)  to override T and N_UE while keeping scale large.
"""
import importlib.util
import sys
import time
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

    # Load base config and main helpers
    cfg_mod = load_module(CODE / 'config.py', 'base_config')
    main_mod = load_module(CODE / 'main.py', 'main_module')
    ripple_mod = load_module(CODE / 'ripple.py', 'ripple_module')

    CONFIG = dict(cfg_mod.CONFIG)
    # Standard scale by default; allow env overrides
    import os
    T = int(os.environ.get('RIPPLE_T', CONFIG.get('T', 2000)))
    N_UE = int(os.environ.get('RIPPLE_N_UE', CONFIG.get('N_UE', 100)))

    # Fair-timing overrides (in-memory, do not change files)
    C = dict(CONFIG)
    C['T'] = T
    C['N_UE'] = N_UE
    C['show_progress'] = False
    # Avoid strict Z checks in case radiomap Z differs; we will use loaded Z
    C['Z'] = None
    # Disable HARQ to isolate scheduler timing
    C['enable_harq_full'] = False
    C['enable_harq_deferral'] = False
    # Equal-power everywhere for fairness/runtime focus
    C['baseline_dl_power_model'] = 'equal_prb'
    C['rm_dl_power_model'] = 'equal_prb'

    # Build inputs similarly to run_once, but only once (no time-varying series)
    rng = __import__('numpy').random.default_rng(C.get('seed', 101))
    R_xyz_dbm, X, Y, Z = main_mod.select_radio_map(C)
    ue_pos = main_mod.generate_ue_positions(N_UE, X, Y, rng)
    L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue = main_mod.compute_geometry_and_beam(C, X, Y, ue_pos)
    noise_dbm, _ = main_mod.resolve_noise_and_prb_bw(C)
    P_tx_dbm = main_mod.apply_open_loop_power_control(C, L_fs_per_ue, G_rx_per_ue)
    cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb = main_mod.compute_caps(
        R_xyz_dbm, ue_pos,
        P_tx_dbm=P_tx_dbm,
        L_fs_db=L_fs_per_ue,
        G_rx_db=G_rx_per_ue,
        shadow_db_std=C.get('shadow_std_db', 7.0),
        N0_dbm=noise_dbm,
        rx_nf_db=C.get('rx_nf_db', 0.0),
        impl_loss_db=C.get('impl_loss_db', 0.0),
        seed=C.get('seed', 101),
        elevation_deg=elev_deg_per_ue,
        channel_model=C.get('channel_model', '3gpp_ntn'),
        channel_params=C.get('channel_params'),
        channel_profile=C.get('ntn_channel_profile', 's_band_handheld_urban'),
    )

    # MCS params (keep consistent with main)
    mcs_params = {
        'olla_offset_db': C.get('csi_olla_offset_db', 0.0),
        'mcs_table': C.get('csi_mcs_table', 'legacy'),
        'residual_freq_hz': C.get('residual_freq_hz', 0.0),
        'scs_khz': C.get('scs_khz', 30),
    }

    # RadioMap scheduler timing
    t0 = time.perf_counter()
    se_rm = main_mod.pf_schedule_radiomap_blocks(
        cap, T, beta=C['pf_beta'],
        snr_lin=snr_lin,
        overhead_eff=C.get('overhead_eff', 1.0),
        use_mcs=C.get('use_mcs', False),
        power_split=C.get('power_split', False),
        se_metric_override=None,
        max_prbs_per_ue=C.get('rm_max_prbs_per_ue', C.get('max_prbs_per_ue')),
        mcs_params=mcs_params,
        se_metric_time=None,
        snr_lin_time=None,
        eesm_beta_db=float(C.get('rm_sched_eesm_beta_db', C.get('sched_eesm_beta_db', 1.0))),
        require_contiguous=bool(C.get('sched_require_contiguous', True)),
        rng=rng,
        ue_mask_time=None,
        harq_mgr=None,
        dl_power_model='equal_prb',
        P_tot_dbm=None,
        P_ref_dbm=C.get('P_tx_dbm'),
        p_min_dbm=None,
        p_max_dbm=None,
        record_assignments=False,
        assignments_out=None,
        config=C,
    )
    t1 = time.perf_counter()

    # RIPPLE scheduler timing
    ripple_beta = float(C.get('ripple_eesm_beta_db', C.get('rm_sched_eesm_beta_db', C.get('sched_eesm_beta_db', 1.0))))
    t2 = time.perf_counter()
    se_rp = ripple_mod.ripple_schedule(
        cap=cap,
        T=T,
        beta=C['pf_beta'],
        snr_lin=snr_lin,
        overhead_eff=C.get('overhead_eff', 1.0),
        use_mcs=C.get('use_mcs', False),
        mcs_params=mcs_params,
        snr_lin_time=None,
        I_total_dbm=I_total_dbm,
        eesm_beta_db=ripple_beta,
        rng=rng,
        ue_mask_time=None,
        harq_mgr=None,
        record_assignments=False,
        assignments_out=None,
        config=C,
    )
    t3 = time.perf_counter()

    dt_rm = t1 - t0
    dt_rp = t3 - t2
    ratio = dt_rm / dt_rp if dt_rp > 0 else float('inf')

    print('\n=== RIPPLE vs RadioMap benchmark (scheduler only) ===')
    print(f'Scale: T={T}, N_UE={N_UE}, Z={Z}')
    print(f'RadioMap avg SE: {se_rm:.4f} bits/s/Hz, time: {dt_rm:.3f} s')
    print(f'RIPPLE   avg SE: {se_rp:.4f} bits/s/Hz, time: {dt_rp:.3f} s')
    print(f'Runtime speedup (RM/RIPPLE): {ratio:.2f}x')


if __name__ == '__main__':
    main()

