import numpy as np
from copy import deepcopy

from main import run_once
from config import CONFIG
from csi import sinr_to_cqi, cqi_to_se


def run_with(overrides):
    cfg = deepcopy(CONFIG)
    cfg.update(overrides)
    return run_once(cfg)


# DL-only: remove UL power-control test


def test_csi_delay_changes_outcome():
    # Create time variation to make delay observable
    base = run_with({
        "enable_time_varying": True,
        "rm_flicker_db_std": 2.0,
        "csi_delay_ttis": 0,
        "baseline_csi_delay_ttis": 0,
        "rm_csi_delay_ttis": 0,
        "seed": 22,
    })
    delayed = run_with({
        "enable_time_varying": True,
        "rm_flicker_db_std": 2.0,
        "csi_delay_ttis": 5,
        "baseline_csi_delay_ttis": 5,
        "rm_csi_delay_ttis": 5,
        "seed": 22,
    })
    print("[CSI delay] 0 vs 5 TTIs:", base["avg_se_radiomap"], delayed["avg_se_radiomap"])
    # Not asserting sign, but ensure some difference occurs
    assert abs(base["avg_se_radiomap"] - delayed["avg_se_radiomap"]) > 1e-6


def test_olla_offset():
    # Negative OLLA should reduce SE vs zero when using MCS mapping
    zero = run_with({"csi_olla_offset_db": 0.0, "use_mcs": True, "seed": 33})
    neg = run_with({"csi_olla_offset_db": -1.0, "use_mcs": True, "seed": 33})
    print("[OLLA] 0 vs -1 dB:", zero["avg_se_radiomap"], neg["avg_se_radiomap"])
    assert neg["avg_se_radiomap"] <= zero["avg_se_radiomap"] + 1e-9


def test_residual_freq_penalty():
    # Residual frequency error should reduce SE due to ICI penalty
    no_ici = run_with({"residual_freq_hz": 0.0, "seed": 44})
    ici = run_with({"residual_freq_hz": 1200.0, "seed": 44})
    print("[ICI] 0 Hz vs 1200 Hz:", no_ici["avg_se_radiomap"], ici["avg_se_radiomap"])
    assert ici["avg_se_radiomap"] <= no_ici["avg_se_radiomap"] + 1e-9


def test_orbit_dynamics_and_doppler():
    # With zero speed, Doppler should be ~0 and tau in [~1ms, ~10ms]
    zero = run_with({
        "enable_time_varying": True,
        "enable_orbit_dynamics": True,
        "sat_ground_speed_kms": 0.0,
        "doppler_residual_fraction": 0.0,
        "seed": 66,
    })
    assert zero["tau_time"] is not None and zero["fd_time"] is not None
    tau_ms = zero["tau_time"][0] * 1e3
    fd_abs = np.abs(zero["fd_time"][0])
    print("[Orbit] tau_ms range:", tau_ms.min(), tau_ms.max(), "fd max:", fd_abs.max())
    assert tau_ms.min() > 0.5 and tau_ms.max() < 15.0
    assert fd_abs.max() < 1e-6

    # Non-zero speed with residual Doppler should reduce SE (ICI penalty)
    no_resid = run_with({
        "enable_time_varying": True,
        "enable_orbit_dynamics": True,
        "sat_ground_speed_kms": 7.5,
        "doppler_residual_fraction": 0.0,
        "seed": 67,
    })
    resid = run_with({
        "enable_time_varying": True,
        "enable_orbit_dynamics": True,
        "sat_ground_speed_kms": 7.5,
        "doppler_residual_fraction": 0.1,
        "seed": 67,
    })
    print("[Orbit Doppler] 0 vs 10% residual:", no_resid["avg_se_radiomap"], resid["avg_se_radiomap"])
    assert resid["avg_se_radiomap"] <= no_resid["avg_se_radiomap"] + 1e-9


def test_mcs_table_equivalence():
    # Legacy and nr_64qam are identical in this initial drop
    leg = run_with({"csi_mcs_table": "legacy", "seed": 55})
    nr = run_with({"csi_mcs_table": "nr_64qam", "seed": 55})
    print("[MCS] legacy vs nr_64qam:", leg["avg_se_radiomap"], nr["avg_se_radiomap"])
    assert abs(leg["avg_se_radiomap"] - nr["avg_se_radiomap"]) < 1e-12


def test_cqi_mapping_boundaries():
    # Check a few thresholds for NR 64QAM table
    thr = [-6.8, -6.7, -2.3, 22.7, 25.0]
    c = sinr_to_cqi(np.array(thr), table="nr_64qam")
    # Below first threshold -> CQI 0; equal to threshold -> CQI >=1
    assert c[0] == 0 and c[1] >= 1 and c[-2] <= 15 and c[-1] == 15
    se = cqi_to_se(c, table="nr_64qam")
    assert se[0] == 0.0 and se[-1] > 5.0


# DL minimal preset: HO gating removed


def test_harq_full_basic():
    # Smoke test: full HARQ + TBS/BLER/OLLA runs without errors and yields finite SE
    out = run_with({
        "enable_time_varying": True,
        "enable_orbit_dynamics": False,
        "enable_harq_full": True,
        "harq_max_procs": 8,
        "harq_ack_delay_ttis": 5,
        "T": 30,
        "N_UE": 10,
        "seed": 88,
    })
    assert np.isfinite(out["avg_se_radiomap"]) and out["avg_se_radiomap"] >= 0.0

if __name__ == "__main__":
    # UL power-control test removed in DL-only configuration
    test_csi_delay_changes_outcome()
    test_olla_offset()
    test_residual_freq_penalty()
    test_orbit_dynamics_and_doppler()
    test_mcs_table_equivalence()
    test_cqi_mapping_boundaries()
    # HO gating test removed in minimal DL-only configuration
    test_harq_full_basic()
    print("All feature tests passed.")
