import numpy as np
from copy import deepcopy

from main import run_once, CONFIG


def run_with(overrides):
    cfg = deepcopy(CONFIG)
    cfg.update(overrides)
    return run_once(cfg)


def test_power_control_effect():
    # With this open-loop setting, P_tx is typically below P_max, so throughput should not exceed baseline.
    base = run_with({"enable_power_control": False, "seed": 11})
    pc = run_with({"enable_power_control": True, "pc_P0_dbm": -90.0, "pc_alpha": 0.8, "pc_M_ref": 1, "seed": 11})
    print("[PC] baseline vs PC:", base["avg_se_radiomap"], pc["avg_se_radiomap"])
    assert pc["avg_se_radiomap"] <= base["avg_se_radiomap"] + 1e-9


def test_csi_delay_changes_outcome():
    # Create time variation to make delay observable
    base = run_with({
        "enable_time_varying": True,
        "rm_flicker_db_std": 2.0,
        "csi_delay_ttis": 0,
        "seed": 22,
    })
    delayed = run_with({
        "enable_time_varying": True,
        "rm_flicker_db_std": 2.0,
        "csi_delay_ttis": 5,
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


def test_mcs_table_equivalence():
    # Legacy and nr_64qam are identical in this initial drop
    leg = run_with({"csi_mcs_table": "legacy", "seed": 55})
    nr = run_with({"csi_mcs_table": "nr_64qam", "seed": 55})
    print("[MCS] legacy vs nr_64qam:", leg["avg_se_radiomap"], nr["avg_se_radiomap"])
    assert abs(leg["avg_se_radiomap"] - nr["avg_se_radiomap"]) < 1e-12


if __name__ == "__main__":
    test_power_control_effect()
    test_csi_delay_changes_outcome()
    test_olla_offset()
    test_residual_freq_penalty()
    test_mcs_table_equivalence()
    print("All feature tests passed.")

