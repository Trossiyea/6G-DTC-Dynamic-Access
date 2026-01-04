#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a single scenario with NS-GBS enabled.
"""

import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = SCRIPT_DIR / "code"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


SCENARIOS = {
    "toronto_single": "test/config_toronto_single.py",
    "toronto_constellation": "test/config_toronto_constellation.py",
    "shanghai_single": "test/config_shanghai_single.py",
    "shanghai_constellation": "test/config_shanghai_constellation.py",
    "toronto_125m": "test/config_toronto_single_125m.py",
    "toronto_150m": "test/config_toronto_single_150m.py",
    "shanghai_125m": "test/config_shanghai_single_125m.py",
    "shanghai_150m": "test/config_shanghai_single_150m.py",
}


def load_config_from_file(config_path: Path):
    spec = importlib.util.spec_from_file_location("test_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def load_base_config():
    config_path = SCRIPT_DIR / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("base_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def main():
    parser = argparse.ArgumentParser(description="Run a scenario with NS-GBS enabled.")
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), default="toronto_single")
    parser.add_argument("--config", help="Path to a custom config file")
    parser.add_argument("--model", required=True, help="Path to NS-GBS model (.pt or .ts)")
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--N_UE", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    base_cfg = load_base_config()
    cfg = base_cfg.copy()

    if args.config:
        cfg.update(load_config_from_file(Path(args.config)))
    else:
        cfg.update(load_config_from_file(SCRIPT_DIR / SCENARIOS[args.scenario]))

    if args.T is not None:
        cfg["T"] = int(args.T)
    if args.N_UE is not None:
        cfg["N_UE"] = int(args.N_UE)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    cfg["show_progress"] = not args.no_progress
    cfg["scheduler_kind"] = "nsgbs"
    cfg["nsgbs_model_path"] = args.model

    from main import run_once

    out = run_once(cfg)
    print("NS-GBS avg SE:", out.get("avg_se_radiomap"))
    print("Baseline avg SE:", out.get("avg_se_baseline_default"))
    print("Gain (%):", out.get("improvement_vs_default_pct"))


if __name__ == "__main__":
    main()
