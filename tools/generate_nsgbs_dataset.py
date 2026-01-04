#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate NS-GBS dataset by running a scenario and recording per-step candidates.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


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
    spec = importlib.util.spec_from_file_location("nsgbs_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def load_base_config():
    config_path = SCRIPT_DIR / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("base_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def to_jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Generate NS-GBS dataset")
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), default="toronto_single")
    parser.add_argument("--config", help="Path to a custom config file")
    parser.add_argument("--out", default="output/datasets/nsgbs_dataset.npz")
    parser.add_argument("--T", type=int, default=200, help="Override TTI count")
    parser.add_argument("--N_UE", type=int, default=50, help="Override UE count")
    parser.add_argument("--stride", type=int, default=1, help="Sample every N steps")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--topB", type=int, default=4)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    base = load_base_config()
    cfg = base.copy()

    if args.config:
        custom = load_config_from_file(Path(args.config))
        cfg.update(custom)
    else:
        cfg_path = SCRIPT_DIR / SCENARIOS[args.scenario]
        cfg.update(load_config_from_file(cfg_path))

    if args.T is not None:
        cfg["T"] = int(args.T)
    if args.N_UE is not None:
        cfg["N_UE"] = int(args.N_UE)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    cfg["show_progress"] = not args.no_progress
    cfg["scheduler_kind"] = "heuristic"
    cfg["nsgbs_collect_dataset"] = True
    cfg["nsgbs_collect_stride"] = int(max(1, args.stride))
    cfg["nsgbs_collect_max_samples"] = args.max_samples
    cfg["nsgbs_topB"] = int(max(1, args.topB))
    cfg["nsgbs_window"] = int(max(1, args.window))

    dataset = []
    cfg["nsgbs_dataset_out"] = dataset

    from main import run_once

    run_once(cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features = np.array([s["features"] for s in dataset], dtype=object)
    labels = np.array([s["label"] for s in dataset], dtype=np.int64)
    actions = np.array([s["actions"] for s in dataset], dtype=object)
    deltas = np.array([s["deltas"] for s in dataset], dtype=object)
    t_idx = np.array([s["t_idx"] for s in dataset], dtype=np.int32)
    step = np.array([s["step"] for s in dataset], dtype=np.int32)

    np.savez_compressed(
        out_path,
        features=features,
        labels=labels,
        actions=actions,
        deltas=deltas,
        t_idx=t_idx,
        step=step,
    )

    meta = {
        "scenario": args.scenario if not args.config else str(args.config),
        "out": str(out_path),
        "num_samples": int(len(dataset)),
        "feature_dim": int(features[0].shape[1]) if len(dataset) > 0 else 0,
        "config": to_jsonable(cfg),
    }
    with open(str(out_path) + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print(f"[NS-GBS] saved dataset: {out_path} (samples={len(dataset)})")


if __name__ == "__main__":
    main()
