#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化脚本：分别绘制上海与多伦多场景（两张图）。
每张图含两个子图：
- 左：单星场景（Single）下 Baseline vs RadioMap 的 SE 对比（20 MHz, 30 kHz SCS）
- 右：多星场景（Constellation）下 Baseline vs RadioMap 的 SE 对比（20 MHz, 30 kHz SCS）

说明：
- 使用现有电磁地图（dBm）与默认 SCS=30 kHz（PRB BW=360 kHz）。
- Z 由地图决定；系统带宽由 PRB_BW×Z 计算（典型 ≈20 MHz）。
- 多星需要 skyfield/sgp4 依赖；若不可用，将在图中标注“不可用”。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import importlib.util
from typing import Dict, Tuple
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR / "code"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# 专利图风格：Times New Roman
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})


def _load_main_module():
    main_path = SCRIPT_DIR / "code" / "main.py"
    spec = importlib.util.spec_from_file_location("main_module", main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['main_module'] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


def _load_base_config() -> Dict:
    cfg_path = SCRIPT_DIR / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("base_config", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return dict(module.CONFIG)


def _run_once_for_city(city: str, single: bool, base_cfg: Dict, main_module) -> Tuple[float, float]:
    cfg = dict(base_cfg)
    # 20 MHz / 30 kHz SCS
    cfg["scs_khz"] = 30.0
    # 让 Z 由地图决定
    cfg["Z"] = None
    # 关闭进度/JSON
    cfg["show_progress"] = False
    cfg["write_json_report"] = False
    cfg["enable_constellation"] = (not single)

    # 城市参数
    if city.lower().startswith("toronto"):
        cfg["radio_map_mat_path"] = str(SCRIPT_DIR / "radio_map" / "Toronto" / "RadioMap" / "RM_toronto125_dBm.mat")
        cfg["radio_map_mat_var"] = "XdB_recon_tensor"
        cfg["radio_map_units"] = "dBm"
        cfg["ref_lat_deg"] = 43.65108
        cfg["ref_lon_deg"] = -79.34702
        # 与测试配置保持一致（Starlink 单星可见窗口）
        cfg["carrier_freq_GHz"] = 2.19
        cfg["tle_name"] = "STARLINK-11090 [DTC]"
        cfg["tle_lines"] = [
            "1 59422C 24065B   25266.77548611  .00029064  00000+0  23954-3 0  2662",
            "2 59422  53.1572 196.6800 0001379  82.5466  65.8632 15.69667376    15",
        ]
        cfg["orbit_start_datetime"] = "2025-10-07T21:38:31.574982+00:00"
    elif city.lower().startswith("shanghai"):
        cfg["radio_map_mat_path"] = str(SCRIPT_DIR / "radio_map" / "Shanghai" / "RadioMap" / "RM_shanghai125_dBm.mat")
        cfg["radio_map_mat_var"] = "XdB_recon_tensor"
        cfg["radio_map_units"] = "dBm"
        cfg["ref_lat_deg"] = 31.2304
        cfg["ref_lon_deg"] = 121.4737
        # 与测试配置保持一致（Satnet 单星可见窗口）
        cfg["carrier_freq_GHz"] = 1.90
        cfg["tle_name"] = "SATNET-590-00004 [DTC]"
        cfg["tle_lines"] = [
            "1     3U 00000A   25274.57027601 .00000000  00000-0 0 000           3",
            "2     3  85.0000   0.0000 0000000   0.0000  24.0000 14.92546055    05",
        ]
        cfg["orbit_start_datetime"] = "2025-10-08T02:19:16.314794+00:00"
    else:
        raise ValueError(f"未知城市: {city}")

    # 运行
    if cfg.get("enable_constellation", False):
        out = main_module.run_constellation(cfg)
    else:
        out = main_module.run_once(cfg)
    return float(out["avg_se_baseline_default"]), float(out["avg_se_radiomap"])


def _plot_city(city: str, se_single: Tuple[float, float], se_multi: Tuple[float, float], save: bool) -> None:
    base_single, rm_single = se_single
    base_multi, rm_multi = se_multi

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=220)

    # 子图1：Single 场景内 Baseline vs RadioMap
    x = np.arange(2)
    bars1 = ax1.bar(x, [base_single, rm_single], width=0.6,
                    color=["#34495e", "#e67e22"], edgecolor='black', alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["3GPP", "This Work"])
    ax1.set_ylabel("Spectral Efficiency (bits/s/Hz)")
    ax1.set_title("Single Satellite (20 MHz, 30 kHz)", pad=10)
    ax1.grid(True, axis='y', alpha=0.25)
    for b in bars1:
        h = b.get_height()
        if np.isfinite(h):
            ax1.text(b.get_x() + b.get_width()/2., h, f"{h:.3f}", ha='center', va='bottom', fontsize=9)

    # 子图2：Constellation 场景内 Baseline vs RadioMap
    bars2 = ax2.bar(x, [base_multi, rm_multi], width=0.6,
                    color=["#16a085", "#8e44ad"], edgecolor='black', alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["3GPP", "This Work"])
    ax2.set_ylabel("Spectral Efficiency (bits/s/Hz)")
    ax2.set_title("Constellation (20 MHz, 30 kHz)", pad=10)
    ax2.grid(True, axis='y', alpha=0.25)
    for b in bars2:
        h = b.get_height()
        if np.isfinite(h):
            ax2.text(b.get_x() + b.get_width()/2., h, f"{h:.3f}", ha='center', va='bottom', fontsize=9)

    # fig.suptitle(f"{city.title()} – Baseline vs RadioMap", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out = OUTPUT_DIR / f"se_{city.lower()}_single_vs_constellation_20mhz_30khz.png"
    if save:
        fig.savefig(out, bbox_inches='tight')
        print(f"✅ 已保存: {out}")
    plt.close(fig)


def _cache_file_path(fast: bool) -> Path:
    name = "city_se_20mhz_30khz_fast.json" if fast else "city_se_20mhz_30khz.json"
    return RESULTS_DIR / name


def _load_cache(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(path: Path, data: Dict) -> None:
    meta = data.get('meta', {})
    meta['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    data['meta'] = meta
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="上海/多伦多：20MHz×30kHz，单星与多星频谱效率对比（带结果缓存）")
    ap.add_argument('--save', action='store_true', help='保存图片到 output/')
    ap.add_argument('--fast', action='store_true', help='快速模式（减小 T/N_UE）')
    ap.add_argument('--refresh-cache', action='store_true', help='忽略缓存并重新计算')
    args = ap.parse_args()

    main_module = _load_main_module()
    base_cfg = _load_base_config()
    # 统一字体/风格在上面已设定

    # 快速模式：减少计算成本（不改变相对趋势）
    if args.fast:
        base_cfg["N_UE"] = min(60, int(base_cfg.get("N_UE", 100)))
        base_cfg["T"] = min(300, int(base_cfg.get("T", 2000)))
        base_cfg["show_progress"] = False

    # 缓存路径
    cpath = _cache_file_path(args.fast)
    cache = {} if args.refresh_cache else _load_cache(cpath)
    if cache:
        print(f"🗂️  使用缓存: {cpath}")

    def get_from_cache(city: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        d = cache.get('data', {}).get(city, {}) if isinstance(cache, dict) else {}
        def to_tuple(obj):
            if not isinstance(obj, dict):
                return (np.nan, np.nan)
            b = obj.get('baseline_se', None)
            r = obj.get('radiomap_se', None)
            b = float(b) if b is not None else np.nan
            r = float(r) if r is not None else np.nan
            return (b, r)
        return to_tuple(d.get('single')), to_tuple(d.get('constellation'))

    def put_to_cache(city: str, single: Tuple[float, float], multi: Tuple[float, float]) -> None:
        if 'data' not in cache or not isinstance(cache.get('data'), dict):
            cache['data'] = {}
        if city not in cache['data']:
            cache['data'][city] = {}
        cache['data'][city]['single'] = {
            'baseline_se': None if not np.isfinite(single[0]) else float(single[0]),
            'radiomap_se': None if not np.isfinite(single[1]) else float(single[1]),
        }
        cache['data'][city]['constellation'] = {
            'baseline_se': None if not np.isfinite(multi[0]) else float(multi[0]),
            'radiomap_se': None if not np.isfinite(multi[1]) else float(multi[1]),
        }

    # 多星依赖可能缺失；优先读取缓存，不足部分再跑
    def ensure_pair(city: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # 若已有缓存（且未要求刷新），直接返回缓存值以避免长时间运行
        if cache and not args.refresh_cache:
            s_cached, m_cached = get_from_cache(city)
            sc_any = np.isfinite(np.asarray(s_cached, dtype=float)).any()
            mc_any = np.isfinite(np.asarray(m_cached, dtype=float)).any()
            if sc_any or mc_any:
                return s_cached, m_cached
        # 否则计算并写入缓存
        s_val = _run_once_for_city(city, True, base_cfg, main_module)
        try:
            m_val = _run_once_for_city(city, False, base_cfg, main_module)
        except Exception as e:
            print(f"[WARN] 多星场景不可用（{city}）：{e}")
            m_val = (np.nan, np.nan)
        put_to_cache(city, s_val, m_val)
        return s_val, m_val

    sing_tor, mult_tor = ensure_pair("Toronto")
    sing_sha, mult_sha = ensure_pair("Shanghai")

    # 写入缓存
    cache.setdefault('meta', {})
    cache['meta'].update({
        'fast': bool(args.fast),
        'seed': base_cfg.get('seed'),
        'scs_khz': 30.0,
        'N_UE': base_cfg.get('N_UE'),
        'T': base_cfg.get('T'),
        'script': 'visualize_city_se_20mhz_30khz.py',
    })
    _save_cache(cpath, cache)

    # 绘图
    _plot_city("Toronto", sing_tor, mult_tor, save=args.save)
    _plot_city("Shanghai", sing_sha, mult_sha, save=args.save)

    print("✅ 可视化完成！图片已输出至 output/ 目录（如启用 --save）")


if __name__ == '__main__':
    sys.exit(main())
