#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专利可视化脚本：多伦多电磁地图 + Starlink 场景，
不同带宽配置 (10/20/30/40 MHz) 与 15 kHz / 30 kHz 子载波间隔下的频谱效率对比。

产出两张图：
- 单星（OrbitModel）
- 多星（ConstellationOrbit）

说明与假设：
- 仅有 20 MHz、30 kHz SCS 的电磁地图（Radio Map）。本脚本对其它带宽/SCS做近似：
  1) 频域轴按比例拉伸/压缩，超过带宽部分按频谱形状平铺（tile）后插值。
  2) PRB 宽度变化按比例缩放干扰功率：P_PRB ∝ SCS_kHz（dBm 上相当于加 10log10(SCS_new/SCS_base)）。
  3) 噪声功率使用 kTB 根据 PRB 带宽自适应（由现有代码根据 SCS 自动计算）。

用法示例：
  python3 visualize_bw_scs_se.py \
    --map radio_map/Toronto/RadioMap/RM_toronto125_dBm.mat \
    --varname XdB_recon_tensor --base-scs-khz 30 \
    --bandwidths 10 20 30 40 --scs 15 30 \
    --save

可选加速参数：
  --fast 将 T 与 N_UE 降低以加快运行；--no-constellation 跳过多星图。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import importlib.util
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# 路径与模块加载
SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR / "code"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    return dict(module.CONFIG)  # copy


def _load_radiomap_from_mat(mat_path: Path, varname: str) -> np.ndarray:
    # 复用 main.py 的 MAT 读取逻辑
    import h5py
    from scipy.io import loadmat
    p = str(mat_path)
    # 尝试 v7.3
    try:
        with h5py.File(p, 'r') as f:
            if varname in f:
                return np.array(f[varname], dtype=float)
            keys = list(f.keys())
            if not keys:
                raise KeyError(f"No dataset in {p}")
            return np.array(f[keys[0]], dtype=float)
    except Exception:
        pass
    # 回退传统 MAT
    data = loadmat(p)
    keys = [k for k in data.keys() if not k.startswith('__')]
    if varname in data:
        return np.array(data[varname], dtype=float)
    if not keys:
        raise KeyError(f"No data in {p}; raw keys={list(data.keys())}")
    return np.array(data[keys[0]], dtype=float)


def _resample_tile_scale_radiomap(
    R_dbm: np.ndarray,
    base_scs_khz: float,
    target_bw_mhz: float,
    target_scs_khz: float,
) -> np.ndarray:
    """沿频率轴(Z)做尺寸变换并按PRB带宽缩放功率。

    - 频域形状：以 base Z 为模板，若目标带宽更宽则按整段平铺后插值到目标 Z。
    - PRB带宽缩放：功率线性域乘以 (target_scs/base_scs)。
    """
    assert R_dbm.ndim == 3, f"RadioMap must be 3D, got {R_dbm.shape}"
    X, Y, Zb = R_dbm.shape

    # 基础/目标 PRB 宽度和 PRB 数
    prb_bw_base_hz = 12.0 * float(base_scs_khz) * 1e3
    prb_bw_tgt_hz = 12.0 * float(target_scs_khz) * 1e3
    bw_base_hz = prb_bw_base_hz * Zb
    Z_target = int(np.floor((float(target_bw_mhz) * 1e6) / prb_bw_tgt_hz))
    Z_target = max(1, Z_target)

    # 频域轴处理：
    # r = 目标总带宽 / 基础总带宽
    r = (float(target_bw_mhz) * 1e6) / max(1.0, bw_base_hz)
    # 将基础在 [0,1) 上，平铺到 [0,r) 轴
    ext_mult = int(np.ceil(max(1.0, r)))

    # 转线性功率
    R_mw = 10.0 ** (np.asarray(R_dbm, dtype=float) / 10.0)
    R_ext = np.tile(R_mw, (1, 1, ext_mult))  # [X,Y,Zb*ext_mult]

    # 轴坐标
    fb = np.linspace(0.0, 1.0, Zb, endpoint=False)
    f_ext = np.linspace(0.0, r, Zb * ext_mult, endpoint=False)
    f_tgt = np.linspace(0.0, r, Z_target, endpoint=False)

    # 插值（逐 XY，为稳妥与内存友好）
    R_tgt = np.empty((X, Y, Z_target), dtype=float)
    for xi in range(X):
        # 向量化 Y 维：对每个 y 一次性插值
        # 先 reshape 为 [Y, Z_ext]
        mat = R_ext[xi, :, :]
        # 对每行进行一维插值
        # np.interp 仅支持 1D；此处循环 Y（典型 Y 不大）
        for yi in range(Y):
            R_tgt[xi, yi, :] = np.interp(f_tgt, f_ext, mat[yi, :])

    # PRB 宽度缩放：
    scale = prb_bw_tgt_hz / prb_bw_base_hz
    R_tgt *= scale

    R_tgt_dbm = 10.0 * np.log10(np.maximum(R_tgt, 1e-30))
    return R_tgt_dbm


def _run_se_for_case(
    main_module,
    base_config: Dict,
    R_dbm_base: np.ndarray,
    base_scs_khz: float,
    bw_mhz: float,
    scs_khz: float,
    use_constellation: bool,
    fast: bool,
) -> Tuple[float, float]:
    """返回 (baseline_se, radiomap_se)。"""
    # 准备 Radio Map（按目标带宽/SCS 生成）
    R_tgt_dbm = _resample_tile_scale_radiomap(R_dbm_base, base_scs_khz, bw_mhz, scs_khz)
    X, Y, Zt = R_tgt_dbm.shape

    # 复制配置并覆盖关键参数
    cfg = dict(base_config)
    cfg["Z"] = int(Zt)
    cfg["scs_khz"] = float(scs_khz)
    cfg["enable_constellation"] = bool(use_constellation)
    # 关闭进度条与 JSON 报告
    cfg["show_progress"] = False
    cfg["write_json_report"] = False
    # 加速参数
    if fast:
        cfg["N_UE"] = min(60, int(cfg.get("N_UE", 100)))
        cfg["T"] = min(300, int(cfg.get("T", 2000)))

    # Monkey-patch: 用我们构造的 R 替代 select_radio_map 返回
    orig_fn = main_module.select_radio_map

    def _patched_select_radio_map(config: Dict):
        return R_tgt_dbm, X, Y, Zt

    main_module.select_radio_map = _patched_select_radio_map  # type: ignore

    try:
        if use_constellation:
            out = main_module.run_constellation(cfg)
        else:
            out = main_module.run_once(cfg)
    finally:
        # 恢复
        main_module.select_radio_map = orig_fn  # type: ignore

    base_se = float(out["avg_se_baseline_default"])
    rm_se = float(out["avg_se_radiomap"])
    return base_se, rm_se


def plot_se_matrix(
    ax_l, ax_r,
    bw_list: List[float],
    scs_list: List[float],
    se_baseline: Dict[Tuple[float, float], float],
    se_radiomap: Dict[Tuple[float, float], float],
    title_prefix: str,
):
    # 左：Baseline；右：RadioMap。X 轴为带宽组，每组两根柱（15k/30k）。
    labels = [f"{int(bw)} MHz" for bw in bw_list]
    x = np.arange(len(bw_list))
    width = 0.35

    def _vals(se_map):
        v15 = [se_map.get((bw, 15.0), np.nan) for bw in bw_list]
        v30 = [se_map.get((bw, 30.0), np.nan) for bw in bw_list]
        return v15, v30

    b15, b30 = _vals(se_baseline)
    r15, r30 = _vals(se_radiomap)

    bars1 = ax_l.bar(x - width/2, b15, width, label='15 kHz', color='#2980b9', edgecolor='black', alpha=0.85)
    bars2 = ax_l.bar(x + width/2, b30, width, label='30 kHz', color='#27ae60', edgecolor='black', alpha=0.85)
    ax_l.set_title(f"{title_prefix} – Baseline", pad=12)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels)
    ax_l.set_ylabel('Spectral Efficiency (bits/s/Hz)')
    ax_l.grid(True, axis='y', alpha=0.25)
    ax_l.legend()
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if np.isfinite(h):
            ax_l.text(bar.get_x() + bar.get_width()/2., h, f"{h:.3f}", ha='center', va='bottom', fontsize=9)

    bars3 = ax_r.bar(x - width/2, r15, width, label='15 kHz', color='#c0392b', edgecolor='black', alpha=0.85)
    bars4 = ax_r.bar(x + width/2, r30, width, label='30 kHz', color='#8e44ad', edgecolor='black', alpha=0.85)
    ax_r.set_title(f"{title_prefix} – RadioMap", pad=12)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels)
    ax_r.set_ylabel('Spectral Efficiency (bits/s/Hz)')
    ax_r.grid(True, axis='y', alpha=0.25)
    ax_r.legend()
    for bar in list(bars3) + list(bars4):
        h = bar.get_height()
        if np.isfinite(h):
            ax_r.text(bar.get_x() + bar.get_width()/2., h, f"{h:.3f}", ha='center', va='bottom', fontsize=9)


def main():
    ap = argparse.ArgumentParser(description="带宽×SCS 频谱效率对比可视化（多伦多 + Starlink）")
    ap.add_argument('--map', type=str, required=True, help='Radio Map .mat 路径')
    ap.add_argument('--varname', type=str, default='XdB_recon_tensor', help='MAT 变量名')
    ap.add_argument('--base-scs-khz', type=float, default=30.0, help='电磁地图对应的基础 SCS (kHz)')
    ap.add_argument('--bandwidths', type=float, nargs='+', default=[10, 20, 30, 40], help='带宽(MHz)列表')
    ap.add_argument('--scs', type=float, nargs='+', default=[15, 30], help='SCS(kHz) 列表')
    ap.add_argument('--fast', action='store_true', help='快速模式（减小 T/N_UE）')
    ap.add_argument('--no-constellation', action='store_true', help='仅生成单星图')
    ap.add_argument('--save', action='store_true', help='保存图片到 output/')
    args = ap.parse_args()

    # 加载模块与基础配置
    main_module = _load_main_module()
    base_config = _load_base_config()
    base_config["radio_map_mat_path"] = str(args.map)
    base_config["radio_map_mat_var"] = str(args.varname)
    base_config["radio_map_units"] = "dBm"
    base_config["show_progress"] = False
    # 为对比统一设置 HARQ/调度等（使用配置默认）

    # 载入基础 Radio Map（20 MHz, 30 kHz 假设）
    R_dbm_base = _load_radiomap_from_mat(Path(args.map), args.varname)
    if R_dbm_base.ndim != 3:
        raise ValueError(f"Radio Map 必须为3D张量，得到 {R_dbm_base.shape}")

    bw_list = list(dict.fromkeys([float(b) for b in args.bandwidths]))
    scs_list = list(dict.fromkeys([float(s) for s in args.scs]))

    # 单星
    se_base_single: Dict[Tuple[float, float], float] = {}
    se_rm_single: Dict[Tuple[float, float], float] = {}

    for bw in bw_list:
        for scs in scs_list:
            try:
                bse, rse = _run_se_for_case(
                    main_module, base_config, R_dbm_base, args.base_scs_khz, bw, scs,
                    use_constellation=False, fast=args.fast,
                )
                se_base_single[(bw, scs)] = bse
                se_rm_single[(bw, scs)] = rse
            except Exception as e:
                print(f"[WARN] 单星运行失败: BW={bw}MHz, SCS={scs}kHz: {e}")
                se_base_single[(bw, scs)] = np.nan
                se_rm_single[(bw, scs)] = np.nan

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
    plot_se_matrix(ax1, ax2, bw_list, scs_list, se_base_single, se_rm_single, title_prefix="Single-Satellite")
    fig1.suptitle("Toronto Radio Map + Starlink – Single Satellite", y=0.98)
    fig1.tight_layout(rect=[0, 0.02, 1, 0.95])
    out1 = OUTPUT_DIR / "toronto_starlink_single_se_bw_scs.png"
    if args.save:
        fig1.savefig(out1, bbox_inches='tight')
        print(f"✅ 已保存: {out1}")

    # 多星（如未禁用）
    if not args.no_constellation:
        se_base_multi: Dict[Tuple[float, float], float] = {}
        se_rm_multi: Dict[Tuple[float, float], float] = {}
        for bw in bw_list:
            for scs in scs_list:
                try:
                    bse, rse = _run_se_for_case(
                        main_module, base_config, R_dbm_base, args.base_scs_khz, bw, scs,
                        use_constellation=True, fast=args.fast,
                    )
                    se_base_multi[(bw, scs)] = bse
                    se_rm_multi[(bw, scs)] = rse
                except Exception as e:
                    print(f"[WARN] 多星运行失败: BW={bw}MHz, SCS={scs}kHz: {e}")
                    se_base_multi[(bw, scs)] = np.nan
                    se_rm_multi[(bw, scs)] = np.nan

        fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
        plot_se_matrix(bx1, bx2, bw_list, scs_list, se_base_multi, se_rm_multi, title_prefix="Constellation")
        fig2.suptitle("Toronto Radio Map + Starlink – Constellation", y=0.98)
        fig2.tight_layout(rect=[0, 0.02, 1, 0.95])
        out2 = OUTPUT_DIR / "toronto_starlink_constellation_se_bw_scs.png"
        if args.save:
            fig2.savefig(out2, bbox_inches='tight')
            print(f"✅ 已保存: {out2}")

    plt.show()


if __name__ == '__main__':
    sys.exit(main())
