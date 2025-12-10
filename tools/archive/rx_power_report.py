#!/usr/bin/env python3
"""
统计每个 UE 的接收功率并可视化。

说明
- 直接复用 code/ 目录下现有的计算流程：几何与波束 -> 热噪声 -> compute_caps。
- 输出：
  1) 每 UE 的接收功率直方图（dBm）
  2) UE 在栅格上的散点图，按接收功率着色（dBm）
  3) CSV：ue_idx,x,y,P_rx_dbm

运行
- 从仓库根目录执行：
    python tools/rx_power_report.py [--show] [--seed 101] [--out output]

依赖
- 使用项目自带的 code/ 下模块。为便于导入，脚本会把 code/ 加入 sys.path。
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


def _add_code_to_syspath() -> None:
    """确保可以从 code/ 目录导入模块。"""
    here = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.abspath(os.path.join(here, "..", "code"))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


def main(argv: list[str] | None = None) -> int:
    _add_code_to_syspath()

    # 延迟导入项目模块（在 sys.path 修改之后）
    from config import CONFIG  # type: ignore
    from main import (
        select_radio_map,
        generate_ue_positions,
        compute_geometry_and_beam,
        resolve_noise_and_prb_bw,
        apply_open_loop_power_control,
        compute_caps,
    )  # type: ignore

    parser = argparse.ArgumentParser(description="统计每个UE接收功率并可视化")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 CONFIG['seed']")
    parser.add_argument("--show", action="store_true", help="显示图像（否则仅保存）")
    parser.add_argument("--out", default=None, help="输出目录（默认使用 CONFIG['plot_dir'] 或 output）")
    args = parser.parse_args(argv)

    cfg = dict(CONFIG)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    # 目录与绘图开关
    out_dir = args.out or cfg.get("plot_dir", "output")
    os.makedirs(out_dir, exist_ok=True)
    show_plots = bool(args.show or cfg.get("show_plots", False))

    # 1) 读取/生成无线电干扰图 + 随机 UE 位置
    R_xyz_dbm, X, Y, Z = select_radio_map(cfg)
    rng = np.random.default_rng(cfg["seed"])
    ue_pos = generate_ue_positions(cfg["N_UE"], X, Y, rng)

    # 2) 几何/波束、噪声与功率
    L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue = compute_geometry_and_beam(cfg, X, Y, ue_pos)
    noise_dbm, _ = resolve_noise_and_prb_bw(cfg)
    P_tx_per_ue_dbm = apply_open_loop_power_control(cfg, L_fs_per_ue, G_rx_per_ue)

    # 3) 计算接收功率（宽带 per‑UE）与每 PRB SINR/容量（本脚本只取 P_rx_dbm 做统计）
    cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb = compute_caps(
        R_xyz_dbm,
        ue_pos,
        P_tx_dbm=P_tx_per_ue_dbm,
        L_fs_db=L_fs_per_ue,
        G_rx_db=G_rx_per_ue,
        shadow_db_std=cfg["shadow_std_db"],
        N0_dbm=noise_dbm,
        rx_nf_db=cfg.get("rx_nf_db", 0.0),
        impl_loss_db=cfg.get("impl_loss_db", 0.0),
        seed=cfg["seed"],
        elevation_deg=elev_deg_per_ue,
        channel_model=cfg.get("channel_model", "3gpp_ntn"),
        channel_params=cfg.get("channel_params"),
        channel_profile=cfg.get("ntn_channel_profile", "s_band_handheld_urban"),
    )

    # 4) 统计与保存 CSV
    ue_idx = np.arange(P_rx_dbm.size, dtype=int)
    x = ue_pos[:, 0]
    y = ue_pos[:, 1]
    stats = {
        "N_UE": int(P_rx_dbm.size),
        "P_rx_dbm_min": float(np.min(P_rx_dbm)),
        "P_rx_dbm_mean": float(np.mean(P_rx_dbm)),
        "P_rx_dbm_median": float(np.median(P_rx_dbm)),
        "P_rx_dbm_max": float(np.max(P_rx_dbm)),
    }
    print("接收功率统计 (dBm): min={:.2f}, p50={:.2f}, mean={:.2f}, max={:.2f} (N_UE={})".format(
        stats["P_rx_dbm_min"], stats["P_rx_dbm_median"], stats["P_rx_dbm_mean"], stats["P_rx_dbm_max"], stats["N_UE"],
    ))

    csv_path = os.path.join(out_dir, "rx_power_per_ue.csv")
    with open(csv_path, "w") as f:
        f.write("ue_idx,x,y,P_rx_dbm\n")
        for i in range(ue_idx.size):
            f.write(f"{int(ue_idx[i])},{int(x[i])},{int(y[i])},{float(P_rx_dbm[i]):.6f}\n")
    print(f"已保存 CSV: {csv_path}")

    # 5) 可视化：直方图
    plt.figure(figsize=(6, 4))
    plt.hist(P_rx_dbm, bins=20, edgecolor='black')
    plt.xlabel("接收功率 P_rx (dBm)")
    plt.ylabel("UE 数量")
    plt.title("每 UE 接收功率分布")
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "rx_power_hist.png")
    plt.savefig(hist_path, dpi=140, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"已保存图像: {hist_path}")

    # 6) 可视化：散点（叠加干扰图的频域中位数背景，便于空间直观对比）
    R_med = np.median(R_xyz_dbm, axis=2)
    plt.figure(figsize=(6, 6))
    plt.imshow(R_med.T, origin='lower', aspect='equal', cmap='viridis')
    sc = plt.scatter(x, y, c=P_rx_dbm, cmap='plasma', edgecolor='white', linewidths=0.5)
    plt.colorbar(sc, label='P_rx (dBm)')
    plt.title("UE 位置与接收功率（背景为干扰中位数 dBm）")
    plt.xlabel("x 索引")
    plt.ylabel("y 索引")
    plt.tight_layout()
    sc_path = os.path.join(out_dir, "rx_power_scatter.png")
    plt.savefig(sc_path, dpi=140, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"已保存图像: {sc_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

