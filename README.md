NR-NTN Radio Map–aware Uplink Simulator (Direct-to-Satellite)

Overview
- NR/NTN 上行仿真，支持 Radio Map 感知的 PRB 调度与 3GPP 关键流程近似：功控、CSI/CQI 周期/时延、HARQ ACK 延后、NTN 频率/定时预补偿、A3 类测量/切换，以及 Skyfield/SGP4 轨道与多波束（量化/跳波束）几何。
- 输出平均频谱效率、改进比，以及多普勒/时延等时序指标与图表。

What's New (P0)
- 38.214 MCS/TBS + BLER/OLLA + HARQ 全流程（保持默认关闭，向后兼容）
  - 新增 `code/link_adapt.py`：
    - 近似 Table‑1/2/3 MCS 候选集、38.214 风格 TBS 计算（小/大 TBS 分段、CB 分段、8bit 对齐）
    - BLER（AWGN logistic 可配置 slope/margin）、OLLA（±步长、目标 BLER）
    - EESM 与多次传输软合并（effective SINR 合并）；RE/PRB 计算（按 DMRS/OH/CP）
    - 3GPP MCS 表加载：`register_mcs_tables_from_file(path)`，支持 `mcs_table_kind='3gpp_table_1/2/3'`
  - 新增 `HarqManagerFull`（ACK/NACK、RV 循环、软合并、OLLA 更新、TBS 记账、统计输出）
  - 调度器块路径集成（`pf_schedule_radiomap_blocks`）：
    - 每 TTI 消费 ACK 比特计入吞吐；重传 UE 优先级；新传/重传统一交由 HARQ 管理
  - 配置扩展（默认值不变）：`enable_harq_full`、`harq_target_bler`、`harq_max_retx`、
    `olla_step_up_db/olla_step_down_db/olla_init_offset_db`、`bler_slope_db/bler_margin_db`、
    `pusch_dmrs_sym_per_slot/dmrs_re_per_sym_per_prb/oh_prb`、`mcs_3gpp_table_path`、`mcs_table_kind`

3GPP MCS 表（外部 JSON）
- 位置：`docs/mcs_tables_38_214.json`（你已提供，已验证可用）
- 模板：`docs/mcs_tables_38_214_template.json`（字段说明与示例）
- 使用：
  - 配置 `mcs_3gpp_table_path: "docs/mcs_tables_38_214.json"`
  - 配置 `mcs_table_kind: "3gpp_table_1" | "3gpp_table_2" | "3gpp_table_3"`
  - JSON 条目：`{"idx": <int>, "Qm": <2|4|6|8>, "R_x1024": <int或保留>}`（保留项会自动跳过）

主要能力（按阶段）
- Stage‑1：轨道/波束移动、Doppler/TA 预补偿、CSI 时延、功控、RM 感知调度
- Stage‑2：HARQ ACK 延后门控、CQI 周期化（hold‑last）、PTRS CFO 跟踪预算推导
- Stage‑3：
  - Skyfield 轨道：基于 TLE 的子星点与 GCRS 速度，径向多普勒 vr=vsat·r̂
  - 多波束骨架：波束指向量化/跳波束
  - A3 类切换：SSB/CSI‑RS 测量周期 + 滞后 + 测量 TTT，HO 完成后更新“服务波束中心”，几何/增益随之变化

环境要求
- 推荐在已有 conda 环境 ns3env 中运行（已安装 numpy、scipy、matplotlib、skyfield、sgp4 等）。
- 若遇到 Matplotlib 缓存权限告警，可先设：`export MPLCONFIGDIR=$(mktemp -d)`。

快速开始
- 单次数值测量（使用 STARLINK DTC TLE，自动以“现在”为 TTI=0 并自动居中）：
  - `conda run -n ns3env bash -lc 'export MPLCONFIGDIR=$(mktemp -d); python tools/measure_starlink_dtc.py'`
  - 环境变量覆盖：`MEAS_T=120 MEAS_N_UE=50`。

长时长测量 + 图表
- 运行：
  - `conda run -n ns3env bash -lc 'export MPLCONFIGDIR=$(mktemp -d); MEAS_T=600 MEAS_N_UE=100 python tools/measure_starlink_dtc_plots.py'`
  - 输出目录：`output/skyfield_measure_<timestamp>/`
  - 生成文件：
    - `summary.json`（T、N_UE、avg_se、改进比、HO/RACH 事件数、中心经纬）
    - `ho_counts.png`、`rach_counts.png`（事件直方图）
    - `tau_percentiles.png`、`doppler_percentiles.png`（5/50/95 百分位）
    - `radio_map_median.png`（干扰图频域中值）
    - `beam_center_track.png`（前 1000 TTI 波束中心轨迹，Skyfield 可用时）

TLE 探针与中心建议
- `conda run -n ns3env bash -lc 'python tools/tle_probe.py "<tle1>" "<tle2>" 1'`
- 输出包含 6 个样本（lat/lon/alt）与 `suggest_grid_center`；可将中心写入 `config.py` 的 `grid_center_lat_deg/lon_deg`，或继续使用 `auto_grid_center_from_tle=True` 自动居中。

关键配置（config.py）
- 轨道与几何
  - `enable_skyfield_orbit: True`（优先）/`enable_sgp4_orbit: True`/纯几何 `enable_orbit_dynamics: True`
  - `tle_ref_use_now: True`、`auto_grid_center_from_tle: True`、`tle_line1/2`
  - `offaxis_ecef: True`（用 ECEF 角度计算 off-axis）
  - 多波束：`enable_multi_beam`、`n_beams_x/y`、`beam_grid_spacing_px`、`beam_hop_period_ttis`
- A3 测量/触发（替代纯 off-axis TTT）
  - `enable_a3_ho: True`、`ssb_period_ttis`、`a3_hysteresis_db`、`a3_ttt_meas`、`a3_neighbor_k`
- HARQ/CSI/频率/定时
  - HARQ（最小门控）：`enable_harq_deferral`、`harq_max_procs`、`harq_ack_delay_ttis`
  - HARQ（全流程）：`enable_harq_full`、`harq_target_bler`、`harq_max_retx`、
    `olla_step_up_db/olla_step_down_db/olla_init_offset_db`、`harq_retx_priority_bonus`
  - CQI 周期：`enable_cqi_periodicity`、`cqi_period_ttis`、`cqi_offset_ttis`
  - PTRS CFO 预算：`ptrs_cfo_track_hz` 显式，或 `ptrs_symbols_per_slot` + `ptrs_track_k_factor` 推导
  - NTN 预补偿/TA：`enable_ntn_freq_precomp`、`enable_ta_model` 相关参数

MCS/TBS/BLER 配置
- 3GPP 表：`mcs_3gpp_table_path` + `mcs_table_kind='3gpp_table_1/2/3'`
- 近似表：`mcs_table_kind='table_1_64qam'|'table_2_256qam'|'table_3_low_se'`
- TBS/RE：`pusch_dmrs_sym_per_slot`、`dmrs_re_per_sym_per_prb`、`oh_prb`
- BLER/OLLA：`bler_slope_db`、`bler_margin_db`、`harq_target_bler`、`olla_step_up_db/olla_step_down_db`
- EESM：`sched_eesm_beta_db`（块调度）、`baseline_eesm_beta_db`（子带基线）

示例：快速验证（Python 调用）
```python
from copy import deepcopy
from config import CONFIG
from main import run_once

cfg = deepcopy(CONFIG)
cfg.update({
  'enable_time_varying': True,
  'enable_orbit_dynamics': False,
  'mcs_3gpp_table_path': 'docs/mcs_tables_38_214.json',
  'mcs_table_kind': '3gpp_table_2',
  'enable_harq_full': True,
  'harq_ack_delay_ttis': 4,
  'harq_max_procs': 8,
  'harq_target_bler': 0.1,
  'bler_margin_db': 2.0,
  'bler_slope_db': 1.5,
  'sched_eesm_beta_db': 1.5,
  'baseline_eesm_beta_db': 1.5,
  # 频谱异质性/时变
  'K_interferers': 12,
  'rm_flicker_db_std': 2.0,
  # CSI 时延对比
  'baseline_csi_delay_ttis': 8,
  'rm_csi_delay_ttis': 0,
  # 调度：两者均用块调度
  'baseline_block_mode': True,
  'sched_block_mode': True,
  'sched_require_contiguous': True,
  'N_UE': 25,
  'T': 80,
  'seed': 780,
  'save_plots': False,
  'show_plots': False,
})
res = run_once(cfg)
print('Baseline-Default avg SE:', res['avg_se_baseline_default'])
print('RadioMap         avg SE:', res['avg_se_radiomap'])
print('Gain vs Default (%):', res['improvement_vs_default_pct'])
```

参考结果（一次运行）
- Baseline-Default avg SE ≈ 2.00
- RadioMap         avg SE ≈ 2.39
- Gain vs Default  ≈ +19.4%

Radio Map 与可重复性
- `radio_map_mat_path` 可指定外部 3D 干扰图（dBm/mW），默认自生成。
- 固定随机种子 `seed` 以便重现；UE 随机位置 `N_UE` 与地图大小 `X/Y/Z` 可在配置中调整。

典型运行建议
- LEO+Skyfield 场景：启用 `enable_skyfield_orbit: True`，`tle_ref_use_now: True`，`auto_grid_center_from_tle: True`。
- A3 参数基线：`ssb_period_ttis: 20`，`a3_hysteresis_db: 3.0`，`a3_ttt_meas: 2`。
- 需要降低 HO 频度可增大 `ssb_period_ttis`/`a3_ttt_meas`/`a3_hysteresis_db`，或调整多波束密度/半功率角。

目录结构
- `code/`：核心库与仿真入口（`main.py`、`orbit_skyfield.py`、`ho.py`、`rach.py`、`harq.py`、`ntn_*` 等）
- `tools/`：
  - `measure_starlink_dtc.py`：快速数值测量
  - `measure_starlink_dtc_plots.py`：长测+图表
  - `tle_probe.py`：TLE 子星点探针
- `radio_map/`：示例干扰数据（MAT）
- `docs/`：背景 TODO 与规范参考

注意事项
- 若使用 Skyfield，请确保 ns3env 已安装 `skyfield`；若缺失，会自动回退到简化轨道。
- Matplotlib 缓存目录建议设置 `MPLCONFIGDIR` 到可写目录，以避免警告并加速导入。

Roadmap（可选）
- 更完整的 ECEF 速度/参考框架统一；A5/事件触发与 L1/L3 滤波；全 ECEF/本地投影映射；真实波束图与旁瓣模型；更精细的 BLER/OLLA/闭环功控建模。

附录：常见问题
- Matplotlib 缓存写权限：`export MPLCONFIGDIR=$(mktemp -d)`
- 3GPP 表加载失败：检查 `mcs_3gpp_table_path` 路径、JSON 格式与是否含有保留条目（保留会被跳过）
- 想要固定 BLER 目标：调整 `bler_margin_db`/`bler_slope_db` 与 `sched_eesm_beta_db`，再用 OLLA 步长微调
