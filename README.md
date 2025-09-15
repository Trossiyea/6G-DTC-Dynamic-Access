NR-NTN Radio Map–aware Downlink Simulator (Direct-to-Satellite)

Overview
- 本仓库已重构为 NR/NTN 下行（DL）仿真：基于 Radio Map 的细粒度 PRB 调度与功率分配（等功率/水位法），支持 MCS/TBS、CSI/CQI 周期/时延、HARQ（可选），并考虑卫星移动性（轨道引起的 FSPL/多普勒/时延）。
- 输出平均频谱效率、相对增益，以及多普勒/时延等时序指标与图表。

What's New (P0)
- 38.214 MCS/TBS + BLER/OLLA + HARQ（默认关闭，向后兼容）
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
    `pdsch_dmrs_sym_per_slot/dmrs_re_per_sym_per_prb/oh_prb`、`mcs_3gpp_table_path`、`mcs_table_kind`

3GPP MCS 表（外部 JSON）
- 位置：`docs/mcs_tables_38_214.json`（你已提供，已验证可用）
- 模板：`docs/mcs_tables_38_214_template.json`（字段说明与示例）
- 使用：
  - 配置 `mcs_3gpp_table_path: "docs/mcs_tables_38_214.json"`
  - 配置 `mcs_table_kind: "3gpp_table_1" | "3gpp_table_2" | "3gpp_table_3"`
  - JSON 条目：`{"idx": <int>, "Qm": <2|4|6|8>, "R_x1024": <int或保留>}`（保留项会自动跳过）

主要能力（按阶段）
- Stage‑1：轨道/波束移动、CSI 时延、Radio Map 感知调度
- Stage‑2：HARQ ACK 延后门控、CQI 周期化（hold‑last）
- Stage‑3：可选（默认关闭），保留轻量 OrbitModel 以模拟卫星移动性

环境要求
- 推荐在已有 conda 环境 ns3env 中运行（已安装 numpy、scipy、matplotlib、skyfield、sgp4 等）。
- 若遇到 Matplotlib 缓存权限告警，可先设：`export MPLCONFIGDIR=$(mktemp -d)`。

快速开始
- 快速开始：`python code/main.py`
  - 环境变量覆盖：`MEAS_T=120 MEAS_N_UE=50`。

长时长测量 + 图表
 运行主程序会生成：增益分布、Radio Map 频域中值、WB vs 最优 PRB 能力对比图。

（最简仓库已移除 TLE 探针脚本与外部轨道依赖）

关键配置（config.py）
- 轨道与几何：`enable_orbit_dynamics`（是否考虑卫星移动性）、`sat_ground_speed_kms`、`sat_heading_deg`、`sat_altitude_km`
- HARQ/CSI/频率/定时
  - HARQ（最小门控）：`enable_harq_deferral`、`harq_max_procs`、`harq_ack_delay_ttis`
  - HARQ（全流程）：`enable_harq_full`、`harq_target_bler`、`harq_max_retx`、
    `olla_step_up_db/olla_step_down_db/olla_init_offset_db`、`harq_retx_priority_bonus`
  - CQI 周期：`enable_cqi_periodicity`、`cqi_period_ttis`、`cqi_offset_ttis`
  - DL 功率分配：等功率/水位法（通过 `dl_power_model`/`P_tot_dbm` 配置）

MCS/TBS/BLER 配置
- 3GPP 表：`mcs_3gpp_table_path` + `mcs_table_kind='3gpp_table_1/2/3'`
- 近似表：`mcs_table_kind='table_1_64qam'|'table_2_256qam'|'table_3_low_se'`
- TBS/RE：`pdsch_dmrs_sym_per_slot`、`dmrs_re_per_sym_per_prb`、`oh_prb`
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
- 开启 `enable_orbit_dynamics` 以考虑卫星移动性；调整 `sat_ground_speed_kms`/`sat_heading_deg`/`sat_altitude_km` 以覆盖目标轨道。

目录结构
- `code/`：核心库与仿真入口（`main.py`、`orbit.py`、`harq.py`、`link_adapt.py`、`ntn_*` 等）
- `tools/`：已移除以保持最简
- `radio_map/`：示例干扰数据（MAT）
- `docs/`：背景 TODO 与规范参考

注意事项
- 外部 Skyfield/SGP4 已移除；使用内置 `OrbitModel` 模拟移动性。
- Matplotlib 缓存目录建议设置 `MPLCONFIGDIR` 到可写目录，以避免警告并加速导入。

Roadmap（可选）
- 更完整的 ECEF 速度/参考框架统一；A5/事件触发与 L1/L3 滤波；全 ECEF/本地投影映射；真实波束图与旁瓣模型；更精细的 BLER/OLLA/DL 功率分配建模。

附录：常见问题
- Matplotlib 缓存写权限：`export MPLCONFIGDIR=$(mktemp -d)`
- 3GPP 表加载失败：检查 `mcs_3gpp_table_path` 路径、JSON 格式与是否含有保留条目（保留会被跳过）
- 想要固定 BLER 目标：调整 `bler_margin_db`/`bler_slope_db` 与 `sched_eesm_beta_db`，再用 OLLA 步长微调
