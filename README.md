NR-NTN Radio Map–aware Uplink Simulator (Direct-to-Satellite)

Overview
- NR/NTN 上行仿真，支持 Radio Map 感知的 PRB 调度与 3GPP 关键流程近似：功控、CSI/CQI 周期/时延、HARQ ACK 延后、NTN 频率/定时预补偿、A3 类测量/切换，以及 Skyfield/SGP4 轨道与多波束（量化/跳波束）几何。
- 输出平均频谱效率、改进比，以及多普勒/时延等时序指标与图表。

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
  - HARQ：`enable_harq_deferral`、`harq_max_procs`、`harq_ack_delay_ttis`
  - CQI 周期：`enable_cqi_periodicity`、`cqi_period_ttis`、`cqi_offset_ttis`
  - PTRS CFO 预算：`ptrs_cfo_track_hz` 显式，或 `ptrs_symbols_per_slot` + `ptrs_track_k_factor` 推导
  - NTN 预补偿/TA：`enable_ntn_freq_precomp`、`enable_ta_model` 相关参数

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

