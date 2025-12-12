NR-NTN Radio Map–Aware Downlink Simulator
=========================================

This repository hosts a self-contained simulator for the NR non-terrestrial
network (NTN) downlink with a focus on direct-to-satellite access. It models a
radio-map-aware scheduler, realistic orbit and beam geometry, per-PRB link
adaptation, and (optionally) HARQ with OLLA. The code base has been trimmed to
downlink-only mechanics so it can be used as a light-weight analysis harness or
as a reference implementation for radio-map-driven NTN research.


Key Capabilities
----------------
- Radio-map aware proportional fair scheduling with contiguous PRB blocks,
  optional water-filling, and time-varying interference maps.
- 3GPP-aligned link adaptation (TS 38.214). MCS tables load from
  `docs/mcs_tables_38_214.json` by default; optional external BLER curves.
- Orbit & beam dynamics via Skyfield/TLE. The simulator tracks slant range,
  Doppler, and propagation delay per UE。单星与星座两种场景均支持。
- HARQ（默认启用 Full 模式）包含进程管理、EESM 软合并、OLLA 及延迟 ACK。
- 清晰的配置接口：CSI 延迟与周期、Radio Map 估计误差/模糊、Doppler 残差，及调度/功率分配参数。


Prerequisites
-------------
- Python 3.9+ with NumPy, SciPy, Matplotlib, and Skyfield/sgp4（用于 TLE 轨道）。
  建议使用 Conda 环境（例如 `ns3env`）。
- To avoid Matplotlib cache permission warnings on shared machines, set
  `export MPLCONFIGDIR=$(mktemp -d)` before running.


Quick Start
-----------

### 方式1: 使用 Make 命令（推荐）
```bash
# 查看所有可用命令
make help

# 列出所有测试场景
make list

# 运行单个测试场景
make test-toronto-single
make test-shanghai-constellation

# 运行所有场景
make test-all
```

### 方式2: 使用 Python 测试脚本
```bash
# 运行单个场景
python run_test.py --scenario toronto_single
python run_test.py -s shanghai_single

# 运行所有场景
python run_test.py --all

# 列出所有可用场景
python run_test.py --list
```

### 方式3: 分辨率对比测试
```bash
# 对比不同 Radio Map 分辨率的影响（125m vs 150m）
python run_resolution_comparison.py --city toronto
python run_resolution_comparison.py --all --save-report
```

### 方式4: 直接运行（使用默认配置）
```bash
python code/main.py
```

详细场景配置请参考 `test/` 目录下的配置文件。


Configuration Highlights
------------------------
全部配置在 `code/config.py`。核心分组如下：

- **Radio Map 与业务规模**：`X/Y/Z`, `N_UE`, `radio_map_mat_path`, `radio_map_mat_var`, `radio_map_units`, `seed`。
  - 说明：使用 MAT 或 HDF5 格式的 Radio Map；Z 维度必须与 PRB 数匹配。
- **频谱与噪声**：`scs_khz`（自动推导 PRB 带宽并按 kTB 计算热噪声）、`noise_temp_K`。
- **几何与轨道**：`enable_orbit_dynamics`、`sat_altitude_km`、`carrier_freq_GHz`、
  `beam_center_xy`、`beam_half_bw_deg`、`beam_edge_drop_db`、`cell_size_km`；
  TLE：`tle_lines` 或 `tle_path`、`tle_name`、`orbit_start_datetime`、
  `ref_lat_deg/ref_lon_deg`、`auto_ref_from_tle`、`map_rotation_deg`。
- **链路预算**：`P_tx_dbm`、`G_rx_db`（波束主瓣值）、`rx_nf_db`、`impl_loss_db`。
- **调度与链路自适应**：`use_mcs`、`csi_mcs_table`（默认 `3gpp_table_1`）、`pf_beta`、
  `overhead_eff`、`sched_require_contiguous`、`baseline_csi_delay_ttis`、
  `rm_csi_delay_ttis`、`baseline_sched_eesm_beta_db`、`rm_sched_eesm_beta_db`。
- **功率分配（按路径）**：`baseline_dl_power_model`、`baseline_P_tot_dbm`、
  `baseline_p_min_dbm`、`baseline_p_max_dbm`、`rm_dl_power_model`、
  `rm_P_tot_dbm`、`rm_p_min_dbm`、`rm_p_max_dbm`、`baseline_max_prbs_per_ue`、
  `rm_max_prbs_per_ue`。
- **HARQ/OLLA/BLER**：`enable_harq_full`（默认 True）、`harq_max_procs`、
  `harq_ack_delay_ttis`、`harq_target_bler`、`harq_max_retx`、`olla_*`、`bler_*`。
- **MCS 表**：`mcs_3gpp_table_path`（默认 `docs/mcs_tables_38_214.json`）、
  `mcs_table_kind`（默认 `3gpp_table_2`）、`csi_mcs_table`（默认 `3gpp_table_1`）。
- **星座模式**：`enable_constellation`（默认 False）、`tle_catalog_path`、
  `constellation_max_ground_radius_km`、`min_elev_deg`、`association_metric`、
  `ho_enabled`、`ho_hyst_db`、`ho_ttt_ttis`。

已移除/不再支持的键：`K_interferers`、`prb_bw_hz`、`noise_dbm`（噪声改用 SCS→kTB）、
`enable_geometry`/`L_fs_db`（始终计算几何）、`csi_delay_ttis`（使用 per-path 延迟）、
全局 `enable_cqi_periodicity`（改用 per-path）、通用功率回退 `dl_power_model`、
`P_tot_dbm`、`p_min_dbm`、`p_max_dbm`、`max_prbs_per_ue`。


Example: custom experiment
--------------------------
```python
from copy import deepcopy
from config import CONFIG
from main import run_once

cfg = deepcopy(CONFIG)
cfg.update({
    "enable_time_varying": True,
    "enable_orbit_dynamics": False,
    "mcs_table_kind": "3gpp_table_2",
    "enable_harq_full": True,
    "harq_ack_delay_ttis": 4,
    "harq_max_procs": 8,
    "harq_target_bler": 0.1,
    "bler_margin_db": 2.0,
    "N_UE": 25,
    "T": 80,
    "seed": 780,
    "save_plots": False,
    "show_plots": False,
})
result = run_once(cfg)
print("Baseline default SE:", result["avg_se_baseline_default"])
print("Radio-map SE:", result["avg_se_radiomap"])
print("Gain vs default (%):", result["improvement_vs_default_pct"])
```


Outputs
-------
`run_once` returns a dictionary with the primary metrics (average SE for both
baselines, relative gains, Jain fairness, etc.) plus optional time-series
snapshots when dynamics are enabled. When `CONFIG['write_json_report']` is
true, the summary is serialized to `output/<report_basename>.json`.


Test Scenarios
--------------
项目包含四个预配置的测试场景：

| 场景 | 配置文件 | 地点 | 类型 | 运行命令 |
|------|---------|------|------|----------|
| Toronto 单卫星 | `test/config_toronto_single.py` | Toronto | 单星 | `make test-toronto-single` |
| Toronto 星座 | `test/config_toronto_constellation.py` | Toronto | 星座 | `make test-toronto-constellation` |
| Shanghai 单卫星 | `test/config_shanghai_single.py` | Shanghai | 单星 | `make test-shanghai-single` |
| Shanghai 星座 | `test/config_shanghai_constellation.py` | Shanghai | 星座 | `make test-shanghai-constellation` |

此外还支持分辨率对比测试（125m vs 150m）：
```bash
python run_resolution_comparison.py --city toronto
python run_resolution_comparison.py --city shanghai
python run_resolution_comparison.py --all --save-report
```

详见 `test/` 目录下各场景配置文件。


Troubleshooting
---------------
- **MCS 表**：确保 `mcs_3gpp_table_path` 指向合法 JSON（默认 `docs/mcs_tables_38_214.json`）。
- **Matplotlib 缓存**：在受限环境下可 `export MPLCONFIGDIR=$(mktemp -d)`。
- **TLE/轨道**：星座模式需要有效 TLE catalog；单星启用 `enable_orbit_dynamics` 时推荐提供 `tle_lines/tle_path`。
- **Radio Map 格式**：支持传统 MAT 和 HDF5 (v7.3) 格式；Z 维度必须与配置的 PRB 数匹配。
- **导入错误**：如遇 `ModuleNotFoundError`，请确保从项目根目录运行命令。


Repository Layout
-----------------
- `code/`: simulation modules (`main.py`, `config.py`, `orbit.py`, `constellation.py`,
  `link_adapt.py`, `harq.py`, `ntn_channel.py`, `csi.py`, `logging_utils.py`,
  `result_schema.py`, etc.). 模块通过 `__init__.py` 暴露公共 API。
- `docs/`: official 38.214 MCS tables (JSON format) and templates.
- `radio_map/`: Radio Map 数据文件（Toronto 和 Shanghai，支持 MAT/HDF5 格式）。
- `test/`: 测试场景配置文件（`config_toronto_single.py`, `config_shanghai_constellation.py` 等）。
- `tles/`: TLE 轨道数据文件（Starlink 和 Satnet 星座）。
- `tools/`: 辅助工具脚本（`find_best_satellite.py`, `diagnose_zero_se.py`, 可视化脚本等）；
  `tools/archive/` 存放归档的旧版脚本。
- `output/`: 生成的报告和图表（按需创建）。
- `run_test.py`: 多场景测试运行脚本。
- `run_resolution_comparison.py`: Radio Map 分辨率对比测试脚本。
- `run_all_tests.sh`: 批量运行所有测试场景。
- `Makefile`: Make 命令快捷方式。


Roadmap (indicative)
--------------------
- Expand orbit modelling with full ECEF frame support and richer beam patterns.
- Integrate real-world interference datasets and per-beam correlation models.
- Extend HARQ statistics and logging for multi-beam / multi-cell studies.


License
-------
This simulator is provided for research and prototyping purposes. Adapt or
extend at your discretion within your project’s licensing constraints.
