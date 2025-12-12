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
- **Modular architecture**: 代码按功能解耦为独立子模块 (`core/`, `data_io/`, `scheduler/`, `config/`, `simulation/`, `link/`)，
  便于单元测试、Web 可视化集成和二次开发。
- **Link layer modularization (Phase 7)**: 统一的链路层子包 (`link/`) 整合了 MCS/CQI/BLER 表管理、
  EESM 计算、TBS 估算、OLLA 自适应、链路自适应和 HARQ 管理，消除代码重复，提升可维护性。
- **SimulationEngine pattern**: 基于类的仿真引擎封装 `run_once()` 和 `run_constellation()` 逻辑，
  支持实时进度回调、状态序列化、阶段通知，适配 Web API 和异步执行场景。
- **Type-safe configuration system**: 基于 dataclass 的配置系统支持字典风格和属性访问、
  IDE 自动补全、JSON Schema 生成，并向后兼容传统用法。
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

### 环境验证（推荐首次运行）
```bash
# 验证 Phase 1-7 模块化架构和依赖
python run_test.py --verify
```

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
全部配置在 `code/config/` 子包。核心分组如下：

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
# 方式1: 传统字典风格（向后兼容）
from copy import deepcopy
from code.config import CONFIG
from code.main import run_once

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

```python
# 方式2: 从场景文件加载（推荐）
from code.config import load_scenario_config
from code.main import run_once

cfg = load_scenario_config("test/config_toronto_single.py")
result = run_once(cfg)
print("Baseline default SE:", result["avg_se_baseline_default"])
print("Radio-map SE:", result["avg_se_radiomap"])
```

```python
# 方式3: 类型安全的属性访问（支持 IDE 自动补全）
from code.config import CONFIG
from code.main import run_once

CONFIG.simulation.N_UE = 25
CONFIG.simulation.T = 80
CONFIG.harq.enable_harq_full = True
CONFIG.harq.harq_target_bler = 0.1
result = run_once(CONFIG)
```

```python
# 方式4: 使用 SimulationEngine (Web 集成推荐)
from code.simulation import SimulationEngine
from code.config import load_scenario_config

# 加载配置
cfg = load_scenario_config("test/config_toronto_single.py")

# 创建引擎并注册进度回调
engine = SimulationEngine(cfg)
engine.on_progress(lambda tti, metrics:
    print(f"TTI {tti}: {metrics['progress']*100:.1f}% complete"))

# 执行仿真
result = engine.run()

# 访问状态对象
state = engine.state
print(f"Final phase: {state.phase}")
print(f"Baseline SE: {result['avg_se_baseline_default']:.4f}")
print(f"RadioMap SE: {result['avg_se_radiomap']:.4f}")
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
- `code/`: simulation modules, organized into sub-packages:
  - `core/`: 核心工具模块
    - `units.py` - 单位转换 (dBm↔mW, thermal noise)
    - `capacity.py` - 容量/SE计算, MCS映射, EESM
  - `data_io/`: 数据输入输出
    - `radiomap.py` - Radio Map 加载 (MAT/HDF5 格式)
  - `scheduler/`: 调度算法
    - `baseline.py` - 3GPP-like 宽带基线 PF 调度器
    - `radiomap.py` - RadioMap 感知连续块调度器 (EESM+MCS)
    - `subband.py` - 子带级基线调度器
    - `power_alloc.py` - DL 功率分配 (water-filling)
  - `config/`: 配置管理系统
    - `schema.py` - 15 个 dataclass 配置组（simulation, radio_map, harq 等）
    - `compat.py` - ConfigDict 向后兼容包装器，支持字典风格访问
    - `loader.py` - JSON/YAML/Python 文件配置加载器
    - `__init__.py` - 导出 CONFIG 实例和公共 API
  - `simulation/`: 仿真引擎 (Phase 5 模块化重构)
    - `state.py` - 可序列化状态容器 (SimulationState, ConstellationState, 支持 Web API)
    - `callbacks.py` - 回调系统 (ProgressCallback, TTIMetrics, CallbackManager, 实时进度通知)
    - `helpers.py` - 辅助函数 (UE 位置生成, 噪声/功率控制, 容量计算, 时变动态)
    - `engine.py` - 单卫星仿真引擎 (SimulationEngine, 封装 run_once 逻辑)
    - `constellation_engine.py` - 多卫星星座仿真引擎 (ConstellationEngine, TTI 循环/切换)
    - `__init__.py` - 导出公共 API (向后兼容 run_once/run_constellation)
  - `link/`: 链路层模块 (Phase 7 模块化重构, 1492 行)
    - `mcs.py` - MCS 表管理 (3GPP TS 38.214, 内置/外部表支持)
    - `cqi.py` - CQI 表和 SINR→CQI→SE 映射
    - `bler.py` - BLER 曲线管理 (sigmoid 模型 + 外部曲线)
    - `eesm.py` - EESM 有效 SINR 计算和 HARQ 软合并
    - `tbs.py` - TBS 计算和 RE 统计 (TS 38.214 §5.1.3.2)
    - `olla.py` - OLLA 自适应偏移控制 (ACK/NACK 反馈)
    - `adaptation.py` - 统一链路自适应接口 (MCS 选择)
    - `harq.py` - HARQ 管理器 (简单/完整两种模式, 进程管理, RV 循环)
    - `__init__.py` - 导出 21 个公共 API
  - 主模块: `main.py` (轻量编排层, 274 行), `orbit.py`, `constellation.py`,
    `ntn_channel.py`, `logging_utils.py`, `result_schema.py` 等。
  - 兼容包装器: `link_adapt.py`, `csi.py`, `harq.py`, `ntn_csi.py` (保持向后兼容性)。
- `docs/`: official 38.214 MCS tables (JSON format) and templates, Phase 7 report.
- `radio_map/`: Radio Map 数据文件（Toronto 和 Shanghai，支持 MAT/HDF5 格式）。
- `test/`: 测试场景配置文件（`config_toronto_single.py`, `config_shanghai_constellation.py` 等）。
- `tles/`: TLE 轨道数据文件（Starlink 和 Satnet 星座）。
- `tools/`: 辅助工具脚本（`find_best_satellite.py`, `diagnose_zero_se.py`, 可视化脚本等）；
  `tools/archive/` 存放归档的旧版脚本。
- `output/`: 生成的报告和图表（按需创建）。
- `run_test.py`: 多场景测试运行脚本 (支持 `--verify` 环境验证)。
- `run_resolution_comparison.py`: Radio Map 分辨率对比测试脚本。
- `run_all_tests.sh`: 批量运行所有测试场景 (支持 `--unit`, `--quick`, `--scenarios` 模式)。
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
