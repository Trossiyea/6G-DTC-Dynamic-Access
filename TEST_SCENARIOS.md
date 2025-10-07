# 测试场景使用指南

本项目支持四种测试场景配置，分别针对不同的地理位置和卫星部署方式。

## 可用场景

| 场景名称 | 配置文件 | 地点 | 类型 | 说明 |
|---------|---------|------|------|------|
| `toronto_single` | `test/config_toronto_single.py` | Toronto | 单卫星 | Starlink-11075单星 |
| `toronto_constellation` | `test/config_toronto_constellation.py` | Toronto | 星座 | Starlink星座 |
| `shanghai_single` | `test/config_shanghai_single.py` | Shanghai | 单卫星 | Satnet-590-00004单星 |
| `shanghai_constellation` | `test/config_shanghai_constellation.py` | Shanghai | 星座 | Satnet星座 |

## 使用方法

### 方法1: 使用 `run_test.py` 脚本

#### 运行单个场景
```bash
python run_test.py --scenario toronto_single
python run_test.py --scenario toronto_constellation
python run_test.py --scenario shanghai_single
python run_test.py --scenario shanghai_constellation
```

或使用简写形式:
```bash
python run_test.py -s toronto_single
```

#### 运行多个场景
```bash
python run_test.py -s toronto_single -s shanghai_single
```

#### 运行所有场景
```bash
python run_test.py --all
```

#### 列出所有可用场景
```bash
python run_test.py --list
```

### 方法2: 使用批量运行脚本

直接运行所有四个场景:
```bash
./run_all_tests.sh
```

或使用 Python 脚本:
```bash
python run_test.py --all
```

### 方法3: 直接在代码中使用

如果您想在 Python 代码中直接使用特定场景配置:

```python
from code.config import CONFIG as BASE_CONFIG
from test.config_toronto_single import CONFIG as TEST_CONFIG
from code.main import run_once

# 合并配置
config = BASE_CONFIG.copy()
config.update(TEST_CONFIG)

# 运行测试
results = run_once(config)

print(f"基线SE: {results['avg_se_baseline_default']:.4f}")
print(f"RadioMap SE: {results['avg_se_radiomap']:.4f}")
print(f"提升: {results['improvement_vs_default_pct']:.2f}%")
```

## 场景配置说明

### Toronto 单卫星场景
- **地点**: Toronto, Canada (43.69°N, 79.37°W)
- **卫星**: STARLINK-11075
- **Radio Map**: `radio_map/Toronto/RadioMap/RM_toronto125_dBm.mat`
- **特点**: 适合测试单星覆盖场景

### Toronto 星座场景
- **地点**: Toronto, Canada
- **卫星**: Starlink星座（从TLE文件加载）
- **TLE文件**: `tles/starlink_DTC_tle.txt`
- **特点**: 适合测试多星协同覆盖

### Shanghai 单卫星场景
- **地点**: Shanghai, China (31.23°N, 121.47°E)
- **卫星**: SATNET-590-00004
- **Radio Map**: `radio_map/Shanghai/RadioMap/RM_shanghai125_dBm.mat`
- **特点**: 适合测试单星覆盖场景

### Shanghai 星座场景
- **地点**: Shanghai, China
- **卫星**: Satnet星座（从TLE文件加载）
- **TLE文件**: `tles/Satnet_DTC.txt`
- **特点**: 适合测试多星协同覆盖

## 输出结果

每个场景运行后会输出:
- **基线平均SE** (avg_se_baseline_default): 3GPP基线方案的频谱效率
- **RadioMap平均SE** (avg_se_radiomap): Radio Map感知方案的频谱效率
- **提升百分比** (improvement_vs_default_pct): 相对于基线的提升百分比

如果配置中启用了 `write_json_report`，还会生成JSON报告文件，文件名格式为: `report_{report_basename}_{timestamp}.json`

## 自定义场景

如果需要添加新的测试场景:

1. 在 `test/` 目录下创建新的配置文件，例如 `config_new_scenario.py`
2. 定义 `CONFIG` 字典，包含场景特定的配置项
3. 在 `run_test.py` 的 `SCENARIOS` 字典中添加映射关系:
   ```python
   SCENARIOS = {
       # ... 现有场景
       'new_scenario': 'test/config_new_scenario.py',
   }
   ```

## 故障排查

### 找不到Radio Map文件
确保配置文件中的 `radio_map_mat_path` 路径正确，且文件存在。

### 找不到TLE文件
确保配置文件中的 `tle_catalog_path` 路径正确，且文件存在。

### 内存不足
如果遇到内存问题，可以尝试:
- 减少 `N_UE` (用户数量)
- 减少 `T` (时间步数)
- 禁用时间变化: `enable_time_varying: False`
