# 寻找最佳卫星和仿真时间工具

## 📋 问题说明

**原始脚本的限制**:
- `find_overpass_times.py` 只能分析**单颗卫星**（TLE文件的最后一颗）
- `Satnet_DTC.txt` 包含 **6,080颗卫星**
- 需要找到**最适合仿真的卫星**和**最佳开始时间**

## ✅ 解决方案

我创建了新脚本 `find_best_satellite.py`，它可以：

1. ✨ 分析 TLE 文件中的**所有卫星**
2. 🎯 找出在指定时间窗口内**仰角最高的卫星**
3. ⏰ 给出**最佳仿真开始时间**
4. 📊 显示排名前N的卫星供选择

## 🚀 使用方法

### 基本用法

```bash
# 需要先安装依赖
pip install numpy skyfield sgp4

# 运行脚本 - 使用默认参数（上海坐标）
python tools/find_best_satellite.py --tle-path tles/Satnet_DTC.txt
```

### 指定您的坐标

```bash
python tools/find_best_satellite.py \
    --tle-path tles/Satnet_DTC.txt \
    --lat 31.2304 \
    --lon 121.4737
```

### 完整参数示例

```bash
python tools/find_best_satellite.py \
    --tle-path tles/Satnet_DTC.txt \
    --lat 31.2304 \
    --lon 121.4737 \
    --hours 24 \
    --min-elev 60 \
    --top-n 10 \
    --step-sec 30
```

### 快速测试（只分析前100颗卫星）

```bash
python tools/find_best_satellite.py \
    --tle-path tles/Satnet_DTC.txt \
    --lat 31.2304 \
    --lon 121.4737 \
    --max-sats 100
```

## 📝 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tle-path` | TLE文件路径 | **必需** |
| `--lat` | 地面纬度（度） | 31.2304（上海） |
| `--lon` | 地面经度（度） | 121.4737（上海） |
| `--start` | 搜索开始时间（ISO8601格式） | 当前时间(UTC) |
| `--hours` | 搜索时间窗口（小时） | 24.0 |
| `--step-sec` | 采样间隔（秒）| 30.0 |
| `--min-elev` | 最小仰角阈值（度） | 60.0 |
| `--top-n` | 显示前N个最佳卫星 | 10 |
| `--max-sats` | 限制分析的卫星数量（测试用） | None（全部） |

## 📤 输出示例

```
================================================================================
Finding Best Satellite for Simulation
================================================================================
TLE File: tles/Satnet_DTC.txt
Ground Location: lat=31.2304°, lon=121.4737°
Search Window: 2025-10-07T09:40:00+00:00Z + 24.0 hours
Minimum Elevation: 60.0°
Step: 30.0 seconds
================================================================================

Loading satellites from TLE file...
Found 6080 satellites in TLE file.

Analyzing satellites (this may take a while)...
  Progress: 100/6080 satellites analyzed...
  Progress: 200/6080 satellites analyzed...
  ...

Analysis complete!
================================================================================

✅ Found 245 satellite(s) with elevation >= 60.0°

Top 10 Best Satellites:

 1. Satellite #  1234
    Peak Elevation: 87.45°
    Peak Time: 2025-10-07T14:23:45Z
    Time Offset: +4.73 hours from start

 2. Satellite #  2456
    Peak Elevation: 85.12°
    Peak Time: 2025-10-07T18:56:12Z
    Time Offset: +9.27 hours from start

 ... (更多结果)

================================================================================
🎯 RECOMMENDATION FOR SIMULATION:
================================================================================

✨ Best Satellite: #1234
   Peak Elevation: 87.45°
   Optimal Start Time: 2025-10-07T14:23:45Z
   Suggested Sim Start: 2025-10-07T14:18:45Z
                        (5 minutes before peak)

📋 TLE Lines for Satellite #1234:
   1  1234U 00000A   25274.57027601 .00000000  00000-0 0 000           1
   2  1234  85.0000   0.0000 0000000   0.0000   0.0000 14.92546055    07

================================================================================
You can use this satellite and start time in your simulation config!
================================================================================
```

## ⚡ 性能说明

- **全量分析 6,080 颗卫星**大约需要 **5-10 分钟**（取决于电脑性能）
- 使用 `--max-sats` 可以快速测试（如 `--max-sats 100` 约需 10-20 秒）
- 增大 `--step-sec` 可以加快速度（但会降低精度）

## 🔍 工作原理

1. **加载所有卫星**: 从 TLE 文件读取所有卫星的轨道参数
2. **逐一分析**: 对每颗卫星计算指定时间窗口内的仰角轨迹
3. **筛选排序**: 找出超过最小仰角阈值的卫星，按峰值仰角排序
4. **推荐结果**: 给出仰角最高的卫星及其最佳过顶时间

## 💡 使用建议

1. **首次使用**: 建议先用 `--max-sats 100` 快速测试
2. **调整参数**: 
   - 如果没找到合适的卫星，降低 `--min-elev`（如改为 50 或 40）
   - 增加 `--hours`（如改为 48 或 72）以扩大搜索窗口
3. **仿真配置**: 将推荐的卫星TLE和开始时间写入您的仿真配置文件

## 🔗 相关文件

- 原始脚本: `tools/find_overpass_times.py` (单卫星分析)
- 新脚本: `tools/find_best_satellite.py` (多卫星分析)
- TLE数据: `tles/Satnet_DTC.txt` (6,080颗卫星)
