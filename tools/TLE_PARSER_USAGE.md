# TLE Parser 使用指南

基于 Skyfield 库的 TLE (Two-Line Element Set) 轨道根数解析器。

## 功能特性

- ✅ 完整解析 TLE 格式轨道根数
- ✅ 使用 Skyfield 进行精确轨道传播
- ✅ 计算轨道特性（周期、高度、速度等）
- ✅ 地面轨迹计算
- ✅ 卫星可见性分析（过顶预测）
- ✅ 支持批量处理 TLE 文件
- ✅ JSON 格式输出

## 安装依赖

```bash
pip install skyfield numpy
```

## 使用方法

### 1. 基础解析

#### 解析单个 TLE（命令行输入）

```bash
python tools/tle_parser.py \
  --line1 "1     3U 00000A   25274.57027601 .00000000  00000-0 0 000           3" \
  --line2 "2     3  85.0000   0.0000 0000000   0.0000  24.0000 14.92546055    05" \
  --name "SATNET-590-00004"
```

**输出示例：**
```
================================================================================
卫星: SATNET-590-00004
================================================================================

【基本信息】
  编目号:           3
  国际编号:         00000A
  轨道类型:         LEO (低地球轨道), 极轨
  历元时间:         2025-10-01 13:41:11.847260 UTC

【Keplerian 轨道根数】
  倾角 (i):            85.0000°
  升交点赤经 (Ω):       0.0000°
  离心率 (e):        0.0000000
  近地点幅角 (ω):       0.0000°
  平近点角 (M):        24.0000°
  平均运动 (n):     14.92546055 revs/day

【轨道特性】
  轨道周期:              96.48 分钟 (1.61 小时)
  半长轴:              6968.14 km
  近地点高度:           590.00 km
  远地点高度:           590.00 km
  平均高度:             590.00 km
  轨道速度:              7.563 km/s
  地面轨迹速度:          0.659 km/s
```

#### 从文件批量读取

```bash
# 读取整个 TLE 目录
python tools/tle_parser.py --file tles/Satnet_DTC.txt --summary
```

**输出示例：**
```
✓ 成功从文件读取 6080 个TLE

1. SATNET-590-00001 [DTC]
   编目号: 1, 类型: LEO (低地球轨道), 极轨
   高度: 590.0 km, 倾角: 85.00°, 周期: 96.48 min
   历元: 2025-10-01 13:41:11 UTC

2. SATNET-590-00002 [DTC]
   编目号: 2, 类型: LEO (低地球轨道), 极轨
   高度: 590.0 km, 倾角: 85.00°, 周期: 96.48 min
   历元: 2025-10-01 13:41:11 UTC
...
```

#### 只处理第一个卫星

```bash
python tools/tle_parser.py --file tles/starlink_DTC_tle.txt --first
```

---

### 2. 轨道计算

#### 计算地面轨迹

```bash
python tools/tle_parser.py \
  --file tles/Satnet_DTC.txt \
  --first \
  --compute \
  --hours 24 \
  --points 100
```

**参数说明：**
- `--compute, -c`: 启用轨道计算
- `--hours`: 计算时长（小时，默认 24）
- `--points`: 轨迹采样点数（默认 100）
- `--start`: 起始时间（ISO 格式，例如 `2025-10-08T12:00:00`）

**输出示例：**
```
【轨道计算】(24.0 小时, 100 个采样点)
  起始时间: 2025-10-01T13:41:11.847260+00:00
  结束时间: 2025-10-02T13:41:11.847260+00:00
  起始位置: 纬度 14.2345°, 经度 -168.7654°, 高度 590.00 km
  结束位置: 纬度 -12.5678°, 经度 175.1234°, 高度 590.00 km
```

---

### 3. 可见性分析（过顶预测）

#### 计算特定地点的可见性

```bash
# 上海地区
python tools/tle_parser.py \
  --file tles/Satnet_DTC.txt \
  --first \
  --location 31.23 121.47 \
  --hours 24 \
  --min-elevation 20

# Toronto 地区
python tools/tle_parser.py \
  --file tles/starlink_DTC_tle.txt \
  --first \
  --location 43.69 -79.37 \
  --hours 48 \
  --min-elevation 10
```

**参数说明：**
- `--location LAT LON`: 观测点坐标（纬度 经度）
- `--hours`: 搜索时长（小时）
- `--min-elevation`: 最小仰角（度，默认 10）

**输出示例：**
```
【可见性分析】
  观测点: 纬度 31.2300°, 经度 121.4700°
  时间范围: 24.0 小时
  最小仰角: 20.0°
  过顶次数: 15

  过顶详情:
    1. 升起: 13:45:23, 最高: 13:48:15 (仰角 80.3°, 方位 92.2°), 落下: 13:51:08, 持续 5.8 分钟
    2. 升起: 15:16:49, 最高: 15:19:41 (仰角 62.5°, 方位 278.4°), 落下: 15:22:34, 持续 5.8 分钟
    3. 升起: 16:48:15, 最高: 16:51:07 (仰角 45.2°, 方位 135.6°), 落下: 16:53:59, 持续 5.7 分钟
    ...
```

#### 摘要模式显示可见性

```bash
python tools/tle_parser.py \
  --file tles/Satnet_DTC.txt \
  --summary \
  --location 31.23 121.47 \
  --hours 24
```

**输出示例：**
```
1. SATNET-590-00001 [DTC]
   编目号: 1, 类型: LEO (低地球轨道), 极轨
   高度: 590.0 km, 倾角: 85.00°, 周期: 96.48 min
   历元: 2025-10-01 13:41:11 UTC
   过顶次数 (24.0h): 15 次
   最佳仰角: 87.3° @ 2025-10-01T20:12:45
```

---

### 4. JSON 输出

#### 基础 JSON 输出

```bash
python tools/tle_parser.py \
  --line1 "..." \
  --line2 "..." \
  --json
```

**JSON 结构：**
```json
{
  "satellite_name": "SATNET-590-00004",
  "catalog_number": 3,
  "classification": "U",
  "intl_designator": "00000A",
  "epoch": "2025-10-01T13:41:11.847260+00:00",
  "orbit_type": "LEO (低地球轨道), 极轨",
  "keplerian_elements": {
    "inclination_deg": 85.0,
    "raan_deg": 0.0,
    "eccentricity": 0.0,
    "arg_perigee_deg": 0.0,
    "mean_anomaly_deg": 24.0,
    "mean_motion_revs_per_day": 14.92546055
  },
  "orbital_parameters": {
    "period_minutes": 96.48,
    "semi_major_axis_km": 6968.14,
    "perigee_altitude_km": 590.00,
    "apogee_altitude_km": 590.00,
    "mean_altitude_km": 590.00,
    "orbital_velocity_km_s": 7.563,
    "ground_track_velocity_km_s": 0.659
  },
  "perturbation": {
    "mean_motion_derivative": 0.0,
    "bstar_drag": 0.0,
    "revolution_number": 0
  }
}
```

#### JSON 输出包含轨道计算

```bash
python tools/tle_parser.py \
  --file tles/Satnet_DTC.txt \
  --first \
  --json \
  --compute \
  --hours 2 \
  --points 10
```

**JSON 会额外包含：**
```json
{
  ...
  "ground_track": [
    {
      "time": "2025-10-01T13:41:11.847260+00:00",
      "position_eci_km": [x, y, z],
      "velocity_eci_km_s": [vx, vy, vz],
      "latitude_deg": 14.2345,
      "longitude_deg": -168.7654,
      "altitude_km": 590.00,
      "speed_km_s": 7.563
    },
    ...
  ]
}
```

#### JSON 输出包含可见性分析

```bash
python tools/tle_parser.py \
  --file tles/Satnet_DTC.txt \
  --first \
  --json \
  --location 31.23 121.47 \
  --hours 24
```

**JSON 会额外包含：**
```json
{
  ...
  "observer_location": {
    "latitude": 31.23,
    "longitude": 121.47
  },
  "passes": [
    {
      "rise_time": "2025-10-01T13:45:23.123456+00:00",
      "peak_time": "2025-10-01T13:48:15.654321+00:00",
      "peak_elevation_deg": 80.3,
      "peak_azimuth_deg": 92.2,
      "peak_distance_km": 612.5,
      "set_time": "2025-10-01T13:51:08.987654+00:00",
      "duration_seconds": 345.8
    },
    ...
  ]
}
```

---

## Python API 使用

### 基础解析

```python
from tools.tle_parser import TLEAnalyzer

# 创建 TLE 分析器
line1 = "1     3U 00000A   25274.57027601 .00000000  00000-0 0 000           3"
line2 = "2     3  85.0000   0.0000 0000000   0.0000  24.0000 14.92546055    05"

tle = TLEAnalyzer(line1, line2, name="SATNET-590-00004")

# 打印详细信息
print(tle)

# 获取字典格式
data = tle.to_dict()
print(f"轨道周期: {data['orbital_parameters']['period_minutes']:.2f} 分钟")
print(f"轨道高度: {data['orbital_parameters']['mean_altitude_km']:.2f} km")
print(f"轨道类型: {data['orbit_type']}")
```

### 轨道计算

```python
from datetime import datetime

# 计算地面轨迹
start = datetime(2025, 10, 8, 12, 0, 0)
track = tle.compute_ground_track(start_time=start, hours=2, points=50)

for point in track[::10]:  # 每10个点打印一次
    print(f"时间: {point['time']}")
    print(f"  位置: ({point['latitude_deg']:.4f}°, {point['longitude_deg']:.4f}°)")
    print(f"  高度: {point['altitude_km']:.2f} km")
    print()
```

### 可见性分析

```python
# 计算上海地区可见性
passes = tle.compute_passes_over_location(
    lat=31.23,
    lon=121.47,
    start_time=start,
    hours=24,
    min_elevation_deg=20.0
)

print(f"未来24小时过顶次数: {len(passes)}")

for i, p in enumerate(passes[:5], 1):
    print(f"\n过顶 {i}:")
    print(f"  升起时间: {p['rise_time']}")
    print(f"  最高仰角: {p['peak_elevation_deg']:.1f}° @ {p['peak_time']}")
    print(f"  方位角: {p['peak_azimuth_deg']:.1f}°")
    print(f"  持续时间: {p['duration_seconds']/60:.1f} 分钟")
```

### 批量处理文件

```python
from tools.tle_parser import parse_tle_file

# 读取整个 TLE 文件
tles = parse_tle_file('tles/Satnet_DTC.txt')

print(f"共读取 {len(tles)} 个卫星")

# 筛选特定高度范围的卫星
leo_sats = [t for t in tles if t.mean_altitude_km < 2000]
print(f"LEO 卫星数量: {len(leo_sats)}")

# 找到倾角最大的卫星
highest_inc = max(tles, key=lambda t: t.inclination_deg)
print(f"最大倾角: {highest_inc.name} - {highest_inc.inclination_deg:.2f}°")
```

---

## 命令行参数完整列表

### 输入选项
- `--line1 TEXT`: TLE 第一行
- `--line2 TEXT`: TLE 第二行
- `--name TEXT`: 卫星名称（可选）
- `--file PATH`, `-f PATH`: 从文件读取 TLE

### 输出选项
- `--json`: 输出 JSON 格式
- `--summary`, `-s`: 只显示摘要信息
- `--first`: 只处理第一个卫星

### 轨道计算选项
- `--compute`, `-c`: 启用轨道轨迹计算
- `--hours FLOAT`: 计算时长（小时，默认 24）
- `--points INT`: 轨迹采样点数（默认 100）
- `--start ISO_TIME`: 起始时间（ISO 格式）

### 可见性计算选项
- `--location LAT LON`: 观测点坐标（纬度 经度）
- `--min-elevation FLOAT`: 最小仰角（度，默认 10）

---

## TLE 格式说明

### 第一行格式
```
1 NNNNNC NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
```

- **列 01**: 行号（固定为 "1"）
- **列 03-07**: 卫星编目号（Catalog Number）
- **列 08**: 分类（U=未分类, C=机密, S=秘密）
- **列 10-17**: 国际编号（International Designator）
- **列 19-32**: 历元时间（年份 + 年积日）
- **列 34-43**: 平均运动一阶导数（revs/day²）
- **列 45-52**: 平均运动二阶导数（revs/day³）
- **列 54-61**: BSTAR 拖拽项
- **列 63**: 星历类型（0=SGP, 4=SGP4）
- **列 65-68**: 根数序号
- **列 69**: 校验和

### 第二行格式
```
2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN
```

- **列 01**: 行号（固定为 "2"）
- **列 03-07**: 卫星编目号（与第一行相同）
- **列 09-16**: 倾角（degrees）
- **列 18-25**: 升交点赤经（degrees）
- **列 27-33**: 离心率（无小数点）
- **列 35-42**: 近地点幅角（degrees）
- **列 44-51**: 平近点角（degrees）
- **列 53-63**: 平均运动（revs/day）
- **列 64-68**: 革命编号
- **列 69**: 校验和

---

## 常见场景示例

### 场景 1: 快速查看卫星基本信息

```bash
python tools/tle_parser.py \
  --file tles/starlink_DTC_tle.txt \
  --first \
  --summary
```

### 场景 2: 寻找最佳观测时机

```bash
# Toronto 地区，未来 48 小时，仰角 > 30°
python tools/tle_parser.py \
  --file tles/starlink_DTC_tle.txt \
  --first \
  --location 43.69 -79.37 \
  --hours 48 \
  --min-elevation 30
```

### 场景 3: 批量导出 JSON 数据

```bash
python tools/tle_parser.py \
  --file tles/Satnet_DTC.txt \
  --json \
  > satnet_orbit_data.json
```

### 场景 4: 对比两个星座

```bash
# Starlink
python tools/tle_parser.py \
  --file tles/starlink_DTC_tle.txt \
  --summary \
  --location 43.69 -79.37 \
  --hours 24

# Satnet
python tools/tle_parser.py \
  --file tles/Satnet_DTC.txt \
  --summary \
  --location 31.23 121.47 \
  --hours 24
```

---

## 注意事项

1. **TLE 时效性**：TLE 数据会随时间衰减，建议定期更新（尤其是 LEO 卫星）
2. **历元时间**：计算时应尽量接近 TLE 的历元时间以保证精度
3. **最小仰角**：建议设置 10°-20° 以避免地平线附近的大气折射影响
4. **计算时长**：对于 LEO 卫星，建议不超过 7 天；GEO 可更长

---

## 故障排查

### 问题：`ModuleNotFoundError: No module named 'skyfield'`

**解决方案：**
```bash
pip install skyfield numpy
```

### 问题：TLE 解析失败

**检查：**
1. TLE 格式是否正确（两行，每行 69 字符）
2. 第一行是否以 "1 " 开头
3. 第二行是否以 "2 " 开头
4. 编目号是否匹配

### 问题：可见性计算无结果

**可能原因：**
1. 时间范围内卫星未经过该地区
2. 最小仰角设置过高
3. 卫星轨道倾角与观测点纬度不兼容

**解决方案：**
- 增加搜索时长 `--hours`
- 降低最小仰角 `--min-elevation`
- 检查卫星轨道类型是否覆盖目标地区

---

## 参考资料

- [Skyfield 官方文档](https://rhodesmill.org/skyfield/)
- [TLE 格式说明（CelesTrak）](https://celestrak.com/NORAD/documentation/tle-fmt.php)
- [SGP4/SDP4 模型](https://www.celestrak.com/publications/AIAA/2006-6753/)
