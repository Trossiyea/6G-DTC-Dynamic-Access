# TLE Parser 快速参考

## 🚀 一句话上手

```bash
python tools/tle_parser.py --line1 "1 ..." --line2 "2 ..." --name "卫星名称"
```

## 📋 常用命令速查

| 功能 | 命令 |
|------|------|
| **解析单个TLE** | `python tools/tle_parser.py --line1 "..." --line2 "..."` |
| **读取TLE文件** | `python tools/tle_parser.py --file tles/xxx.txt --summary` |
| **只看第一个** | 加参数 `--first` |
| **JSON输出** | 加参数 `--json` |
| **计算轨道** | 加参数 `--compute --hours 24 --points 100` |
| **可见性分析** | 加参数 `--location 31.23 121.47 --hours 24` |

## 🌍 实战示例

### 上海地区可见性
```bash
python tools/tle_parser.py \
  --file tles/Satnet_DTC.txt --first \
  --location 31.23 121.47 --hours 24 --min-elevation 20
```

### Toronto地区可见性
```bash
python tools/tle_parser.py \
  --file tles/starlink_DTC_tle.txt --first \
  --location 43.69 -79.37 --hours 48 --min-elevation 10
```

### 批量导出JSON
```bash
python tools/tle_parser.py --file tles/Satnet_DTC.txt --json > output.json
```

## 🔍 TLE 参数速查

| 参数 | 含义 | 单位 |
|------|------|------|
| **倾角 (i)** | 轨道面与赤道面夹角 | 度 (°) |
| **RAAN (Ω)** | 升交点赤经 | 度 (°) |
| **离心率 (e)** | 椭圆程度 (0=圆) | 无量纲 |
| **近地点幅角 (ω)** | 近地点位置角 | 度 (°) |
| **平近点角 (M)** | 卫星在轨道上的位置 | 度 (°) |
| **平均运动 (n)** | 每天绕地球圈数 | revs/day |

## 🎯 轨道类型判断

- **i < 10°**: 赤道轨道
- **10° < i < 80°**: 倾斜轨道
- **80° < i < 100°**: 极轨
- **高度 < 2000 km**: LEO
- **2000 km < 高度 < 35786 km**: MEO
- **高度 ≈ 35786 km**: GEO

## 💡 提示

- TLE数据会老化，LEO卫星建议7天内更新
- 可见性计算建议最小仰角设为10°-20°
- 历元时间越近，计算越准确
