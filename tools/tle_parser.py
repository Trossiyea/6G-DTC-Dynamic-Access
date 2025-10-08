#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TLE (Two-Line Element Set) Parser using Skyfield

使用Skyfield库解析并分析卫星轨道根数（TLE格式）。
支持命令行输入、文件读取、轨道计算和可视化。

Dependencies:
    pip install skyfield numpy

Usage:
    # 解析单个TLE
    python tle_parser.py --line1 "1 ..." --line2 "2 ..."
    
    # 从文件读取
    python tle_parser.py --file tles/starlink_DTC_tle.txt
    
    # 计算未来24小时轨道
    python tle_parser.py --file tles/Satnet_DTC.txt --compute --hours 24
    
    # 计算特定地点的可见性
    python tle_parser.py --file tles/starlink_DTC_tle.txt --location 43.69 -79.37 --hours 24
    
    # 输出JSON格式
    python tle_parser.py --line1 "..." --line2 "..." --json
"""

import sys
import argparse
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

try:
    from skyfield.api import load, EarthSatellite, wgs84
    from skyfield.timelib import Time
    import numpy as np
except ImportError:
    print("错误: 需要安装 skyfield 库", file=sys.stderr)
    print("请运行: pip install skyfield numpy", file=sys.stderr)
    sys.exit(1)


class TLEAnalyzer:
    """基于Skyfield的TLE分析器"""
    
    # 地球参数
    EARTH_RADIUS_KM = 6378.137
    EARTH_MU = 398600.4418  # km^3/s^2 (WGS-84)
    
    def __init__(self, line1: str, line2: str, name: Optional[str] = None):
        """
        初始化TLE分析器
        
        Args:
            line1: TLE第一行
            line2: TLE第二行  
            name: 卫星名称（可选）
        """
        self.line1 = line1.strip()
        self.line2 = line2.strip()
        self.name = name.strip() if name else f"SAT-{self.line1[2:7].strip()}"
        
        # 使用Skyfield创建卫星对象
        self.ts = load.timescale()
        try:
            self.satellite = EarthSatellite(self.line1, self.line2, self.name, self.ts)
        except Exception as e:
            raise ValueError(f"无效的TLE格式: {e}")
        
        # 解析基本参数
        self._parse_parameters()
    
    def _parse_parameters(self):
        """解析TLE中的轨道参数"""
        # 从Skyfield的satellite.model获取SGP4/SDP4模型参数
        model = self.satellite.model
        
        # 基本信息
        self.catalog_number = int(self.line1[2:7].strip())
        self.classification = self.line1[7]
        self.intl_designator = self.line1[9:17].strip()
        
        # 历元时间
        self.epoch = self.satellite.epoch
        self.epoch_datetime = self.epoch.utc_datetime()
        
        # Keplerian轨道根数 (从model获取)
        self.inclination_deg = model.inclo * 180.0 / np.pi  # rad to deg
        self.raan_deg = model.nodeo * 180.0 / np.pi
        self.eccentricity = model.ecco
        self.arg_perigee_deg = model.argpo * 180.0 / np.pi
        self.mean_anomaly_deg = model.mo * 180.0 / np.pi
        self.mean_motion_revs_per_day = model.no_kozai * 1440.0 / (2.0 * np.pi)  # rad/min to revs/day
        
        # 摄动参数
        self.mean_motion_derivative = float(self.line1[33:43])
        self.bstar = self._parse_exp_notation(self.line1[53:61].strip())
        
        # 其他
        self.revolution_number = int(self.line2[63:68].strip()) if self.line2[63:68].strip() else 0
        
        # 计算派生参数
        self._compute_derived_params()
    
    def _parse_exp_notation(self, s: str) -> float:
        """解析TLE指数记号"""
        if not s or s == '00000-0' or s == '0':
            return 0.0
        
        sign = -1 if s[0] == '-' else 1
        s = s.lstrip('+-')
        
        if '-' in s:
            mantissa, exp = s.split('-')
            exp = -int(exp)
        elif '+' in s:
            mantissa, exp = s.split('+')
            exp = int(exp)
        else:
            mantissa = s
            exp = 0
        
        value = float(mantissa) * (10 ** (exp - len(mantissa) + 1))
        return sign * value
    
    def _compute_derived_params(self):
        """计算派生轨道参数"""
        # 轨道周期
        self.period_minutes = 1440.0 / self.mean_motion_revs_per_day
        
        # 半长轴 (km) - 从mean motion推导
        n_rad_per_min = self.mean_motion_revs_per_day * 2 * np.pi / 1440.0
        self.semi_major_axis_km = (self.EARTH_MU / ((n_rad_per_min / 60.0) ** 2)) ** (1.0/3.0)
        
        # 近地点和远地点高度
        self.perigee_altitude_km = self.semi_major_axis_km * (1 - self.eccentricity) - self.EARTH_RADIUS_KM
        self.apogee_altitude_km = self.semi_major_axis_km * (1 + self.eccentricity) - self.EARTH_RADIUS_KM
        self.mean_altitude_km = (self.perigee_altitude_km + self.apogee_altitude_km) / 2
        
        # 轨道速度 (km/s)
        self.orbital_velocity_km_s = np.sqrt(self.EARTH_MU / self.semi_major_axis_km)
        
        # 地面轨迹速度
        self.ground_track_velocity_km_s = self.orbital_velocity_km_s * np.cos(np.radians(self.inclination_deg))
    
    def get_orbit_type(self) -> str:
        """判断轨道类型"""
        alt = self.mean_altitude_km
        inc = self.inclination_deg
        
        # 高度分类
        if alt < 2000:
            orbit_class = "LEO (低地球轨道)"
        elif alt < 35786:
            orbit_class = "MEO (中地球轨道)"
        elif 35686 < alt < 35886:
            orbit_class = "GEO (地球同步轨道)"
        else:
            orbit_class = "HEO (高地球轨道)"
        
        # 倾角分类
        if inc < 10:
            inc_class = "赤道轨道"
        elif inc < 80:
            inc_class = "倾斜轨道"
        elif inc < 100:
            inc_class = "极轨"
        else:
            inc_class = "逆行轨道"
        
        return f"{orbit_class}, {inc_class}"
    
    def compute_position_at_time(self, dt: datetime) -> Dict:
        """
        计算指定时间的卫星位置
        
        Args:
            dt: datetime对象
        
        Returns:
            包含位置、速度、经纬度等信息的字典
        """
        t = self.ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        
        # 地心坐标系位置和速度
        geocentric = self.satellite.at(t)
        position = geocentric.position.km  # [x, y, z] km
        velocity = geocentric.velocity.km_per_s  # [vx, vy, vz] km/s
        
        # 地理坐标（经纬度高度）
        subpoint = wgs84.subpoint(geocentric)
        latitude = subpoint.latitude.degrees
        longitude = subpoint.longitude.degrees
        altitude = subpoint.elevation.km
        
        return {
            'time': dt.isoformat(),
            'position_eci_km': position.tolist(),
            'velocity_eci_km_s': velocity.tolist(),
            'latitude_deg': latitude,
            'longitude_deg': longitude,
            'altitude_km': altitude,
            'speed_km_s': float(np.linalg.norm(velocity)),
        }
    
    def compute_ground_track(self, start_time: Optional[datetime] = None, 
                            hours: float = 24, points: int = 100) -> List[Dict]:
        """
        计算地面轨迹
        
        Args:
            start_time: 起始时间（默认为epoch）
            hours: 计算时长（小时）
            points: 轨迹点数
        
        Returns:
            轨迹点列表，每个点包含时间、经纬度、高度等
        """
        if start_time is None:
            start_time = self.epoch_datetime
        
        track = []
        delta = timedelta(hours=hours)
        end_time = start_time + delta
        
        time_points = [start_time + (delta * i / points) for i in range(points + 1)]
        
        for dt in time_points:
            pos_info = self.compute_position_at_time(dt)
            track.append(pos_info)
        
        return track
    
    def compute_passes_over_location(self, lat: float, lon: float, 
                                    start_time: Optional[datetime] = None,
                                    hours: float = 24, 
                                    min_elevation_deg: float = 10.0) -> List[Dict]:
        """
        计算卫星过顶事件（passes）
        
        Args:
            lat: 观测点纬度（度）
            lon: 观测点经度（度）
            start_time: 起始时间
            hours: 搜索时长（小时）
            min_elevation_deg: 最小仰角（度）
        
        Returns:
            过顶事件列表
        """
        if start_time is None:
            start_time = self.epoch_datetime
        
        # 创建观测点
        location = wgs84.latlon(lat, lon)
        
        # 时间范围
        t0 = self.ts.utc(start_time.year, start_time.month, start_time.day,
                        start_time.hour, start_time.minute, start_time.second)
        end_time = start_time + timedelta(hours=hours)
        t1 = self.ts.utc(end_time.year, end_time.month, end_time.day,
                        end_time.hour, end_time.minute, end_time.second)
        
        # 找到过顶事件
        t, events = self.satellite.find_events(location, t0, t1, altitude_degrees=min_elevation_deg)
        
        passes = []
        current_pass = {}
        
        for ti, event in zip(t, events):
            dt = ti.utc_datetime()
            
            if event == 0:  # 升起
                current_pass = {
                    'rise_time': dt.isoformat(),
                    'rise_time_dt': dt,
                }
            elif event == 1:  # 最高点
                if current_pass:
                    # 计算最高点的仰角和方位角
                    difference = self.satellite - location
                    topocentric = difference.at(ti)
                    alt, az, distance = topocentric.altaz()
                    
                    current_pass['peak_time'] = dt.isoformat()
                    current_pass['peak_elevation_deg'] = float(alt.degrees)
                    current_pass['peak_azimuth_deg'] = float(az.degrees)
                    current_pass['peak_distance_km'] = float(distance.km)
            elif event == 2:  # 落下
                if current_pass:
                    current_pass['set_time'] = dt.isoformat()
                    current_pass['set_time_dt'] = dt
                    
                    # 计算过顶持续时间
                    if 'rise_time_dt' in current_pass:
                        duration = (dt - current_pass['rise_time_dt']).total_seconds()
                        current_pass['duration_seconds'] = duration
                        del current_pass['rise_time_dt']
                        del current_pass['set_time_dt']
                    
                    passes.append(current_pass)
                    current_pass = {}
        
        return passes
    
    def to_dict(self) -> Dict:
        """返回完整的轨道参数字典"""
        return {
            'satellite_name': self.name,
            'catalog_number': self.catalog_number,
            'classification': self.classification,
            'intl_designator': self.intl_designator,
            'epoch': self.epoch_datetime.isoformat(),
            'orbit_type': self.get_orbit_type(),
            
            # Keplerian元素
            'keplerian_elements': {
                'inclination_deg': float(self.inclination_deg),
                'raan_deg': float(self.raan_deg),
                'eccentricity': float(self.eccentricity),
                'arg_perigee_deg': float(self.arg_perigee_deg),
                'mean_anomaly_deg': float(self.mean_anomaly_deg),
                'mean_motion_revs_per_day': float(self.mean_motion_revs_per_day),
            },
            
            # 轨道特性
            'orbital_parameters': {
                'period_minutes': float(self.period_minutes),
                'semi_major_axis_km': float(self.semi_major_axis_km),
                'perigee_altitude_km': float(self.perigee_altitude_km),
                'apogee_altitude_km': float(self.apogee_altitude_km),
                'mean_altitude_km': float(self.mean_altitude_km),
                'orbital_velocity_km_s': float(self.orbital_velocity_km_s),
                'ground_track_velocity_km_s': float(self.ground_track_velocity_km_s),
            },
            
            # 摄动参数
            'perturbation': {
                'mean_motion_derivative': self.mean_motion_derivative,
                'bstar_drag': self.bstar,
                'revolution_number': self.revolution_number,
            },
        }
    
    def __str__(self) -> str:
        """格式化输出"""
        lines = [
            "=" * 80,
            f"卫星: {self.name}",
            "=" * 80,
            "",
            "【基本信息】",
            f"  编目号:           {self.catalog_number}",
            f"  国际编号:         {self.intl_designator}",
            f"  轨道类型:         {self.get_orbit_type()}",
            f"  历元时间:         {self.epoch_datetime.strftime('%Y-%m-%d %H:%M:%S.%f UTC')[:-3]}",
            "",
            "【Keplerian 轨道根数】",
            f"  倾角 (i):         {self.inclination_deg:>10.4f}°",
            f"  升交点赤经 (Ω):   {self.raan_deg:>10.4f}°",
            f"  离心率 (e):       {self.eccentricity:>10.7f}",
            f"  近地点幅角 (ω):   {self.arg_perigee_deg:>10.4f}°",
            f"  平近点角 (M):     {self.mean_anomaly_deg:>10.4f}°",
            f"  平均运动 (n):     {self.mean_motion_revs_per_day:>10.8f} revs/day",
            "",
            "【轨道特性】",
            f"  轨道周期:         {self.period_minutes:>10.2f} 分钟 ({self.period_minutes/60:.2f} 小时)",
            f"  半长轴:           {self.semi_major_axis_km:>10.2f} km",
            f"  近地点高度:       {self.perigee_altitude_km:>10.2f} km",
            f"  远地点高度:       {self.apogee_altitude_km:>10.2f} km",
            f"  平均高度:         {self.mean_altitude_km:>10.2f} km",
            f"  轨道速度:         {self.orbital_velocity_km_s:>10.3f} km/s",
            f"  地面轨迹速度:     {self.ground_track_velocity_km_s:>10.3f} km/s",
            "",
            "【摄动参数】",
            f"  平均运动一阶导数: {self.mean_motion_derivative:>15.11e} revs/day²",
            f"  BSTAR 拖拽项:     {self.bstar:>15.11e} 1/Earth radii",
            f"  革命编号:         {self.revolution_number}",
            "=" * 80,
        ]
        return '\n'.join(lines)


def parse_tle_file(filepath: str) -> List[TLEAnalyzer]:
    """
    从文件解析多个TLE
    
    Args:
        filepath: TLE文件路径
    
    Returns:
        TLEAnalyzer对象列表
    """
    results = []
    with open(filepath, 'r') as f:
        lines = [line.rstrip('\n\r') for line in f.readlines()]
    
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        
        if lines[i].startswith('1 '):
            # 没有名称行
            if i + 1 < len(lines) and lines[i + 1].startswith('2 '):
                try:
                    tle = TLEAnalyzer(lines[i], lines[i + 1])
                    results.append(tle)
                except Exception as e:
                    print(f"警告: 跳过无效TLE (行 {i+1}): {e}", file=sys.stderr)
                i += 2
            else:
                i += 1
        else:
            # 可能有名称行
            name = lines[i]
            if i + 2 < len(lines) and lines[i + 1].startswith('1 ') and lines[i + 2].startswith('2 '):
                try:
                    tle = TLEAnalyzer(lines[i + 1], lines[i + 2], name)
                    results.append(tle)
                except Exception as e:
                    print(f"警告: 跳过无效TLE {name} (行 {i+1}): {e}", file=sys.stderr)
                i += 3
            else:
                i += 1
    
    return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='TLE Parser using Skyfield - 精确解析和分析卫星轨道',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 解析单个TLE
  %(prog)s --line1 "1 25544U ..." --line2 "2 25544  51.6..."
  
  # 从文件读取并显示摘要
  %(prog)s --file tles/starlink_DTC_tle.txt --summary
  
  # 计算未来24小时的轨道
  %(prog)s --file tles/Satnet_DTC.txt --compute --hours 24 --points 100
  
  # 计算对Toronto的可见性
  %(prog)s --file tles/starlink_DTC_tle.txt --location 43.69 -79.37 --hours 48
  
  # 输出JSON格式
  %(prog)s --line1 "..." --line2 "..." --json
        """
    )
    
    # 输入选项
    input_group = parser.add_argument_group('输入选项')
    input_group.add_argument('--line1', type=str, help='TLE第一行')
    input_group.add_argument('--line2', type=str, help='TLE第二行')
    input_group.add_argument('--name', type=str, help='卫星名称')
    input_group.add_argument('--file', '-f', type=str, help='从文件读取TLE')
    
    # 输出选项
    output_group = parser.add_argument_group('输出选项')
    output_group.add_argument('--json', action='store_true', help='输出JSON格式')
    output_group.add_argument('--summary', '-s', action='store_true', help='只显示摘要')
    output_group.add_argument('--first', action='store_true', help='只处理第一个卫星')
    
    # 计算选项
    compute_group = parser.add_argument_group('轨道计算选项')
    compute_group.add_argument('--compute', '-c', action='store_true', help='计算轨道轨迹')
    compute_group.add_argument('--hours', type=float, default=24, help='计算时长（小时，默认24）')
    compute_group.add_argument('--points', type=int, default=100, help='轨迹点数（默认100）')
    compute_group.add_argument('--start', type=str, help='起始时间 (ISO格式: 2025-10-08T12:00:00)')
    
    # 可见性计算
    visibility_group = parser.add_argument_group('可见性计算选项')
    visibility_group.add_argument('--location', nargs=2, type=float, metavar=('LAT', 'LON'),
                                 help='观测点坐标（纬度 经度，例如: 43.69 -79.37）')
    visibility_group.add_argument('--min-elevation', type=float, default=10.0,
                                 help='最小仰角（度，默认10）')
    
    args = parser.parse_args()
    
    # 读取TLE数据
    tles = []
    
    if args.file:
        try:
            tles = parse_tle_file(args.file)
            print(f"✓ 成功从文件读取 {len(tles)} 个TLE\n", file=sys.stderr)
        except Exception as e:
            print(f"✗ 错误: 读取文件失败 - {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.line1 and args.line2:
        try:
            tle = TLEAnalyzer(args.line1, args.line2, args.name)
            tles.append(tle)
        except Exception as e:
            print(f"✗ 错误: 解析TLE失败 - {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(0)
    
    if not tles:
        print("✗ 错误: 没有找到有效的TLE数据", file=sys.stderr)
        sys.exit(1)
    
    # 只处理第一个
    if args.first:
        tles = tles[:1]
    
    # 解析起始时间
    start_time = None
    if args.start:
        try:
            start_time = datetime.fromisoformat(args.start.replace('Z', '+00:00'))
        except Exception as e:
            print(f"✗ 警告: 无效的起始时间格式 - {e}", file=sys.stderr)
    
    # 输出结果
    for i, tle in enumerate(tles, 1):
        if args.json:
            # JSON输出
            output = tle.to_dict()
            
            # 添加轨道计算
            if args.compute:
                track = tle.compute_ground_track(start_time, args.hours, args.points)
                output['ground_track'] = track
            
            # 添加可见性计算
            if args.location:
                lat, lon = args.location
                passes = tle.compute_passes_over_location(lat, lon, start_time, 
                                                         args.hours, args.min_elevation)
                output['passes'] = passes
                output['observer_location'] = {'latitude': lat, 'longitude': lon}
            
            if len(tles) == 1:
                print(json.dumps(output, indent=2, ensure_ascii=False))
            else:
                if i == 1:
                    print("[")
                print(json.dumps(output, indent=2, ensure_ascii=False), end='')
                if i < len(tles):
                    print(",")
                else:
                    print("\n]")
        
        elif args.summary:
            # 摘要输出
            print(f"{i}. {tle.name}")
            print(f"   编目号: {tle.catalog_number}, 类型: {tle.get_orbit_type()}")
            print(f"   高度: {tle.mean_altitude_km:.1f} km, 倾角: {tle.inclination_deg:.2f}°, 周期: {tle.period_minutes:.2f} min")
            print(f"   历元: {tle.epoch_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # 如果要计算可见性
            if args.location:
                lat, lon = args.location
                passes = tle.compute_passes_over_location(lat, lon, start_time, 
                                                         args.hours, args.min_elevation)
                print(f"   过顶次数 ({args.hours}h): {len(passes)} 次")
                if passes:
                    best_pass = max(passes, key=lambda p: p.get('peak_elevation_deg', 0))
                    print(f"   最佳仰角: {best_pass.get('peak_elevation_deg', 0):.1f}° "
                          f"@ {best_pass.get('peak_time', 'N/A')[:19]}")
            
            if i < len(tles):
                print()
        
        else:
            # 详细输出
            if len(tles) > 1:
                print(f"\n{'='*80}")
                print(f"第 {i}/{len(tles)} 个卫星")
                print(f"{'='*80}\n")
            
            print(tle)
            
            # 计算轨道轨迹
            if args.compute:
                print(f"\n【轨道计算】({args.hours} 小时, {args.points} 个采样点)")
                track = tle.compute_ground_track(start_time, args.hours, args.points)
                print(f"  起始时间: {track[0]['time']}")
                print(f"  结束时间: {track[-1]['time']}")
                print(f"  起始位置: 纬度 {track[0]['latitude_deg']:.4f}°, "
                      f"经度 {track[0]['longitude_deg']:.4f}°, "
                      f"高度 {track[0]['altitude_km']:.2f} km")
                print(f"  结束位置: 纬度 {track[-1]['latitude_deg']:.4f}°, "
                      f"经度 {track[-1]['longitude_deg']:.4f}°, "
                      f"高度 {track[-1]['altitude_km']:.2f} km")
            
            # 计算可见性
            if args.location:
                lat, lon = args.location
                print(f"\n【可见性分析】")
                print(f"  观测点: 纬度 {lat:.4f}°, 经度 {lon:.4f}°")
                print(f"  时间范围: {args.hours} 小时")
                print(f"  最小仰角: {args.min_elevation}°")
                
                passes = tle.compute_passes_over_location(lat, lon, start_time, 
                                                         args.hours, args.min_elevation)
                print(f"  过顶次数: {len(passes)}")
                
                if passes:
                    print(f"\n  过顶详情:")
                    for j, p in enumerate(passes[:10], 1):  # 最多显示前10次
                        print(f"    {j}. 升起: {p['rise_time'][11:19]}, "
                              f"最高: {p['peak_time'][11:19]} "
                              f"(仰角 {p['peak_elevation_deg']:.1f}°, "
                              f"方位 {p['peak_azimuth_deg']:.1f}°), "
                              f"落下: {p['set_time'][11:19]}, "
                              f"持续 {p['duration_seconds']/60:.1f} 分钟")
                    
                    if len(passes) > 10:
                        print(f"    ... 还有 {len(passes) - 10} 次过顶")
            
            if i < len(tles):
                print()


if __name__ == '__main__':
    main()
