#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置验证脚本 - 检查四个场景的配置是否正确
"""

import sys
from pathlib import Path
import importlib.util

SCRIPT_DIR = Path(__file__).parent.absolute()

SCENARIOS = {
    'toronto_single': 'test/config_toronto_single.py',
    'toronto_constellation': 'test/config_toronto_constellation.py',
    'shanghai_single': 'test/config_shanghai_single.py',
    'shanghai_constellation': 'test/config_shanghai_constellation.py',
}


def load_config_from_file(config_path):
    """从文件路径加载配置"""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def load_base_config():
    """加载基础配置文件"""
    config_path = SCRIPT_DIR / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("base_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def verify_scenario(scenario_name, config_path):
    """验证单个场景配置"""
    print(f"\n{'='*70}")
    print(f"验证场景: {scenario_name}")
    print(f"配置文件: {config_path}")
    print(f"{'='*70}")
    
    # 加载并合并配置
    base_config = load_base_config()
    test_config = load_config_from_file(config_path)
    config = base_config.copy()
    config.update(test_config)
    
    # 检查关键配置
    enable_constellation = config.get("enable_constellation")
    tle_catalog_path = config.get("tle_catalog_path")
    tle_name = config.get("tle_name")
    tle_lines = config.get("tle_lines")
    ref_lat = config.get("ref_lat_deg")
    ref_lon = config.get("ref_lon_deg")
    radio_map = config.get("radio_map_mat_path")
    
    print(f"\n配置详情:")
    print(f"  轨道模式:")
    print(f"    enable_constellation: {enable_constellation}")
    
    errors = []
    warnings = []
    
    if 'constellation' in scenario_name:
        # 星座场景检查
        if not enable_constellation:
            errors.append("❌ 星座场景应该设置 enable_constellation=True")
        else:
            print(f"    ✓ 星座模式已启用")
        
        if not tle_catalog_path:
            errors.append("❌ 星座场景应该设置 tle_catalog_path")
        else:
            print(f"    ✓ TLE目录: {tle_catalog_path}")
        
        if tle_name is not None:
            errors.append("❌ 星座场景应该设置 tle_name=None")
        else:
            print(f"    ✓ 单星TLE已禁用 (tle_name=None)")
            
        if tle_lines is not None:
            errors.append("❌ 星座场景应该设置 tle_lines=None")
        else:
            print(f"    ✓ 单星TLE已禁用 (tle_lines=None)")
    else:
        # 单星场景检查
        if enable_constellation:
            errors.append("❌ 单星场景应该设置 enable_constellation=False")
        else:
            print(f"    ✓ 单星模式已启用")
        
        if tle_catalog_path is not None:
            errors.append("❌ 单星场景应该设置 tle_catalog_path=None")
        else:
            print(f"    ✓ TLE目录已禁用 (tle_catalog_path=None)")
        
        if not tle_name:
            errors.append("❌ 单星场景应该设置 tle_name")
        else:
            print(f"    ✓ 卫星名称: {tle_name}")
        
        if not tle_lines:
            errors.append("❌ 单星场景应该设置 tle_lines")
        else:
            print(f"    ✓ TLE已配置 ({len(tle_lines)} 行)")
    
    # 检查地理位置
    print(f"\n  地理位置:")
    print(f"    ref_lat_deg: {ref_lat}")
    print(f"    ref_lon_deg: {ref_lon}")
    
    if 'toronto' in scenario_name:
        if abs(ref_lat - 43.69) > 0.1 or abs(ref_lon - (-79.37)) > 0.1:
            warnings.append("⚠️  地理坐标似乎不在Toronto附近")
        else:
            print(f"    ✓ Toronto坐标正确")
    elif 'shanghai' in scenario_name:
        if abs(ref_lat - 31.23) > 0.1 or abs(ref_lon - 121.47) > 0.1:
            warnings.append("⚠️  地理坐标似乎不在Shanghai附近")
        else:
            print(f"    ✓ Shanghai坐标正确")
    
    # 检查Radio Map
    print(f"\n  Radio Map:")
    print(f"    {radio_map}")
    
    if 'toronto' in scenario_name and 'Toronto' not in radio_map:
        warnings.append("⚠️  Radio Map路径似乎不匹配Toronto")
    elif 'shanghai' in scenario_name and 'Shanghai' not in radio_map:
        warnings.append("⚠️  Radio Map路径似乎不匹配Shanghai")
    else:
        print(f"    ✓ Radio Map路径匹配")
    
    # 输出结果
    print(f"\n{'='*70}")
    if errors:
        print("❌ 配置错误:")
        for error in errors:
            print(f"  {error}")
        print(f"{'='*70}")
        return False
    elif warnings:
        print("⚠️  警告:")
        for warning in warnings:
            print(f"  {warning}")
        print(f"{'='*70}")
        return True
    else:
        print("✅ 配置检查通过!")
        print(f"{'='*70}")
        return True


def main():
    print("\n" + "="*70)
    print("场景配置验证工具")
    print("="*70)
    
    all_passed = True
    for scenario_name, config_path in SCENARIOS.items():
        passed = verify_scenario(scenario_name, config_path)
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ 所有场景配置验证通过!")
    else:
        print("❌ 部分场景配置存在错误，请修复")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
