#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radio Map分辨率影响对比测试

该脚本专门用于评估不同Radio Map空间分辨率对系统性能的影响。
测试125m vs 150m两种分辨率在上海和多伦多单星场景下的性能差异。

使用方法:
  python run_resolution_comparison.py --city toronto    # 仅测试多伦多
  python run_resolution_comparison.py --city shanghai   # 仅测试上海
  python run_resolution_comparison.py --all             # 测试所有城市
"""

import argparse
import sys
import os
import importlib.util
from pathlib import Path
import json
from datetime import datetime

# 将当前目录和 code 目录添加到 Python 路径
SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR / "code"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# 分辨率对比场景配置
RESOLUTION_SCENARIOS = {
    'toronto': {
        '125m': {
            'config_path': 'test/config_toronto_single_125m.py',
            'description': 'Toronto - 125m分辨率',
            'resolution': 125,
        },
        '150m': {
            'config_path': 'test/config_toronto_single_150m.py',
            'description': 'Toronto - 150m分辨率',
            'resolution': 150,
        }
    },
    'shanghai': {
        '125m': {
            'config_path': 'test/config_shanghai_single_125m.py',
            'description': 'Shanghai - 125m分辨率',
            'resolution': 125,
        },
        '150m': {
            'config_path': 'test/config_shanghai_single_150m.py',
            'description': 'Shanghai - 150m分辨率',
            'resolution': 150,
        }
    }
}


def load_config_from_file(config_path):
    """从文件路径加载配置"""
    spec = importlib.util.spec_from_file_location("test_config", config_path)
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


def load_main_module():
    """加载main模块"""
    main_path = SCRIPT_DIR / "code" / "main.py"
    spec = importlib.util.spec_from_file_location("main_module", main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['main_module'] = module
    spec.loader.exec_module(module)
    return module


def run_scenario(scenario_name, config_path, resolution):
    """运行单个场景"""
    print(f"\n{'='*70}")
    print(f"运行场景: {scenario_name}")
    print(f"分辨率:   {resolution}m")
    print(f"配置文件: {config_path}")
    print(f"{'='*70}\n")
    
    # 加载基础配置和场景配置
    base_config = load_base_config()
    test_config = load_config_from_file(config_path)
    
    # 合并配置（场景配置覆盖基础配置）
    config = base_config.copy()
    config.update(test_config)
    
    # 加载并运行 main 模块
    main_module = load_main_module()
    
    # 运行单星模式
    results = main_module.run_once(config)
    
    # 输出结果摘要
    print(f"\n{'-'*70}")
    print(f"场景 [{scenario_name} - {resolution}m] 结果:")
    print(f"  基线平均SE:     {results['avg_se_baseline_default']:.4f} bits/s/Hz")
    print(f"  RadioMap平均SE: {results['avg_se_radiomap']:.4f} bits/s/Hz")
    print(f"  提升百分比:     {results['improvement_vs_default_pct']:.2f}%")
    print(f"{'-'*70}\n")
    
    return results


def compare_resolutions(city, results_125m, results_150m):
    """对比两种分辨率的结果"""
    print(f"\n{'='*70}")
    print(f"{city.upper()} - 分辨率影响对比分析")
    print(f"{'='*70}\n")
    
    # 提取关键指标
    baseline_125 = results_125m['avg_se_baseline_default']
    radiomap_125 = results_125m['avg_se_radiomap']
    gain_125 = results_125m['improvement_vs_default_pct']
    
    baseline_150 = results_150m['avg_se_baseline_default']
    radiomap_150 = results_150m['avg_se_radiomap']
    gain_150 = results_150m['improvement_vs_default_pct']
    
    print("📊 性能对比:")
    print(f"\n  {'指标':<25} {'125m':<15} {'150m':<15} {'差异':<15}")
    print(f"  {'-'*70}")
    print(f"  {'基线SE (bits/s/Hz)':<25} {baseline_125:<15.4f} {baseline_150:<15.4f} {baseline_150-baseline_125:+.4f}")
    print(f"  {'RadioMap SE (bits/s/Hz)':<25} {radiomap_125:<15.4f} {radiomap_150:<15.4f} {radiomap_150-radiomap_125:+.4f}")
    print(f"  {'提升百分比 (%)':<25} {gain_125:<15.2f} {gain_150:<15.2f} {gain_150-gain_125:+.2f}")
    
    # 计算分辨率带来的相对改善
    gain_improvement = ((gain_150 - gain_125) / max(gain_125, 1e-9)) * 100.0 if gain_125 > 0 else 0.0
    
    print(f"\n💡 关键发现:")
    print(f"  • 150m分辨率相比125m分辨率:")
    print(f"    - RadioMap SE变化: {radiomap_150-radiomap_125:+.4f} bits/s/Hz ({((radiomap_150/radiomap_125-1)*100):+.2f}%)")
    print(f"    - 提升效果变化: {gain_150-gain_125:+.2f} 百分点")
    
    if abs(gain_150 - gain_125) < 1.0:
        print(f"    - 结论: 分辨率差异对性能影响较小 (<1%)")
    elif gain_150 > gain_125:
        print(f"    - 结论: 更高分辨率(150m)带来更大性能提升")
    else:
        print(f"    - 结论: 较低分辨率(125m)已足够，更高分辨率收益有限")
    
    # 数据密度对比
    if 'R_xyz_dbm' in results_125m and 'R_xyz_dbm' in results_150m:
        import numpy as np
        shape_125 = results_125m['R_xyz_dbm'].shape
        shape_150 = results_150m['R_xyz_dbm'].shape
        
        total_points_125 = shape_125[0] * shape_125[1] * shape_125[2]
        total_points_150 = shape_150[0] * shape_150[1] * shape_150[2]
        
        print(f"\n📐 Radio Map尺寸对比:")
        print(f"  • 125m分辨率: {shape_125[0]} × {shape_125[1]} × {shape_125[2]} = {total_points_125:,} 点")
        print(f"  • 150m分辨率: {shape_150[0]} × {shape_150[1]} × {shape_150[2]} = {total_points_150:,} 点")
        print(f"  • 数据量比例: 150m/125m = {total_points_150/total_points_125:.2f}x")
    
    print(f"\n{'='*70}\n")
    
    return {
        'city': city,
        'baseline_125m': baseline_125,
        'baseline_150m': baseline_150,
        'radiomap_125m': radiomap_125,
        'radiomap_150m': radiomap_150,
        'gain_125m': gain_125,
        'gain_150m': gain_150,
        'gain_delta': gain_150 - gain_125,
        'relative_gain_improvement': gain_improvement,
    }


def save_comparison_report(comparison_results, output_path='results/resolution_comparison.json'):
    """保存对比结果到JSON文件"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Radio Map分辨率影响对比测试',
        'comparisons': comparison_results,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 对比报告已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Radio Map分辨率影响对比测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_resolution_comparison.py --city toronto       # 测试多伦多
  python run_resolution_comparison.py --city shanghai      # 测试上海
  python run_resolution_comparison.py --all                # 测试所有城市
  python run_resolution_comparison.py --all --save-report  # 测试并保存报告
        """
    )
    
    parser.add_argument(
        '--city',
        choices=['toronto', 'shanghai'],
        help='选择要测试的城市'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='测试所有城市'
    )
    
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='保存对比报告到JSON文件'
    )
    
    parser.add_argument(
        '--output',
        default='results/resolution_comparison.json',
        help='报告输出路径（默认: results/resolution_comparison.json）'
    )
    
    args = parser.parse_args()
    
    # 确定要测试的城市
    if args.all:
        cities_to_test = ['toronto', 'shanghai']
    elif args.city:
        cities_to_test = [args.city]
    else:
        parser.print_help()
        return 1
    
    # 运行测试
    all_comparison_results = []
    
    for city in cities_to_test:
        print(f"\n{'#'*70}")
        print(f"# 开始测试: {city.upper()}")
        print(f"{'#'*70}")
        
        scenarios = RESOLUTION_SCENARIOS[city]
        
        # 运行125m分辨率场景
        scenario_125 = scenarios['125m']
        try:
            results_125m = run_scenario(
                city,
                scenario_125['config_path'],
                scenario_125['resolution']
            )
        except Exception as e:
            print(f"❌ 错误: {city} 125m场景运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 运行150m分辨率场景
        scenario_150 = scenarios['150m']
        try:
            results_150m = run_scenario(
                city,
                scenario_150['config_path'],
                scenario_150['resolution']
            )
        except Exception as e:
            print(f"❌ 错误: {city} 150m场景运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 对比分析
        comparison = compare_resolutions(city, results_125m, results_150m)
        all_comparison_results.append(comparison)
    
    # 生成总结报告
    if len(all_comparison_results) > 0:
        print(f"\n{'='*70}")
        print("📋 总体分析摘要")
        print(f"{'='*70}\n")
        
        for comp in all_comparison_results:
            print(f"{comp['city'].upper()}:")
            print(f"  125m分辨率 - RadioMap提升: {comp['gain_125m']:.2f}%")
            print(f"  150m分辨率 - RadioMap提升: {comp['gain_150m']:.2f}%")
            print(f"  提升差异: {comp['gain_delta']:+.2f} 百分点")
            print()
        
        # 保存报告
        if args.save_report:
            save_comparison_report(all_comparison_results, args.output)
        
        print(f"{'='*70}\n")
        print("✅ 所有测试完成!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
