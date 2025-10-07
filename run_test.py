#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本 - 支持多场景配置切换
使用方法:
  python run_test.py --scenario toronto_single
  python run_test.py --scenario toronto_constellation
  python run_test.py --scenario shanghai_single
  python run_test.py --scenario shanghai_constellation
  python run_test.py --all  # 运行所有场景
"""

import argparse
import sys
import os
import importlib.util
from pathlib import Path

# 将当前目录和 code 目录添加到 Python 路径
SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR / "code"

# 添加项目根目录到 sys.path
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 添加 code 目录到 sys.path，这样 main.py 中的导入就能正常工作
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# 场景配置映射
SCENARIOS = {
    'toronto_single': 'test/config_toronto_single.py',
    'toronto_constellation': 'test/config_toronto_constellation.py',
    'shanghai_single': 'test/config_shanghai_single.py',
    'shanghai_constellation': 'test/config_shanghai_constellation.py',
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


def run_scenario(scenario_name, config_path):
    """运行单个场景"""
    print(f"\n{'='*70}")
    print(f"运行场景: {scenario_name}")
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
    
    # 根据配置决定调用哪个函数
    if bool(config.get("enable_constellation", False)):
        # 星座模式
        results = main_module.run_constellation(config)
        print(f"\nℹ️  使用星座模式 (ConstellationOrbit)")
    else:
        # 单星模式
        results = main_module.run_once(config)
        print(f"\nℹ️  使用单星模式 (OrbitModel)")
    
    # 输出结果摘要
    print(f"\n{'-'*70}")
    print(f"场景 [{scenario_name}] 结果:")
    print(f"  基线平均SE: {results['avg_se_baseline_default']:.4f} bits/s/Hz")
    print(f"  RadioMap平均SE: {results['avg_se_radiomap']:.4f} bits/s/Hz")
    print(f"  提升百分比: {results['improvement_vs_default_pct']:.2f}%")
    print(f"{'-'*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='运行多场景测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_test.py --scenario toronto_single
  python run_test.py --scenario shanghai_constellation
  python run_test.py --all
  python run_test.py -s toronto_single -s shanghai_single
        """
    )
    
    parser.add_argument(
        '-s', '--scenario',
        action='append',
        choices=list(SCENARIOS.keys()),
        help='选择要运行的场景（可多次指定）'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='运行所有场景'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用场景'
    )
    
    args = parser.parse_args()
    
    # 列出场景
    if args.list:
        print("\n可用场景:")
        for name, path in SCENARIOS.items():
            print(f"  - {name:30s} -> {path}")
        print()
        return 0
    
    # 确定要运行的场景
    if args.all:
        scenarios_to_run = list(SCENARIOS.keys())
    elif args.scenario:
        scenarios_to_run = args.scenario
    else:
        parser.print_help()
        return 1
    
    # 运行场景
    all_results = {}
    for scenario_name in scenarios_to_run:
        config_path = SCENARIOS[scenario_name]
        try:
            results = run_scenario(scenario_name, config_path)
            all_results[scenario_name] = results
        except Exception as e:
            print(f"错误: 场景 [{scenario_name}] 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总所有结果
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("所有场景结果汇总:")
        print(f"{'='*70}")
        for scenario_name, results in all_results.items():
            print(f"\n{scenario_name}:")
            print(f"  基线SE:    {results['avg_se_baseline_default']:.4f} bits/s/Hz")
            print(f"  RadioMap:  {results['avg_se_radiomap']:.4f} bits/s/Hz")
            print(f"  提升:      {results['improvement_vs_default_pct']:+.2f}%")
        print(f"\n{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
