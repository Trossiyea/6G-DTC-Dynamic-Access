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
from pathlib import Path
import time
from datetime import datetime, timedelta

# 将当前目录和 code 目录添加到 Python 路径
SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR / "code"

# 添加项目根目录到 sys.path
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 添加 code 目录到 sys.path，这样 main.py 中的导入就能正常工作
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# 导入重构后的配置系统
from code.config import load_scenario_config
from code import main as main_module

# 场景配置映射
SCENARIOS = {
    # Original scenarios
    'toronto_single': 'test/config_toronto_single.py',
    'toronto_constellation': 'test/config_toronto_constellation.py',
    'shanghai_single': 'test/config_shanghai_single.py',
    'shanghai_constellation': 'test/config_shanghai_constellation.py',
    
    # Resolution comparison scenarios
    'toronto_125m': 'test/config_toronto_single_125m.py',
    'toronto_150m': 'test/config_toronto_single_150m.py',
    'shanghai_125m': 'test/config_shanghai_single_125m.py',
    'shanghai_150m': 'test/config_shanghai_single_150m.py',
}




def run_scenario(scenario_name, config_path, current_idx=None, total_count=None):
    """运行单个场景"""
    start_time = time.time()
    
    # 进度信息
    progress_info = ""
    if current_idx is not None and total_count is not None:
        progress_info = f" [{current_idx}/{total_count}]"
    
    print(f"\n{'='*70}")
    print(f"🚀 运行场景{progress_info}: {scenario_name}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📄 配置文件: {config_path}")
    print(f"{'='*70}\n")
    
    # 加载场景配置（自动合并默认配置）
    print("📝 [1/3] 加载配置文件...")
    config = load_scenario_config(config_path)
    print("✓ 配置加载完成\n")
    
    # 根据配置决定调用哪个函数
    sim_start_time = time.time()
    if bool(config.get("enable_constellation", False)):
        # 星座模式
        print("🛰️  [2/3] 运行仿真 (星座模式)...")
        print(f"ℹ️  模式: ConstellationOrbit")
        print(f"ℹ️  UE数量: {config.get('N_UE', 'N/A')}")
        print(f"ℹ️  TTI数量: {config.get('T', 'N/A')}")
        print(f"ℹ️  中心频率: {config.get('carrier_freq_GHz', 'N/A')} GHz")
        print()
        results = main_module.run_constellation(config)
    else:
        # 单星模式
        print("🛰️  [2/3] 运行仿真 (单星模式)...")
        print(f"ℹ️  模式: OrbitModel")
        print(f"ℹ️  UE数量: {config.get('N_UE', 'N/A')}")
        print(f"ℹ️  TTI数量: {config.get('T', 'N/A')}")
        print(f"ℹ️  中心频率: {config.get('carrier_freq_GHz', 'N/A')} GHz")
        print()
        results = main_module.run_once(config)

    sim_elapsed = time.time() - sim_start_time
    print(f"\n✓ 仿真完成 (耗时: {sim_elapsed:.1f}秒)\n")

    # 输出结果摘要
    print("📊 [3/3] 生成结果报告...")
    total_elapsed = time.time() - start_time
    
    print(f"\n{'-'*70}")
    print(f"✅ 场景 [{scenario_name}] 完成")
    print(f"{'-'*70}")
    print(f"📈 性能结果:")
    print(f"  基线平均SE:     {results['avg_se_baseline_default']:.4f} bits/s/Hz")
    print(f"  RadioMap平均SE: {results['avg_se_radiomap']:.4f} bits/s/Hz")
    print(f"  提升百分比:     {results['improvement_vs_default_pct']:+.2f}%")
    # 追加吞吐量指标
    try:
        bw_hz = float(results.get('system_bandwidth_hz', 0.0))
        if bw_hz > 0:
            print(f"  系统带宽:       {bw_hz/1e6:.3f} MHz")
        tb = results.get('total_throughput_baseline_bps', None)
        tm = results.get('total_throughput_radiomap_bps', None)
        if tb is not None and tm is not None:
            print(f"  基线总吞吐量:   {tb/1e6:.3f} Mbps")
            print(f"  RadioMap总吞吐量: {tm/1e6:.3f} Mbps")
        aub = results.get('avg_ue_throughput_baseline_bps', None)
        aum = results.get('avg_ue_throughput_radiomap_bps', None)
        if aub is not None and aum is not None:
            print(f"  UE平均吞吐量(基线): {aub/1e6:.3f} Mbps/UE")
            print(f"  UE平均吞吐量(RM):  {aum/1e6:.3f} Mbps/UE")
    except Exception:
        pass
    print(f"\n⏱️  运行时间:")
    print(f"  总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
    print(f"  仿真时间: {sim_elapsed:.1f}秒")
    print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    print(f"\n{'#'*70}")
    print(f"# 测试批次概览")
    print(f"{'#'*70}")
    print(f"📋 计划运行 {len(scenarios_to_run)} 个场景")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")
    
    all_results = {}
    start_time_all = time.time()
    
    for idx, scenario_name in enumerate(scenarios_to_run, 1):
        config_path = SCENARIOS[scenario_name]
        try:
            results = run_scenario(scenario_name, config_path, idx, len(scenarios_to_run))
            all_results[scenario_name] = results
            
            # 显示整体进度
            elapsed = time.time() - start_time_all
            avg_time_per_scenario = elapsed / idx
            remaining = (len(scenarios_to_run) - idx) * avg_time_per_scenario
            eta = datetime.now() + timedelta(seconds=remaining)
            
            print(f"📊 整体进度: {idx}/{len(scenarios_to_run)} ({idx*100//len(scenarios_to_run)}%)")
            if idx < len(scenarios_to_run):
                print(f"⏱️  预计剩余时间: {remaining/60:.1f}分钟")
                print(f"🎯 预计完成时间: {eta.strftime('%H:%M:%S')}")
            print()
            
        except Exception as e:
            print(f"❌ 错误: 场景 [{scenario_name}] 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总所有结果
    total_elapsed_all = time.time() - start_time_all
    
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("📊 所有场景结果汇总")
        print(f"{'='*70}")
        for scenario_name, results in all_results.items():
            print(f"\n{scenario_name}:")
            print(f"  基线SE:    {results['avg_se_baseline_default']:.4f} bits/s/Hz")
            print(f"  RadioMap:  {results['avg_se_radiomap']:.4f} bits/s/Hz")
            print(f"  提升:      {results['improvement_vs_default_pct']:+.2f}%")
            try:
                bw_hz = float(results.get('system_bandwidth_hz', 0.0))
                if bw_hz > 0:
                    print(f"  带宽:      {bw_hz/1e6:.3f} MHz")
                tb = results.get('total_throughput_baseline_bps', None)
                tm = results.get('total_throughput_radiomap_bps', None)
                if tb is not None and tm is not None:
                    print(f"  基线吞吐量: {tb/1e6:.3f} Mbps")
                    print(f"  RM吞吐量:  {tm/1e6:.3f} Mbps")
                aub = results.get('avg_ue_throughput_baseline_bps', None)
                aum = results.get('avg_ue_throughput_radiomap_bps', None)
                if aub is not None and aum is not None:
                    print(f"  UE均吞吐(基线): {aub/1e6:.3f} Mbps/UE")
                    print(f"  UE均吞吐(RM):  {aum/1e6:.3f} Mbps/UE")
            except Exception:
                pass
        print(f"\n{'='*70}")
        print(f"\n⏱️  总计耗时: {total_elapsed_all:.1f}秒 ({total_elapsed_all/60:.1f}分钟)")
        print(f"✅ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"✅ 成功运行 {len(all_results)}/{len(scenarios_to_run)} 个场景")
        print(f"{'='*70}\n")
    elif len(all_results) == 1:
        print(f"\n⏱️  总计耗时: {total_elapsed_all:.1f}秒 ({total_elapsed_all/60:.1f}分钟)")
        print(f"✅ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
