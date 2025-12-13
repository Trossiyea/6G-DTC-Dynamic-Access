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
  python run_test.py --verify  # 验证环境和模块

Requirements:
  - Phase 1-7 模块化重构已完成
  - code/link/ 链路层模块可用
  - code/simulation/ 仿真引擎可用
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


def verify_environment():
    """验证环境和关键模块是否可用（Phase 6-7 增强）"""
    print("\n" + "="*70)
    print("🔍 环境验证 (Phase 6-7 模块化架构)")
    print("="*70 + "\n")

    checks_passed = 0
    checks_total = 0

    # Check 1: Python version
    checks_total += 1
    print("  [1/9] Python 版本...", end=" ")
    if sys.version_info >= (3, 9):
        print("✓ OK", f"({sys.version.split()[0]})")
        checks_passed += 1
    else:
        print(f"✗ FAIL (需要 Python 3.9+, 当前 {sys.version.split()[0]})")

    # Check 2: Core dependencies
    checks_total += 1
    print("  [2/9] 核心依赖...", end=" ")
    try:
        import numpy as np
        import scipy
        import matplotlib
        print(f"✓ OK (numpy {np.__version__})")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ FAIL ({e})")

    # Check 3: Configuration system (Phase 4)
    checks_total += 1
    print("  [3/9] 配置系统 (Phase 4)...", end=" ")
    try:
        from code.config import load_scenario_config, CONFIG
        print("✓ OK")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ FAIL ({e})")

    # Check 4: NTN module (Phase 6)
    checks_total += 1
    print("  [4/9] NTN 模块 (Phase 6)...", end=" ")
    try:
        from code.ntn import (
            OrbitModel, ConstellationOrbit, sample_3gpp_ntn_fading,
            beam_gain_db, fspl_db, map_xy_to_latlon, BeamManager
        )
        print("✓ OK (5 个子模块)")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ FAIL ({e})")

    # Check 5: Link module (Phase 7)
    checks_total += 1
    print("  [5/9] 链路层模块 (Phase 7)...", end=" ")
    try:
        from code.link import (
            MCS, HarqManager, HarqManagerFull,
            choose_mcs_from_sinr, calc_tbs_bits,
            eff_sinr_eesm_db, OLLA
        )
        print("✓ OK (8 个子模块)")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ FAIL ({e})")

    # Check 6: Backward compatibility
    checks_total += 1
    print("  [6/9] 向后兼容性...", end=" ")
    try:
        from code.link_adapt import MCS, choose_mcs_from_sinr
        from code.csi import get_nr_cqi_table
        from code.harq import HarqManager
        from code.orbit import OrbitModel
        from code.constellation import ConstellationOrbit
        from code.ntn_channel import sample_3gpp_ntn_fading
        print("✓ OK")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ FAIL ({e})")

    # Check 7: Scheduler modules (Phase 3)
    checks_total += 1
    print("  [7/9] 调度器模块 (Phase 3)...", end=" ")
    try:
        from code.scheduler.baseline import pf_schedule_baseline
        from code.scheduler.radiomap import pf_schedule_radiomap_blocks
        print("✓ OK")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ FAIL ({e})")

    # Check 8: Simulation engines (Phase 5)
    checks_total += 1
    print("  [8/9] 仿真引擎 (Phase 5)...", end=" ")
    try:
        from code.simulation.engine import SimulationEngine
        from code.simulation.constellation_engine import ConstellationEngine
        print("✓ OK")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ FAIL ({e})")

    # Check 9: Main API
    checks_total += 1
    print("  [9/9] 主接口 API...", end=" ")
    try:
        from code import main as main_module
        assert hasattr(main_module, 'run_once'), "run_once() 不存在"
        assert hasattr(main_module, 'run_constellation'), "run_constellation() 不存在"
        print("✓ OK")
        checks_passed += 1
    except (ImportError, AssertionError) as e:
        print(f"✗ FAIL ({e})")

    # Summary
    print(f"\n{'='*70}")
    if checks_passed == checks_total:
        print(f"✅ 环境验证通过 ({checks_passed}/{checks_total})")
    else:
        print(f"⚠️  环境验证失败 ({checks_passed}/{checks_total})")
        print(f"请修复失败的检查项后重试")
    print("="*70 + "\n")

    return checks_passed == checks_total


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

    # 延迟导入以避免启动时开销
    from code.config import load_scenario_config
    from code import main as main_module

    # 加载场景配置（自动合并默认配置）
    print("📝 [1/3] 加载配置文件...")
    try:
        config = load_scenario_config(config_path)
        print("✓ 配置加载完成\n")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        raise

    # 根据配置决定调用哪个函数
    sim_start_time = time.time()
    try:
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
    except Exception as e:
        print(f"\n✗ 仿真运行失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    sim_elapsed = time.time() - sim_start_time
    print(f"\n✓ 仿真完成 (耗时: {sim_elapsed:.1f}秒)\n")

    # 输出结果摘要
    print("📊 [3/3] 生成结果报告...")
    total_elapsed = time.time() - start_time

    print(f"\n{'-'*70}")
    print(f"✅ 场景 [{scenario_name}] 完成")
    print(f"{'-'*70}")
    print(f"📈 性能结果:")

    # 基本 SE 指标
    try:
        baseline_se = results.get('avg_se_baseline_default', results.get('bl_se_mean', 0))
        radiomap_se = results.get('avg_se_radiomap', results.get('rm_se_mean', 0))
        improvement = results.get('improvement_vs_default_pct',
                                 (radiomap_se/baseline_se - 1) * 100 if baseline_se > 0 else 0)

        print(f"  基线平均SE:     {baseline_se:.4f} bits/s/Hz")
        print(f"  RadioMap平均SE: {radiomap_se:.4f} bits/s/Hz")
        print(f"  提升百分比:     {improvement:+.2f}%")
    except Exception as e:
        print(f"  ⚠️  SE 指标解析失败: {e}")

    # 吞吐量指标
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
    except Exception as e:
        print(f"  ℹ️  吞吐量指标不可用")

    # HARQ 统计 (Phase 7)
    try:
        harq_stats_rm = results.get('harq_stats_radiomap', {})
        if harq_stats_rm and isinstance(harq_stats_rm, dict):
            ack = harq_stats_rm.get('ack_count', 0)
            nack = harq_stats_rm.get('nack_count', 0)
            if ack + nack > 0:
                print(f"  HARQ统计(RM):")
                print(f"    ACK:  {ack}")
                print(f"    NACK: {nack}")
                print(f"    初传成功率: {harq_stats_rm.get('initial_ack_count', 0) / (harq_stats_rm.get('initial_ack_count', 0) + harq_stats_rm.get('initial_nack_count', 1)) * 100:.1f}%")
                avg_retx = harq_stats_rm.get('avg_retx_per_acked', 0)
                if avg_retx > 0:
                    print(f"    平均重传次数: {avg_retx:.2f}")
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
        description='运行多场景测试 (Phase 7 优化版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_test.py --scenario toronto_single
  python run_test.py --scenario shanghai_constellation
  python run_test.py --all
  python run_test.py -s toronto_single -s shanghai_single
  python run_test.py --verify  # 验证环境
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

    parser.add_argument(
        '--verify',
        action='store_true',
        help='验证环境和模块（Phase 7 增强）'
    )

    args = parser.parse_args()

    # 验证环境
    if args.verify:
        success = verify_environment()
        return 0 if success else 1

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
    print(f"# 测试批次概览 (Phase 7 模块化架构)")
    print(f"{'#'*70}")
    print(f"📋 计划运行 {len(scenarios_to_run)} 个场景")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    all_results = {}
    start_time_all = time.time()
    failures = []

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
            failures.append(scenario_name)
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
            try:
                baseline_se = results.get('avg_se_baseline_default', results.get('bl_se_mean', 0))
                radiomap_se = results.get('avg_se_radiomap', results.get('rm_se_mean', 0))
                improvement = results.get('improvement_vs_default_pct',
                                         (radiomap_se/baseline_se - 1) * 100 if baseline_se > 0 else 0)

                print(f"  基线SE:    {baseline_se:.4f} bits/s/Hz")
                print(f"  RadioMap:  {radiomap_se:.4f} bits/s/Hz")
                print(f"  提升:      {improvement:+.2f}%")

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
            except Exception as e:
                print(f"  ⚠️  结果解析失败: {e}")

        print(f"\n{'='*70}")
        print(f"\n⏱️  总计耗时: {total_elapsed_all:.1f}秒 ({total_elapsed_all/60:.1f}分钟)")
        print(f"✅ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"✅ 成功运行 {len(all_results)}/{len(scenarios_to_run)} 个场景")
        if failures:
            print(f"⚠️  失败场景: {', '.join(failures)}")
        print(f"{'='*70}\n")
    elif len(all_results) == 1:
        print(f"\n⏱️  总计耗时: {total_elapsed_all:.1f}秒 ({total_elapsed_all/60:.1f}分钟)")
        print(f"✅ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Return non-zero if any failures
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
