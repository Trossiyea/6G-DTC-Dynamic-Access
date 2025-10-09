#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试结果可视化脚本

用于生成柱状图展示：
1. 四个主要场景 (toronto_single, toronto_constellation, shanghai_single, shanghai_constellation)
   的 Baseline vs RadioMap 性能对比
2. 分辨率对比结果 (125m vs 150m)

使用方法:
  # 可视化四个主场景（需要先运行测试）
  python visualize_results.py --scenarios
  
  # 可视化分辨率对比（需要先运行分辨率对比测试）
  python visualize_results.py --resolution
  
  # 两者都可视化
  python visualize_results.py --all
  
  # 从JSON文件读取数据并可视化
  python visualize_results.py --resolution --input results/resolution_comparison.json
"""

import argparse
import sys
import json
import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR / "code"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR / "output"

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


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


def run_scenario_for_visualization(scenario_name, config_path):
    """运行单个场景并返回结果"""
    print(f"  运行场景: {scenario_name}...")
    
    base_config = load_base_config()
    test_config = load_config_from_file(config_path)
    
    config = base_config.copy()
    config.update(test_config)
    
    main_module = load_main_module()
    
    if bool(config.get("enable_constellation", False)):
        results = main_module.run_constellation(config)
    else:
        results = main_module.run_once(config)
    
    return results


def visualize_four_scenarios(run_tests=True, save_data=True):
    """
    可视化四个主要场景的结果
    
    Args:
        run_tests: 是否运行测试（否则从已保存的数据读取）
        save_data: 是否保存结果数据
    """
    print("\n" + "="*70)
    print("📊 四场景性能对比可视化")
    print("="*70 + "\n")
    
    scenarios = {
        'toronto_single': {
            'name': 'Toronto\nSingle',
            'config': 'test/config_toronto_single.py',
            'color': '#3498db',  # 蓝色
        },
        'toronto_constellation': {
            'name': 'Toronto\nConstellation',
            'config': 'test/config_toronto_constellation.py',
            'color': '#9b59b6',  # 紫色
        },
        'shanghai_single': {
            'name': 'Shanghai\nSingle',
            'config': 'test/config_shanghai_single.py',
            'color': '#e74c3c',  # 红色
        },
        'shanghai_constellation': {
            'name': 'Shanghai\nConstellation',
            'config': 'test/config_shanghai_constellation.py',
            'color': '#f39c12',  # 橙色
        },
    }
    
    results_data = {}
    data_file = RESULTS_DIR / "four_scenarios_results.json"
    
    if run_tests:
        print("🔄 开始运行四个场景测试...\n")
        
        for key, info in scenarios.items():
            try:
                results = run_scenario_for_visualization(key, info['config'])
                results_data[key] = {
                    'baseline_se': float(results['avg_se_baseline_default']),
                    'radiomap_se': float(results['avg_se_radiomap']),
                    'improvement_pct': float(results['improvement_vs_default_pct']),
                }
                print(f"  ✓ {key}: Baseline={results_data[key]['baseline_se']:.4f}, "
                      f"RadioMap={results_data[key]['radiomap_se']:.4f}, "
                      f"提升={results_data[key]['improvement_pct']:.2f}%\n")
            except Exception as e:
                print(f"  ✗ {key} 运行失败: {e}\n")
                continue
        
        if save_data and results_data:
            with open(data_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'results': results_data
                }, f, indent=2)
            print(f"✅ 结果已保存至: {data_file}\n")
    
    else:
        # 从文件读取
        if data_file.exists():
            print(f"📂 从文件读取数据: {data_file}\n")
            with open(data_file, 'r') as f:
                data = json.load(f)
                results_data = data['results']
        else:
            print(f"❌ 错误: 找不到数据文件 {data_file}")
            print("请先运行: python visualize_results.py --scenarios --run")
            return
    
    if not results_data:
        print("❌ 没有可用的结果数据")
        return
    
    # 准备绘图数据
    scenario_names = [scenarios[k]['name'] for k in results_data.keys()]
    baseline_values = [results_data[k]['baseline_se'] for k in results_data.keys()]
    radiomap_values = [results_data[k]['radiomap_se'] for k in results_data.keys()]
    improvement_pcts = [results_data[k]['improvement_pct'] for k in results_data.keys()]
    colors = [scenarios[k]['color'] for k in results_data.keys()]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== 子图1: Baseline vs RadioMap SE对比 ==========
    x = np.arange(len(scenario_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_values, width, 
                    label='Baseline', color='#95a5a6', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, radiomap_values, width,
                    label='RadioMap', color=colors, alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline vs RadioMap Performance Comparison', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, fontsize=10)
    
    # 创建包含Baseline和各场景RadioMap的图例
    legend_labels = [k.replace('_', ' ').title() for k in results_data.keys()]
    legend_handles = [mpatches.Patch(color='#95a5a6', label='Baseline', alpha=0.8)]
    legend_handles.extend([mpatches.Patch(color=scenarios[k]['color'], 
                                          label=f"RadioMap - {legend_labels[i]}", alpha=0.8) 
                          for i, k in enumerate(results_data.keys())])
    ax1.legend(handles=legend_handles, fontsize=9, loc='upper left', 
              framealpha=0.9, edgecolor='black', ncol=1)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ========== 子图2: RadioMap相对于Baseline的提升百分比 ==========
    bars3 = ax2.bar(x, improvement_pcts, color=colors, alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax2.set_title('RadioMap Gain over Baseline', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names, fontsize=10)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 为每个场景创建图例
    legend_labels = [k.replace('_', ' ').title() for k in results_data.keys()]
    legend_patches = [mpatches.Patch(color=scenarios[k]['color'], label=legend_labels[i], alpha=0.8) 
                     for i, k in enumerate(results_data.keys())]
    ax2.legend(handles=legend_patches, fontsize=10, loc='upper left', 
              framealpha=0.9, edgecolor='black')
    
    # 添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_file = OUTPUT_DIR / "four_scenarios_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至: {output_file}")
    
    plt.show()
    
    # 打印统计摘要
    print("\n" + "="*70)
    print("📈 统计摘要")
    print("="*70)
    print(f"平均Baseline SE:  {np.mean(baseline_values):.4f} bits/s/Hz")
    print(f"平均RadioMap SE:  {np.mean(radiomap_values):.4f} bits/s/Hz")
    print(f"平均提升百分比:   {np.mean(improvement_pcts):.2f}%")
    print(f"最大提升百分比:   {np.max(improvement_pcts):.2f}% ({scenario_names[np.argmax(improvement_pcts)]})")
    print(f"最小提升百分比:   {np.min(improvement_pcts):.2f}% ({scenario_names[np.argmin(improvement_pcts)]})")
    print("="*70 + "\n")


def visualize_resolution_comparison(input_file=None, run_tests=False):
    """
    可视化分辨率对比结果
    
    Args:
        input_file: JSON结果文件路径（如果提供，则从文件读取）
        run_tests: 是否运行测试（如果True，则运行测试并生成新数据）
    """
    print("\n" + "="*70)
    print("📊 分辨率影响对比可视化")
    print("="*70 + "\n")
    
    if run_tests:
        print("⚠️  运行分辨率对比测试需要使用:")
        print("   python run_resolution_comparison.py --all --save-report")
        print("\n请先运行上述命令，然后使用 --input 参数指定生成的JSON文件\n")
        return
    
    # 确定数据文件路径
    if input_file is None:
        input_file = RESULTS_DIR / "resolution_comparison.json"
    else:
        input_file = Path(input_file)
    
    if not input_file.exists():
        print(f"❌ 错误: 找不到数据文件 {input_file}")
        print("\n请先运行:")
        print("  python run_resolution_comparison.py --all --save-report")
        return
    
    # 读取数据
    print(f"📂 从文件读取数据: {input_file}\n")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    comparisons = data.get('comparisons', [])
    if not comparisons:
        print("❌ 错误: 数据文件中没有对比结果")
        return
    
    # 准备绘图数据
    cities = [c['city'].upper() for c in comparisons]
    
    # 数据提取
    baseline_125 = [c['baseline_125m'] for c in comparisons]
    baseline_150 = [c['baseline_150m'] for c in comparisons]
    radiomap_125 = [c['radiomap_125m'] for c in comparisons]
    radiomap_150 = [c['radiomap_150m'] for c in comparisons]
    gain_125 = [c['gain_125m'] for c in comparisons]
    gain_150 = [c['gain_150m'] for c in comparisons]
    
    # 创建图表 - 仅2个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(cities))
    bar_width = 0.25
    
    # ========== 子图1: SE对比 (Baseline, RadioMap 125m, RadioMap 150m) ==========
    # 添加 Baseline 的平均值作为参考
    baseline_avg = [(baseline_125[i] + baseline_150[i]) / 2 for i in range(len(baseline_125))]
    
    bars1 = ax1.bar(x - bar_width, baseline_avg, bar_width,
                    label='Baseline', color='#95a5a6', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x, radiomap_125, bar_width,
                    label='RadioMap 125m', color='#3498db', alpha=0.8, edgecolor='black')
    bars3 = ax1.bar(x + bar_width, radiomap_150, bar_width,
                    label='RadioMap 150m', color='#e67e22', alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('City', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Comparison: Baseline vs RadioMap Resolutions',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cities, fontsize=11)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars3:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # ========== 子图2: RadioMap增益对比 (125m vs 150m) - 使用统一Baseline ==========
    # 重新计算增益：使用平均Baseline作为统一基准
    gain_125_unified = [((radiomap_125[i] - baseline_avg[i]) / baseline_avg[i] * 100) 
                        for i in range(len(cities))]
    gain_150_unified = [((radiomap_150[i] - baseline_avg[i]) / baseline_avg[i] * 100) 
                        for i in range(len(cities))]
    
    width = 0.35
    bars4 = ax2.bar(x - width/2, gain_125_unified, width,
                    label='RadioMap 125m', color='#3498db', alpha=0.8, edgecolor='black')
    bars5 = ax2.bar(x + width/2, gain_150_unified, width,
                    label='RadioMap 150m', color='#e67e22', alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('RadioMap Gain over Unified Baseline (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('City', fontsize=12, fontweight='bold')
    ax2.set_title('RadioMap Gain: 125m vs 150m Resolution\n(Relative to Averaged Baseline)',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cities, fontsize=11)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars5:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_file = OUTPUT_DIR / "resolution_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至: {output_file}")
    
    plt.show()
    
    # 打印统计摘要
    print("\n" + "="*70)
    print("📈 分辨率影响统计摘要")
    print("="*70)
    for i, comp in enumerate(comparisons):
        baseline_avg_val = (comp['baseline_125m'] + comp['baseline_150m']) / 2
        gain_125_unified_val = ((comp['radiomap_125m'] - baseline_avg_val) / baseline_avg_val * 100)
        gain_150_unified_val = ((comp['radiomap_150m'] - baseline_avg_val) / baseline_avg_val * 100)
        gain_delta_unified = gain_150_unified_val - gain_125_unified_val
        
        print(f"\n{comp['city'].upper()}:")
        print(f"  Baseline (平均): {baseline_avg_val:.4f} bits/s/Hz")
        print(f"  RadioMap 125m:   {comp['radiomap_125m']:.4f} bits/s/Hz (增益: {gain_125_unified_val:+.2f}%)")
        print(f"  RadioMap 150m:   {comp['radiomap_150m']:.4f} bits/s/Hz (增益: {gain_150_unified_val:+.2f}%)")
        print(f"  增益差异 (150m - 125m): {gain_delta_unified:+.2f} 百分点")
        
        se_improvement = comp['radiomap_150m'] - comp['radiomap_125m']
        if abs(gain_delta_unified) < 1.0:
            conclusion = "分辨率差异对性能影响较小"
        elif gain_delta_unified > 0:
            conclusion = f"更高分辨率(150m)带来更大提升 (SE提高{se_improvement:+.4f})"
        else:
            conclusion = f"较低分辨率(125m)性能更优 (SE降低{se_improvement:+.4f})"
        print(f"  结论: {conclusion}")
    
    print("\n" + "="*70)
    print("💡 说明: 增益百分比基于统一的平均Baseline计算，保证了两个子图结论一致。")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='测试结果可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 可视化四个主场景（直接运行测试）
  python visualize_results.py --scenarios --run
  
  # 可视化四个主场景（从已保存数据）
  python visualize_results.py --scenarios
  
  # 可视化分辨率对比（从默认路径）
  python visualize_results.py --resolution
  
  # 可视化分辨率对比（指定JSON文件）
  python visualize_results.py --resolution --input results/resolution_comparison.json
  
  # 两者都可视化
  python visualize_results.py --all
        """
    )
    
    parser.add_argument(
        '--scenarios',
        action='store_true',
        help='可视化四个主场景结果'
    )
    
    parser.add_argument(
        '--resolution',
        action='store_true',
        help='可视化分辨率对比结果'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='可视化所有结果'
    )
    
    parser.add_argument(
        '--run',
        action='store_true',
        help='运行测试（仅用于四场景可视化）'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='分辨率对比结果JSON文件路径'
    )
    
    args = parser.parse_args()
    
    if not (args.scenarios or args.resolution or args.all):
        parser.print_help()
        return 1
    
    try:
        if args.all or args.scenarios:
            visualize_four_scenarios(run_tests=args.run, save_data=True)
        
        if args.all or args.resolution:
            visualize_resolution_comparison(input_file=args.input, run_tests=False)
        
        print("\n✅ 可视化完成!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
