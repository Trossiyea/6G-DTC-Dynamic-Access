#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEEE TMC 论文绘图专用脚本
Performance Evaluation: Spectral Efficiency Gains

功能：
1. 运行多伦多单星 (Scenario A) 和 多伦多星座 (Scenario B) 的仿真。
2. 提取 'avg_se_baseline_default' 和 'avg_se_radiomap' 指标。
3. 绘制符合 IEEE 图表规范的 Spectral Efficiency 对比图和增益图。
"""

import sys
import os
import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime

# Setup logging to both console and file
LOG_DIR = Path(__file__).parent / "output"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"simulation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 环境路径设置 (引入 code 模块)
# ==========================================
SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR / "code"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# ==========================================
# 2. 动态加载模块 (main, config)
# ==========================================
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

try:
    main_module = load_module("code.main", CODE_DIR / "main.py")
    config_module = load_module("code.config", CODE_DIR / "config.py")
    run_once = main_module.run_once
    run_constellation_func = main_module.run_constellation
    BASE_CONFIG = config_module.CONFIG
except ImportError as e:
    print(f"Error loading project modules: {e}")
    sys.exit(1)

def load_scenario_config(scenario_name):
    """加载特定场景的配置文件"""
    config_path = SCRIPT_DIR / "test" / f"config_{scenario_name}.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    spec = importlib.util.spec_from_file_location(f"test.config_{scenario_name}", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG

# ==========================================
# 3. 仿真运行逻辑
# ==========================================
def run_simulation(scenario_key):
    logger.info(f"[{scenario_key}] Loading configuration...")
    try:
        scenario_cfg = load_scenario_config(scenario_key)
        
        # 合并配置
        run_config = BASE_CONFIG.copy()
        run_config.update(scenario_cfg)
        
        # 强制不在运行过程中显示图表，只保存数据
        run_config["show_plots"] = False
        run_config["save_plots"] = False # 我们自己画图
        run_config["show_progress"] = True # 显示主进度条
        
        logger.info(f"[{scenario_key}] Running simulation (T={run_config.get('T')} TTIs)...")
        
        if run_config.get("enable_constellation", False):
            logger.info(f"  > Mode: Constellation")
            results = run_constellation_func(run_config)
        else:
            logger.info(f"  > Mode: Single Satellite")
            results = run_once(run_config)
        
        baseline_se = results['avg_se_baseline_default']
        baseline_mr_se = results.get('avg_se_baseline_mr', 0.0) # Check keys
        radiomap_se = results['avg_se_radiomap']
        gain = results['improvement_vs_default_pct']
        gain_vs_mr = (radiomap_se - baseline_mr_se) / max(1e-9, baseline_mr_se) * 100.0 if baseline_mr_se > 0 else 0.0
        
        logger.info(f"[{scenario_key}] Done.")
        logger.info(f"  > Baseline PF SE: {baseline_se:.4f} bps/Hz")
        logger.info(f"  > Baseline MR SE: {baseline_mr_se:.4f} bps/Hz")
        logger.info(f"  > RadioMap SE:    {radiomap_se:.4f} bps/Hz")
        logger.info(f"  > Gain (vs PF):   {gain:.2f}%")
        logger.info(f"  > Gain (vs MR):   {gain_vs_mr:.2f}%")
        
        return baseline_se, baseline_mr_se, radiomap_se, gain, gain_vs_mr
        
    except Exception as e:
        logger.error(f"Simulation failed for {scenario_key}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0, 0.0, 0.0

# ==========================================
# 4. 绘图逻辑 (IEEE Style)
# ==========================================
def apply_ieee_style():
    """配置 Matplotlib 以符合 IEEE 刊物风格"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
        'axes.axisbelow': True # Grid behind bars
    })

def plot_se_comparison(scenarios, base_vals, mr_vals, prop_vals, filename="Fig3_SE_Comparison.png"):
    """绘制 Baseline(PF) vs Baseline(MR) vs Proposed 的对比柱状图"""
    apply_ieee_style()
    
    x = np.arange(len(scenarios))
    width = 0.25 # Slimmer bars for 3 items
    
    fig, ax = plt.subplots(figsize=(7, 5)) # Slightly wider
    
    # Colors
    # Baseline PF: Grayscale/Blueish
    colors_base = ["#BBBBBB", "#929191"] 
    # Baseline MR: Lighter/Different texture? Maybe a light dashed pattern or distinct color.
    # Let's use a nice distinct blue/teal for MR to contrast with Grey PF and Orange/Purple Proposed.
    colors_mr = ["#76D7C4", "#48C9B0"] # Teal-ish

    # Proposed: Red/Orange gradients
    colors_prop = ["#F09164", "#978BE6"] 

    rects1 = ax.bar(x - width, base_vals, width, label='Baseline (PF)', 
                    color=colors_base, edgecolor='black', linewidth=0.5, alpha=0.9, hatch='//')
    rects2 = ax.bar(x, mr_vals, width, label='Baseline (MaxRate)', 
                    color=colors_mr, edgecolor='black', linewidth=0.5, alpha=0.9, hatch='..')
    rects3 = ax.bar(x + width, prop_vals, width, label='Proposed RM-DSS', 
                    color=colors_prop, edgecolor='black', linewidth=0.5, alpha=0.9, hatch='oo')

    # Labeling
    ax.set_ylabel('Spectral Efficiency (bits/s/Hz)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, max(max(prop_vals), max(mr_vals))*1.4) 
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_base[0], edgecolor='black', hatch='//', label='Baseline PF (Single)'),
        Patch(facecolor=colors_base[1], edgecolor='black', hatch='//', label='Baseline PF (Constel.)'),
        Patch(facecolor=colors_mr[0], edgecolor='black', hatch='..', label='Baseline MR (Single)'),
        Patch(facecolor=colors_mr[1], edgecolor='black', hatch='..', label='Baseline MR (Constel.)'),
        Patch(facecolor=colors_prop[0], edgecolor='black', hatch='oo', label='Proposed (Single)'),
        Patch(facecolor=colors_prop[1], edgecolor='black', hatch='oo', label='Proposed (Constel.)'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15),
              frameon=False, fontsize=9, ncol=3)

    # 标注数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    # plt.show()
    
    viridis_derived = [
    "#440154",  # 深紫
    "#31688E",  # 中蓝
    "#35B779",  # 青绿
    "#FDE725",  # 黄
]
    tol_bright = [
    "#4477AA",  # 蓝
    "#EE6677",  # 玫瑰红
    "#228833",  # 深绿
    "#CCBB44",  # 金黄
    "#66CCEE",  # 青色
    "#AA3377",  # 紫红
    "#BBBBBB",  # 灰
]
    nature_palette = [
    "#0077BB",  # 海军蓝（主色）
    "#EE7733",  # 橙色
    "#33BBEE",  # 天蓝
    "#CC3311",  # 深橙红
    "#009988",  # 青绿色
    "#DDCC77",  # 沙黄
    "#AA4499",  # 紫色
]



def plot_gain_analysis(scenarios, gains, filename="Fig4_SE_Gain.png"):
    """绘制性能增益百分比图"""
    apply_ieee_style()
    
    x = np.arange(len(scenarios))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Nature Journal Colors
    # Gain: Teal/Green gradients (#00A087, #006E5D)
    colors_gain = ['#F09164', '#978BE6'] # Lighter green for Single, Darker for Constellation
    bars = ax.bar(x, gains, width, color=colors_gain, edgecolor='white', alpha=0.9)

    ax.set_ylabel('Spectral Efficiency Gain (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, max(gains) * 1.3)
    
    # 在柱子上标注具体百分比
    for i, rect in enumerate(bars):
        height = rect.get_height()
        ax.annotate(f'+{height:.3f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', weight='bold')

    # 添加平均线（可选）
    # avg_gain = np.mean(gains)
    # ax.axhline(y=avg_gain, color='black', linestyle='--', linewidth=1, label=f'Avg Gain: {avg_gain:.1f}%')
    # ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    # plt.show()

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    print("=== IEEE TMC Paper: Spectral Efficiency Evaluation ===\n")
    
    # 定义测试列表
    target_scenarios = [
        ("toronto_single", "Scenario A:\nToronto Single-Sat"),
        ("toronto_constellation", "Scenario B:\nToronto Constellation")
    ]
    
    base_se_list = []
    mr_se_list = []
    rm_se_list = []
    gain_list = []
    gain_mr_list = []
    labels = []
    
    # 依次运行
    for key, label in target_scenarios:
        base, mr, rm, g, g_mr = run_simulation(key)
        base_se_list.append(base)
        mr_se_list.append(mr)
        rm_se_list.append(rm)
        gain_list.append(g)
        gain_mr_list.append(g_mr)
        labels.append(label)
        
    print("\nAll simulations completed. Generating plots...")
    
    # 绘制两张图
    plot_se_comparison(labels, base_se_list, mr_se_list, rm_se_list, filename="IEEE_TMC_Fig3_SE_Comparison_v4.png")
    plot_gain_analysis(labels, gain_list, filename="IEEE_TMC_Fig4_SE_Gain_v4.png")
    
    print("\nProcess finished.")