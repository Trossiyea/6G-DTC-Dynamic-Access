#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UE Scaling Analysis Script
分析不同 UE 数量 (25, 50, 75, 100) 下三种调度算法的性能对比

输出:
- output/UE_Scaling_SE_Comparison.png: 频谱效率对比图
- output/ue_scaling_analysis_log_YYYYMMDD_HHMMSS.txt: 日志文件
"""

import sys
import os
import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime

# ==========================================
# 1. 环境配置
# ==========================================
SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR / "code"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# 日志配置
LOG_FILE = OUTPUT_DIR / f"ue_scaling_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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
# 2. 加载模块
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
    run_constellation = main_module.run_constellation
    BASE_CONFIG = config_module.CONFIG
except ImportError as e:
    logger.error(f"Error loading project modules: {e}")
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
# 3. 仿真运行
# ==========================================
def run_simulation_with_ue_count(scenario_key: str, n_ue: int):
    """运行指定 UE 数量的仿真"""
    logger.info(f"[{scenario_key}] Running with N_UE={n_ue}...")
    try:
        scenario_cfg = load_scenario_config(scenario_key)
        
        # 合并配置并覆盖 N_UE
        run_config = BASE_CONFIG.copy()
        run_config.update(scenario_cfg)
        run_config["N_UE"] = n_ue
        run_config["show_plots"] = False
        run_config["save_plots"] = False
        run_config["show_progress"] = True
        # 禁用记录以加速
        run_config["record_assignments"] = False
        run_config["record_ue_thr"] = False
        
        logger.info(f"  > T={run_config.get('T')} TTIs, N_UE={n_ue}")
        
        # 根据是否为星座模式选择运行函数
        if run_config.get("enable_constellation", False):
            logger.info(f"  > Mode: Constellation")
            results = run_constellation(run_config)
        else:
            logger.info(f"  > Mode: Single Satellite")
            results = run_once(run_config)
        
        baseline_pf = results['avg_se_baseline_default']
        baseline_mr = results.get('avg_se_baseline_mr', 0.0)
        radiomap = results['avg_se_radiomap']
        
        logger.info(f"  > Baseline PF: {baseline_pf:.4f} bps/Hz")
        logger.info(f"  > Baseline MR: {baseline_mr:.4f} bps/Hz")
        logger.info(f"  > RadioMap:    {radiomap:.4f} bps/Hz")
        
        return baseline_pf, baseline_mr, radiomap
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0

# ==========================================
# 4. 绘图
# ==========================================
def plot_ue_scaling(ue_counts, pf_results, mr_results, rm_results, scenario_label, filename):
    """绘制 UE 数量 vs 频谱效率柱状图（分组）"""
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
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(ue_counts))
    width = 0.25  # 柱子宽度
    
    # 绘制三组柱子
    bars_pf = ax.bar(x - width, pf_results, width, label='Baseline PF', 
                     color='#BBBBBB', edgecolor='white', linewidth=0.5, hatch='//')
    bars_mr = ax.bar(x, mr_results, width, label='Baseline MR', 
                     color='#48C9B0', edgecolor='white', linewidth=0.5, hatch='//')
    bars_rm = ax.bar(x + width, rm_results, width, label='Proposed RM-DSS', 
                     color='#E74C3C', edgecolor='white', linewidth=0.5, hatch='\\')
    
    # 在柱子上标注数值
    def autolabel(bars, color='black'):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color=color)
    
    autolabel(bars_pf, '#666666')
    autolabel(bars_mr, '#2E7D5E')
    autolabel(bars_rm, '#B03A2E')
    
    ax.set_xlabel('Number of UEs')
    ax.set_ylabel('Spectral Efficiency (bits/s/Hz)')
    ax.set_title(f'SE vs Number of UEs - {scenario_label}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N_UE={n}' for n in ue_counts])
    ax.legend(loc='upper left')
    ax.set_ylim(0, max(max(pf_results), max(mr_results), max(rm_results)) * 1.2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {filename}")
    plt.close()

# ==========================================
# 5. 主程序
# ==========================================
def run_scenario_analysis(scenario_key: str, scenario_label: str, ue_counts: list, output_filename: str):
    """运行单个场景的 UE 数量分析"""
    logger.info(f"\n{'#'*60}")
    logger.info(f"# Scenario: {scenario_label}")
    logger.info(f"{'#'*60}")
    
    pf_results = []
    mr_results = []
    rm_results = []
    
    for n_ue in ue_counts:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing with N_UE = {n_ue}")
        logger.info('='*50)
        
        pf, mr, rm = run_simulation_with_ue_count(scenario_key, n_ue)
        pf_results.append(pf)
        mr_results.append(mr)
        rm_results.append(rm)
    
    # 汇总结果
    logger.info(f"\n{'='*50}")
    logger.info(f"Summary Results - {scenario_label}:")
    logger.info('='*50)
    logger.info(f"{'N_UE':<10} {'PF':<12} {'MR':<12} {'Proposed':<12}")
    logger.info('-'*46)
    for i, n_ue in enumerate(ue_counts):
        logger.info(f"{n_ue:<10} {pf_results[i]:<12.4f} {mr_results[i]:<12.4f} {rm_results[i]:<12.4f}")
    
    # 绘图
    output_file = OUTPUT_DIR / output_filename
    plot_ue_scaling(ue_counts, pf_results, mr_results, rm_results, scenario_label, output_file)
    
    return pf_results, mr_results, rm_results

if __name__ == "__main__":
    logger.info("=== UE Scaling Analysis ===\n")
    
    # 配置
    UE_COUNTS = [25, 50, 75, 100]
    
    # ==========================================
    # Scenario 1: 单星场景
    # ==========================================
    run_scenario_analysis(
        scenario_key="toronto_single",
        scenario_label="Toronto Single Satellite",
        ue_counts=UE_COUNTS,
        output_filename="UE_Scaling_SE_SingleSat.png"
    )
    
    # ==========================================
    # Scenario 2: 星座场景
    # ==========================================
    run_scenario_analysis(
        scenario_key="toronto_constellation",
        scenario_label="Toronto Constellation",
        ue_counts=UE_COUNTS,
        output_filename="UE_Scaling_SE_Constellation.png"
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("All scenarios completed!")
    logger.info(f"Log saved to: {LOG_FILE}")
    logger.info(f"{'='*60}")
