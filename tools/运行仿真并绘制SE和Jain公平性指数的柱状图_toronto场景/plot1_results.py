
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# 基于跑的csv数据文件，绘图代码（绘制SE、Jain对比的柱状图）

# Compute PROJECT_ROOT dynamically
_script_dir = Path(__file__).resolve().parent
# Auto-detect if script is nested in output_toronto/jain or flat in tools/scripts
if _script_dir.name == 'jain' and _script_dir.parent.name.startswith('output'):
    PROJECT_ROOT = _script_dir.parent.parent
else:
    PROJECT_ROOT = _script_dir.parent if _script_dir.name in ['tools', 'scripts'] else _script_dir

# 配置
single_csv = str(PROJECT_ROOT / "output_toronto/jain/output_images_v1_radiomap_est_error_db_1.5/single_kpis_vs_nue_data_20260126_210635.csv")
const_csv = str(PROJECT_ROOT / "output_toronto/jain/output_images_v1_radiomap_est_error_db_1.5/constellation_kpis_vs_nue_data_20260126_215321.csv")
output_dir = str(PROJECT_ROOT / "output_toronto/jain/output_images")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 统一绘图风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    # 去除顶部和右侧边框
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 高级配色 (柔和优雅 - 暖色基线 vs 绿色提议)
# Baselines (Warm): 
# PF: Muted Coral/Red
# MR: Muted Orange/Gold
# Proposed (Green): Muted/Sage Green
color_pf = "#F8B0A5"   # Warm Brick Red
color_mr = '#F4A582'   # Warm Salmon/Orange
#color_rm = "#6DC0C3"   # Cadet Blue / Sage Green-ish (Changing to a distinct Green)
color_rm = "#57CEB6"   # Soft Green (Teal-ish Green) - looks professional

def plot_bar_chart(df, y_col_pf, y_col_mr, y_col_rm, ylabel, title, filename):
    n_ue = df['N_UE'].unique()
    x = np.arange(len(n_ue))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6)) # Increased size slightly for labels

    # 绘制柱状图，添加纹理和颜色
    # Baseline: 白色反斜线纹理 '\\'
    # Proposed: 白色斜线纹理 '//'
    
    # PF (Base)
    rects1 = ax.bar(x - width, df[y_col_pf], width, label='Baseline PF', 
                    color=color_pf, edgecolor='white', hatch='\\\\', zorder=3)
    
    # MR (Base)
    rects2 = ax.bar(x, df[y_col_mr], width, label='Baseline MR', 
                    color=color_mr, edgecolor='white', hatch='\\\\', zorder=3)
    
    # RadioMap (Proposed)
    rects3 = ax.bar(x + width, df[y_col_rm], width, label='Proposed RM-DSS', 
                    color=color_rm, edgecolor='white', hatch='//', zorder=3)

    # 去除柱子的黑色边框 (edgecolor='white' 已设置)

    # 添加数值标签 (3位小数)
    for rects in [rects1, rects2, rects3]:
        ax.bar_label(rects, fmt='%.3f', padding=3, fontsize=9, color='black')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_ue])
    ax.set_xlabel('Number of UEs')
    
    # 图例
    ax.legend(loc='best', frameon=False)
    
    # 网格线 (仅水平)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    # 调整布局并保存
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

def process_and_plot():
    # 1. Load Data
    try:
        df_single = pd.read_csv(single_csv, comment='#')
        df_const = pd.read_csv(const_csv, comment='#')
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # 2. Plot Single Satellite - Spectral Efficiency
    plot_bar_chart(
        df_single, 
        'SE_3GPP_PF', 'SE_3GPP_MR', 'SE_RadioMap',
        'Spectral Efficiency (bits/s/Hz)',
        'Spectral Efficiency vs Number of UEs\n(Toronto Single Satellite)',
        'Single_SE_Bar.png'
    )

    # 3. Plot Single Satellite - Jain's Index
    plot_bar_chart(
        df_single, 
        'Jain_3GPP_PF', 'Jain_3GPP_MR', 'Jain_RadioMap',
        "Jain's Fairness Index",
        "Jain's Fairness Index vs Number of UEs\n(Toronto Single Satellite)",
        'Single_Jain_Bar.png'
    )

    # 4. Plot Constellation - Spectral Efficiency
    plot_bar_chart(
        df_const, 
        'SE_3GPP_PF', 'SE_3GPP_MR', 'SE_RadioMap',
        'Spectral Efficiency (bits/s/Hz)',
        'Spectral Efficiency vs Number of UEs\n(Toronto Constellation)',
        'Constellation_SE_Bar.png'
    )

    # 5. Plot Constellation - Jain's Index
    plot_bar_chart(
        df_const, 
        'Jain_3GPP_PF', 'Jain_3GPP_MR', 'Jain_RadioMap',
        "Jain's Fairness Index",
        "Jain's Fairness Index vs Number of UEs\n(Toronto Constellation)",
        'Constellation_Jain_Bar.png'
    )

if __name__ == "__main__":
    process_and_plot()
