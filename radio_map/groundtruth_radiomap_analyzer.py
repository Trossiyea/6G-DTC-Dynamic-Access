#!/usr/bin/env python3
"""
GroundTruth和RadioMap数据相似性分析脚本
功能：
1. 加载GroundTruth和RadioMap数据
2. 进行数据预处理和单位统一
3. 计算多种相似性指标
4. 生成可视化分析结果
5. 输出详细的分析报告
"""

import argparse
import os
import sys
import numpy as np
import h5py
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
# import seaborn as sns
import json
from datetime import datetime


def load_groundtruth_data(filepath):
    """
    加载GroundTruth数据（npy格式）
    
    Args:
        filepath: npy文件路径
    
    Returns:
        numpy.ndarray: GroundTruth数据
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GroundTruth文件不存在: {filepath}")
    
    data = np.load(filepath)
    print(f"📂 加载GroundTruth: {filepath}")
    print(f"   - 形状: {data.shape}")
    print(f"   - 数据类型: {data.dtype}")
    print(f"   - 数值范围: [{np.min(data):.6e}, {np.max(data):.6e}]")
    
    return data


def load_radiomap_data(filepath):
    """
    加载RadioMap数据（mat格式）
    
    Args:
        filepath: mat文件路径
    
    Returns:
        numpy.ndarray: RadioMap数据
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"RadioMap文件不存在: {filepath}")
    
    # 尝试使用h5py加载（v7.3格式）
    try:
        with h5py.File(filepath, 'r') as f:
            # 查找数值数据
            for key in f.keys():
                data = f[key]
                if isinstance(data, h5py.Dataset) and data.dtype.kind in ['f', 'i']:
                    data_array = np.array(data)
                    print(f"📂 加载RadioMap: {filepath}")
                    print(f"   - 键名: {key}")
                    print(f"   - 形状: {data_array.shape}")
                    print(f"   - 数据类型: {data_array.dtype}")
                    print(f"   - 数值范围: [{np.min(data_array):.2f}, {np.max(data_array):.2f}]")
                    return data_array, key
    except Exception as e:
        print(f"⚠️  h5py加载失败，尝试scipy: {e}")
    
    # 尝试使用scipy加载（传统格式）
    try:
        mat_data = loadmat(filepath)
        # 过滤掉MATLAB内部变量
        user_vars = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        
        # 选择最大的数值数组
        numeric_vars = {k: v for k, v in user_vars.items() 
                       if isinstance(v, np.ndarray) and v.dtype.kind in ['f', 'i']}
        
        if not numeric_vars:
            raise ValueError("未找到数值数据")
        
        # 选择最大的数组
        key = max(numeric_vars.keys(), key=lambda k: numeric_vars[k].size)
        data_array = numeric_vars[key]
        
        print(f"📂 加载RadioMap: {filepath}")
        print(f"   - 键名: {key}")
        print(f"   - 形状: {data_array.shape}")
        print(f"   - 数据类型: {data_array.dtype}")
        print(f"   - 数值范围: [{np.min(data_array):.2f}, {np.max(data_array):.2f}]")
        return data_array, key
        
    except Exception as e:
        raise RuntimeError(f"无法加载RadioMap文件 {filepath}: {e}")


def preprocess_data(gt_data, rm_data, rm_key=None):
    """
    预处理数据，统一单位和维度
    
    Args:
        gt_data: GroundTruth数据
        rm_data: RadioMap数据
        rm_key: RadioMap数据键名
    
    Returns:
        tuple: (处理后的GT数据, 处理后的RM数据)
    """
    print(f"\n🔄 数据预处理:")
    
    # 1. 单位统一：将GT数据从W转换为dBm，RM数据从dBW转换为dBm
    print(f"   1. 单位统一:")
    
    # GT数据：W -> dBm
    gt_watts = np.maximum(gt_data, 1e-20)  # 避免log(0)
    gt_dbm = 10 * np.log10(gt_watts * 1000)
    print(f"      - GT: W -> dBm, 范围: [{np.min(gt_dbm):.2f}, {np.max(gt_dbm):.2f}] dBm")
    
    # RM数据：dBW -> dBm
    rm_dbm = rm_data + 30
    print(f"      - RM: dBW -> dBm, 范围: [{np.min(rm_dbm):.2f}, {np.max(rm_dbm):.2f}] dBm")
    
    # 2. 维度统一
    print(f"   2. 维度统一:")
    print(f"      - GT原始形状: {gt_data.shape}")
    print(f"      - RM原始形状: {rm_data.shape}")
    
    # 检查维度是否匹配
    if gt_data.shape != rm_data.shape:
        print(f"      ⚠️  维度不匹配，尝试重排...")
        
        # 尝试不同的维度重排
        possible_shapes = [
            gt_data.shape,
            (gt_data.shape[1], gt_data.shape[2], gt_data.shape[0]),  # (y,z,x)
            (gt_data.shape[2], gt_data.shape[0], gt_data.shape[1]),  # (z,x,y)
        ]
        
        for i, shape in enumerate(possible_shapes):
            if shape == rm_data.shape:
                if i == 1:  # (y,z,x)
                    gt_dbm = np.transpose(gt_dbm, (1, 2, 0))
                    print(f"      - 应用维度重排: (x,y,z) -> (y,z,x)")
                elif i == 2:  # (z,x,y)
                    gt_dbm = np.transpose(gt_dbm, (2, 0, 1))
                    print(f"      - 应用维度重排: (x,y,z) -> (z,x,y)")
                break
        else:
            print(f"      ❌ 无法匹配维度，使用最小公共形状")
            # 使用最小公共形状
            min_shape = tuple(min(gt_data.shape[i], rm_data.shape[i]) for i in range(3))
            gt_dbm = gt_dbm[:min_shape[0], :min_shape[1], :min_shape[2]]
            rm_dbm = rm_dbm[:min_shape[0], :min_shape[1], :min_shape[2]]
            print(f"      - 裁剪到公共形状: {min_shape}")
    
    print(f"      - 最终GT形状: {gt_dbm.shape}")
    print(f"      - 最终RM形状: {rm_dbm.shape}")
    
    return gt_dbm, rm_dbm


def calculate_similarity_metrics(gt_data, rm_data):
    """
    计算多种相似性指标
    
    Args:
        gt_data: GroundTruth数据
        rm_data: RadioMap数据
    
    Returns:
        dict: 相似性指标字典
    """
    print(f"\n📊 计算相似性指标:")
    
    # 展平数据用于计算
    gt_flat = gt_data.flatten()
    rm_flat = rm_data.flatten()
    
    # 移除无效值
    valid_mask = np.isfinite(gt_flat) & np.isfinite(rm_flat)
    gt_valid = gt_flat[valid_mask]
    rm_valid = rm_flat[valid_mask]
    
    print(f"   - 有效数据点: {len(gt_valid)} / {len(gt_flat)}")
    
    if len(gt_valid) == 0:
        raise ValueError("没有有效的数据点用于计算相似性")
    
    metrics = {}
    
    # 1. 相关系数
    try:
        pearson_corr, pearson_p = pearsonr(gt_valid, rm_valid)
        metrics['pearson_correlation'] = {
            'value': pearson_corr,
            'p_value': pearson_p,
            'interpretation': '线性相关性'
        }
        print(f"   - Pearson相关系数: {pearson_corr:.4f} (p={pearson_p:.2e})")
    except Exception as e:
        print(f"   - Pearson相关系数计算失败: {e}")
        metrics['pearson_correlation'] = {'value': np.nan, 'error': str(e)}
    
    try:
        spearman_corr, spearman_p = spearmanr(gt_valid, rm_valid)
        metrics['spearman_correlation'] = {
            'value': spearman_corr,
            'p_value': spearman_p,
            'interpretation': '单调相关性'
        }
        print(f"   - Spearman相关系数: {spearman_corr:.4f} (p={spearman_p:.2e})")
    except Exception as e:
        print(f"   - Spearman相关系数计算失败: {e}")
        metrics['spearman_correlation'] = {'value': np.nan, 'error': str(e)}
    
    # 2. 回归指标
    try:
        # 手动计算R²
        ss_res = np.sum((gt_valid - rm_valid) ** 2)
        ss_tot = np.sum((gt_valid - np.mean(gt_valid)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        metrics['r2_score'] = {
            'value': r2,
            'interpretation': '决定系数，越接近1越好'
        }
        print(f"   - R²决定系数: {r2:.4f}")
    except Exception as e:
        print(f"   - R²决定系数计算失败: {e}")
        metrics['r2_score'] = {'value': np.nan, 'error': str(e)}
    
    # 3. 误差指标
    try:
        # 手动计算MSE
        mse = np.mean((gt_valid - rm_valid) ** 2)
        rmse = np.sqrt(mse)
        metrics['mse'] = {
            'value': mse,
            'interpretation': '均方误差，越小越好'
        }
        metrics['rmse'] = {
            'value': rmse,
            'interpretation': '均方根误差，越小越好'
        }
        print(f"   - MSE: {mse:.4f}")
        print(f"   - RMSE: {rmse:.4f}")
    except Exception as e:
        print(f"   - MSE/RMSE计算失败: {e}")
        metrics['mse'] = {'value': np.nan, 'error': str(e)}
        metrics['rmse'] = {'value': np.nan, 'error': str(e)}
    
    try:
        # 手动计算MAE
        mae = np.mean(np.abs(gt_valid - rm_valid))
        metrics['mae'] = {
            'value': mae,
            'interpretation': '平均绝对误差，越小越好'
        }
        print(f"   - MAE: {mae:.4f}")
    except Exception as e:
        print(f"   - MAE计算失败: {e}")
        metrics['mae'] = {'value': np.nan, 'error': str(e)}
    
    # 4. 统计指标
    try:
        mean_diff = np.mean(gt_valid - rm_valid)
        std_diff = np.std(gt_valid - rm_valid)
        metrics['mean_difference'] = {
            'value': mean_diff,
            'interpretation': '平均差异，越接近0越好'
        }
        metrics['std_difference'] = {
            'value': std_diff,
            'interpretation': '差异标准差，越小越好'
        }
        print(f"   - 平均差异: {mean_diff:.4f}")
        print(f"   - 差异标准差: {std_diff:.4f}")
    except Exception as e:
        print(f"   - 统计指标计算失败: {e}")
        metrics['mean_difference'] = {'value': np.nan, 'error': str(e)}
        metrics['std_difference'] = {'value': np.nan, 'error': str(e)}
    
    # 5. 数据范围比较
    try:
        gt_range = np.max(gt_valid) - np.min(gt_valid)
        rm_range = np.max(rm_valid) - np.min(rm_valid)
        range_ratio = rm_range / gt_range if gt_range != 0 else np.nan
        metrics['data_range_comparison'] = {
            'gt_range': gt_range,
            'rm_range': rm_range,
            'range_ratio': range_ratio,
            'interpretation': '数据范围比较，ratio越接近1越好'
        }
        print(f"   - GT数据范围: {gt_range:.2f}")
        print(f"   - RM数据范围: {rm_range:.2f}")
        print(f"   - 范围比例: {range_ratio:.4f}")
    except Exception as e:
        print(f"   - 数据范围比较失败: {e}")
        metrics['data_range_comparison'] = {'error': str(e)}
    
    return metrics


def create_visualizations(gt_data, rm_data, output_dir, prefix="analysis"):
    """
    创建可视化分析图表
    
    Args:
        gt_data: GroundTruth数据
        rm_data: RadioMap数据
        output_dir: 输出目录
        prefix: 文件前缀
    """
    print(f"\n📈 生成可视化图表:")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 展平数据
    gt_flat = gt_data.flatten()
    rm_flat = rm_data.flatten()
    
    # 移除无效值
    valid_mask = np.isfinite(gt_flat) & np.isfinite(rm_flat)
    gt_valid = gt_flat[valid_mask]
    rm_valid = rm_flat[valid_mask]
    
    # 1. 散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(gt_valid, rm_valid, alpha=0.5, s=1)
    
    # 添加对角线
    min_val = min(np.min(gt_valid), np.min(rm_valid))
    max_val = max(np.max(gt_valid), np.max(rm_valid))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美匹配线')
    
    plt.xlabel('GroundTruth (dBm)')
    plt.ylabel('RadioMap (dBm)')
    plt.title('GroundTruth vs RadioMap 散点图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    if len(gt_valid) > 0:
        corr = np.corrcoef(gt_valid, rm_valid)[0, 1]
        plt.text(0.05, 0.95, f'相关系数: {corr:.4f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"{prefix}_scatter_plot.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   - 散点图: {scatter_path}")
    
    # 2. 差异分布直方图
    plt.figure(figsize=(10, 6))
    diff = gt_valid - rm_valid
    plt.hist(diff, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(diff), color='red', linestyle='--', linewidth=2, label=f'平均差异: {np.mean(diff):.2f}')
    plt.xlabel('差异 (GT - RM) [dBm]')
    plt.ylabel('频次')
    plt.title('GroundTruth与RadioMap差异分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    diff_path = os.path.join(output_dir, f"{prefix}_difference_histogram.png")
    plt.savefig(diff_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   - 差异分布图: {diff_path}")
    
    # 3. 数据分布对比
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(gt_valid, bins=50, alpha=0.7, label='GroundTruth', color='blue')
    plt.hist(rm_valid, bins=50, alpha=0.7, label='RadioMap', color='orange')
    plt.xlabel('功率 [dBm]')
    plt.ylabel('频次')
    plt.title('数据分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([gt_valid, rm_valid], labels=['GroundTruth', 'RadioMap'])
    plt.ylabel('功率 [dBm]')
    plt.title('数据分布箱线图')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_path = os.path.join(output_dir, f"{prefix}_distribution_comparison.png")
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   - 分布对比图: {dist_path}")
    
    # 4. 3D切片可视化（如果数据是3D的）
    if len(gt_data.shape) == 3:
        fig = plt.figure(figsize=(15, 5))
        
        # 选择中间切片
        mid_slice = gt_data.shape[2] // 2
        
        # GT数据切片
        plt.subplot(1, 3, 1)
        plt.imshow(gt_data[:, :, mid_slice], cmap='viridis', aspect='auto')
        plt.colorbar(label='功率 [dBm]')
        plt.title(f'GroundTruth (切片 {mid_slice})')
        
        # RM数据切片
        plt.subplot(1, 3, 2)
        plt.imshow(rm_data[:, :, mid_slice], cmap='viridis', aspect='auto')
        plt.colorbar(label='功率 [dBm]')
        plt.title(f'RadioMap (切片 {mid_slice})')
        
        # 差异切片
        plt.subplot(1, 3, 3)
        diff_slice = gt_data[:, :, mid_slice] - rm_data[:, :, mid_slice]
        plt.imshow(diff_slice, cmap='RdBu_r', aspect='auto')
        plt.colorbar(label='差异 [dBm]')
        plt.title(f'差异 (GT - RM, 切片 {mid_slice})')
        
        plt.tight_layout()
        slice_path = os.path.join(output_dir, f"{prefix}_3d_slices.png")
        plt.savefig(slice_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   - 3D切片图: {slice_path}")


def generate_report(gt_file, rm_file, metrics, output_dir, prefix="analysis"):
    """
    生成分析报告
    
    Args:
        gt_file: GroundTruth文件路径
        rm_file: RadioMap文件路径
        metrics: 相似性指标
        output_dir: 输出目录
        prefix: 文件前缀
    """
    print(f"\n📄 生成分析报告:")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成JSON报告
    report = {
        'analysis_info': {
            'timestamp': datetime.now().isoformat(),
            'groundtruth_file': gt_file,
            'radiomap_file': rm_file,
            'analysis_type': 'GroundTruth vs RadioMap Similarity'
        },
        'similarity_metrics': metrics,
        'summary': {
            'overall_similarity': '待评估',
            'recommendations': []
        }
    }
    
    # 评估整体相似性
    recommendations = []
    
    if 'pearson_correlation' in metrics and not np.isnan(metrics['pearson_correlation']['value']):
        corr = metrics['pearson_correlation']['value']
        if corr > 0.9:
            report['summary']['overall_similarity'] = '优秀'
            recommendations.append('数据高度相关，RadioMap质量很好')
        elif corr > 0.7:
            report['summary']['overall_similarity'] = '良好'
            recommendations.append('数据相关性较好，RadioMap质量可接受')
        elif corr > 0.5:
            report['summary']['overall_similarity'] = '一般'
            recommendations.append('数据相关性一般，建议进一步优化RadioMap')
        else:
            report['summary']['overall_similarity'] = '较差'
            recommendations.append('数据相关性较低，需要重新评估RadioMap生成方法')
    
    if 'rmse' in metrics and not np.isnan(metrics['rmse']['value']):
        rmse = metrics['rmse']['value']
        if rmse < 5:
            recommendations.append('RMSE较小，预测精度较高')
        elif rmse < 10:
            recommendations.append('RMSE中等，预测精度可接受')
        else:
            recommendations.append('RMSE较大，预测精度需要改进')
    
    report['summary']['recommendations'] = recommendations
    
    # 保存JSON报告
    json_path = os.path.join(output_dir, f"{prefix}_report.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"   - JSON报告: {json_path}")
    
    # 生成文本报告
    txt_path = os.path.join(output_dir, f"{prefix}_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("GroundTruth vs RadioMap 相似性分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"分析时间: {report['analysis_info']['timestamp']}\n")
        f.write(f"GroundTruth文件: {gt_file}\n")
        f.write(f"RadioMap文件: {rm_file}\n\n")
        
        f.write("相似性指标:\n")
        f.write("-" * 20 + "\n")
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'value' in metric_data:
                value = metric_data['value']
                interpretation = metric_data.get('interpretation', '')
                if not np.isnan(value):
                    f.write(f"{metric_name}: {value:.4f} ({interpretation})\n")
                else:
                    f.write(f"{metric_name}: 计算失败\n")
        
        f.write(f"\n整体评估: {report['summary']['overall_similarity']}\n\n")
        
        f.write("建议:\n")
        f.write("-" * 10 + "\n")
        for rec in recommendations:
            f.write(f"- {rec}\n")
    
    print(f"   - 文本报告: {txt_path}")


def analyze_groundtruth_radiomap(gt_file, rm_file, output_dir=None):
    """
    分析GroundTruth和RadioMap数据的相似性
    
    Args:
        gt_file: GroundTruth文件路径
        rm_file: RadioMap文件路径
        output_dir: 输出目录
    
    Returns:
        dict: 分析结果
    """
    print("🔍 GroundTruth vs RadioMap 相似性分析")
    print("=" * 60)
    
    # 生成输出目录
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(gt_file))[0]
        output_dir = f"analysis_{base_name}"
    
    # 加载数据
    gt_data = load_groundtruth_data(gt_file)
    rm_data, rm_key = load_radiomap_data(rm_file)
    
    # 数据预处理
    gt_processed, rm_processed = preprocess_data(gt_data, rm_data, rm_key)
    
    # 计算相似性指标
    metrics = calculate_similarity_metrics(gt_processed, rm_processed)
    
    # 生成可视化
    prefix = os.path.splitext(os.path.basename(gt_file))[0]
    create_visualizations(gt_processed, rm_processed, output_dir, prefix)
    
    # 生成报告
    generate_report(gt_file, rm_file, metrics, output_dir, prefix)
    
    print(f"\n✅ 分析完成!")
    print(f"   - 输出目录: {output_dir}")
    print(f"   - 整体相似性: {metrics.get('pearson_correlation', {}).get('value', 'N/A'):.4f}")
    
    return {
        'metrics': metrics,
        'output_dir': output_dir,
        'gt_shape': gt_processed.shape,
        'rm_shape': rm_processed.shape
    }


def main():
    parser = argparse.ArgumentParser(
        description="分析GroundTruth和RadioMap数据的相似性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 分析单个文件对
  python groundtruth_radiomap_analyzer.py GT_shanghai125.npy RM_shanghai125.mat
  
  # 指定输出目录
  python groundtruth_radiomap_analyzer.py GT_shanghai125.npy RM_shanghai125.mat -o analysis_results
  
  # 批量分析
  python groundtruth_radiomap_analyzer.py -d /path/to/data -o /path/to/output
        """
    )
    
    # 输入文件
    parser.add_argument("groundtruth", nargs='?', type=str,
                       help="GroundTruth文件路径（npy格式）")
    parser.add_argument("radiomap", nargs='?', type=str,
                       help="RadioMap文件路径（mat格式）")
    
    # 输出目录
    parser.add_argument("-o", "--output", type=str,
                       help="输出目录路径")
    
    # 批量处理
    parser.add_argument("-d", "--directory", type=str,
                       help="批量处理模式，指定包含GT和RM文件的目录")
    
    args = parser.parse_args()
    
    try:
        if args.directory:
            # 批量处理模式
            print("🔄 批量分析模式")
            # 这里可以添加批量处理逻辑
            print("批量处理功能待实现")
        else:
            # 单文件分析模式
            if not args.groundtruth or not args.radiomap:
                parser.print_help()
                sys.exit(1)
            
            analyze_groundtruth_radiomap(args.groundtruth, args.radiomap, args.output)
            
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
