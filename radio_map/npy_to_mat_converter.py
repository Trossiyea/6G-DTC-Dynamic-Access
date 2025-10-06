#!/usr/bin/env python3
"""
NPY到MAT文件转换脚本
功能：
1. 将npy文件转换为mat文件
2. 维度重排：从(x,y,z)转换为(y,z,x)
3. 功率单位转换：从W转换为dBm
"""

import argparse
import os
import sys
import numpy as np
from scipy.io import savemat
import json


def watts_to_dbm(power_watts):
    """
    将功率从瓦特(W)转换为分贝毫瓦(dBm)
    
    公式: P_dBm = 10 * log10(P_W * 1000)
    
    Args:
        power_watts: 功率值（瓦特）
    
    Returns:
        功率值（dBm）
    """
    # 避免log(0)的情况
    power_watts = np.maximum(power_watts, 1e-20)
    return 10 * np.log10(power_watts * 1000)


def convert_npy_to_mat(input_npy_path, output_mat_path=None, variable_name='data'):
    """
    将npy文件转换为mat文件，并进行维度重排和单位转换
    
    Args:
        input_npy_path: 输入npy文件路径
        output_mat_path: 输出mat文件路径（可选，默认自动生成）
        variable_name: mat文件中的变量名
    
    Returns:
        str: 输出文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_npy_path):
        raise FileNotFoundError(f"输入文件不存在: {input_npy_path}")
    
    # 加载npy文件
    print(f"📂 正在加载npy文件: {input_npy_path}")
    data = np.load(input_npy_path)
    
    print(f"📊 原始数据信息:")
    print(f"   - 形状: {data.shape}")
    print(f"   - 数据类型: {data.dtype}")
    print(f"   - 数值范围: [{np.min(data):.6e}, {np.max(data):.6e}]")
    print(f"   - 内存大小: {data.nbytes / 1024 / 1024:.2f} MB")
    
    # 检查数据维度
    if len(data.shape) != 3:
        print(f"⚠️  警告: 数据维度为{len(data.shape)}D，期望3D数据(x,y,z)")
        print(f"   当前形状: {data.shape}")
    
    # 维度重排: (x,y,z) -> (y,z,x)
    print(f"🔄 执行维度重排: {data.shape} -> ", end="")
    if len(data.shape) == 3:
        # 假设原始维度为(x,y,z)，重排为(y,z,x)
        data_reordered = np.transpose(data, (1, 2, 0))
        print(f"{data_reordered.shape}")
    else:
        print("跳过维度重排（非3D数据）")
        data_reordered = data
    
    # 功率单位转换: W -> dBm
    print(f"⚡ 执行功率单位转换: W -> dBm")
    print(f"   转换前数值范围: [{np.min(data_reordered):.6e}, {np.max(data_reordered):.6e}]")
    
    data_dbm = watts_to_dbm(data_reordered)
    
    print(f"   转换后数值范围: [{np.min(data_dbm):.2f}, {np.max(data_dbm):.2f}] dBm")
    
    # 生成输出文件路径
    if output_mat_path is None:
        base_name = os.path.splitext(input_npy_path)[0]
        output_mat_path = f"{base_name}_converted.mat"
    
    # 保存为mat文件
    print(f"💾 正在保存mat文件: {output_mat_path}")
    
    # 准备保存的数据字典
    mat_data = {
        variable_name: data_dbm,
        'original_shape': data.shape,
        'converted_shape': data_dbm.shape,
        'units': 'dBm',
        'conversion_info': {
            'original_units': 'W',
            'dimension_reorder': 'x,y,z -> y,z,x',
            'power_conversion': 'P_dBm = 10 * log10(P_W * 1000)'
        }
    }
    
    # 保存mat文件
    savemat(output_mat_path, mat_data)
    
    print(f"✅ 转换完成!")
    print(f"   - 输出文件: {output_mat_path}")
    print(f"   - 变量名: {variable_name}")
    print(f"   - 最终形状: {data_dbm.shape}")
    print(f"   - 最终单位: dBm")
    
    return output_mat_path


def batch_convert(input_dir, output_dir=None, pattern="*.npy"):
    """
    批量转换目录中的npy文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选）
        pattern: 文件匹配模式
    """
    import glob
    
    if output_dir is None:
        output_dir = input_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有npy文件
    npy_files = glob.glob(os.path.join(input_dir, pattern))
    
    if not npy_files:
        print(f"❌ 在目录 {input_dir} 中未找到匹配 {pattern} 的文件")
        return
    
    print(f"🔍 找到 {len(npy_files)} 个npy文件")
    
    for npy_file in npy_files:
        try:
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(npy_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_converted.mat")
            
            print(f"\n{'='*60}")
            print(f"处理文件: {npy_file}")
            
            convert_npy_to_mat(npy_file, output_file)
            
        except Exception as e:
            print(f"❌ 处理文件 {npy_file} 时出错: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ 批量转换完成!")


def main():
    parser = argparse.ArgumentParser(
        description="将NPY文件转换为MAT文件，支持维度重排和功率单位转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换单个文件
  python npy_to_mat_converter.py input.npy
  
  # 指定输出文件名
  python npy_to_mat_converter.py input.npy -o output.mat
  
  # 指定变量名
  python npy_to_mat_converter.py input.npy -v power_data
  
  # 批量转换目录中的所有npy文件
  python npy_to_mat_converter.py -d /path/to/npy/files
  
  # 批量转换并指定输出目录
  python npy_to_mat_converter.py -d /path/to/npy/files -o /path/to/output
        """
    )
    
    # 输入文件或目录
    parser.add_argument("input", nargs='?', type=str, 
                       help="输入npy文件路径或包含npy文件的目录")
    
    # 输出文件或目录
    parser.add_argument("-o", "--output", type=str,
                       help="输出mat文件路径或输出目录")
    
    # 变量名
    parser.add_argument("-v", "--variable", type=str, default="data",
                       help="mat文件中的变量名 (默认: data)")
    
    # 批量处理模式
    parser.add_argument("-d", "--directory", action="store_true",
                       help="批量处理模式，处理目录中的所有npy文件")
    
    # 文件匹配模式
    parser.add_argument("-p", "--pattern", type=str, default="*.npy",
                       help="批量处理时的文件匹配模式 (默认: *.npy)")
    
    args = parser.parse_args()
    
    # 检查输入参数
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"❌ 输入路径不存在: {args.input}")
        sys.exit(1)
    
    try:
        if args.directory or os.path.isdir(args.input):
            # 批量处理模式
            batch_convert(args.input, args.output, args.pattern)
        else:
            # 单文件处理模式
            convert_npy_to_mat(args.input, args.output, args.variable)
            
    except Exception as e:
        print(f"❌ 转换过程中出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
