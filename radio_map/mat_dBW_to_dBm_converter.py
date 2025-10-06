#!/usr/bin/env python3
"""
MAT文件dBW到dBm转换脚本
功能：
1. 将mat文件中的功率数据从dBW转换为dBm
2. 支持MATLAB v7.3格式的mat文件（使用h5py）
3. 支持传统mat文件格式（使用scipy.io）
4. 批量处理功能
"""

import argparse
import os
import sys
import numpy as np
from scipy.io import loadmat, savemat
import h5py
import json


def dbw_to_dbm(power_dbw):
    """
    将功率从分贝瓦(dBW)转换为分贝毫瓦(dBm)
    
    公式: P_dBm = P_dBW + 30
    
    因为: 1W = 1000mW
    所以: 10*log10(1000) = 30dB
    
    Args:
        power_dbw: 功率值（dBW）
    
    Returns:
        功率值（dBm）
    """
    return power_dbw + 30


def detect_mat_format(filepath):
    """
    检测MAT文件的格式
    
    Returns:
        str: 'v7.3' 或 'legacy'
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(128)
            if header.startswith(b'\x89HDF'):
                return 'v7.3'
            elif header.startswith(b'MATLAB'):
                return 'legacy'
            else:
                # 尝试用scipy加载，如果失败则可能是v7.3
                try:
                    loadmat(filepath)
                    return 'legacy'
                except Exception as e:
                    if 'HDF reader' in str(e) or 'v7.3' in str(e):
                        return 'v7.3'
                    else:
                        return 'v7.3'
    except Exception as e:
        print(f"⚠️  无法检测文件格式，默认使用v7.3: {e}")
        return 'v7.3'


def load_mat_file(filepath):
    """
    加载MAT文件，自动检测格式
    
    Returns:
        tuple: (data_dict, format_type)
    """
    # 首先尝试用h5py加载（v7.3格式）
    try:
        data_dict = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data = f[key]
                if isinstance(data, h5py.Dataset):
                    data_dict[key] = np.array(data)
                else:
                    data_dict[key] = f"Group with {len(data)} items"
        return data_dict, 'v7.3'
    except Exception as e:
        print(f"⚠️  h5py加载失败，尝试scipy: {e}")
    
    # 如果h5py失败，尝试用scipy加载（传统格式）
    try:
        data_dict = loadmat(filepath)
        # 过滤掉MATLAB内部变量
        data_dict = {k: v for k, v in data_dict.items() if not k.startswith('__')}
        return data_dict, 'legacy'
    except Exception as e:
        raise RuntimeError(f"无法加载MAT文件 {filepath}: {e}")


def save_mat_file(filepath, data_dict, format_type='v7.3'):
    """
    保存MAT文件
    
    Args:
        filepath: 输出文件路径
        data_dict: 数据字典
        format_type: 文件格式类型
    """
    if format_type == 'v7.3':
        # 使用h5py保存v7.3格式
        with h5py.File(filepath, 'w') as f:
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                else:
                    # 对于非数组数据，转换为字符串
                    f.attrs[key] = str(value)
    else:
        # 使用scipy保存传统格式
        savemat(filepath, data_dict)


def convert_mat_dBw_to_dBm(input_mat_path, output_mat_path=None, data_key=None):
    """
    将mat文件中的dBW数据转换为dBm
    
    Args:
        input_mat_path: 输入mat文件路径
        output_mat_path: 输出mat文件路径（可选，默认自动生成）
        data_key: 要转换的数据键名（可选，自动检测）
    
    Returns:
        str: 输出文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_mat_path):
        raise FileNotFoundError(f"输入文件不存在: {input_mat_path}")
    
    # 加载mat文件
    print(f"📂 正在加载mat文件: {input_mat_path}")
    data_dict, format_type = load_mat_file(input_mat_path)
    
    print(f"📊 文件格式: MATLAB {format_type}")
    print(f"📋 文件内容:")
    
    # 显示文件内容
    numeric_keys = []
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in ['f', 'i']:
            print(f"   - {key}: shape={value.shape}, dtype={value.dtype}, range=[{np.min(value):.2f}, {np.max(value):.2f}]")
            numeric_keys.append(key)
        else:
            print(f"   - {key}: {type(value).__name__}")
    
    # 自动选择要转换的数据键
    if data_key is None:
        if len(numeric_keys) == 1:
            data_key = numeric_keys[0]
            print(f"🔍 自动选择数据键: {data_key}")
        elif len(numeric_keys) > 1:
            # 选择最大的数组
            largest_key = max(numeric_keys, key=lambda k: data_dict[k].size)
            data_key = largest_key
            print(f"🔍 自动选择最大数组: {data_key}")
        else:
            raise ValueError("未找到可转换的数值数据")
    else:
        if data_key not in data_dict:
            raise KeyError(f"数据键 '{data_key}' 不存在")
        if not isinstance(data_dict[data_key], np.ndarray):
            raise TypeError(f"数据键 '{data_key}' 不是数值数组")
    
    # 获取要转换的数据
    original_data = data_dict[data_key]
    print(f"📊 原始数据信息:")
    print(f"   - 键名: {data_key}")
    print(f"   - 形状: {original_data.shape}")
    print(f"   - 数据类型: {original_data.dtype}")
    print(f"   - 数值范围: [{np.min(original_data):.2f}, {np.max(original_data):.2f}] dBW")
    print(f"   - 内存大小: {original_data.nbytes / 1024 / 1024:.2f} MB")
    
    # 执行单位转换: dBW -> dBm
    print(f"⚡ 执行功率单位转换: dBW -> dBm")
    converted_data = dbw_to_dbm(original_data)
    
    print(f"   转换后数值范围: [{np.min(converted_data):.2f}, {np.max(converted_data):.2f}] dBm")
    
    # 更新数据字典
    data_dict[data_key] = converted_data
    
    # 添加转换信息
    conversion_info = {
        'original_units': 'dBW',
        'converted_units': 'dBm',
        'conversion_formula': 'P_dBm = P_dBW + 30',
        'converted_key': data_key,
        'conversion_timestamp': np.datetime64('now').astype(str)
    }
    
    # 将转换信息添加到数据字典
    for key, value in conversion_info.items():
        if format_type == 'v7.3':
            # 对于v7.3格式，将字符串信息存储为属性
            data_dict[f'conversion_{key}'] = np.array([value], dtype='S')
        else:
            data_dict[f'conversion_{key}'] = value
    
    # 生成输出文件路径
    if output_mat_path is None:
        base_name = os.path.splitext(input_mat_path)[0]
        output_mat_path = f"{base_name}_dBm.mat"
    
    # 保存转换后的mat文件
    print(f"💾 正在保存mat文件: {output_mat_path}")
    save_mat_file(output_mat_path, data_dict, format_type)
    
    print(f"✅ 转换完成!")
    print(f"   - 输出文件: {output_mat_path}")
    print(f"   - 转换的数据键: {data_key}")
    print(f"   - 最终形状: {converted_data.shape}")
    print(f"   - 最终单位: dBm")
    print(f"   - 文件格式: MATLAB {format_type}")
    
    return output_mat_path


def batch_convert(input_dir, output_dir=None, pattern="*.mat", data_key=None):
    """
    批量转换目录中的mat文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选）
        pattern: 文件匹配模式
        data_key: 要转换的数据键名
    """
    import glob
    
    if output_dir is None:
        output_dir = input_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有mat文件
    mat_files = glob.glob(os.path.join(input_dir, pattern))
    
    if not mat_files:
        print(f"❌ 在目录 {input_dir} 中未找到匹配 {pattern} 的文件")
        return
    
    print(f"🔍 找到 {len(mat_files)} 个mat文件")
    
    for mat_file in mat_files:
        try:
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(mat_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_dBm.mat")
            
            print(f"\n{'='*60}")
            print(f"处理文件: {mat_file}")
            
            convert_mat_dBw_to_dBm(mat_file, output_file, data_key)
            
        except Exception as e:
            print(f"❌ 处理文件 {mat_file} 时出错: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ 批量转换完成!")


def main():
    parser = argparse.ArgumentParser(
        description="将MAT文件中的功率数据从dBW转换为dBm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换单个文件
  python mat_dBW_to_dBm_converter.py input.mat
  
  # 指定输出文件名
  python mat_dBW_to_dBm_converter.py input.mat -o output.mat
  
  # 指定要转换的数据键名
  python mat_dBW_to_dBm_converter.py input.mat -k XdB_recon_tensor
  
  # 批量转换目录中的所有mat文件
  python mat_dBW_to_dBm_converter.py -d /path/to/mat/files
  
  # 批量转换并指定输出目录
  python mat_dBW_to_dBm_converter.py -d /path/to/mat/files -o /path/to/output
        """
    )
    
    # 输入文件或目录
    parser.add_argument("input", nargs='?', type=str, 
                       help="输入mat文件路径或包含mat文件的目录")
    
    # 输出文件或目录
    parser.add_argument("-o", "--output", type=str,
                       help="输出mat文件路径或输出目录")
    
    # 数据键名
    parser.add_argument("-k", "--key", type=str,
                       help="要转换的数据键名（可选，自动检测）")
    
    # 批量处理模式
    parser.add_argument("-d", "--directory", action="store_true",
                       help="批量处理模式，处理目录中的所有mat文件")
    
    # 文件匹配模式
    parser.add_argument("-p", "--pattern", type=str, default="*.mat",
                       help="批量处理时的文件匹配模式 (默认: *.mat)")
    
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
            batch_convert(args.input, args.output, args.pattern, args.key)
        else:
            # 单文件处理模式
            convert_mat_dBw_to_dBm(args.input, args.output, args.key)
            
    except Exception as e:
        print(f"❌ 转换过程中出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
