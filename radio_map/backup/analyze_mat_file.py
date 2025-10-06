#!/usr/bin/env python3
"""
MAT文件格式和内容分析脚本
用于分析MATLAB .mat文件的结构、变量和数据类型
"""

import argparse
import os
import sys
import numpy as np
from scipy.io import loadmat
import h5py
import json


def analyze_with_scipy(filepath):
    """使用scipy.io.loadmat分析MAT文件"""
    print("=== 使用 scipy.io.loadmat 分析 ===")
    try:
        mdict = loadmat(filepath)
        print(f"✅ 成功加载文件: {filepath}")
        
        # 过滤掉MATLAB内部变量
        user_vars = {k: v for k, v in mdict.items() if not k.startswith('__')}
        
        print(f"📊 用户变量数量: {len(user_vars)}")
        print(f"📋 变量列表:")
        
        for var_name, var_data in user_vars.items():
            print(f"  - {var_name}:")
            print(f"    类型: {type(var_data)}")
            if hasattr(var_data, 'shape'):
                print(f"    形状: {var_data.shape}")
                print(f"    数据类型: {var_data.dtype}")
                if var_data.size > 0:
                    print(f"    数值范围: [{np.min(var_data):.6f}, {np.max(var_data):.6f}]")
                    print(f"    内存大小: {var_data.nbytes / 1024 / 1024:.2f} MB")
            else:
                print(f"    内容: {var_data}")
            print()
        
        return user_vars
        
    except Exception as e:
        print(f"❌ scipy.io.loadmat 失败: {e}")
        return None


def analyze_with_h5py(filepath):
    """使用h5py分析MAT文件"""
    print("=== 使用 h5py 分析 ===")
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"✅ 成功打开HDF5文件: {filepath}")
            
            def print_structure(name, obj):
                indent = "  " * (name.count('/'))
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}📄 {name.split('/')[-1]} (Dataset):")
                    print(f"{indent}  形状: {obj.shape}")
                    print(f"{indent}  数据类型: {obj.dtype}")
                    if obj.size > 0:
                        try:
                            data = np.array(obj)
                            print(f"{indent}  数值范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
                            print(f"{indent}  内存大小: {data.nbytes / 1024 / 1024:.2f} MB")
                        except:
                            print(f"{indent}  无法读取数值范围")
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name.split('/')[-1]} (Group):")
            
            print("📋 文件结构:")
            f.visititems(print_structure)
            
            # 获取顶级变量
            top_level_vars = {}
            for key in f.keys():
                try:
                    data = f[key]
                    if isinstance(data, h5py.Dataset):
                        top_level_vars[key] = np.array(data)
                    else:
                        top_level_vars[key] = f"Group with {len(data)} items"
                except Exception as e:
                    top_level_vars[key] = f"Error reading: {e}"
            
            return top_level_vars
            
    except Exception as e:
        print(f"❌ h5py 分析失败: {e}")
        return None


def get_file_info(filepath):
    """获取文件基本信息"""
    print("=== 文件基本信息 ===")
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return
    
    stat = os.stat(filepath)
    print(f"📁 文件路径: {filepath}")
    print(f"📏 文件大小: {stat.st_size / 1024 / 1024:.2f} MB")
    print(f"📅 修改时间: {stat.st_mtime}")
    
    # 尝试检测MAT文件版本
    try:
        with open(filepath, 'rb') as f:
            header = f.read(128)
            if header.startswith(b'MATLAB'):
                print(f"🔍 MATLAB版本: {header[:20].decode('ascii', errors='ignore')}")
            elif header.startswith(b'\x89HDF'):
                print("🔍 检测到HDF5格式 (MATLAB v7.3)")
            else:
                print("🔍 未知文件格式")
    except Exception as e:
        print(f"❌ 无法读取文件头: {e}")

def save_analysis_report(filepath, scipy_data, h5py_data):
    """保存分析报告到JSON文件"""
    report = {
        "filepath": filepath,
        "scipy_analysis": {},
        "h5py_analysis": {}
    }
    
    if scipy_data:
        for var_name, var_data in scipy_data.items():
            if hasattr(var_data, 'shape'):
                report["scipy_analysis"][var_name] = {
                    "type": str(type(var_data)),
                    "shape": var_data.shape,
                    "dtype": str(var_data.dtype),
                    "size_mb": var_data.nbytes / 1024 / 1024
                }
            else:
                report["scipy_analysis"][var_name] = {
                    "type": str(type(var_data)),
                    "content": str(var_data)
                }
    
    if h5py_data:
        for var_name, var_data in h5py_data.items():
            if isinstance(var_data, np.ndarray):
                report["h5py_analysis"][var_name] = {
                    "type": "numpy.ndarray",
                    "shape": var_data.shape,
                    "dtype": str(var_data.dtype),
                    "size_mb": var_data.nbytes / 1024 / 1024
                }
            else:
                report["h5py_analysis"][var_name] = {
                    "type": str(type(var_data)),
                    "content": str(var_data)
                }
    
    # 保存报告
    report_path = filepath.replace('.mat', '_analysis.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 分析报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="分析MATLAB .mat文件的格式和内容"
    )
    parser.add_argument("input_mat", type=str, help="输入.mat文件路径")
    parser.add_argument(
        "--save-report", 
        action="store_true", 
        help="保存分析报告到JSON文件"
    )
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input_mat):
        print(f"❌ 文件不存在: {args.input_mat}")
        sys.exit(1)
    
    print(f"🔍 开始分析MAT文件: {args.input_mat}")
    print("=" * 60)
    
    # 获取文件基本信息
    get_file_info(args.input_mat)
    print()
    
    # 使用scipy分析
    scipy_data = analyze_with_scipy(args.input_mat)
    print()
    
    # 使用h5py分析
    h5py_data = analyze_with_h5py(args.input_mat)
    print()
    
    # 保存分析报告
    if args.save_report:
        save_analysis_report(args.input_mat, scipy_data, h5py_data)
    
    print("=" * 60)
    print("✅ 分析完成!")


if __name__ == "__main__":
    main()
