#!/usr/bin/env python3
"""
快速分析 .npy 文件的简化脚本
"""

import numpy as np
import sys
from pathlib import Path

def quick_analyze(file_path):
    """快速分析 .npy 文件"""
    try:
        data = np.load(file_path)
        
        print(f"文件: {Path(file_path).name}")
        print(f"形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"总元素数: {data.size:,}")
        print(f"文件大小: {data.nbytes / (1024*1024):.2f} MB")
        
        if np.issubdtype(data.dtype, np.number):
            print(f"数值范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
            print(f"平均值: {np.mean(data):.6f}")
            print(f"标准差: {np.std(data):.6f}")
            
            # 检查是否有非零值
            non_zero = np.count_nonzero(data)
            print(f"非零元素: {non_zero:,} ({non_zero/data.size*100:.1f}%)")
            
            # 检查唯一值数量
            unique_count = len(np.unique(data))
            print(f"唯一值数量: {unique_count:,}")
            
            if unique_count <= 20:
                unique_values = np.unique(data)
                print(f"唯一值: {unique_values}")
        
        return data
        
    except Exception as e:
        print(f"错误: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python quick_analyze_npy.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"文件不存在: {file_path}")
        sys.exit(1)
    
    quick_analyze(file_path)
