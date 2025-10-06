# MAT文件dBW到dBm转换工具

## 功能说明

这个脚本可以将MAT文件中的功率数据从dBW转换为dBm，支持以下功能：

1. **功率单位转换**: 将功率从分贝瓦(dBW)转换为分贝毫瓦(dBm)
2. **自动格式检测**: 支持MATLAB v7.3格式（使用h5py）和传统格式（使用scipy.io）
3. **自动数据检测**: 自动识别要转换的数值数据
4. **单文件转换**: 支持转换单个mat文件
5. **批量转换**: 支持批量处理目录中的所有mat文件

## 转换公式

- **功率转换**: `P_dBm = P_dBW + 30`
- **原理**: 因为 1W = 1000mW，所以 10*log10(1000) = 30dB

## 使用方法

### 单文件转换

```bash
# 基本用法
python mat_dBW_to_dBm_converter.py input.mat

# 指定输出文件名
python mat_dBW_to_dBm_converter.py input.mat -o output.mat

# 指定要转换的数据键名
python mat_dBW_to_dBm_converter.py input.mat -k XdB_recon_tensor
```

### 批量转换

```bash
# 转换目录中的所有mat文件
python mat_dBW_to_dBm_converter.py -d /path/to/mat/files

# 批量转换并指定输出目录
python mat_dBW_to_dBm_converter.py -d /path/to/mat/files -o /path/to/output
```

## 输出文件内容

转换后的MAT文件包含以下内容：

- **原始数据**: 转换后的功率数据（单位：dBm）
- **转换信息**: 自动添加的转换元数据
  - `conversion_original_units`: 原始单位（'dBW'）
  - `conversion_converted_units`: 转换后单位（'dBm'）
  - `conversion_conversion_formula`: 转换公式
  - `conversion_converted_key`: 转换的数据键名
  - `conversion_conversion_timestamp`: 转换时间戳

## 示例

### 转换RM_shanghai125.mat

```bash
python mat_dBW_to_dBm_converter.py RM_shanghai125.mat -o RM_shanghai125_dBm.mat
```

**转换结果**:
- 原始数值范围: `[-213.09, -52.34] dBW`
- 转换后数值范围: `[-183.09, -22.34] dBm`
- 数据形状: `(40, 40, 51)`
- 文件格式: MATLAB v7.3

### 转换RM_shanghai150.mat

```bash
python mat_dBW_to_dBm_converter.py RM_shanghai150.mat -o RM_shanghai150_dBm.mat
```

**转换结果**:
- 原始数值范围: `[-163.82, -53.41] dBW`
- 转换后数值范围: `[-133.82, -23.41] dBm`
- 数据形状: `(34, 34, 51)`
- 文件格式: MATLAB v7.3

## 支持的文件格式

### MATLAB v7.3格式
- 使用HDF5格式存储
- 支持大文件和复杂数据结构
- 使用h5py库读取和写入

### 传统MAT格式
- 使用MATLAB传统格式
- 兼容性更好
- 使用scipy.io库读取和写入

## 自动检测功能

脚本会自动：
1. **检测文件格式**: 自动识别MATLAB v7.3或传统格式
2. **选择数据键**: 自动选择最大的数值数组进行转换
3. **处理错误**: 如果一种格式失败，自动尝试另一种格式

## 注意事项

1. **环境要求**: 确保已安装必要的Python包：`numpy`, `scipy`, `h5py`
2. **数据安全**: 转换过程不会修改原始文件，会创建新的输出文件
3. **重复转换**: 避免对已转换的文件再次转换，会导致数值错误
4. **内存使用**: 大文件会占用相应内存，注意系统资源

## 环境要求

- Python 3.7+
- numpy
- scipy
- h5py

## 故障排除

### 常见错误

1. **"Please use HDF reader for matlab v7.3 files"**
   - 解决方案: 脚本会自动处理，无需手动干预

2. **"未找到可转换的数值数据"**
   - 解决方案: 使用 `-k` 参数指定数据键名

3. **内存不足**
   - 解决方案: 分批处理大文件，或增加系统内存

### 调试模式

脚本会显示详细的转换信息，包括：
- 文件格式检测结果
- 数据结构和范围
- 转换过程状态
- 输出文件信息
