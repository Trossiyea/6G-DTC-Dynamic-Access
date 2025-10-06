# NPY到MAT文件转换工具

## 功能说明

这个脚本可以将NPY文件转换为MAT文件，并进行以下处理：

1. **维度重排**: 将原始维度 `(x,y,z)` 重排为 `(y,z,x)`
2. **功率单位转换**: 将功率从瓦特(W)转换为分贝毫瓦(dBm)

## 转换公式

- **功率转换**: `P_dBm = 10 * log10(P_W * 1000)`
- **维度重排**: `(x,y,z) → (y,z,x)`

## 使用方法

### 单文件转换

```bash
# 基本用法
python npy_to_mat_converter.py input.npy

# 指定输出文件名
python npy_to_mat_converter.py input.npy -o output.mat

# 指定变量名
python npy_to_mat_converter.py input.npy -v power_data
```

### 批量转换

```bash
# 转换目录中的所有npy文件
python npy_to_mat_converter.py -d /path/to/npy/files

# 批量转换并指定输出目录
python npy_to_mat_converter.py -d /path/to/npy/files -o /path/to/output
```

## 输出文件内容

转换后的MAT文件包含以下变量：

- `data`: 转换后的数据（维度重排 + 单位转换）
- `original_shape`: 原始数据形状
- `converted_shape`: 转换后数据形状
- `units`: 数据单位（'dBm'）
- `conversion_info`: 转换信息（原始单位、维度重排方式、转换公式）

## 示例

### 转换GT_shanghai125.npy

```bash
python npy_to_mat_converter.py GT_shanghai125.npy -o GT_shanghai125_converted.mat
```

**转换结果**:
- 原始形状: `(51, 40, 40)`
- 转换后形状: `(40, 40, 51)`
- 原始数值范围: `[4.91e-22, 5.84e-06] W`
- 转换后数值范围: `[-170.00, -22.34] dBm`

### 转换GT_shanghai150.npy

```bash
python npy_to_mat_converter.py GT_shanghai150.npy -o GT_shanghai150_converted.mat
```

**转换结果**:
- 原始形状: `(51, 34, 34)`
- 转换后形状: `(34, 34, 51)`
- 原始数值范围: `[1.91e-17, 4.56e-06] W`
- 转换后数值范围: `[-137.19, -23.41] dBm`

## 注意事项

1. 确保已安装必要的Python包：`numpy`, `scipy`
2. 脚本会自动处理数值稳定性（避免log(0)的情况）
3. 转换过程中会显示详细的进度信息
4. 支持任意维度的数据，但主要针对3D数据优化

## 环境要求

- Python 3.7+
- numpy
- scipy
