# 测试结果可视化指南

本指南介绍如何使用 `visualize_results.py` 脚本可视化仿真结果。

## 📊 可视化内容

### 1. 四场景性能对比
展示以下四个场景的 Baseline vs RadioMap 性能对比：
- **Toronto Single**: Toronto 单星场景
- **Toronto Constellation**: Toronto 星座场景
- **Shanghai Single**: Shanghai 单星场景
- **Shanghai Constellation**: Shanghai 星座场景

**输出图表包含：**
- 左图：Baseline 和 RadioMap 的绝对 SE 值对比
- 右图：RadioMap 相对于 Baseline 的提升百分比

### 2. 分辨率影响对比
展示 125m vs 150m 两种 Radio Map 分辨率的性能影响

**输出图表包含：**
- 左上：Baseline SE 对比 (125m vs 150m)
- 右上：RadioMap SE 对比 (125m vs 150m)
- 左下：RadioMap 增益对比 (125m vs 150m)
- 右下：增益差异 (150m - 125m)

---

## 🚀 快速开始

### 方式一：运行测试并可视化（推荐用于首次运行）

```bash
# 1. 运行四个主场景测试并可视化
python visualize_results.py --scenarios --run

# 2. 运行分辨率对比测试
python run_resolution_comparison.py --all --save-report

# 3. 可视化分辨率对比结果
python visualize_results.py --resolution
```

### 方式二：从已有数据可视化

```bash
# 如果已经运行过测试，可以直接从保存的数据可视化

# 可视化四场景结果（从 results/four_scenarios_results.json）
python visualize_results.py --scenarios

# 可视化分辨率对比（从 results/resolution_comparison.json）
python visualize_results.py --resolution

# 一次性可视化所有结果
python visualize_results.py --all
```

---

## 📝 详细使用说明

### 命令行参数

```
--scenarios       可视化四个主场景结果
--resolution      可视化分辨率对比结果
--all             可视化所有结果（等同于 --scenarios --resolution）
--run             运行测试（仅用于四场景可视化）
--input PATH      指定分辨率对比JSON文件路径（默认: results/resolution_comparison.json）
```

### 使用场景

#### 场景 1: 首次运行 - 四场景对比

```bash
# Step 1: 运行测试并立即可视化
python visualize_results.py --scenarios --run
```

**说明：**
- 会自动运行四个场景的测试
- 结果保存到 `results/four_scenarios_results.json`
- 立即生成可视化图表
- 图表保存到 `output/four_scenarios_comparison.png`

**预计耗时：** 取决于配置，单场景约 5-15 分钟，总计 20-60 分钟

#### 场景 2: 已有测试结果 - 重新可视化

```bash
# 从已保存的数据重新生成图表
python visualize_results.py --scenarios
```

**说明：**
- 不运行测试，直接从 JSON 读取数据
- 适合调整图表样式或重新生成图表
- 几秒内完成

#### 场景 3: 分辨率对比可视化

```bash
# Step 1: 运行分辨率对比测试（如果还没运行）
python run_resolution_comparison.py --all --save-report

# Step 2: 可视化结果
python visualize_results.py --resolution
```

**说明：**
- 读取 `results/resolution_comparison.json`
- 生成 4 个子图的对比图表
- 图表保存到 `output/resolution_comparison.png`

#### 场景 4: 使用自定义数据文件

```bash
# 指定自定义的分辨率对比JSON文件
python visualize_results.py --resolution --input my_results/custom_comparison.json
```

#### 场景 5: 一次性可视化所有结果

```bash
# 必须确保两个JSON文件都存在
python visualize_results.py --all
```

---

## 📂 文件结构

### 输入文件

1. **四场景结果数据**
   - 路径：`results/four_scenarios_results.json`
   - 来源：运行 `visualize_results.py --scenarios --run` 自动生成
   - 或：运行 `run_test.py` 手动测试后自动保存

2. **分辨率对比数据**
   - 路径：`results/resolution_comparison.json`
   - 来源：运行 `run_resolution_comparison.py --all --save-report` 生成

### 输出文件

1. **四场景对比图**
   - 路径：`output/four_scenarios_comparison.png`
   - 格式：高分辨率 PNG (300 DPI)
   - 尺寸：16×6 英寸

2. **分辨率对比图**
   - 路径：`output/resolution_comparison.png`
   - 格式：高分辨率 PNG (300 DPI)
   - 尺寸：16×12 英寸

---

## 🎨 图表说明

### 四场景对比图表解读

**左图：Baseline vs RadioMap Performance Comparison**
- X 轴：四个场景名称
- Y 轴：频谱效率 (bits/s/Hz)
- 灰色柱：Baseline 算法性能
- 彩色柱：RadioMap 算法性能（不同场景用不同颜色区分）
- 数值标签：显示具体 SE 值（保留 3 位小数）

**右图：RadioMap Gain over Baseline**
- X 轴：四个场景名称
- Y 轴：提升百分比 (%)
- 彩色柱：RadioMap 相对于 Baseline 的提升
- 红色虚线：0% 基准线
- 数值标签：显示具体提升百分比（带正负号）

### 分辨率对比图表解读

**左上：Baseline Performance: 125m vs 150m Resolution**
- 对比两种分辨率下 Baseline 算法的性能
- 蓝色：125m 分辨率
- 红色：150m 分辨率

**右上：RadioMap Performance: 125m vs 150m Resolution**
- 对比两种分辨率下 RadioMap 算法的性能
- 绿色：125m 分辨率
- 深绿：150m 分辨率

**左下：RadioMap Gain over Baseline: 125m vs 150m**
- 对比两种分辨率下 RadioMap 的相对增益
- 紫色：125m 分辨率增益
- 深紫：150m 分辨率增益
- 红色虚线：0% 基准线

**右下：Resolution Impact: Gain(150m) - Gain(125m)**
- 显示分辨率变化对增益的影响
- 绿色柱：正值，表示 150m 分辨率增益更高
- 红色柱：负值，表示 125m 分辨率增益更高
- 黑色实线：0 基准线

---

## 💡 最佳实践

### 1. 完整工作流程

```bash
# Step 1: 运行四个主场景测试
python run_test.py --all

# Step 2: 可视化四场景结果
python visualize_results.py --scenarios

# Step 3: 运行分辨率对比测试
python run_resolution_comparison.py --all --save-report

# Step 4: 可视化分辨率对比
python visualize_results.py --resolution

# Step 5: 一次性重新生成所有图表
python visualize_results.py --all
```

### 2. 快速测试流程（用于演示）

```bash
# 使用 --run 参数直接运行测试并可视化（更快）
python visualize_results.py --scenarios --run
```

### 3. 批量处理

```bash
# 创建自动化脚本
cat << 'EOF' > run_all_and_visualize.sh
#!/bin/bash
set -e

echo "Running all test scenarios..."
python run_test.py --all

echo "Running resolution comparison..."
python run_resolution_comparison.py --all --save-report

echo "Generating visualizations..."
python visualize_results.py --all

echo "All done! Check output/ directory for figures."
ls -lh output/*.png
EOF

chmod +x run_all_and_visualize.sh
./run_all_and_visualize.sh
```

---

## 🔧 自定义和扩展

### 修改图表样式

编辑 `visualize_results.py`，可以调整：
- **颜色方案**：修改 `color` 参数（第 118-133 行）
- **图表尺寸**：修改 `figsize` 参数（第 190、312 行）
- **字体大小**：修改 `fontsize` 参数
- **DPI**：修改 `dpi` 参数（第 243、419 行）

### 添加新场景

在 `visualize_results.py` 的 `scenarios` 字典中添加：

```python
scenarios = {
    ...
    'new_scenario': {
        'name': 'New\nScenario',
        'config': 'test/config_new_scenario.py',
        'color': '#hexcolor',
    },
}
```

---

## 📊 示例输出

### 四场景对比示例统计

```
📈 统计摘要
======================================================================
平均Baseline SE:  2.3456 bits/s/Hz
平均RadioMap SE:  2.5678 bits/s/Hz
平均提升百分比:   9.47%
最大提升百分比:   12.34% (Shanghai
Constellation)
最小提升百分比:   7.89% (Toronto
Single)
======================================================================
```

### 分辨率对比示例统计

```
📈 分辨率影响统计摘要
======================================================================

TORONTO:
  125m分辨率 - RadioMap增益: +10.25%
  150m分辨率 - RadioMap增益: +10.53%
  增益差异 (150m - 125m):   +0.28 百分点
  结论: 分辨率差异影响较小

SHANGHAI:
  125m分辨率 - RadioMap增益: +11.47%
  150m分辨率 - RadioMap增益: +11.92%
  增益差异 (150m - 125m):   +0.45 百分点
  结论: 分辨率差异影响较小
======================================================================
```

---

## ⚠️ 故障排查

### 问题 1: `FileNotFoundError: results/four_scenarios_results.json`

**原因：** 没有运行测试或数据文件不存在

**解决方案：**
```bash
# 方式 1: 使用 --run 参数
python visualize_results.py --scenarios --run

# 方式 2: 先运行测试
python run_test.py --all
python visualize_results.py --scenarios
```

### 问题 2: `FileNotFoundError: results/resolution_comparison.json`

**原因：** 没有运行分辨率对比测试

**解决方案：**
```bash
python run_resolution_comparison.py --all --save-report
python visualize_results.py --resolution
```

### 问题 3: 图表显示中文乱码

**原因：** 系统缺少中文字体

**解决方案：**
- macOS: 自动使用 'Arial Unicode MS'（无需额外配置）
- Linux: 安装 `sudo apt-get install fonts-wqy-zenhei`
- Windows: 脚本会自动使用 SimHei 字体

### 问题 4: `ModuleNotFoundError: No module named 'matplotlib'`

**解决方案：**
```bash
pip install matplotlib numpy
```

### 问题 5: 测试运行很慢

**说明：**
- 单个场景根据配置可能需要 5-15 分钟
- 四个场景总计约 20-60 分钟
- 建议先用小参数测试，确认可行后再运行完整测试

**加速建议：**
- 减少 `T` (TTI数量) 和 `N_UE` (用户数)
- 使用 `--first` 只测试第一个卫星（constellation 模式）

---

## 📚 相关文档

- `README.md`: 项目总体说明
- `QUICKSTART.md`: 快速开始指南
- `TEST_SCENARIOS.md`: 测试场景详解
- `VERIFICATION.md`: 验证和测试指南

---

## 💬 提示

1. **图表保存位置**: 所有图表自动保存在 `output/` 目录
2. **数据保存位置**: 所有 JSON 数据自动保存在 `results/` 目录
3. **高分辨率**: 图表默认 300 DPI，适合论文和演示使用
4. **实时显示**: 运行后会自动弹出图表窗口（关闭窗口后继续执行）
5. **批量处理**: 可以使用 `--all` 参数一次性生成所有图表
