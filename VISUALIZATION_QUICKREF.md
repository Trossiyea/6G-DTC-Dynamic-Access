# 可视化快速参考卡片 📊

## 🎯 一句话上手

```bash
# 方式1: 运行测试并可视化（首次）
python visualize_results.py --scenarios --run

# 方式2: 从已有数据可视化（快速）
python visualize_results.py --scenarios
python visualize_results.py --resolution
```

## 📋 常用命令

| 功能 | 命令 |
|------|------|
| **四场景对比（运行测试）** | `python visualize_results.py --scenarios --run` |
| **四场景对比（已有数据）** | `python visualize_results.py --scenarios` |
| **分辨率对比** | `python visualize_results.py --resolution` |
| **全部可视化** | `python visualize_results.py --all` |

## 🔄 完整工作流

```bash
# 1. 运行四场景测试
python run_test.py --all

# 2. 运行分辨率对比
python run_resolution_comparison.py --all --save-report

# 3. 生成所有图表
python visualize_results.py --all

# 查看结果
ls -lh output/*.png
```

## 📁 输入/输出文件

**输入数据：**
- `results/four_scenarios_results.json` (四场景)
- `results/resolution_comparison.json` (分辨率对比)

**输出图表：**
- `output/four_scenarios_comparison.png` (16×6 英寸, 300 DPI)
- `output/resolution_comparison.png` (16×12 英寸, 300 DPI)

## 💡 使用技巧

- 首次运行用 `--run` 参数
- 调整样式用 `--scenarios` 或 `--resolution` (从JSON)
- 图表自动保存，同时弹出窗口显示
- 关闭窗口后脚本继续执行

## ⚡ 故障排查速查

| 错误 | 解决方案 |
|------|---------|
| 找不到JSON文件 | 先运行测试生成数据 |
| 缺少matplotlib | `pip install matplotlib numpy` |
| 测试很慢 | 减少T和N_UE参数，或用小数据集测试 |

