# 快速开始指南

## 运行测试场景的三种方式

### 🚀 方式1: 使用 Make (最简单)

```bash
# 查看帮助
make help

# 列出所有场景
make list

# 运行单个场景
make test-toronto-single
make test-toronto-constellation
make test-shanghai-single
make test-shanghai-constellation

# 运行所有场景
make test-all

# 通用命令
make test SCENARIO=toronto_single
```

### 🐍 方式2: 使用 Python 脚本

```bash
# 运行单个场景
python run_test.py --scenario toronto_single
python run_test.py -s shanghai_constellation

# 运行多个场景
python run_test.py -s toronto_single -s shanghai_single

# 运行所有场景
python run_test.py --all

# 列出所有场景
python run_test.py --list
```

### 📜 方式3: 使用 Shell 脚本

```bash
# 运行所有场景
./run_all_tests.sh
```

## 可用的测试场景

| 场景 | 命令 | 说明 |
|------|------|------|
| Toronto 单卫星 | `make test-toronto-single` | Starlink单星覆盖测试 |
| Toronto 星座 | `make test-toronto-constellation` | Starlink星座覆盖测试 |
| Shanghai 单卫星 | `make test-shanghai-single` | Satnet单星覆盖测试 |
| Shanghai 星座 | `make test-shanghai-constellation` | Satnet星座覆盖测试 |

## 示例输出

运行测试后会看到类似以下输出:

```
======================================================================
运行场景: toronto_single
配置文件: test/config_toronto_single.py
======================================================================

[Orbit] Using Skyfield/TLE orbit: STARLINK-11075 [DTC] (start=2025-10-08T09:22:45.447459+00:00)...

----------------------------------------------------------------------
场景 [toronto_single] 结果:
  基线平均SE: 3.2456 bits/s/Hz
  RadioMap平均SE: 3.8921 bits/s/Hz
  提升百分比: 19.92%
----------------------------------------------------------------------
```

## 查看详细文档

- **完整测试指南**: 查看 [TEST_SCENARIOS.md](TEST_SCENARIOS.md)
- **主项目README**: 查看 [README.md](README.md)

## 常用命令速查

```bash
# 最常用 - 运行Toronto单星测试
make test-toronto-single

# 对比两个城市的单星场景
python run_test.py -s toronto_single -s shanghai_single

# 运行所有场景并对比结果
make test-all

# 清理Python缓存
make clean
```

## 首次使用验证

运行以下命令验证安装：

```bash
# 1. 验证脚本可以列出场景
python run_test.py --list

# 2. 验证 Make 命令可用
make help

# 3. 验证所有场景配置是否正确
make verify

# 4. 运行一个简单测试（可能需要几分钟）
make test-toronto-single
```

如果遇到 `ModuleNotFoundError`，请查看 [VERIFICATION.md](VERIFICATION.md) 了解详细的故障排查步骤。

## 提示

1. **首次运行**: 确保所有依赖已安装 (`requirements.txt`)
2. **数据文件**: 确保Radio Map和TLE文件已就位
3. **内存**: 如果内存不足，可以在配置文件中减少 `N_UE` 或 `T` 值
4. **结果保存**: 如需保存结果，配置中启用 `write_json_report: True`
5. **工作目录**: 始终从项目根目录运行命令（即 `DL/` 目录）
