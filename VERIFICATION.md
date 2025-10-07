# 验证指南

## 快速验证

### 1. 验证脚本导入正常
```bash
python run_test.py --list
```

预期输出：
```
可用场景:
  - toronto_single                 -> test/config_toronto_single.py
  - toronto_constellation          -> test/config_toronto_constellation.py
  - shanghai_single                -> test/config_shanghai_single.py
  - shanghai_constellation         -> test/config_shanghai_constellation.py
```

### 2. 验证单个场景运行
```bash
python run_test.py --scenario toronto_single
```

预期输出示例：
```
======================================================================
运行场景: toronto_single
配置文件: test/config_toronto_single.py
======================================================================

[Orbit] Using Skyfield/TLE orbit: STARLINK-11075 [DTC] (start=2025-10-08T09:22:45.447459+00:00)...

----------------------------------------------------------------------
场景 [toronto_single] 结果:
  基线平均SE: 3.xxxx bits/s/Hz
  RadioMap平均SE: 3.xxxx bits/s/Hz
  提升百分比: xx.xx%
----------------------------------------------------------------------
```

## 解决的问题

### 原问题：ModuleNotFoundError
```
ModuleNotFoundError: No module named 'code.config'; 'code' is not a package
```

### 解决方案
修改了 `run_test.py`，使用以下策略：

1. **添加路径到 sys.path**
   ```python
   SCRIPT_DIR = Path(__file__).parent.absolute()
   CODE_DIR = SCRIPT_DIR / "code"
   sys.path.insert(0, str(SCRIPT_DIR))
   sys.path.insert(0, str(CODE_DIR))
   ```

2. **使用 importlib 直接加载模块**
   ```python
   def load_base_config():
       config_path = SCRIPT_DIR / "code" / "config.py"
       spec = importlib.util.spec_from_file_location("base_config", config_path)
       module = importlib.util.module_from_spec(spec)
       spec.loader.exec_module(module)
       return module.CONFIG
   ```

3. **main.py 保持不变**
   - main.py 中的相对导入（如 `from csi import ...`）仍然正常工作
   - 因为 `CODE_DIR` 已经添加到 `sys.path`

## 验证 main.py 未受影响

直接运行 main.py 仍然应该正常工作：

```bash
cd code
python main.py
```

或者从项目根目录：

```bash
python code/main.py
```

## 目录结构说明

```
DL/
├── code/
│   ├── main.py          # 主程序
│   ├── config.py        # 基础配置
│   ├── csi.py           # CSI模块
│   ├── orbit.py         # 轨道模块
│   └── ...              # 其他模块
├── test/
│   ├── config_toronto_single.py
│   ├── config_toronto_constellation.py
│   ├── config_shanghai_single.py
│   └── config_shanghai_constellation.py
├── run_test.py          # 测试运行脚本
├── run_all_tests.sh     # 批量运行脚本
└── Makefile             # Make 快捷命令
```

## 路径解析说明

### run_test.py 的导入策略

1. **项目根目录** (`SCRIPT_DIR`) 添加到 `sys.path[0]`
   - 允许导入 `test/` 下的配置

2. **code 目录** (`CODE_DIR`) 添加到 `sys.path[0]`
   - 允许 main.py 中的模块导入（如 `from csi import ...`）

3. **使用 importlib 直接加载**
   - 避免依赖 `__init__.py`
   - 不影响原有的代码结构

### main.py 的导入方式（未改变）

main.py 使用相对导入：
```python
from csi import sinr_to_se_mcs
from config import CONFIG
from orbit import compute_geometry_and_beam
# ...
```

这些导入在以下两种情况都能工作：
1. 直接运行 `python code/main.py`
2. 通过 `run_test.py` 调用

## 常见问题排查

### 问题1: 找不到 Radio Map 文件
**错误**: `FileNotFoundError: Radio Map file not found: radio_map/...`

**解决**: 
- 确保从项目根目录运行命令
- 检查配置文件中的相对路径

### 问题2: 找不到 TLE 文件
**错误**: 找不到 TLE 文件

**解决**: 确保 `tles/` 目录存在且包含所需文件

### 问题3: 导入错误
**错误**: `ImportError` 或 `ModuleNotFoundError`

**解决**: 
```bash
# 检查 Python 路径
python -c "import sys; print('\n'.join(sys.path))"

# 验证文件存在
ls code/main.py
ls code/config.py
```

## Make 命令快捷方式

从项目根目录运行：

```bash
make help                          # 查看帮助
make list                          # 列出场景
make test-toronto-single           # 运行特定场景
make test-all                      # 运行所有场景
make clean                         # 清理缓存
```

## 测试清单

- [ ] `python run_test.py --list` 输出所有场景
- [ ] `make help` 显示帮助信息
- [ ] `python run_test.py --scenario toronto_single` 成功运行
- [ ] `python code/main.py` 仍然能直接运行（如果有默认配置）
- [ ] 所有导入错误已解决
