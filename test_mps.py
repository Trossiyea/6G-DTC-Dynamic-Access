import torch
import platform

print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"Mac Architecture: {platform.machine()}") # 应为 arm64

# 1. 检查 MPS 是否编译进 PyTorch
is_built = torch.backends.mps.is_built()
print(f"MPS built: {is_built}")

# 2. 检查 MPS 是否在当前机器可用
is_available = torch.backends.mps.is_available()
print(f"MPS available: {is_available}")

if is_available:
    try:
        # 3. 尝试将张量放入 MPS 设备
        device = torch.device("mps")
        x = torch.ones(5, device=device)
        print("✅ 成功在 MPS 上创建张量")
        
        # 4. 尝试简单计算
        y = x * 2
        print(f"✅ 计算测试通过: {y}")
        
    except Exception as e:
        print(f"❌ MPS 可用但运行出错: {e}")
else:
    print("❌ MPS 当前不可用")
