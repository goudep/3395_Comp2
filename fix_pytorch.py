"""
修复PyTorch DLL加载错误的脚本
在命令行运行: python fix_pytorch.py
"""

import sys
import subprocess

print("="*60)
print("PyTorch DLL加载错误修复脚本")
print("="*60)

print("\n步骤1: 卸载现有的PyTorch...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "-y"])
    print("✓ 卸载完成")
except Exception as e:
    print(f"⚠️  卸载过程出现警告: {e}")

print("\n步骤2: 清理pip缓存...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "cache", "purge"])
    print("✓ 缓存清理完成")
except:
    print("⚠️  缓存清理跳过")

print("\n步骤3: 安装PyTorch CPU版本（使用官方索引）...")
try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", 
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "--no-cache-dir"
    ])
    print("✓ PyTorch安装完成")
except Exception as e:
    print(f"✗ 安装失败: {e}")
    print("\n请尝试手动安装:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

print("\n步骤4: 验证安装...")
try:
    import torch
    x = torch.tensor([1.0])
    print(f"✓ PyTorch验证成功！版本: {torch.__version__}")
    print(f"✓ 设备: {'CUDA可用' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    print(f"✗ 验证失败: {e}")
    print("\n如果问题仍然存在，请尝试:")
    print("1. 安装Visual C++ Redistributable:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("\n2. 使用conda安装（如果可用）:")
    print("   conda install pytorch torchvision -c pytorch")
    sys.exit(1)

print("\n" + "="*60)
print("修复完成！现在可以运行notebook了。")
print("="*60)

