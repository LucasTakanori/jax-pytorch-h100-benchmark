#!/usr/bin/env python3
"""
Setup verification script for 594Project.

Checks that all required dependencies are installed and hardware is properly detected.
Run this after setting up your environment to ensure everything works.
"""

import sys
import importlib
from typing import List, Tuple


def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True, f"✅ {package_name or module_name} installed"
    except ImportError as e:
        return False, f"❌ {package_name or module_name} not found: {e}"


def check_pytorch() -> List[str]:
    """Check PyTorch installation and backends."""
    results = []
    
    success, msg = check_import('torch', 'PyTorch')
    results.append(msg)
    if not success:
        return results
    
    try:
        import torch
        results.append(f"  Version: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            results.append(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            results.append(f"  CUDA Version: {torch.version.cuda}")
            results.append(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                results.append(f"    GPU {i}: {props.name} ({memory_gb:.2f} GB)")
        else:
            results.append("  ⚠️  CUDA not available (expected on macOS/CPU-only systems)")
        
        # Check MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            results.append("  ✅ MPS (Apple Silicon GPU) available")
        else:
            results.append("  ⚠️  MPS not available (expected on non-Apple systems)")
        
        # Check CPU
        results.append(f"  ✅ CPU backend available")
        
    except Exception as e:
        results.append(f"  ❌ Error checking PyTorch: {e}")
    
    return results


def check_jax() -> List[str]:
    """Check JAX installation and backends."""
    results = []
    
    success, msg = check_import('jax', 'JAX')
    results.append(msg)
    if not success:
        return results
    
    try:
        import jax
        
        # JAX version
        try:
            version = jax.__version__
            results.append(f"  Version: {version}")
        except:
            results.append("  Version: (unknown)")
        
        # Check devices
        devices = jax.devices()
        results.append(f"  Available devices: {len(devices)}")
        
        # Group by platform
        platforms = {}
        for device in devices:
            platform = device.platform
            if platform not in platforms:
                platforms[platform] = []
            platforms[platform].append(device)
        
        for platform, devs in platforms.items():
            results.append(f"  ✅ {platform.upper()}: {len(devs)} device(s)")
            for dev in devs:
                results.append(f"    - {dev}")
        
        # Check if XLA is working
        try:
            import jax.numpy as jnp
            x = jnp.array([1.0, 2.0, 3.0])
            y = x * 2
            _ = y.block_until_ready()
            results.append("  ✅ XLA compilation working")
        except Exception as e:
            results.append(f"  ❌ XLA test failed: {e}")
        
    except Exception as e:
        results.append(f"  ❌ Error checking JAX: {e}")
    
    return results


def check_flax() -> List[str]:
    """Check Flax installation."""
    results = []
    
    success, msg = check_import('flax', 'Flax')
    results.append(msg)
    if not success:
        return results
    
    try:
        import flax
        try:
            version = flax.__version__
            results.append(f"  Version: {version}")
        except:
            pass
    except Exception as e:
        results.append(f"  ❌ Error: {e}")
    
    return results


def check_other_deps() -> List[str]:
    """Check other required dependencies."""
    results = []
    
    deps = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('psutil', 'psutil'),
    ]
    
    for module, name in deps:
        success, msg = check_import(module, name)
        results.append(msg)
    
    return results


def test_device_utilities() -> List[str]:
    """Test the device detection utilities."""
    results = []
    
    try:
        sys.path.insert(0, '.')
        from utils.device import detect_all_devices
        
        results.append("✅ Device utilities module loaded")
        
        detected = detect_all_devices()
        
        if 'pytorch' in detected and 'error' not in detected['pytorch']:
            results.append("\nPyTorch Device Detection:")
            info = detected['pytorch']['info']
            results.append(f"  Device: {info.name}")
            results.append(f"  Type: {info.device_type.upper()}")
            results.append(f"  Available: {info.available}")
            if info.memory_gb:
                results.append(f"  Memory: {info.memory_gb:.2f} GB")
            if info.compute_capability:
                results.append(f"  Compute Capability: {info.compute_capability}")
            results.append("  ✅ PyTorch device detection working")
        elif 'pytorch' in detected:
            results.append(f"  ⚠️  PyTorch device detection: {detected['pytorch'].get('error', 'unknown error')}")
        
        if 'jax' in detected and 'error' not in detected['jax']:
            results.append("\nJAX Device Detection:")
            info = detected['jax']['info']
            results.append(f"  Device: {info.name}")
            results.append(f"  Type: {info.device_type.upper()}")
            results.append(f"  Available: {info.available}")
            if info.memory_gb:
                results.append(f"  Memory: {info.memory_gb:.2f} GB")
            results.append("  ✅ JAX device detection working")
        elif 'jax' in detected:
            results.append(f"  ⚠️  JAX device detection: {detected['jax'].get('error', 'unknown error')}")
        
    except Exception as e:
        results.append(f"  ❌ Error testing device utilities: {e}")
        import traceback
        results.append(f"  Traceback: {traceback.format_exc()}")
    
    return results


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("594Project Setup Verification")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Core dependencies
    print("Checking core dependencies...")
    print("-" * 60)
    for msg in check_other_deps():
        print(msg)
        if "❌" in msg:
            all_checks_passed = False
    print()
    
    # PyTorch
    print("Checking PyTorch...")
    print("-" * 60)
    for msg in check_pytorch():
        print(msg)
        if "❌" in msg:
            all_checks_passed = False
    print()
    
    # JAX
    print("Checking JAX...")
    print("-" * 60)
    for msg in check_jax():
        print(msg)
        if "❌" in msg:
            all_checks_passed = False
    print()
    
    # Flax
    print("Checking Flax...")
    print("-" * 60)
    for msg in check_flax():
        print(msg)
        if "❌" in msg:
            all_checks_passed = False
    print()
    
    # Device utilities
    print("Testing device detection utilities...")
    print("-" * 60)
    for msg in test_device_utilities():
        print(msg)
        if "❌" in msg:
            all_checks_passed = False
    print()
    
    # Summary
    print("=" * 60)
    if all_checks_passed:
        print("✅ All checks passed! Your environment is ready.")
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        print("   See docs/setup.md for troubleshooting help.")
    print("=" * 60)
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())

