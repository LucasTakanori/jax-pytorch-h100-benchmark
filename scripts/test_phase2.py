#!/usr/bin/env python3
"""
Test script for Phase 2 components.

Tests all utilities, models, and data loading to ensure everything works correctly.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_timing_utilities():
    """Test timing utilities."""
    print("=" * 60)
    print("Testing Timing Utilities")
    print("=" * 60)
    
    try:
        from utils.timing import LatencyStats, calculate_throughput
        
        # Test LatencyStats
        durations = [0.01, 0.02, 0.015, 0.025, 0.012, 0.018, 0.022, 0.014]
        stats = LatencyStats.from_durations(durations)
        
        print(f"✅ LatencyStats created successfully")
        print(f"   Mean: {stats.mean_ms:.3f}ms")
        print(f"   p50: {stats.p50_ms:.3f}ms")
        print(f"   p95: {stats.p95_ms:.3f}ms")
        print(f"   Count: {stats.count}")
        
        # Test throughput calculation
        throughput = calculate_throughput(batch_size=16, latency_s=0.02)
        print(f"✅ Throughput calculation: {throughput:.2f} images/sec")
        
        return True
    except Exception as e:
        print(f"❌ Timing utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_utilities():
    """Test memory utilities."""
    print("\n" + "=" * 60)
    print("Testing Memory Utilities")
    print("=" * 60)
    
    try:
        from utils.memory import get_current_memory, track_memory
        
        # Test current memory
        mem = get_current_memory('pytorch')
        print(f"✅ Current memory: {mem:.2f} MB")
        
        # Test memory tracker context manager
        with track_memory('pytorch') as tracker:
            # Simulate some memory usage
            import numpy as np
            arr = np.random.rand(1000, 1000)
            del arr
        
        peak = tracker.get_peak_usage_mb()
        print(f"✅ Memory tracker: Peak = {peak:.2f} MB")
        
        return True
    except Exception as e:
        print(f"❌ Memory utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading utilities."""
    print("\n" + "=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    
    try:
        from utils.data import get_synthetic_batch, get_preprocessing
        
        # Test PyTorch synthetic data
        print("Testing PyTorch synthetic data...")
        import torch
        device = torch.device('cpu')
        torch_data = get_synthetic_batch(
            batch_size=4,
            input_shape=(3, 224, 224),
            framework='pytorch',
            device=device,
            seed=42
        )
        print(f"✅ PyTorch synthetic batch: shape={torch_data.shape}, dtype={torch_data.dtype}")
        
        # Test JAX synthetic data
        print("Testing JAX synthetic data...")
        import jax.numpy as jnp
        jax_data = get_synthetic_batch(
            batch_size=4,
            input_shape=(224, 224, 3),
            framework='jax',
            seed=42
        )
        print(f"✅ JAX synthetic batch: shape={jax_data.shape}, dtype={jax_data.dtype}")
        
        # Test preprocessing
        preprocess_torch = get_preprocessing('pytorch')
        preprocessed = preprocess_torch(torch_data)
        print(f"✅ PyTorch preprocessing: output shape={preprocessed.shape}")
        
        preprocess_jax = get_preprocessing('jax')
        preprocessed_jax = preprocess_jax(jax_data)
        print(f"✅ JAX preprocessing: output shape={preprocessed_jax.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pytorch_models():
    """Test PyTorch model loading."""
    print("\n" + "=" * 60)
    print("Testing PyTorch Models")
    print("=" * 60)
    
    try:
        from models.torch_zoo import get_torch_model
        import torch
        
        device = torch.device('cpu')
        
        # Test ResNet-50
        print("Testing ResNet-50...")
        model, preprocess, input_shape, metadata = get_torch_model(
            'resnet50',
            pretrained=False,  # Use random weights for faster loading
            device=device
        )
        print(f"✅ ResNet-50 loaded: input_shape={input_shape}")
        print(f"   Metadata: {metadata['name']}, ~{metadata['params']:,} params")
        
        # Test forward pass
        x = torch.randn(2, *input_shape, device=device)
        x = preprocess(x)
        with torch.no_grad():
            output = model(x)
        print(f"✅ ResNet-50 forward pass: output shape={output.shape}")
        
        # Test ViT-Base
        print("\nTesting ViT-Base...")
        model_vit, preprocess_vit, input_shape_vit, metadata_vit = get_torch_model(
            'vit_b_16',
            pretrained=False,
            device=device
        )
        print(f"✅ ViT-Base loaded: input_shape={input_shape_vit}")
        print(f"   Metadata: {metadata_vit['name']}, ~{metadata_vit['params']:,} params")
        
        # Test forward pass
        x_vit = torch.randn(2, *input_shape_vit, device=device)
        x_vit = preprocess_vit(x_vit)
        with torch.no_grad():
            output_vit = model_vit(x_vit)
        print(f"✅ ViT-Base forward pass: output shape={output_vit.shape}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jax_models():
    """Test JAX/Flax model loading."""
    print("\n" + "=" * 60)
    print("Testing JAX/Flax Models")
    print("=" * 60)
    
    try:
        from models.jax_flax_zoo import get_flax_model
        import jax
        import jax.numpy as jnp
        
        rng_key = jax.random.PRNGKey(42)
        
        # Test ResNet-50
        print("Testing ResNet-50...")
        apply_fn, params, preprocess, metadata = get_flax_model(
            'resnet50',
            input_shape=(224, 224, 3),
            rng_key=rng_key,
            num_classes=1000
        )
        print(f"✅ ResNet-50 loaded: input_shape={metadata['input_shape']}")
        print(f"   Metadata: {metadata['name']}, ~{metadata['params']:,} params")
        
        # Test forward pass
        x = jnp.ones((2, 224, 224, 3), dtype=jnp.float32)
        x = preprocess(x)
        output = apply_fn(params, x, train=False)
        output.block_until_ready()  # Ensure computation completes
        print(f"✅ ResNet-50 forward pass: output shape={output.shape}")
        
        # Test ViT-Base
        print("\nTesting ViT-Base...")
        apply_fn_vit, params_vit, preprocess_vit, metadata_vit = get_flax_model(
            'vit_b_16',
            input_shape=(224, 224, 3),
            rng_key=rng_key,
            num_classes=1000
        )
        print(f"✅ ViT-Base loaded: input_shape={metadata_vit['input_shape']}")
        print(f"   Metadata: {metadata_vit['name']}, ~{metadata_vit['params']:,} params")
        
        # Test forward pass
        x_vit = jnp.ones((2, 224, 224, 3), dtype=jnp.float32)
        x_vit = preprocess_vit(x_vit)
        output_vit = apply_fn_vit(params_vit, x_vit, train=False)
        output_vit.block_until_ready()
        print(f"✅ ViT-Base forward pass: output shape={output_vit.shape}")
        
        return True
    except Exception as e:
        print(f"❌ JAX/Flax models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_logging():
    """Test CSV logging utilities."""
    print("\n" + "=" * 60)
    print("Testing CSV Logging")
    print("=" * 60)
    
    try:
        from utils.logging import BenchmarkLogger, create_result_dict
        from utils.timing import LatencyStats
        import tempfile
        import os
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = BenchmarkLogger(output_dir=tmpdir, filename='test_results.csv')
            
            # Create a test result
            stats = LatencyStats.from_durations([0.01, 0.02, 0.015, 0.025])
            result = create_result_dict(
                framework='pytorch',
                model='resnet50',
                device='cpu',
                device_type='cpu',
                batch_size=16,
                input_shape=(3, 224, 224),
                dtype='float32',
                warmup_iterations=5,
                measurement_iterations=20,
                latency_stats=stats,
                throughput_ips=800.0,
                memory_mb=512.0,
                compilation_time_ms=None
            )
            
            # Append result
            logger.append_result(result)
            print(f"✅ Result appended to CSV: {logger.get_filepath()}")
            
            # Read back results
            df = logger.read_results()
            print(f"✅ CSV read back: {len(df)} row(s)")
            if len(df) > 0:
                print(f"   Columns: {list(df.columns)}")
                print(f"   Framework: {df.iloc[0]['framework']}")
                print(f"   Model: {df.iloc[0]['model']}")
            
            # Verify file exists
            assert logger.get_filepath().exists(), "CSV file should exist"
            print(f"✅ CSV file exists at: {logger.get_filepath()}")
        
        return True
    except Exception as e:
        print(f"❌ CSV logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_integration():
    """Test device detection integration."""
    print("\n" + "=" * 60)
    print("Testing Device Integration")
    print("=" * 60)
    
    try:
        from utils.device import get_torch_device, get_jax_device, print_device_info
        
        # Test PyTorch device
        torch_device, torch_info = get_torch_device()
        print("✅ PyTorch device detected:")
        print_device_info(torch_info)
        
        # Test JAX device
        jax_device, jax_info = get_jax_device()
        print("\n✅ JAX device detected:")
        print_device_info(jax_info)
        
        return True
    except Exception as e:
        print(f"❌ Device integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Phase 2 Component Testing")
    print("=" * 60)
    print()
    
    results = []
    
    # Run all tests
    results.append(("Timing Utilities", test_timing_utilities()))
    results.append(("Memory Utilities", test_memory_utilities()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("PyTorch Models", test_pytorch_models()))
    results.append(("JAX/Flax Models", test_jax_models()))
    results.append(("CSV Logging", test_csv_logging()))
    results.append(("Device Integration", test_device_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 2 components are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

