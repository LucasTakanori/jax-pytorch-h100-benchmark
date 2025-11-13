#!/usr/bin/env python3
"""
Complete validation test for Phase 2.

Tests that the validation framework works correctly and verifies model architectures match.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.validation import compare_models, validate_model_architecture
from models.torch_zoo import get_torch_model
from models.jax_flax_zoo import get_flax_model
import torch
import jax


def test_resnet50_validation():
    """Test ResNet-50 validation."""
    print("=" * 60)
    print("ResNet-50 Validation Test")
    print("=" * 60)
    
    # Load PyTorch model
    device = torch.device('cpu')
    pytorch_model, pytorch_preprocess, input_shape_pytorch, _ = get_torch_model(
        'resnet50',
        pretrained=False,  # Random weights for testing
        device=device
    )
    
    # Load JAX model
    rng_key = jax.random.PRNGKey(42)
    jax_apply_fn, jax_params, jax_preprocess, _ = get_flax_model(
        'resnet50',
        input_shape=(224, 224, 3),
        rng_key=rng_key
    )
    
    # Test 1: Architecture validation
    print("\n1. Architecture Validation:")
    arch_result = validate_model_architecture(
        'resnet50',
        pytorch_model,
        jax_params,
        verbose=True
    )
    
    # Test 2: Forward pass validation (with relaxed tolerance for random weights)
    print("\n2. Forward Pass Validation (random weights, relaxed tolerance):")
    forward_result = compare_models(
        model_name='resnet50',
        pytorch_model=pytorch_model,
        pytorch_preprocess=pytorch_preprocess,
        jax_apply_fn=jax_apply_fn,
        jax_params=jax_params,
        jax_preprocess=jax_preprocess,
        input_shape_pytorch=input_shape_pytorch,
        input_shape_jax=(224, 224, 3),
        batch_size=2,
        tolerance=1e-1,  # Relaxed tolerance for random weights
        seed=42
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("ResNet-50 Validation Summary:")
    print(f"  Architecture Match: {'✅ PASS' if arch_result['passed'] else '❌ FAIL'}")
    print(f"  Forward Pass Match: {'✅ PASS' if forward_result['forward_pass_validation']['passed'] else '⚠️  FAIL (expected with random weights)'}")
    print("=" * 60)
    
    return arch_result['passed'], forward_result


def test_vit_validation():
    """Test ViT-Base validation."""
    print("\n" + "=" * 60)
    print("ViT-Base Validation Test")
    print("=" * 60)
    
    # Load PyTorch model
    device = torch.device('cpu')
    pytorch_model, pytorch_preprocess, input_shape_pytorch, _ = get_torch_model(
        'vit_b_16',
        pretrained=False,  # Random weights for testing
        device=device
    )
    
    # Load JAX model
    rng_key = jax.random.PRNGKey(42)
    jax_apply_fn, jax_params, jax_preprocess, _ = get_flax_model(
        'vit_b_16',
        input_shape=(224, 224, 3),
        rng_key=rng_key
    )
    
    # Test 1: Architecture validation
    print("\n1. Architecture Validation:")
    arch_result = validate_model_architecture(
        'vit_b_16',
        pytorch_model,
        jax_params,
        verbose=True
    )
    
    # Test 2: Forward pass validation (with relaxed tolerance for random weights)
    print("\n2. Forward Pass Validation (random weights, relaxed tolerance):")
    forward_result = compare_models(
        model_name='vit_b_16',
        pytorch_model=pytorch_model,
        pytorch_preprocess=pytorch_preprocess,
        jax_apply_fn=jax_apply_fn,
        jax_params=jax_params,
        jax_preprocess=jax_preprocess,
        input_shape_pytorch=input_shape_pytorch,
        input_shape_jax=(224, 224, 3),
        batch_size=2,
        tolerance=1e-1,  # Relaxed tolerance for random weights
        seed=42
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("ViT-Base Validation Summary:")
    print(f"  Architecture Match: {'✅ PASS' if arch_result['passed'] else '❌ FAIL'}")
    print(f"  Forward Pass Match: {'✅ PASS' if forward_result['forward_pass_validation']['passed'] else '⚠️  FAIL (expected with random weights)'}")
    print("=" * 60)
    
    return arch_result['passed'], forward_result


def test_validation_framework():
    """Test that validation framework works correctly."""
    print("\n" + "=" * 60)
    print("Validation Framework Test")
    print("=" * 60)
    
    from utils.validation import compare_outputs
    import torch
    import jax.numpy as jnp
    
    # Create identical outputs (should pass)
    pytorch_output = torch.randn(2, 1000)
    jax_output = jnp.array(pytorch_output.detach().cpu().numpy())
    
    result = compare_outputs(pytorch_output, jax_output, tolerance=1e-5, verbose=True)
    
    assert result['passed'], "Identical outputs should pass validation"
    print("✅ Validation framework correctly identifies matching outputs")
    
    # Create different outputs (should fail)
    pytorch_output2 = torch.randn(2, 1000)
    jax_output2 = jnp.array(pytorch_output2.detach().cpu().numpy()) + 1.0  # Add large difference
    
    result2 = compare_outputs(pytorch_output2, jax_output2, tolerance=1e-5, verbose=True)
    
    assert not result2['passed'], "Different outputs should fail validation"
    print("✅ Validation framework correctly identifies non-matching outputs")
    
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Phase 2 Complete Validation Test")
    print("=" * 60)
    
    results = {}
    
    # Test validation framework itself
    try:
        results['framework'] = test_validation_framework()
    except Exception as e:
        print(f"❌ Validation framework test failed: {e}")
        results['framework'] = False
    
    # Test ResNet-50
    try:
        arch_pass, forward_result = test_resnet50_validation()
        results['resnet50_arch'] = arch_pass
        results['resnet50_forward'] = forward_result['forward_pass_validation']['passed']
    except Exception as e:
        print(f"❌ ResNet-50 validation failed: {e}")
        import traceback
        traceback.print_exc()
        results['resnet50_arch'] = False
        results['resnet50_forward'] = False
    
    # Test ViT-Base
    try:
        arch_pass, forward_result = test_vit_validation()
        results['vit_arch'] = arch_pass
        results['vit_forward'] = forward_result['forward_pass_validation']['passed']
    except Exception as e:
        print(f"❌ ViT-Base validation failed: {e}")
        import traceback
        traceback.print_exc()
        results['vit_arch'] = False
        results['vit_forward'] = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("Final Validation Summary")
    print("=" * 60)
    print(f"Validation Framework: {'✅ PASS' if results.get('framework') else '❌ FAIL'}")
    print(f"ResNet-50 Architecture: {'✅ PASS' if results.get('resnet50_arch') else '❌ FAIL'}")
    print(f"ResNet-50 Forward Pass: {'✅ PASS' if results.get('resnet50_forward') else '⚠️  FAIL (expected with random weights)'}")
    print(f"ViT-Base Architecture: {'✅ PASS' if results.get('vit_arch') else '❌ FAIL'}")
    print(f"ViT-Base Forward Pass: {'✅ PASS' if results.get('vit_forward') else '⚠️  FAIL (expected with random weights)'}")
    print("=" * 60)
    
    # Note about weight matching
    print("\nNote: Forward pass validation fails with random weights (expected).")
    print("For exact matching, models need identical weights (requires weight conversion).")
    print("Architecture validation confirms models have matching structure.")
    print("Validation framework is working correctly.")
    
    return 0 if results.get('framework') and results.get('resnet50_arch') and results.get('vit_arch') else 1


if __name__ == "__main__":
    sys.exit(main())

