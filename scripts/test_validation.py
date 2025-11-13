#!/usr/bin/env python3
"""
Quick test for validation framework.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.validation import compare_models
from models.torch_zoo import get_torch_model
from models.jax_flax_zoo import get_flax_model
import torch
import jax

def test_resnet50_validation():
    """Test ResNet-50 validation."""
    print("Testing ResNet-50 validation...")
    
    # Load PyTorch model
    device = torch.device('cpu')
    pytorch_model, pytorch_preprocess, input_shape_pytorch, _ = get_torch_model(
        'resnet50',
        pretrained=False,
        device=device
    )
    
    # Load JAX model
    rng_key = jax.random.PRNGKey(42)
    jax_apply_fn, jax_params, jax_preprocess, _ = get_flax_model(
        'resnet50',
        input_shape=(224, 224, 3),
        rng_key=rng_key
    )
    
    # Compare models
    result = compare_models(
        model_name='resnet50',
        pytorch_model=pytorch_model,
        pytorch_preprocess=pytorch_preprocess,
        jax_apply_fn=jax_apply_fn,
        jax_params=jax_params,
        jax_preprocess=jax_preprocess,
        input_shape_pytorch=input_shape_pytorch,
        input_shape_jax=(224, 224, 3),
        batch_size=2,
        tolerance=1e-3,  # Relaxed tolerance for random weights
        seed=42
    )
    
    return result

if __name__ == "__main__":
    test_resnet50_validation()

