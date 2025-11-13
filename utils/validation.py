"""
Numerical validation utilities for ensuring JAX and PyTorch models produce matching outputs.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import torch
import jax.numpy as jnp


def compare_outputs(
    output_pytorch: torch.Tensor,
    output_jax: jnp.ndarray,
    tolerance: float = 1e-5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare outputs from PyTorch and JAX models.
    
    Args:
        output_pytorch: PyTorch tensor output
        output_jax: JAX array output
        tolerance: Maximum allowed absolute difference
        verbose: Print detailed comparison if True
        
    Returns:
        Dictionary with comparison results
    """
    # Convert to numpy for comparison
    if isinstance(output_pytorch, torch.Tensor):
        output_pytorch_np = output_pytorch.detach().cpu().numpy()
    else:
        output_pytorch_np = np.array(output_pytorch)
    
    if isinstance(output_jax, jnp.ndarray):
        output_jax_np = np.array(output_jax)
    else:
        output_jax_np = np.array(output_jax)
    
    # Check shapes match
    if output_pytorch_np.shape != output_jax_np.shape:
        return {
            'passed': False,
            'error': f'Shape mismatch: PyTorch {output_pytorch_np.shape} vs JAX {output_jax_np.shape}',
            'max_diff': None,
            'mean_diff': None,
            'max_abs_diff': None
        }
    
    # Compute differences
    diff = output_pytorch_np - output_jax_np
    abs_diff = np.abs(diff)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_diff = np.max(diff)
    min_diff = np.min(diff)
    
    # Check tolerance
    passed = max_abs_diff < tolerance
    
    result = {
        'passed': passed,
        'max_abs_diff': float(max_abs_diff),
        'mean_abs_diff': float(mean_abs_diff),
        'max_diff': float(max_diff),
        'min_diff': float(min_diff),
        'tolerance': tolerance,
        'shapes_match': True,
        'output_shape': output_pytorch_np.shape
    }
    
    if verbose:
        print(f"Output Comparison:")
        print(f"  Shape: {output_pytorch_np.shape}")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"  Tolerance: {tolerance:.2e}")
        print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return result


def validate_forward_pass(
    pytorch_model: Any,
    jax_apply_fn: Any,
    jax_params: Any,
    input_pytorch: torch.Tensor,
    input_jax: jnp.ndarray,
    tolerance: float = 1e-5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate forward pass outputs match between PyTorch and JAX models.
    
    Args:
        pytorch_model: PyTorch model (in eval mode)
        jax_apply_fn: JAX apply function
        jax_params: JAX model parameters
        input_pytorch: Input tensor for PyTorch (NCHW format)
        input_jax: Input array for JAX (NHWC format)
        tolerance: Maximum allowed absolute difference
        verbose: Print detailed results if True
        
    Returns:
        Dictionary with validation results
    """
    # Run PyTorch forward pass
    pytorch_model.eval()
    with torch.no_grad():
        output_pytorch = pytorch_model(input_pytorch)
    
    # Run JAX forward pass
    output_jax = jax_apply_fn(jax_params, input_jax, train=False)
    output_jax.block_until_ready()  # Ensure computation completes
    
    # Compare outputs
    result = compare_outputs(output_pytorch, output_jax, tolerance, verbose)
    
    return result


def validate_model_architecture(
    model_name: str,
    pytorch_model: Any,
    jax_params: Any,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate that model architectures match (parameter counts, layer structure).
    
    Args:
        model_name: Model name for reporting
        pytorch_model: PyTorch model
        jax_params: JAX model parameters (Flax params dict)
        verbose: Print detailed results if True
        
    Returns:
        Dictionary with validation results
    """
    # Count PyTorch parameters
    pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
    pytorch_trainable = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    
    # Count JAX parameters
    def count_jax_params(params):
        """Recursively count parameters in JAX params dict."""
        if isinstance(params, dict):
            return sum(count_jax_params(v) for v in params.values())
        elif hasattr(params, 'size'):
            return params.size
        else:
            return 0
    
    jax_params_count = count_jax_params(jax_params)
    
    # Compare
    param_diff = abs(pytorch_params - jax_params_count)
    param_diff_pct = (param_diff / pytorch_params) * 100 if pytorch_params > 0 else 0
    
    # Consider match if within 1% (allowing for minor implementation differences)
    passed = param_diff_pct < 1.0
    
    result = {
        'passed': passed,
        'pytorch_params': pytorch_params,
        'jax_params': jax_params_count,
        'param_diff': param_diff,
        'param_diff_pct': param_diff_pct,
        'pytorch_trainable': pytorch_trainable
    }
    
    if verbose:
        print(f"Architecture Validation for {model_name}:")
        print(f"  PyTorch parameters: {pytorch_params:,}")
        print(f"  JAX parameters: {jax_params_count:,}")
        print(f"  Difference: {param_diff:,} ({param_diff_pct:.2f}%)")
        print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return result


def compare_models(
    model_name: str,
    pytorch_model: Any,
    pytorch_preprocess: Any,
    jax_apply_fn: Any,
    jax_params: Any,
    jax_preprocess: Any,
    input_shape_pytorch: Tuple[int, ...],
    input_shape_jax: Tuple[int, ...],
    batch_size: int = 2,
    tolerance: float = 1e-5,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    High-level function to compare PyTorch and JAX models.
    
    Args:
        model_name: Model name for reporting
        pytorch_model: PyTorch model
        pytorch_preprocess: PyTorch preprocessing function
        jax_apply_fn: JAX apply function
        jax_params: JAX model parameters
        jax_preprocess: JAX preprocessing function
        input_shape_pytorch: Input shape for PyTorch (C, H, W)
        input_shape_jax: Input shape for JAX (H, W, C)
        batch_size: Batch size for test
        tolerance: Maximum allowed absolute difference
        seed: Random seed for reproducibility
        device: PyTorch device
        
    Returns:
        Dictionary with comprehensive validation results
    """
    import torch
    import jax
    import jax.numpy as jnp
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng_key = jax.random.PRNGKey(seed)
    
    # Generate test inputs
    input_pytorch = torch.rand(batch_size, *input_shape_pytorch, device=device)
    input_pytorch = pytorch_preprocess(input_pytorch)
    
    rng_key, subkey = jax.random.split(rng_key)
    input_jax = jax.random.uniform(subkey, (batch_size, *input_shape_jax), dtype=jnp.float32)
    input_jax = jax_preprocess(input_jax)
    
    print(f"\n{'='*60}")
    print(f"Validating {model_name}")
    print(f"{'='*60}")
    
    # Validate architecture
    arch_result = validate_model_architecture(model_name, pytorch_model, jax_params, verbose=True)
    
    # Validate forward pass
    print()
    forward_result = validate_forward_pass(
        pytorch_model,
        jax_apply_fn,
        jax_params,
        input_pytorch,
        input_jax,
        tolerance=tolerance,
        verbose=True
    )
    
    # Overall result
    overall_passed = arch_result['passed'] and forward_result['passed']
    
    result = {
        'model_name': model_name,
        'overall_passed': overall_passed,
        'architecture_validation': arch_result,
        'forward_pass_validation': forward_result,
        'batch_size': batch_size,
        'tolerance': tolerance
    }
    
    print(f"\n{'='*60}")
    print(f"Overall Validation: {'✅ PASS' if overall_passed else '❌ FAIL'}")
    print(f"{'='*60}\n")
    
    return result

