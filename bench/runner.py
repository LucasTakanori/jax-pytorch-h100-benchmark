"""
Unified benchmark runner for PyTorch and JAX models.

Provides a single interface for running benchmarks across frameworks, models, and batch sizes.
"""

import sys
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.device import get_torch_device, get_jax_device, DeviceInfo
from utils.timing import LatencyStats, measure_latency, calculate_throughput, measure_compilation_time
from utils.memory import get_peak_memory, reset_memory_stats, track_memory
from utils.logging import BenchmarkLogger, create_result_dict
from utils.data import get_synthetic_batch, get_preprocessing


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    framework: str  # 'pytorch' or 'jax'
    model_name: str  # 'resnet50' or 'vit_b_16'
    device_prefer: Optional[str] = None  # Preferred device type
    batch_sizes: List[int] = None  # Default: [1, 8, 32, 128]
    warmup_iterations: int = 10
    measurement_iterations: int = 50
    input_shape: Optional[Tuple[int, ...]] = None  # Will be inferred from model
    dtype: str = 'float32'
    data_source: str = 'synthetic'  # 'synthetic' or path to ImageNet-100
    csv_output_dir: str = 'results/raw'
    verbose: bool = True
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 8, 32, 128]


def run_pytorch_benchmark(config: BenchmarkConfig) -> List[Dict[str, Any]]:
    """
    Run benchmark for PyTorch model.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        List of result dictionaries (one per batch size)
    """
    import torch
    from models.torch_zoo import get_torch_model
    
    if config.verbose:
        print(f"\n{'='*60}")
        print(f"PyTorch Benchmark: {config.model_name}")
        print(f"{'='*60}")
    
    # Get device
    device, device_info = get_torch_device(config.device_prefer)
    if config.verbose:
        from utils.device import print_device_info
        print_device_info(device_info)
        print()
    
    # Load model
    model, preprocess_fn, input_shape, metadata = get_torch_model(
        config.model_name,
        pretrained=False,  # Use random weights for consistency
        device=device
    )
    
    if config.input_shape is None:
        config.input_shape = input_shape
    
    # Reset memory stats
    reset_memory_stats('pytorch', device)
    
    results = []
    
    for batch_size in config.batch_sizes:
        if config.verbose:
            print(f"Benchmarking batch size: {batch_size}")
        
        # Prepare input
        x = get_synthetic_batch(
            batch_size=batch_size,
            input_shape=config.input_shape,
            framework='pytorch',
            dtype=config.dtype,
            device=device,
            seed=42
        )
        x = preprocess_fn(x)
        
        # Define inference function
        def inference_fn():
            with torch.no_grad():
                return model(x)
        
        # Measure compilation time (PyTorch JIT if applicable, otherwise 0)
        compilation_time_ms = 0.0  # PyTorch eager mode doesn't have separate compilation
        
        # Warmup
        for _ in range(config.warmup_iterations):
            inference_fn()
        
        # Measure memory
        reset_memory_stats('pytorch', device)
        with track_memory('pytorch', device) as mem_tracker:
            # Measure latency
            latency_stats = measure_latency(
                inference_fn,
                warmup_iterations=0,  # Already warmed up
                measurement_iterations=config.measurement_iterations,
                framework='pytorch',
                device=device,
                verbose=False
            )
        
        peak_memory_mb = mem_tracker.get_peak_usage_mb()
        
        # Calculate throughput
        throughput = calculate_throughput(batch_size, latency_stats.mean_ms / 1000.0)
        
        # Create result
        result = create_result_dict(
            framework='pytorch',
            model=config.model_name,
            device=str(device),
            device_type=device_info.device_type,
            batch_size=batch_size,
            input_shape=config.input_shape,
            dtype=config.dtype,
            warmup_iterations=config.warmup_iterations,
            measurement_iterations=config.measurement_iterations,
            latency_stats=latency_stats,
            throughput_ips=throughput,
            memory_mb=peak_memory_mb,
            compilation_time_ms=compilation_time_ms
        )
        
        results.append(result)
        
        if config.verbose:
            print(f"  Latency (p50): {latency_stats.p50_ms:.3f}ms")
            print(f"  Throughput: {throughput:.2f} images/sec")
            print(f"  Memory: {peak_memory_mb:.2f} MB")
            print()
    
    return results


def run_jax_benchmark(config: BenchmarkConfig) -> List[Dict[str, Any]]:
    """
    Run benchmark for JAX/Flax model.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        List of result dictionaries (one per batch size)
    """
    import jax
    import jax.numpy as jnp
    from models.jax_flax_zoo import get_flax_model
    
    if config.verbose:
        print(f"\n{'='*60}")
        print(f"JAX/Flax Benchmark: {config.model_name}")
        print(f"{'='*60}")
    
    # Get device
    device, device_info = get_jax_device(config.device_prefer)
    if config.verbose:
        from utils.device import print_device_info
        print_device_info(device_info)
        print()
    
    # Load model
    rng_key = jax.random.PRNGKey(42)
    
    # Determine input shape (JAX uses HWC format)
    if config.input_shape is None:
        # Convert from CHW to HWC
        if config.model_name == 'resnet50':
            input_shape_jax = (224, 224, 3)
        elif config.model_name == 'vit_b_16':
            input_shape_jax = (224, 224, 3)
        else:
            input_shape_jax = (224, 224, 3)
    else:
        # Convert CHW to HWC
        C, H, W = config.input_shape
        input_shape_jax = (H, W, C)
    
    apply_fn, params, preprocess_fn, metadata = get_flax_model(
        config.model_name,
        input_shape=input_shape_jax,
        rng_key=rng_key,
        num_classes=1000
    )
    
    results = []
    
    for batch_size in config.batch_sizes:
        if config.verbose:
            print(f"Benchmarking batch size: {batch_size}")
        
        # Prepare input
        x = get_synthetic_batch(
            batch_size=batch_size,
            input_shape=input_shape_jax,
            framework='jax',
            dtype=config.dtype,
            seed=42
        )
        x = preprocess_fn(x)
        
        # JIT compile (first call)
        @jax.jit
        def jitted_apply(params, x):
            return apply_fn(params, x, train=False)
        
        # Measure compilation time
        def compile_fn():
            _ = jitted_apply(params, x).block_until_ready()
        
        compilation_time_ms = measure_compilation_time(compile_fn, framework='jax', verbose=False)
        
        # Define inference function
        def inference_fn():
            return jitted_apply(params, x).block_until_ready()
        
        # Warmup (already compiled, but do a few more iterations)
        for _ in range(config.warmup_iterations):
            inference_fn()
        
        # Measure memory (JAX uses process memory)
        with track_memory('jax') as mem_tracker:
            # Measure latency
            latency_stats = measure_latency(
                inference_fn,
                warmup_iterations=0,  # Already warmed up
                measurement_iterations=config.measurement_iterations,
                framework='jax',
                device=None,  # JAX doesn't need device for sync
                verbose=False
            )
        
        peak_memory_mb = mem_tracker.get_peak_usage_mb()
        
        # Calculate throughput
        throughput = calculate_throughput(batch_size, latency_stats.mean_ms / 1000.0)
        
        # Create result
        result = create_result_dict(
            framework='jax',
            model=config.model_name,
            device=str(device),
            device_type=device_info.device_type,
            batch_size=batch_size,
            input_shape=input_shape_jax,
            dtype=config.dtype,
            warmup_iterations=config.warmup_iterations,
            measurement_iterations=config.measurement_iterations,
            latency_stats=latency_stats,
            throughput_ips=throughput,
            memory_mb=peak_memory_mb,
            compilation_time_ms=compilation_time_ms
        )
        
        results.append(result)
        
        if config.verbose:
            print(f"  Latency (p50): {latency_stats.p50_ms:.3f}ms")
            print(f"  Throughput: {throughput:.2f} images/sec")
            print(f"  Memory: {peak_memory_mb:.2f} MB")
            print(f"  Compilation: {compilation_time_ms:.2f}ms")
            print()
    
    return results


def run_inference_benchmark(config: BenchmarkConfig) -> List[Dict[str, Any]]:
    """
    Run inference benchmark based on configuration.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        List of result dictionaries
    """
    if config.framework == 'pytorch':
        return run_pytorch_benchmark(config)
    elif config.framework == 'jax':
        return run_jax_benchmark(config)
    else:
        raise ValueError(f"Unknown framework: {config.framework}")


def run_benchmark_suite(
    models: List[str] = None,
    frameworks: List[str] = None,
    batch_sizes: List[int] = None,
    csv_output_dir: str = 'results/raw',
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Run a complete benchmark suite across models and frameworks.
    
    Args:
        models: List of model names (default: ['resnet50', 'vit_b_16'])
        frameworks: List of frameworks (default: ['pytorch', 'jax'])
        batch_sizes: List of batch sizes (default: [1, 8, 32, 128])
        csv_output_dir: Directory for CSV output
        verbose: Print progress if True
        
    Returns:
        List of all result dictionaries
    """
    if models is None:
        models = ['resnet50', 'vit_b_16']
    if frameworks is None:
        frameworks = ['pytorch', 'jax']
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 128]
    
    all_results = []
    logger = BenchmarkLogger(output_dir=csv_output_dir)
    
    for framework in frameworks:
        for model_name in models:
            config = BenchmarkConfig(
                framework=framework,
                model_name=model_name,
                batch_sizes=batch_sizes,
                csv_output_dir=csv_output_dir,
                verbose=verbose
            )
            
            results = run_inference_benchmark(config)
            all_results.extend(results)
            
            # Log results
            logger.append_results(results)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmark Suite Complete")
        print(f"{'='*60}")
        print(f"Total configurations: {len(all_results)}")
        print(f"Results saved to: {logger.get_filepath()}")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference benchmarks')
    parser.add_argument('--framework', choices=['pytorch', 'jax', 'both'], default='both')
    parser.add_argument('--model', choices=['resnet50', 'vit_b_16', 'both'], default='both')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 32, 128])
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output-dir', default='results/raw')
    parser.add_argument('--verbose', action='store_true', default=True)
    
    args = parser.parse_args()
    
    frameworks = ['pytorch', 'jax'] if args.framework == 'both' else [args.framework]
    models = ['resnet50', 'vit_b_16'] if args.model == 'both' else [args.model]
    
    run_benchmark_suite(
        models=models,
        frameworks=frameworks,
        batch_sizes=args.batch_sizes,
        csv_output_dir=args.output_dir,
        verbose=args.verbose
    )

