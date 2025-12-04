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
from utils.energy import EnergyTracker, EnergyStats
from utils.profiling import NVMLProfiler, ProfileStats

# JAX/Flax imports
try:
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint
    from flax.training import train_state
except ImportError:
    pass


def _wandb_requested() -> bool:
    """Return True if Weights & Biases logging should run (default)."""
    flag = os.environ.get("WANDB_DISABLED")
    if flag is None:
        return True
    return flag.lower() not in {"1", "true", "yes"}


def _maybe_init_wandb(models: List[str], frameworks: List[str], batch_sizes: List[int], verbose: bool):
    """
    Initialize Weights & Biases logging if available and not disabled.
    Returns a dict with wandb module and run, or None.
    """
    if not _wandb_requested():
        if verbose:
            print("W&B logging disabled via WANDB_DISABLED.")
        return None
    try:
        import wandb
    except ImportError:
        if verbose:
            print("W&B not installed; skipping experiment logging.")
        return None
    if "WANDB_MODE" not in os.environ and not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")
        if verbose:
            print("W&B API key not found; defaulting to offline mode.")
    project = os.environ.get("WANDB_PROJECT", "594Project-benchmarks")
    entity = os.environ.get("WANDB_ENTITY")
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            config={
                "models": models,
                "frameworks": frameworks,
                "batch_sizes": batch_sizes,
            },
            settings=wandb.Settings(start_method="thread"),
        )
    except Exception as exc:
        if verbose:
            print(f"Failed to initialize W&B logging: {exc}")
        return None
    if verbose:
        print(f"W&B logging enabled (project={project}).")
    return {"wandb": wandb, "run": run}


def _sanitize_for_wandb(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert benchmark result dict into W&B-friendly scalars."""
    sanitized = {}
    for key, value in row.items():
        if isinstance(value, (tuple, list)):
            sanitized[key] = list(value)
        elif isinstance(value, (LatencyStats,)):
            sanitized[key] = value.mean_ms
        elif isinstance(value, (int, float, str)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    framework: str  # 'pytorch' or 'jax'
    model_name: str  # 'resnet50', 'vit_b_16', 'mobilenet_v3_small', 'efficientnet_b0'
    device_prefer: Optional[str] = None  # Preferred device type
    batch_sizes: List[int] = None  # Default: [1, 8, 32, 128]
    warmup_iterations: int = 10
    measurement_iterations: int = 50
    input_shape: Optional[Tuple[int, ...]] = None  # Will be inferred from model
    dtype: str = 'float32'
    data_source: str = 'synthetic'  # 'synthetic' or path to ImageNet-100
    csv_output_dir: str = 'results/raw'
    verbose: bool = True
    track_energy: bool = True  # Track energy consumption
    track_profiling: bool = True  # Track detailed profiling metrics
    checkpoint_path: Optional[str] = None  # Path to checkpoint to load

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
    
    # Load checkpoint if provided
    if config.checkpoint_path:
        if config.verbose:
            print(f"Loading checkpoint from {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    
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
        
        # Measure compilation time (PyTorch eager mode doesn't have separate compilation)
        compilation_time_ms = 0.0

        # Warmup
        for _ in range(config.warmup_iterations):
            inference_fn()

        # Prepare tracking
        energy_tracker = None
        nvml_profiler = None

        if config.track_energy and device.type == 'cuda':
            energy_tracker = EnergyTracker(device_id=device.index if device.index is not None else 0)

        if config.track_profiling and device.type == 'cuda':
            nvml_profiler = NVMLProfiler(device_id=device.index if device.index is not None else 0)

        # Start tracking
        if energy_tracker:
            energy_tracker.start()
        if nvml_profiler:
            nvml_profiler.start_sampling()

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

            # Sample profiling periodically
            if nvml_profiler:
                for _ in range(5):  # Take a few more samples
                    inference_fn()
                    nvml_profiler.collect_sample()

        # Stop tracking
        energy_stats = None
        profile_stats = None

        if energy_tracker:
            energy_tracker.stop()
            energy_stats = energy_tracker.get_stats()
            energy_tracker.cleanup()

        if nvml_profiler:
            profile_stats = nvml_profiler.stop_sampling()
            nvml_profiler.cleanup()

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

        # Add energy and profiling metrics
        if energy_stats:
            result['energy_j'] = energy_stats.total_energy_j
            result['avg_power_w'] = energy_stats.avg_power_w
            result['peak_power_w'] = energy_stats.peak_power_w
            result['avg_temp_c'] = energy_stats.avg_temperature_c

        if profile_stats:
            result['gpu_utilization_pct'] = profile_stats.gpu_utilization_pct
            result['memory_utilization_pct'] = profile_stats.memory_utilization_pct
            result['pcie_h2d_gb'] = profile_stats.pcie_host_to_device_gb
            result['pcie_d2h_gb'] = profile_stats.pcie_device_to_host_gb
            result['memory_bandwidth_gb_s'] = profile_stats.memory_bandwidth_gb_s
        
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
        input_shape_jax = (224, 224, 3)
    else:
        # Convert CHW to HWC
        C, H, W = config.input_shape
        input_shape_jax = (H, W, C)
    
    # Clear any previous JAX compilations and force GC to prevent memory corruption
    # This is critical for avoiding CUDA_ERROR_ILLEGAL_ADDRESS when switching models
    try:
        import jax
        jax.clear_caches()
    except:
        pass
        
    import gc
    gc.collect()

    # Get model and preprocessing
    apply_fn, params, preprocess_fn, metadata = get_flax_model(
        config.model_name,
        input_shape=input_shape_jax,
        rng_key=rng_key,
        num_classes=1000
    )
    
    # Load checkpoint if provided
    if config.checkpoint_path:
        if config.verbose:
            print(f"Loading checkpoint from {config.checkpoint_path}")
        
        # Create a dummy train state to restore into
        # We need to replicate the TrainState structure from training_runner.py
        class TrainState(train_state.TrainState):
            batch_stats: Any
            
        # Initialize state with dummy optimizer (we only need params and batch_stats)
        import optax
        tx = optax.sgd(0.1)
        
        # Split params into params and batch_stats if BatchNormalization is used
        if 'batch_stats' in params:
            batch_stats = params['batch_stats']
            params_only = params['params']
        else:
            batch_stats = {}
            params_only = params
            
        state = TrainState.create(
            apply_fn=apply_fn,
            params=params_only,
            tx=tx,
            batch_stats=batch_stats
        )
        
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = checkpointer.restore(config.checkpoint_path, item=state)
        
        params = {'params': restored.params}
        if restored.batch_stats:
            params['batch_stats'] = restored.batch_stats
    
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

        # Prepare tracking
        energy_tracker = None
        nvml_profiler = None
        device_id = 0  # Default CUDA device

        if device_info.device_type == 'cuda':
            if device_info.device_id is not None:
                device_id = device_info.device_id

            if config.track_energy:
                energy_tracker = EnergyTracker(device_id=device_id)

            if config.track_profiling:
                nvml_profiler = NVMLProfiler(device_id=device_id)

        # Start tracking
        if energy_tracker:
            energy_tracker.start()
        if nvml_profiler:
            nvml_profiler.start_sampling()

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

            # Sample profiling periodically
            if nvml_profiler:
                for _ in range(5):  # Take a few more samples
                    inference_fn()
                    nvml_profiler.collect_sample()

        # Stop tracking
        energy_stats = None
        profile_stats = None

        if energy_tracker:
            energy_tracker.stop()
            energy_stats = energy_tracker.get_stats()
            energy_tracker.cleanup()

        if nvml_profiler:
            profile_stats = nvml_profiler.stop_sampling()
            nvml_profiler.cleanup()

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

        # Add energy and profiling metrics
        if energy_stats:
            result['energy_j'] = energy_stats.total_energy_j
            result['avg_power_w'] = energy_stats.avg_power_w
            result['peak_power_w'] = energy_stats.peak_power_w
            result['avg_temp_c'] = energy_stats.avg_temperature_c

        if profile_stats:
            result['gpu_utilization_pct'] = profile_stats.gpu_utilization_pct
            result['memory_utilization_pct'] = profile_stats.memory_utilization_pct
            result['pcie_h2d_gb'] = profile_stats.pcie_host_to_device_gb
            result['pcie_d2h_gb'] = profile_stats.pcie_device_to_host_gb
            result['memory_bandwidth_gb_s'] = profile_stats.memory_bandwidth_gb_s
        
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
    wandb_session = _maybe_init_wandb(models, frameworks, batch_sizes, verbose)
    
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
            if wandb_session:
                wandb_module = wandb_session["wandb"]
                for row in results:
                    wandb_module.log(_sanitize_for_wandb(row))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmark Suite Complete")
        print(f"{'='*60}")
        print(f"Total configurations: {len(all_results)}")
        print(f"Results saved to: {logger.get_filepath()}")
    
    if wandb_session:
        wandb_session["run"].finish()
    
    return all_results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference benchmarks')
    parser.add_argument('--framework', choices=['pytorch', 'jax', 'both'], default='both')
    parser.add_argument('--model',
                        choices=['resnet50', 'vit_b_16', 'mobilenet_v3_small', 'efficientnet_b0', 'all'],
                        default='all',
                        help='Model to benchmark (default: all)')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 32, 128])
    parser.add_argument(
        '--iterations', '--measurement-iterations',
        dest='iterations',
        type=int,
        default=50,
        help='Number of measurement iterations (legacy flag --measurement-iterations is also accepted)'
    )
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output-dir', default='results/raw')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--no-energy', action='store_true',
                        help='Disable energy tracking')
    parser.add_argument('--no-profiling', action='store_true',
                        help='Disable detailed profiling')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint to load')
    
    args = parser.parse_args()

    frameworks = ['pytorch', 'jax'] if args.framework == 'both' else [args.framework]

    # Handle model selection
    all_models = ['resnet50', 'vit_b_16', 'mobilenet_v3_small', 'efficientnet_b0']
    if args.model == 'all':
        models = all_models
    else:
        models = [args.model]

    # Run benchmarks for each combination
    logger = BenchmarkLogger(args.output_dir)

    for framework in frameworks:
        for model in models:
            config = BenchmarkConfig(
                framework=framework,
                model_name=model,
                batch_sizes=args.batch_sizes,
                warmup_iterations=args.warmup,
                measurement_iterations=args.iterations,
                csv_output_dir=args.output_dir,
                verbose=args.verbose,
                track_energy=not args.no_energy,
                track_profiling=not args.no_profiling,
                checkpoint_path=args.checkpoint
            )

            results = run_inference_benchmark(config)
            logger.log_results(results)

