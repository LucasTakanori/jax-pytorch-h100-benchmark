# Benchmarking Guide

This guide explains how to use the benchmarking infrastructure in 594Project.

## Verification

The benchmark runner has been tested and verified working. Example output:

```bash
$ python bench/runner.py --framework pytorch --model resnet50 --batch-sizes 1 4 --iterations 10 --warmup 3
============================================================
PyTorch Benchmark: resnet50
============================================================
Device: Apple Silicon GPU (Metal)
  Type: MPS
  Available: True

Benchmarking batch size: 1
  Latency (p50): 5.476ms
  Throughput: 182.39 images/sec
  Memory: 103.06 MB

Benchmarking batch size: 4
  Latency (p50): 12.174ms
  Throughput: 328.16 images/sec
  Memory: 105.67 MB

============================================================
Benchmark Suite Complete
============================================================
Total configurations: 2
Results saved to: results/raw/benchmark_results_20251113_134337.csv
```

**Verification:** ✅ Benchmark runner successfully executes, collects metrics, and logs results to CSV.

## Quick Start

### Using the Command-Line Interface

The simplest way to run benchmarks is using the unified benchmark runner:

```bash
# Run full benchmark suite (all models, all frameworks)
python bench/runner.py --framework both --model both

# Run specific model and framework
python bench/runner.py --framework pytorch --model resnet50

# Custom batch sizes and iterations
python bench/runner.py --framework jax --model vit_b_16 --batch-sizes 1 16 32 64 --iterations 100 --warmup 20
```

### Using Python API

```python
from bench.runner import BenchmarkConfig, run_inference_benchmark, run_benchmark_suite

# Single benchmark
config = BenchmarkConfig(
    framework='pytorch',
    model_name='resnet50',
    batch_sizes=[1, 8, 32, 128],
    warmup_iterations=10,
    measurement_iterations=50
)
results = run_inference_benchmark(config)

# Full suite
all_results = run_benchmark_suite(
    models=['resnet50', 'vit_b_16'],
    frameworks=['pytorch', 'jax'],
    batch_sizes=[1, 8, 32, 128]
)
```

## Benchmark Configuration

### BenchmarkConfig Parameters

- `framework`: 'pytorch' or 'jax'
- `model_name`: 'resnet50' or 'vit_b_16'
- `device_prefer`: Optional device preference ('cuda', 'mps', 'cpu', 'tpu')
- `batch_sizes`: List of batch sizes to test (default: [1, 8, 32, 128])
- `warmup_iterations`: Number of warmup iterations (default: 10)
- `measurement_iterations`: Number of measurement iterations (default: 50)
- `input_shape`: Optional input shape (auto-detected from model if not provided)
- `dtype`: Data type ('float32', 'float16', 'bfloat16')
- `data_source`: 'synthetic' or path to ImageNet-100 dataset
- `csv_output_dir`: Directory for CSV results (default: 'results/raw')
- `verbose`: Print progress (default: True)

## Metrics Collected

Each benchmark run collects the following metrics:

### Latency Metrics (milliseconds)
- `latency_p50_ms`: Median (50th percentile) latency
- `latency_p95_ms`: 95th percentile latency
- `latency_p99_ms`: 99th percentile latency
- `latency_mean_ms`: Mean latency
- `latency_std_ms`: Standard deviation
- `latency_min_ms`: Minimum latency
- `latency_max_ms`: Maximum latency

### Performance Metrics
- `throughput_ips`: Images per second
- `memory_mb`: Peak memory usage in MB
- `compilation_time_ms`: JIT/XLA compilation time (JAX only)

### Metadata
- `framework`: 'pytorch' or 'jax'
- `model`: Model name
- `device`: Device identifier
- `device_type`: 'cpu', 'cuda', 'mps', 'tpu'
- `batch_size`: Batch size used
- `input_shape`: Input shape tuple
- `dtype`: Data type
- `timestamp`: Benchmark timestamp
- `git_commit`: Git commit hash (if available)

## Results Storage

Results are automatically saved to CSV files in `results/raw/` directory with timestamps:

```
results/raw/benchmark_results_20250101_120000.csv
```

### Reading Results

```python
from utils.logging import BenchmarkLogger
import pandas as pd

# Read results
logger = BenchmarkLogger(output_dir='results/raw', filename='benchmark_results_20250101_120000.csv')
df = logger.read_results()

# Filter and analyze
pytorch_resnet = df[(df['framework'] == 'pytorch') & (df['model'] == 'resnet50')]
print(pytorch_resnet[['batch_size', 'latency_p50_ms', 'throughput_ips', 'memory_mb']])
```

## Best Practices

### 1. Warmup Iterations

Always use sufficient warmup iterations to:
- Allow JIT compilation (JAX)
- Warm up GPU/device
- Stabilize memory allocation

**Recommended**: 10-20 warmup iterations

### 2. Measurement Iterations

Use enough iterations for statistical significance:
- **Minimum**: 50 iterations
- **Recommended**: 100+ iterations for production benchmarks
- **For research**: 200+ iterations for high confidence

### 3. Batch Sizes

Test multiple batch sizes to understand scaling:
- **Small batches** (1, 4, 8): Latency-focused applications
- **Medium batches** (16, 32): Balanced throughput/latency
- **Large batches** (64, 128, 256): Throughput-focused applications

### 4. Device Selection

The benchmark runner automatically selects the best available device:
- **CUDA**: NVIDIA GPUs (H100, etc.)
- **MPS**: Apple Silicon GPUs
- **CPU**: Fallback

You can override with `device_prefer` parameter.

### 5. Memory Measurement

Memory measurement accuracy varies by framework:
- **PyTorch CUDA**: Accurate GPU memory via `torch.cuda.max_memory_allocated()`
- **PyTorch MPS**: Estimated via process memory
- **JAX**: Process memory (RSS)

For accurate GPU memory, use PyTorch with CUDA.

## Example Workflows

### Quick Performance Check

```bash
# Quick test with minimal iterations
python bench/runner.py --framework pytorch --model resnet50 --batch-sizes 1 16 --iterations 20 --warmup 5
```

### Production Benchmark

```bash
# Full benchmark with high iteration count
python bench/runner.py \
    --framework both \
    --model both \
    --batch-sizes 1 8 32 128 \
    --iterations 200 \
    --warmup 20 \
    --output-dir results/raw/production
```

### Single Model Deep Dive

```python
from bench.runner import BenchmarkConfig, run_inference_benchmark

# Test ResNet-50 across many batch sizes
config = BenchmarkConfig(
    framework='pytorch',
    model_name='resnet50',
    batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
    warmup_iterations=20,
    measurement_iterations=100
)

results = run_inference_benchmark(config)

# Analyze results
for result in results:
    print(f"Batch {result['batch_size']}: "
          f"{result['latency_p50_ms']:.2f}ms, "
          f"{result['throughput_ips']:.1f} img/s")
```

## Troubleshooting

### Issue: JAX compilation is slow

**Solution**: This is expected. JAX JIT compilation happens on first call and is measured separately. Subsequent calls are fast.

### Issue: Memory measurements seem incorrect

**Solution**: 
- For PyTorch CUDA, ensure you're using CUDA device
- For JAX, memory is process-level, not device-level
- Use `reset_memory_stats()` before measurement

### Issue: Results vary between runs

**Solution**:
- Increase measurement iterations
- Ensure sufficient warmup
- Use deterministic random seeds
- Check for background processes affecting performance

### Issue: CSV file not created

**Solution**:
- Check that `results/raw/` directory exists
- Verify write permissions
- Check `csv_output_dir` path is correct

## Advanced Usage

### Custom Data Loading

```python
from utils.data import create_dataloader

# Use ImageNet-100 instead of synthetic data
config = BenchmarkConfig(
    framework='pytorch',
    model_name='resnet50',
    data_source='data/imagenet100'  # Path to dataset
)
```

### Custom Device Selection

```python
config = BenchmarkConfig(
    framework='pytorch',
    model_name='resnet50',
    device_prefer='cuda'  # Force CUDA even if MPS available
)
```

### Mixed Precision

```python
config = BenchmarkConfig(
    framework='pytorch',
    model_name='resnet50',
    dtype='float16'  # Use FP16 (requires GPU support)
)
```

## Integration with Analysis

Results can be directly used for analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/raw/benchmark_results_20250101_120000.csv')

# Plot latency vs batch size
pytorch_resnet = df[(df['framework'] == 'pytorch') & (df['model'] == 'resnet50')]
plt.plot(pytorch_resnet['batch_size'], pytorch_resnet['latency_p50_ms'])
plt.xlabel('Batch Size')
plt.ylabel('Latency (ms)')
plt.title('ResNet-50 Latency vs Batch Size')
plt.show()
```

See `docs/phases.md` for Phase 4 analysis guidelines.

