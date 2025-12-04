# Deep Learning Framework Benchmark: JAX vs PyTorch

A comprehensive performance comparison of JAX/Flax and PyTorch for computer vision workloads on NVIDIA H100 GPUs.

## Overview

This project benchmarks inference and training performance across multiple CNN and transformer architectures, measuring throughput, latency, memory usage, and energy consumption. The study reveals JAX's 1.5-2× speedup for CNNs but identifies critical stability issues with Vision Transformers due to XLA compiler bugs.

### Key Findings

- **CNN Performance**: JAX achieves 1.98× speedup on ResNet-50, 1.83× on EfficientNet-B0
- **Transformer Challenges**: ViT-Base experiences XLA autotuning failures requiring workarounds
- **Memory Trade-offs**: JAX uses ~10-15% more memory but delivers lower latency
- **Energy Efficiency**: Similar power consumption between frameworks for equivalent workloads

## Repository Structure

```
├── bench/                  # Core benchmarking code
│   ├── runner.py          # Inference benchmark runner  
│   ├── training_runner.py # Training benchmark runner
│   ├── jax_models.py      # JAX/Flax model implementations
│   └── pytorch_models.py  # PyTorch model implementations
├── utils/                 # Utility modules
│   ├── device.py         # Device detection and setup
│   ├── memory.py         # Memory tracking utilities
│   ├── energy.py         # Energy measurement via NVML
│   └── timing.py         # Timing and synchronization
├── models/               # Model architecture definitions
├── analysis/             # Generated analysis outputs
├── report/               # LaTeX report and figures
├── results/              # Raw benchmark CSV results
└── scripts/              # Helper scripts for running benchmarks
```

## Hardware & Software Requirements

### Hardware
- **GPU**: NVIDIA H100 NVL (80GB) or comparable
- **CPU**: AMD EPYC 7763 or similar
- **RAM**: 512GB recommended for full dataset

### Software
- Python 3.10+
- CUDA 12.x
- cuDNN 8.9+

## Installation

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: Energy tracking (requires root/sudo for NVML)
pip install -r requirements_energy.txt
```

### 3. Configure Hugging Face Cache (Optional)

```bash
export HF_HOME=/path/to/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
```

## Quick Start

### Run Inference Benchmark

```bash
# PyTorch inference
python bench/runner.py \
    --framework pytorch \
    --model resnet50 \
    --batch-sizes 1 8 32 128 \
    --output-dir results/inference/pytorch

# JAX inference (with ViT fix)
bash run_jax_inference_fixed.sh
```

### Run Training Benchmark

```bash
# Full training benchmark (both frameworks)
bash run_full_benchmark_with_inference.sh

# Individual training run
python bench/training_runner.py \
    --framework pytorch \
    --model resnet50 \
    --batch-size 32 \
    --epochs 2 \
    --dataset clane9/imagenet-100
```

### Analyze Results

```bash
python analyze_results.py
```

This generates:
- LaTeX tables in `analysis/`
- Throughput scaling plots
- Memory/energy comparison charts

## Supported Models

| Model | Parameters | FLOPs | Architecture Type |
|-------|-----------|-------|-------------------|
| ResNet-50 | 25.6M | 4.1G | CNN (Residual) |
| ViT-Base | 86M | 17.6G | Transformer |
| MobileNetV3-Small | 2.5M | 56M | CNN (Efficient) |
| EfficientNet-B0 | 5.3M | 390M | CNN (Compound Scaled) |

## Known Issues

### ViT XLA Autotuning Bug

**Problem**: JAX ViT-Base crashes with `CUDA_ERROR_ILLEGAL_ADDRESS` on H100 GPUs when XLA autotuning is enabled.

**Workaround**: Disable autotuning for ViT:
```bash
export XLA_FLAGS="--xla_gpu_autotune_level=0"
```

**Impact**: ViT inference runs at 0.56× PyTorch speed instead of expected speedup. Training fails entirely.

**Solution**: See `run_jax_inference_fixed.sh` for conditional autotuning logic.

## Results Summary

### Inference Performance (Batch Size 128)

| Model | PyTorch (img/s) | JAX (img/s) | Speedup |
|-------|----------------|-------------|---------|
| ResNet-50 | 5,915 | 11,739 | 1.98× |
| MobileNetV3 | 37,325 | 56,123 | 1.50× |
| EfficientNet-B0 | 9,027 | 16,510 | 1.83× |
| ViT-Base | 946 | 530 | 0.56× |

### Training Performance (Batch Size 32, Epoch 2)

| Model | Framework | Val Acc (%) | Time/Epoch (s) | Energy (kJ) |
|-------|-----------|-------------|----------------|-------------|
| ResNet-50 | PyTorch | 90.8 | 102.4 | 24.3 |
| ResNet-50 | JAX | 36.3 | 159.1 | 20.7 |
| MobileNetV3 | PyTorch | 80.2 | 90.2 | 9.8 |
| MobileNetV3 | JAX | 11.6 | 165.4 | 15.1 |

*Note: PyTorch models use pre-trained weights; JAX models train from scratch*

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{jax_pytorch_benchmark_2024,
  title={Deep Learning Framework Benchmark: JAX vs PyTorch on NVIDIA H100},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/594Project}}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- ImageNet-100 dataset: clane9/imagenet-100 (Hugging Face)
- NVIDIA for H100 GPU access
- JAX/Flax and PyTorch teams for framework development
