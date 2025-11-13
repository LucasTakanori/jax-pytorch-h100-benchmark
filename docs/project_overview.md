# Project Overview

## Research Question

**How does JAX compare to PyTorch across different neural network architectures and hardware platforms?**

This project provides a systematic evaluation of JAX and PyTorch performance, focusing on understanding *why* performance differences occur, not just *what* they are.

## Scope

### Models
- **ResNet-50** (25M parameters, ~4 GFLOPs) - Baseline CNN architecture
- **Vision Transformer Base** (86M parameters) - Modern attention-based architecture

### Frameworks
- **JAX + XLA** - Native implementations with JIT compilation
- **PyTorch + cuDNN** - Reference implementations from torchvision

### Hardware Platforms
- **CPU** - Baseline performance
- **NVIDIA GPU (H100)** - Primary acceleration platform
- **TPU (optional)** - Google Cloud TPU if access granted

### Configurations
**8 Primary Configurations:**
- ResNet-50: JAX (CPU, GPU), PyTorch (CPU, GPU)
- ViT-Base: JAX (CPU, GPU), PyTorch (CPU, GPU)

**Optional (if TPU access granted):**
- 4 additional TPU configurations

## Metrics Collected

1. **Latency** - Single sample inference time (p50, p95, mean)
2. **Throughput** - Images per second at batch sizes [1, 8, 32, 128]
3. **Memory** - Peak memory usage during inference
4. **Compilation Time** - XLA/JIT compilation overhead
5. **Energy** - Power consumption (where measurable)

## Key Contributions

1. **First systematic JAX vs PyTorch comparison** across diverse hardware
2. **Modern architecture coverage** - Includes Vision Transformers
3. **Deep analysis** - Understand bottlenecks (compute vs memory)
4. **Reproducible methodology** - Open source code and documentation
5. **Hardware-software co-design insights** - Actionable for practitioners

## Project Structure

```
594Project/
├── bench/              # Benchmark scripts
│   ├── bench_infer_torch.py
│   ├── bench_infer_jax.py
│   └── runner.py       # Unified benchmark runner (Phase 2)
├── models/             # Model implementations
│   ├── torch_zoo.py    # PyTorch models (Phase 2)
│   └── jax_flax_zoo.py # JAX/Flax models (Phase 2)
├── utils/              # Utilities
│   ├── device.py       # Device detection (Phase 1) ✅
│   ├── timing.py       # Timing utilities (Phase 2)
│   ├── memory.py       # Memory profiling (Phase 2)
│   ├── logging.py      # CSV logging (Phase 2)
│   ├── data.py         # Data loading (Phase 2)
│   └── validation.py   # Numerical validation (Phase 2)
├── scripts/
│   └── verify_setup.py # Setup verification (Phase 1) ✅
├── data/
│   └── imagenet100/    # ImageNet-100 dataset
├── results/
│   ├── raw/            # CSV benchmark logs (Phase 3)
│   └── figs/           # Plots and visualizations (Phase 4)
└── docs/               # Documentation
    ├── setup.md        # Setup guide (Phase 1) ✅
    ├── tpu_setup.md    # TPU access guide (Phase 1) ✅
    ├── phases.md       # Detailed phase breakdown ✅
    └── project_overview.md # This file ✅
```

## Phase Status

| Phase | Status | Primary Contributor |
|-------|--------|---------------------|
| **Phase 1: Setup** | ✅ Complete | Shashwat S. |
| **Phase 2: Implementation** | 🚧 In Progress | Shashwat S. |
| **Phase 3: Data Collection** | ⏳ Pending | Team (all members) |
| **Phase 4: Analysis** | ⏳ Pending | Team (all members) |
| **Phase 5: Finalization** | ⏳ Pending | Team (all members) |

**Note**: Phases are sequential. Phase 2 is currently in progress. Phases 3-4 will be handled by the team.

**Phase 1 Verification**: Phase 1 completion was verified with successful test runs. See [Project Phases](phases.md#verification-results) for detailed verification outputs showing all checks passed and benchmarks running successfully.

## Team Members

- **Sanchez Shiromizu L.T.** (lsanc68@uic.edu) - JAX implementations
- **Shashwat S.** (ssinha30@uic.edu) - PyTorch baselines
- **Prathyush B.** (pball5@uic.edu) - Benchmarking infrastructure
- **Sai M.** (sbadr4@uic.edu) - Analysis and visualization

## Resources

- **GPU Access**: University cluster with 4x H100NVL
- **Dataset**: ImageNet-100 (~50GB, 100-class subset)
- **Estimated GPU Hours**: ~120 hours total
- **TPU Access**: Optional, via Google Cloud TPU Research Credits

## Success Criteria

### Technical
- ✅ All models implemented and numerically validated
- ✅ All 8 primary configurations benchmarked
- ✅ Minimum 50 iterations per configuration
- ✅ All metrics collected and analyzed
- ✅ Code is reproducible

### Research
- ✅ Clear JAX vs PyTorch performance comparison
- ✅ Understanding of performance differences
- ✅ Hardware-software co-design insights
- ✅ Actionable recommendations

## Documentation

- **[Setup Guide](setup.md)** - Environment setup for local and remote systems
- **[TPU Setup Guide](tpu_setup.md)** - TPU access application and setup
- **[Project Phases](phases.md)** - Detailed phase breakdown with objectives and deliverables
- **[Project Overview](project_overview.md)** - This document

## Quick Links

- **Getting Started**: See [Setup Guide](setup.md)
- **Current Phase**: See [Project Phases](phases.md) for Phase 2 details
- **Verification**: Run `python scripts/verify_setup.py`
- **Test Benchmarks**: Run `python bench/bench_infer_torch.py` and `python bench/bench_infer_jax.py`

## Next Steps

1. **Current**: Complete Phase 2 - Model implementations and benchmarking infrastructure (Shashwat S.)
2. **After Phase 2**: Team will begin Phase 3 - Comprehensive data collection on H100
3. **After Phase 3**: Team will begin Phase 4 - Analysis, visualization, and report writing
4. **After Phase 4**: Team will begin Phase 5 - Finalization and presentation

For detailed information, see [Project Phases](phases.md).

