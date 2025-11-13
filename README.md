# 594Project

**Cross-Platform Performance Analysis of Modern Neural Networks: A JAX-Centric Hardware-Software Co-Design Study**

University of Illinois at Chicago | ECE 594 HW-SW Co-Design for ML Systems

## Overview

This project benchmarks and compares PyTorch and JAX performance across different neural network architectures (ResNet-50, Vision Transformer) and hardware platforms (CPU, GPU, TPU).

## Quick Start

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd 594Project
   ```

2. **Set up environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python scripts/verify_setup.py
   ```

4. **Run a test benchmark:**
   ```bash
   python bench/bench_infer_torch.py
   python bench/bench_infer_jax.py
   ```

### Documentation

- **[Project Overview](docs/project_overview.md)** - Research question, scope, and quick reference
- **[Project Phases](docs/phases.md)** - Detailed breakdown of all 5 project phases, objectives, deliverables, and timeline
- **[Setup Guide](docs/setup.md)** - Complete setup instructions for local (macOS) and remote (H100) systems
- **[TPU Setup Guide](docs/tpu_setup.md)** - Instructions for obtaining and setting up Google Cloud TPU access
- **[Models Documentation](docs/models.md)** - Model implementations guide (ResNet-50, ViT-Base)
- **[Benchmarking Guide](docs/benchmarking.md)** - How to use the benchmarking infrastructure
- **[Validation Guide](docs/validation.md)** - Numerical validation framework and results

## Project Structure

```
594Project/
├── bench/              # Benchmark scripts
│   ├── bench_infer_torch.py
│   ├── bench_infer_jax.py
│   └── runner.py       # Unified benchmark runner ✅
├── models/             # Model implementations
│   ├── __init__.py
│   ├── torch_zoo.py   # PyTorch models (ResNet-50, ViT-Base) ✅
│   └── jax_flax_zoo.py # JAX/Flax models (ResNet-50, ViT-Base) ✅
├── utils/              # Utilities
│   ├── device.py       # Device detection ✅
│   ├── timing.py      # Latency statistics ✅
│   ├── memory.py       # Memory profiling ✅
│   ├── logging.py     # CSV result logging ✅
│   ├── data.py        # Data loading ✅
│   └── validation.py  # Numerical validation ✅
├── scripts/            # Helper scripts
│   └── verify_setup.py # Setup verification ✅
├── data/               # Dataset storage (ImageNet-100)
├── results/             # Benchmark results
│   ├── raw/            # CSV logs
│   └── figs/           # Plots and visualizations
└── docs/               # Documentation
```

## Team Members

- Sanchez Shiromizu L.T. (lsanc68@uic.edu)
- Shashwat S. (ssinha30@uic.edu) - **Phase 1 & 2 Primary Contributor**
- Prathyush B. (pball5@uic.edu)
- Sai M. (sbadr4@uic.edu)

## Project Status

- ✅ **Phase 1: Setup** (Complete - Shashwat S.) - Device detection, verification scripts, documentation. **Sole contributor: Shashwat S.**
- ✅ **Phase 2: Implementation & Infrastructure** (Complete - Shashwat S.) - Model implementations, benchmarking infrastructure. **Sole contributor: Shashwat S.**
- ⏳ **Phase 3: Data Collection** (Pending - Team) - Comprehensive benchmarking across all configurations
- ⏳ **Phase 4: Analysis & Documentation** (Pending - Team) - Performance analysis, visualizations, report writing
- ⏳ **Phase 5: Finalization** (Pending - Team) - Code cleanup, presentation, repository finalization

**Note**: Phases are sequential. Phases 1-2 were completed solely by Shashwat S. Phases 3-4 will be handled by the team.

See [Project Phases](docs/phases.md) for detailed phase breakdown, objectives, and deliverables.

## Contributing

See [Setup Guide](docs/setup.md) for environment setup and collaboration guidelines.
