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

## Project Structure

```
594Project/
├── bench/              # Benchmark scripts
├── models/             # Model implementations (ResNet, ViT)
├── utils/              # Utilities (device detection, timing, logging)
├── scripts/             # Helper scripts (setup verification)
├── data/               # Dataset storage (ImageNet-100)
├── results/             # Benchmark results
│   ├── raw/            # CSV logs
│   └── figs/           # Plots and visualizations
├── docs/               # Documentation
└── tests/              # Unit tests
```

## Team Members

- Sanchez Shiromizu L.T. (lsanc68@uic.edu)
- Shashwat S. (ssinha30@uic.edu)
- Prathyush B. (pball5@uic.edu)
- Sai M. (sbadr4@uic.edu)

## Project Status

- ✅ **Phase 1: Setup** (Complete) - Device detection, verification scripts, documentation
- 🚧 **Phase 2: Implementation & Infrastructure** (In Progress - Shashwat S.) - Model implementations, benchmarking infrastructure
- ⏳ **Phase 3: Data Collection** (Pending - Team) - Comprehensive benchmarking across all configurations
- ⏳ **Phase 4: Analysis & Documentation** (Pending - Team) - Performance analysis, visualizations, report writing
- ⏳ **Phase 5: Finalization** (Pending - Team) - Code cleanup, presentation, repository finalization

**Note**: Phases are sequential. Phase 2 is currently in progress. Phases 3-4 will be handled by the team.

See [Project Phases](docs/phases.md) for detailed phase breakdown, objectives, and deliverables.

## Contributing

See [Setup Guide](docs/setup.md) for environment setup and collaboration guidelines.
