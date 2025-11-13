# Project Phases - Detailed Breakdown

This document provides a comprehensive overview of all project phases, their objectives, deliverables, and implementation details for the 594Project benchmarking framework.

## Phase 1: Setup

### Objectives

- Establish development environment for all team members
- Set up hardware infrastructure (local and remote)
- Create foundational utilities and verification tools
- Document setup process for reproducibility

### Key Deliverables

1. **Device Detection System** (`utils/device.py`)
   - Unified device detection for PyTorch and JAX
   - Support for CPU, CUDA (NVIDIA GPUs), MPS (Apple Silicon), TPU
   - Device synchronization utilities for accurate timing

2. **Setup Verification** (`scripts/verify_setup.py`)
   - Automated environment checking
   - Dependency validation
   - Hardware detection reporting

3. **Documentation**
   - Setup guide for local (macOS) and remote (H100) systems
   - TPU access application documentation
   - Team collaboration guidelines

4. **Updated Requirements**
   - CUDA JAX installation instructions
   - Platform-specific dependency notes

### Tasks Completed

- ✅ Device detection utilities created
- ✅ Setup verification script implemented
- ✅ Documentation for local and remote setup
- ✅ TPU access application process documented
- ✅ Existing benchmarks updated to use device utilities
- ✅ README updated with project structure and quick start

### Success Criteria

- ✅ Verification script passes on local macOS system
- ✅ Verification script passes on remote H100 system (when available)
- ✅ Both benchmark scripts run successfully
- ✅ Device detection correctly identifies available hardware
- ✅ All team members can set up environment following documentation

### Verification Results

Phase 1 completion was verified with the following test outputs:

#### Setup Verification Script Output

```bash
$ python scripts/verify_setup.py
============================================================
594Project Setup Verification
============================================================

Checking core dependencies...
------------------------------------------------------------
✅ NumPy installed
✅ Pandas installed
✅ Matplotlib installed
✅ psutil installed

Checking PyTorch...
------------------------------------------------------------
✅ PyTorch installed
  Version: 2.9.0
  ⚠️  CUDA not available (expected on macOS/CPU-only systems)
  ✅ MPS (Apple Silicon GPU) available
  ✅ CPU backend available

Checking JAX...
------------------------------------------------------------
✅ JAX installed
  Version: 0.8.0
  Available devices: 1
  ✅ CPU: 1 device(s)
    - TFRT_CPU_0
  ✅ XLA compilation working

Checking Flax...
------------------------------------------------------------
✅ Flax installed
  Version: 0.12.0

Testing device detection utilities...
------------------------------------------------------------
✅ Device utilities module loaded

PyTorch Device Detection:
  Device: Apple Silicon GPU (Metal)
  Type: MPS
  Available: True
  ✅ PyTorch device detection working

JAX Device Detection:
  Device: CPU
  Type: CPU
  Available: True
  ✅ JAX device detection working

============================================================
✅ All checks passed! Your environment is ready.
============================================================
```

#### PyTorch Benchmark Test Output

```bash
$ python bench/bench_infer_torch.py
============================================================
PyTorch Inference Benchmark
============================================================
Device: Apple Silicon GPU (Metal)
  Type: MPS
  Available: True

Warming up...
Running benchmark (20 iterations)...
============================================================
✅ ResNet18 ran successfully!
Avg Latency/Batch: 0.0149s
Throughput: 1076.83 images/sec
============================================================
```

#### JAX Benchmark Test Output

```bash
$ python bench/bench_infer_jax.py
============================================================
JAX + Flax Inference Benchmark
============================================================
Device: CPU
  Type: CPU
  Available: True

Warming up (JIT compilation)...
Running benchmark (20 iterations)...
============================================================
✅ JAX + Flax working!
Median Latency: 0.0087s
Mean Latency: 0.0088s
Throughput: 1810.73 images/sec
============================================================
```

**Verification Summary:**
- ✅ All dependencies installed and detected correctly
- ✅ PyTorch MPS (Apple Silicon GPU) detected and working
- ✅ JAX CPU backend detected and working
- ✅ Device detection utilities functioning properly
- ✅ Both benchmark scripts execute successfully
- ✅ Device information displayed correctly
- ✅ Performance metrics calculated and reported

### Team Responsibilities

- **Member 1**: Device detection utilities, PyTorch integration
- **Member 2**: JAX backend detection, verification script
- **Member 3**: Documentation, setup guides
- **Member 4**: TPU access application, requirements management

### Notes

- TPU access application should be submitted early (can be done in parallel with Phase 2)
- Local development can proceed on macOS while waiting for H100 access
- All code should be committed and pushed for team collaboration
- **Status: ✅ COMPLETE** - All deliverables finished and verified

---

## Phase 2: Implementation & Infrastructure

### Objectives

- Implement native JAX/Flax models (ResNet-50, ViT-Base)
- Implement PyTorch reference models
- Create numerical validation framework
- Build comprehensive benchmarking infrastructure
- Begin baseline measurements

### Key Deliverables

1. **Model Implementations** (`models/`)

   **JAX/Flax Models:**
   - `models/jax_flax_zoo.py` - Model registry
   - Native ResNet-50 implementation in Flax
   - Native Vision Transformer Base implementation in Flax
   - Model loading utilities with pretrained weights (if available)

   **PyTorch Models:**
   - `models/torch_zoo.py` - Model registry
   - ResNet-50 (torchvision or custom)
   - Vision Transformer Base (torchvision or custom)
   - Consistent interface with JAX models

2. **Numerical Validation** (`utils/validation.py`)
   - Forward pass comparison between JAX and PyTorch
   - Tolerance checking (1e-5 for FP32)
   - Gradient validation (if needed for training benchmarks)
   - Test suite for model correctness

3. **Benchmarking Infrastructure** (`utils/`)

   **Timing Utilities** (`utils/timing.py`):
   - Latency statistics (p50, p95, mean, std)
   - Warmup period management
   - Synchronization helpers
   - Throughput calculation

   **Memory Profiling** (`utils/memory.py`):
   - Peak memory usage tracking
   - GPU memory monitoring (CUDA, MPS)
   - Memory efficiency metrics

   **Logging System** (`utils/logging.py`):
   - CSV result logging
   - Structured data format
   - Automatic file naming with timestamps
   - Metadata tracking (device, framework, model, batch size, etc.)

4. **Benchmark Runner** (`bench/runner.py`)
   - Unified benchmarking interface
   - Multi-batch-size support [1, 8, 32, 128]
   - Configurable warmup and measurement iterations
   - Framework-agnostic execution
   - Result aggregation

5. **Data Loading** (`utils/data.py`)
   - Synthetic data generation
   - ImageNet-100 dataset loader
   - Preprocessing pipelines (both frameworks)
   - Batch preparation utilities

### Tasks to Complete

- [x] Implement JAX/Flax ResNet-50 ✅
- [x] Implement JAX/Flax ViT-Base ✅
- [x] Implement PyTorch ResNet-50 (or use torchvision) ✅
- [x] Implement PyTorch ViT-Base (or use torchvision) ✅
- [ ] Create numerical validation framework
- [ ] Validate ResNet-50 outputs match (JAX vs PyTorch)
- [ ] Validate ViT-Base outputs match (JAX vs PyTorch)
- [x] Build timing utilities with statistics ✅
- [x] Build memory profiling utilities ✅
- [x] Build CSV logging system ✅
- [ ] Create unified benchmark runner
- [x] Implement data loading (synthetic + ImageNet-100) ✅
- [ ] Run initial baseline benchmarks
- [ ] Document model implementations

### Success Criteria

- ✅ All models implemented and validated
- ✅ Numerical validation passes (1e-5 tolerance)
- ✅ Benchmarking infrastructure complete
- ✅ CSV logging working for all configurations
- ✅ Initial baseline measurements recorded
- ✅ Code is modular and well-documented

### Team Responsibilities

- **Member 1**: JAX/Flax ResNet-50 and ViT-Base implementations
- **Member 2**: PyTorch model implementations and validation
- **Member 3**: Benchmarking infrastructure (timing, memory, logging)
- **Member 4**: Data loading, benchmark runner, initial baselines

### Dependencies

- Requires Phase 1 device detection utilities
- Requires access to ImageNet-100 dataset (or synthetic data)
- May require pretrained weights for validation

### Notes

- Focus on correctness first, optimization later
- Document any deviations from reference implementations
- Keep code modular for easy extension
- Begin testing on local systems before moving to H100
- **Status: 🚧 IN PROGRESS** - Currently being worked on
- **Primary Contributor**: Shashwat S. (ssinha30@uic.edu)

### Progress Update

**Completed Components:**
- ✅ Timing utilities (`utils/timing.py`) - LatencyStats, synchronization, throughput calculation
- ✅ Memory profiling (`utils/memory.py`) - MemoryTracker, peak memory measurement
- ✅ CSV logging (`utils/logging.py`) - BenchmarkLogger with full schema
- ✅ Data loading (`utils/data.py`) - Synthetic data generation and ImageNet-100 loader
- ✅ PyTorch models (`models/torch_zoo.py`) - ResNet-50 and ViT-Base from torchvision
- ✅ JAX/Flax models (`models/jax_flax_zoo.py`) - Native ResNet-50 and ViT-Base implementations

**Remaining Tasks:**
- ⏳ Numerical validation framework (`utils/validation.py`)
- ⏳ Unified benchmark runner (`bench/runner.py`)
- ⏳ Integration testing and baseline measurements

---

## Phase 3: Data Collection

### Objectives

- Conduct comprehensive benchmarking across all configurations
- Collect statistically significant data (50+ iterations per config)
- Measure all required metrics (latency, throughput, memory, compilation time, energy)
- Apply framework-specific optimizations
- Complete profiling analysis

### Key Deliverables

1. **Benchmark Results** (`results/raw/`)
   - CSV files with all benchmark data
   - Multiple batch sizes: [1, 8, 32, 128]
   - Multiple precision levels: FP32, FP16/BF16 (if applicable)
   - All 8 primary configurations:
     - ResNet-50: JAX (CPU, GPU), PyTorch (CPU, GPU)
     - ViT-Base: JAX (CPU, GPU), PyTorch (CPU, GPU)
   - Optional: TPU configurations (if access granted)

2. **Metrics Collected**

   **Latency:**
   - Single sample inference time
   - Percentiles: p50, p95, p99
   - Mean and standard deviation
   - Per-batch-size analysis

   **Throughput:**
   - Images per second
   - Batch processing rate
   - Scaling analysis across batch sizes

   **Memory:**
   - Peak memory usage
   - Memory efficiency (throughput per GB)
   - Memory scaling with batch size

   **Compilation Time:**
   - JAX XLA compilation time (first run)
   - PyTorch JIT compilation time (if used)
   - Impact on development iteration speed

   **Energy (if measurable):**
   - GPU power consumption (nvidia-smi)
   - Energy per inference
   - Energy efficiency comparisons

3. **Profiling Data**
   - XLA profiler outputs (JAX)
   - TensorBoard logs (PyTorch)
   - nvprof traces (GPU)
   - CPU profiling data (perf)

4. **Optimization Experiments**
   - Framework-specific optimizations tested
   - Performance improvements documented
   - Trade-offs analyzed

### Tasks to Complete

- [ ] Run benchmarks for all 8 primary configurations
- [ ] Collect 50+ iterations per configuration
- [ ] Test multiple batch sizes [1, 8, 32, 128]
- [ ] Test multiple precision levels (FP32, FP16/BF16)
- [ ] Measure compilation times
- [ ] Measure memory usage
- [ ] Measure energy consumption (where possible)
- [ ] Run profiling tools (XLA profiler, TensorBoard, nvprof)
- [ ] Apply framework optimizations
- [ ] Document optimization results
- [ ] Validate data quality and consistency
- [ ] Backup all results

### Success Criteria

- ✅ All 8 primary configurations benchmarked
- ✅ Minimum 50 iterations per configuration
- ✅ All metrics collected and logged
- ✅ Profiling data available for analysis
- ✅ Results stored in structured CSV format
- ✅ Data quality validated (no outliers, consistent measurements)

### Team Responsibilities

- **Member 1**: JAX benchmarks (CPU, GPU, TPU if available)
- **Member 2**: PyTorch benchmarks (CPU, GPU)
- **Member 3**: Profiling and optimization experiments
- **Member 4**: Data collection coordination, result validation

### Dependencies

- Requires Phase 2 model implementations (must be complete)
- Requires Phase 2 benchmarking infrastructure (must be complete)
- Requires H100 GPU access for production benchmarks
- Requires ImageNet-100 dataset (or synthetic equivalent)

### Notes

- Coordinate GPU access among team members
- Run benchmarks during off-peak hours if possible
- Keep detailed logs of any issues or anomalies
- Backup results frequently
- Document any system-specific observations
- **Status: ⏳ PENDING** - Waiting for Phase 2 completion
- **Team Responsibility**: All team members will participate in data collection

---

## Phase 4: Analysis & Documentation

### Objectives

- Analyze benchmark results comprehensively
- Understand performance differences between JAX and PyTorch
- Investigate architectural interactions with hardware
- Create visualizations and comparison matrices
- Prepare detailed report with methodology and results

### Key Deliverables

1. **Performance Analysis** (`docs/analysis.md`)
   - Comparative analysis of JAX vs PyTorch
   - Performance breakdown by model, framework, hardware
   - Bottleneck identification (compute vs memory)
   - Scaling analysis across batch sizes
   - Architectural insights (CNN vs Transformer)

2. **Visualizations** (`results/figs/`)
   - Latency comparison plots (by batch size, framework, hardware)
   - Throughput scaling curves
   - Memory usage comparisons
   - Performance heatmaps
   - Roofline plots (if applicable)
   - Compilation time analysis

3. **Plotting Scripts** (`scripts/plot_results.py`)
   - Automated plot generation from CSV data
   - Multiple visualization types
   - Publication-quality figures
   - Configurable styling

4. **Performance Matrices**
   - Comparison tables (JAX vs PyTorch)
   - Speedup factors
   - Efficiency metrics
   - Hardware utilization analysis

5. **Written Report** (`docs/report.md` or separate document)
   - Executive summary
   - Methodology section
   - Results and analysis
   - Discussion of findings
   - Conclusions and recommendations
   - References

6. **Code Documentation**
   - API documentation
   - Usage examples
   - Reproducibility guide

### Tasks to Complete

- [ ] Load and process all benchmark CSV files
- [ ] Generate performance comparison matrices
- [ ] Create latency vs batch size plots
- [ ] Create throughput scaling plots
- [ ] Create memory usage visualizations
- [ ] Analyze compilation time impact
- [ ] Identify performance bottlenecks
- [ ] Compare CNN vs Transformer patterns
- [ ] Analyze JAX XLA compilation benefits
- [ ] Write methodology section
- [ ] Write results section
- [ ] Write discussion and conclusions
- [ ] Create presentation slides (if needed)
- [ ] Finalize code documentation

### Success Criteria

- ✅ All results analyzed and visualized
- ✅ Clear performance comparisons available
- ✅ Bottlenecks identified and explained
- ✅ Report sections complete
- ✅ Visualizations are clear and publication-quality
- ✅ Findings are well-documented

### Team Responsibilities

- **Member 1**: JAX performance analysis, XLA compilation analysis
- **Member 2**: PyTorch performance analysis, cuDNN optimization analysis
- **Member 3**: Visualization creation, plotting scripts
- **Member 4**: Report writing, methodology documentation, overall analysis

### Dependencies

- Requires Phase 3 benchmark results (must be complete)
- Requires profiling data from Phase 3
- May require additional analysis iterations

### Notes

- Focus on answering the core research question: "How does JAX compare to PyTorch?"
- Investigate "why" performance differences occur, not just "what"
- Document all assumptions and methodology choices
- Prepare for presentation delivery
- **Status: ⏳ PENDING** - Waiting for Phase 3 completion
- **Team Responsibility**: All team members will participate in analysis and documentation

---

## Phase 5: Finalization

### Objectives

- Finalize all documentation
- Refine codebase for reproducibility
- Prepare presentation materials
- Integrate feedback
- Complete repository documentation

### Key Deliverables

1. **Finalized Codebase**
   - Code cleanup and optimization
   - Comprehensive documentation
   - Reproducibility guide
   - Example usage scripts

2. **Complete Documentation**
   - Final report
   - API documentation
   - Setup guides (all updated)
   - Troubleshooting guides

3. **Presentation Materials**
   - Slides for project presentation
   - Key findings summary
   - Visualizations for presentation
   - Demo scripts (if applicable)

4. **Repository Finalization**
   - README with complete project overview
   - License file (if applicable)
   - Contribution guidelines
   - Citation information

5. **Reproducibility Package**
   - All code with version tags
   - Requirements with versions
   - Dataset preparation instructions
   - Benchmark execution guide

### Tasks to Complete

- [ ] Review and finalize all code
- [ ] Complete all documentation
- [ ] Create presentation slides
- [ ] Prepare demo (if required)
- [ ] Test full reproducibility
- [ ] Integrate any feedback
- [ ] Final code review
- [ ] Repository cleanup
- [ ] Create final release tag
- [ ] Deliver presentation

### Success Criteria

- ✅ All code is clean and well-documented
- ✅ Full reproducibility demonstrated
- ✅ Presentation is ready
- ✅ Repository is complete and professional
- ✅ All deliverables submitted on time

### Team Responsibilities

- **All Members**: Code review, documentation review
- **Member 1**: Code finalization, reproducibility testing
- **Member 2**: Presentation preparation
- **Member 3**: Repository organization, final documentation
- **Member 4**: Report finalization, presentation delivery

### Dependencies

- Requires completion of all previous phases
- May require feedback integration

### Notes

- Focus on polish and completeness
- Ensure reproducibility for future researchers
- Prepare for questions during presentation
- Document any limitations or future work
- **Status: ⏳ PENDING** - Waiting for Phase 4 completion
- **Team Responsibility**: All team members will participate in finalization

---

## Cross-Phase Considerations

### Reproducibility

- All code should be version controlled
- Use random seeds for deterministic results
- Document all system configurations
- Save environment specifications (requirements.txt with versions)

### Collaboration

- Use Git branches for feature development
- Regular code reviews
- Clear communication of blockers
- Shared documentation of decisions

### Risk Mitigation

- **Hardware Access Issues**: Cloud GPU fallback (Colab Pro, AWS, Azure)
- **TPU Access Delayed**: Continue with CPU/GPU benchmarks (core project complete)
- **Dataset Issues**: Synthetic data fallback, CIFAR-100 alternative
- **Timeline Delays**: Prioritize core configurations, optional extensions can be dropped

### Quality Assurance

- Code reviews at each phase
- Testing on multiple systems
- Validation of numerical correctness
- Statistical significance of results

---

## Success Metrics

### Technical Metrics

- ✅ All models implemented and validated
- ✅ All 8 primary configurations benchmarked
- ✅ Minimum 50 iterations per configuration
- ✅ All metrics collected (latency, throughput, memory, compilation, energy)
- ✅ Results are statistically significant
- ✅ Code is reproducible

### Research Metrics

- ✅ Clear comparison of JAX vs PyTorch performance
- ✅ Understanding of performance differences (not just measurements)
- ✅ Insights into hardware-software co-design
- ✅ Actionable recommendations for practitioners

### Collaboration Metrics

- ✅ All team members can run benchmarks
- ✅ Code is well-documented and maintainable
- ✅ Repository is professional and complete
- ✅ Presentation is clear and compelling

---

## Phase Status Summary

| Phase | Status | Primary Contributor | Notes |
|-------|--------|---------------------|-------|
| Phase 1: Setup | ✅ Complete | Shashwat S. | All deliverables finished |
| Phase 2: Implementation | 🚧 In Progress | Shashwat S. | Currently being worked on |
| Phase 3: Data Collection | ⏳ Pending | Team (all members) | Waiting for Phase 2 |
| Phase 4: Analysis | ⏳ Pending | Team (all members) | Waiting for Phase 3 |
| Phase 5: Finalization | ⏳ Pending | Team (all members) | Waiting for Phase 4 |

**Note**: Phases are sequential - each phase must be completed before the next begins.

---

## Next Steps

1. **Current**: Complete Phase 2 - Model implementations and benchmarking infrastructure
2. **After Phase 2**: Team will begin Phase 3 - Comprehensive data collection
3. **After Phase 3**: Team will begin Phase 4 - Analysis and documentation
4. **After Phase 4**: Team will begin Phase 5 - Finalization and presentation

For detailed implementation guidance, see:
- [Setup Guide](setup.md) - Environment setup
- [TPU Setup Guide](tpu_setup.md) - TPU access
- Individual phase documentation (this file)

