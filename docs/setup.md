# Setup Guide for 594Project

This guide will help you set up the development environment for the 594Project benchmarking framework. The setup works on both **local machines** (macOS with Apple Silicon) and **remote systems** (Linux with H100 GPUs).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Setup (macOS)](#local-setup-macos)
3. [Remote Setup (H100 GPU System)](#remote-setup-h100-gpu-system)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Team Collaboration](#team-collaboration)

## Prerequisites

- Python 3.10 or higher (3.13 recommended)
- Git
- pip (Python package manager)
- For GPU support: CUDA-capable GPU with appropriate drivers (for H100 systems)

## Local Setup (macOS)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd 594Project
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements (CPU-only JAX, PyTorch with MPS support)
pip install -r requirements.txt
```

**Note**: On macOS, PyTorch will automatically use MPS (Metal Performance Shaders) for Apple Silicon GPUs. JAX will use CPU backend.

### Step 4: Verify Installation

```bash
# Run verification script
python scripts/verify_setup.py
```

You should see:
- ✅ PyTorch installed with MPS available
- ✅ JAX installed with CPU backend
- ✅ All dependencies installed

## Remote Setup (H100 GPU System)

### Step 1: SSH into Remote System

```bash
ssh username@h100-system.example.com
```

### Step 2: Clone Repository

```bash
git clone <repository-url>
cd 594Project
```

### Step 3: Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate
```

### Step 4: Install PyTorch with CUDA

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.x support (for H100)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Install JAX with CUDA

```bash
# Install JAX with CUDA 12 support
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Step 6: Install Other Dependencies

```bash
# Install remaining dependencies
pip install numpy pandas matplotlib psutil flax
```

### Step 7: Optional - GPU Monitoring Tools

For energy measurement on NVIDIA GPUs:

```bash
pip install nvidia-ml-py3
```

### Step 8: Verify Installation

```bash
# Run verification script
python scripts/verify_setup.py
```

You should see:
- ✅ PyTorch installed with CUDA available
- ✅ JAX installed with GPU backend
- ✅ H100 GPU detected
- ✅ All dependencies installed

## Verification

### Quick Test

Run the verification script:

```bash
python scripts/verify_setup.py
```

This will check:
- All required packages are installed
- PyTorch backends (CPU, CUDA, MPS)
- JAX backends (CPU, CUDA, TPU)
- Device detection utilities
- Basic functionality tests

### Test Benchmarks

Run the existing benchmark scripts:

```bash
# Test PyTorch benchmark
python bench/bench_infer_torch.py

# Test JAX benchmark
python bench/bench_infer_jax.py
```

Both should run successfully and print latency measurements.

## Troubleshooting

### Issue: PyTorch CUDA not available on H100 system

**Solution:**
1. Check CUDA installation: `nvidia-smi`
2. Verify CUDA version matches PyTorch: `python -c "import torch; print(torch.version.cuda)"`
3. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Issue: JAX CUDA not working

**Solution:**
1. Check CUDA installation: `nvcc --version`
2. Verify JAX CUDA installation:
   ```python
   import jax
   print(jax.devices())  # Should show GPU devices
   ```
3. Reinstall JAX CUDA:
   ```bash
   pip uninstall jax jaxlib
   pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

### Issue: MPS not available on macOS

**Solution:**
- MPS requires Apple Silicon (M1/M2/M3). On Intel Macs, PyTorch will use CPU.
- This is expected behavior and benchmarks will still work.

### Issue: Import errors for device utilities

**Solution:**
1. Ensure you're in the project root directory
2. Check that `utils/device.py` exists
3. Try: `python -c "from utils.device import get_torch_device; print('OK')"`

### Issue: Permission denied on remote system

**Solution:**
- Use virtual environment (recommended)
- Or install with `--user` flag: `pip install --user -r requirements.txt`

## Team Collaboration

### Git Workflow

1. **Create a branch for your work:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and commit:**
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. **Push and create pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Environment Consistency

To ensure all team members have the same environment:

1. **Always use virtual environments** (venv)
2. **Keep requirements.txt updated** when adding dependencies
3. **Run verification script** before committing major changes
4. **Document system-specific setup** in this file

### Sharing Setup Issues

If you encounter setup issues:

1. Check this troubleshooting section
2. Check `docs/tpu_setup.md` for TPU-specific issues
3. Document new issues and solutions here
4. Notify team via GitHub Issues or team chat

## Next Steps

After setup is complete:

1. ✅ Run `scripts/verify_setup.py` - should pass all checks
2. ✅ Test `bench/bench_infer_torch.py` - should run successfully
3. ✅ Test `bench/bench_infer_jax.py` - should run successfully
4. 📖 Read the project proposal for understanding the goals
5. 🚀 Start implementing Phase 2 (Model implementations)

## System-Specific Notes

### macOS (Local Development)
- Uses MPS for GPU acceleration (Apple Silicon only)
- JAX runs on CPU (no CUDA on macOS)
- Good for development and testing
- Not suitable for final H100 benchmarks

### Linux H100 System (Production Benchmarks)
- Uses CUDA for GPU acceleration
- Both PyTorch and JAX can use H100 GPU
- Required for final benchmarking results
- Team members should coordinate GPU access

### TPU Access (Optional)
- See `docs/tpu_setup.md` for TPU setup
- Requires Google Cloud TPU access
- Optional extension to core project

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html)
- [Project README](../README.md)
- [TPU Setup Guide](tpu_setup.md)

## Support

For setup issues:
1. Check troubleshooting section above
2. Review error messages carefully
3. Check GitHub Issues for similar problems
4. Contact team members for help

