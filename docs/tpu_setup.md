# TPU Setup and Access Guide

This document outlines the process for obtaining and setting up Google Cloud TPU access for the 594Project benchmarking study.

## Overview

TPU (Tensor Processing Unit) access is **optional** for this project. The core benchmarking (CPU and GPU) can be completed without TPUs. However, if time and resources permit, TPU benchmarking provides valuable insights into JAX performance on specialized hardware.

## TPU Access Application Process

### 1. Google Cloud TPU Access

Google Cloud provides TPU access through several programs:

#### Option A: Google Cloud Free Tier / Credits
- **Students**: Apply for Google Cloud Education Credits
  - Link: https://edu.google.com/programs/credits/
  - Provides $300+ in credits for students
  - Can be used for TPU access

#### Option B: Google Cloud TPU Research Credits
- **Research Projects**: Apply for TPU Research Credits
  - Link: https://sites.research.google/trc/
  - Specifically for academic research
  - Requires project description and research goals
  - **Recommended for this project**

#### Option C: Google Colab Pro / Pro+
- **Limited TPU Access**: Colab Pro+ provides limited TPU access
  - Not suitable for comprehensive benchmarking
  - Good for initial testing only

### 2. Application Timeline

**Recommended Timeline:**
- **Week 1-2**: Submit TPU Research Credits application
- **Week 3-4**: Wait for approval (typically 1-2 weeks)
- **Week 4-5**: If approved, conduct TPU benchmarks
- **If not approved**: Continue with CPU/GPU benchmarks (core project remains complete)

### 3. Application Details

When applying for TPU Research Credits, include:

**Project Title:**
"Cross-Platform Performance Analysis of Modern Neural Networks: A JAX-Centric Hardware-Software Co-Design Study"

**Research Goals:**
- Compare JAX and PyTorch performance across CPU, GPU, and TPU platforms
- Analyze ResNet-50 and Vision Transformer architectures
- Understand hardware-software co-design implications

**Resource Requirements:**
- TPU v3 or v4 (preferred)
- Estimated usage: 10-20 TPU hours for benchmarking
- Timeline: 4-6 weeks

**Team Information:**
- University: University of Illinois at Chicago
- Course: ECE 594 HW-SW Co-Design for ML Systems
- Team members: [List team member emails]

## TPU Setup Instructions

### Prerequisites

1. **Google Cloud Account**: Create account at https://cloud.google.com/
2. **TPU Access**: Approved TPU Research Credits or Cloud credits
3. **TPU VM**: Create a TPU VM instance (not a Compute Engine VM with TPU attached)

### Step 1: Create TPU VM

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Create TPU VM (example for TPU v3)
gcloud alpha compute tpus tpu-vm create tpu-vm-1 \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-tf-2.13.0
```

### Step 2: Install JAX on TPU VM

```bash
# SSH into TPU VM
gcloud alpha compute tpus tpu-vm ssh tpu-vm-1 --zone=us-central1-a

# Install JAX for TPU
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/jax_tpu_releases.html

# Install other dependencies
pip install flax numpy pandas matplotlib
```

### Step 3: Verify TPU Access

```python
import jax
print(jax.devices())  # Should show TPU devices
```

### Step 4: Test TPU Connection

Run the verification script:
```bash
python scripts/verify_setup.py
```

## Cost Considerations

- **TPU v3-8**: ~$8/hour
- **TPU v4-8**: ~$10-12/hour
- **Estimated total cost**: $80-240 for full benchmarking (10-20 hours)

**Cost Mitigation:**
- Use TPU Research Credits (free if approved)
- Limit benchmarking to essential configurations
- Use preemptible TPUs if available (lower cost)
- Monitor usage closely

## Fallback Plan

If TPU access is **not granted** or **delayed**:

1. **Core project remains complete**: CPU and GPU benchmarks provide sufficient data
2. **TPU becomes optional extension**: Can be added later if access is granted
3. **Focus on JAX vs PyTorch on GPU**: This is the primary research question
4. **Document TPU limitations**: Note in report that TPU evaluation was attempted but not available

## Testing TPU Connectivity

Once TPU is set up, test with a simple script:

```python
import jax
import jax.numpy as jnp

# Check devices
print("TPU Devices:", jax.devices())

# Simple computation test
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
print("Result shape:", y.shape)
print("✅ TPU working!")
```

## Troubleshooting

### Common Issues

1. **"No TPU devices found"**
   - Verify TPU VM is running: `gcloud alpha compute tpus tpu-vm list`
   - Check JAX installation: `pip list | grep jax`
   - Ensure using TPU VM, not regular Compute Engine VM

2. **Connection timeouts**
   - Check firewall rules
   - Verify network connectivity
   - Check TPU VM status

3. **Out of memory errors**
   - Reduce batch size
   - Use smaller models for initial testing
   - Check TPU type (v3-8 vs v4-8 have different memory)

## Resources

- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [JAX on TPU Guide](https://jax.readthedocs.io/en/latest/faq.html#running-jax-on-tpus)
- [TPU Research Credits](https://sites.research.google/trc/)
- [TPU Pricing](https://cloud.google.com/tpu/pricing)

## Team Notes

**Action Items:**
- [ ] Team member 1: Submit TPU Research Credits application (Week 1-2)
- [ ] Team member 2: Set up Google Cloud account and billing
- [ ] Team member 3: Prepare TPU benchmarking scripts (can be done in parallel)
- [ ] Team member 4: Document TPU setup process (this document)

**Status Tracking:**
- Application submitted: [Date]
- Application approved: [Date]
- TPU VM created: [Date]
- First successful benchmark: [Date]

