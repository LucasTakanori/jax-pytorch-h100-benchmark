# Model Implementations Documentation

This document describes the model implementations in the 594Project benchmarking framework.

## Overview

The project implements two neural network architectures in both PyTorch and JAX/Flax:
- **ResNet-50**: 25M parameters, ~4.1 GFLOPs
- **Vision Transformer Base (ViT-Base)**: 86M parameters, ~17.6 GFLOPs

## PyTorch Models

### Location
`models/torch_zoo.py`

### Usage

```python
from models.torch_zoo import get_torch_model
import torch

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ResNet-50
model, preprocess, input_shape, metadata = get_torch_model(
    'resnet50',
    pretrained=True,  # Use ImageNet pretrained weights
    device=device
)

# Load ViT-Base
model_vit, preprocess_vit, input_shape_vit, metadata_vit = get_torch_model(
    'vit_b_16',
    pretrained=True,
    device=device
)

# Run inference
x = torch.randn(1, *input_shape, device=device)
x = preprocess(x)
with torch.no_grad():
    output = model(x)
```

### Implementation Details

- **ResNet-50**: Uses `torchvision.models.resnet50()` with ImageNet pretrained weights
- **ViT-Base**: Uses `torchvision.models.vit_b_16()` with ImageNet pretrained weights
- **Input Format**: NCHW (batch, channels, height, width)
- **Input Shape**: (3, 224, 224)
- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Model Metadata

```python
from models.torch_zoo import get_model_info

info = get_model_info('resnet50')
# Returns: {'name': 'ResNet-50', 'params': 25557032, 'input_shape': (3, 224, 224), 'flops': 4100000000.0}
```

## JAX/Flax Models

### Location
`models/jax_flax_zoo.py`

### Usage

```python
from models.jax_flax_zoo import get_flax_model
import jax
import jax.numpy as jnp

# Initialize model
rng_key = jax.random.PRNGKey(42)
apply_fn, params, preprocess, metadata = get_flax_model(
    'resnet50',
    input_shape=(224, 224, 3),  # HWC format for JAX
    rng_key=rng_key,
    num_classes=1000
)

# JIT compile
@jax.jit
def jitted_apply(params, x):
    return apply_fn(params, x, train=False)

# Run inference
x = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
x = preprocess(x)
output = jitted_apply(params, x)
output.block_until_ready()  # Ensure computation completes
```

### Implementation Details

- **ResNet-50**: Native Flax implementation matching PyTorch architecture
  - Bottleneck blocks with 1x1, 3x3, 1x1 convolutions
  - Batch normalization
  - ReLU activations
  - Global average pooling
  
- **ViT-Base**: Native Flax implementation
  - 12 transformer blocks
  - 768 hidden dimensions
  - 12 attention heads
  - Patch size 16
  - Image size 224x224
  - GELU activations

- **Input Format**: NHWC (batch, height, width, channels)
- **Input Shape**: (224, 224, 3)
- **Preprocessing**: ImageNet normalization (same as PyTorch)

### Architecture Matching

The JAX/Flax implementations are designed to match PyTorch architectures:
- Same layer structure and ordering
- Same kernel sizes and strides
- Same activation functions (ReLU for ResNet, GELU for ViT)
- Same normalization behavior

**Note**: Currently using random initialization. For exact numerical matching, pretrained weights or weight conversion would be needed.

## Model Comparison

### ResNet-50

| Framework | Parameters | Input Format | Implementation |
|-----------|-----------|--------------|----------------|
| PyTorch   | 25,557,032 | NCHW         | torchvision   |
| JAX/Flax  | ~25,636,712 | NHWC        | Native Flax   |

**Parameter Difference**: ~0.31% (acceptable, due to minor implementation differences)

### ViT-Base

| Framework | Parameters | Input Format | Implementation |
|-----------|-----------|--------------|----------------|
| PyTorch   | 86,567,656 | NCHW         | torchvision   |
| JAX/Flax  | ~86M (estimated) | NHWC        | Native Flax   |

## Validation

Use the validation framework to compare models:

```python
from utils.validation import compare_models
from models.torch_zoo import get_torch_model
from models.jax_flax_zoo import get_flax_model

# Load both models
pytorch_model, pytorch_preprocess, input_shape_pytorch, _ = get_torch_model('resnet50', pretrained=False)
jax_apply_fn, jax_params, jax_preprocess, _ = get_flax_model('resnet50', input_shape=(224, 224, 3), rng_key=jax.random.PRNGKey(42))

# Compare
result = compare_models(
    model_name='resnet50',
    pytorch_model=pytorch_model,
    pytorch_preprocess=pytorch_preprocess,
    jax_apply_fn=jax_apply_fn,
    jax_params=jax_params,
    jax_preprocess=jax_preprocess,
    input_shape_pytorch=input_shape_pytorch,
    input_shape_jax=(224, 224, 3),
    batch_size=2,
    tolerance=1e-5
)
```

## Benchmarking

Use the unified benchmark runner:

```python
from bench.runner import BenchmarkConfig, run_inference_benchmark

config = BenchmarkConfig(
    framework='pytorch',
    model_name='resnet50',
    batch_sizes=[1, 8, 32, 128],
    warmup_iterations=10,
    measurement_iterations=50
)

results = run_inference_benchmark(config)
```

Or use the command-line interface:

```bash
# Run single model
python bench/runner.py --framework pytorch --model resnet50

# Run full suite
python bench/runner.py --framework both --model both

# Custom configuration
python bench/runner.py --framework jax --model vit_b_16 --batch-sizes 1 16 32 --iterations 100
```

## Notes

1. **Weight Initialization**: Models use random weights by default. For pretrained PyTorch models, set `pretrained=True`. JAX models currently don't have pretrained weights loaded.

2. **Input Format**: PyTorch uses NCHW, JAX uses NHWC. The data loading utilities handle this conversion automatically.

3. **Device Placement**: PyTorch models are moved to device explicitly. JAX automatically uses available devices (CPU, GPU, TPU).

4. **JIT Compilation**: JAX models should be JIT compiled for best performance. The benchmark runner handles this automatically.

5. **Numerical Validation**: For exact output matching, both models need identical weights. This requires either:
   - Pretrained weights for both frameworks
   - Weight conversion from PyTorch to JAX
   - Identical random initialization (same seed)

## Future Improvements

- [ ] Add pretrained weight loading for JAX models
- [ ] Implement weight conversion from PyTorch to JAX
- [ ] Add more model architectures (ResNet-101, ViT-Large, etc.)
- [ ] Support for different input sizes
- [ ] Mixed precision (FP16/BF16) support

