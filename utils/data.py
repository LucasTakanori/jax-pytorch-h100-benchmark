"""
Data loading utilities for synthetic and real datasets.
"""

import numpy as np
from typing import Tuple, Optional, Union, Callable, Any
import os


def get_synthetic_batch(
    batch_size: int,
    input_shape: Tuple[int, ...],
    framework: str = 'pytorch',
    dtype: str = 'float32',
    device: Optional[Any] = None,
    seed: Optional[int] = None
) -> Any:
    """
    Generate synthetic batch matching ImageNet format.
    
    Args:
        batch_size: Number of samples in batch
        input_shape: Input shape tuple. For PyTorch: (C, H, W), for JAX: (H, W, C)
        framework: 'pytorch' or 'jax'
        dtype: Data type ('float32' or 'float16')
        device: Device object (for PyTorch)
        seed: Random seed for reproducibility
        
    Returns:
        Synthetic batch tensor/array
    """
    if seed is not None:
        np.random.seed(seed)
    
    if framework == 'pytorch':
        import torch
        
        # PyTorch uses NCHW format: (batch, channels, height, width)
        if len(input_shape) == 3:
            # input_shape is (C, H, W)
            C, H, W = input_shape
            shape = (batch_size, C, H, W)
        else:
            raise ValueError(f"Invalid input_shape for PyTorch: {input_shape}")
        
        # Generate random data in [0, 1] range (will be normalized later)
        data = torch.rand(shape, dtype=get_torch_dtype(dtype))
        
        if device is not None:
            data = data.to(device)
        
        return data
    
    elif framework == 'jax':
        import jax
        import jax.numpy as jnp
        
        # JAX uses NHWC format: (batch, height, width, channels)
        if len(input_shape) == 3:
            # input_shape is (H, W, C) for JAX
            H, W, C = input_shape
            shape = (batch_size, H, W, C)
        else:
            raise ValueError(f"Invalid input_shape for JAX: {input_shape}")
        
        # Generate random data in [0, 1] range
        key = jax.random.PRNGKey(seed if seed is not None else 0)
        data = jax.random.uniform(key, shape, dtype=get_jax_dtype(dtype))
        
        return data
    
    else:
        raise ValueError(f"Unknown framework: {framework}")


def get_torch_dtype(dtype_str: str):
    """Convert dtype string to PyTorch dtype."""
    import torch
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def get_jax_dtype(dtype_str: str):
    """Convert dtype string to JAX dtype."""
    import jax.numpy as jnp
    dtype_map = {
        'float32': jnp.float32,
        'float16': jnp.float16,
        'bfloat16': jnp.bfloat16,
    }
    return dtype_map.get(dtype_str, jnp.float32)


def get_preprocessing(framework: str) -> Callable:
    """
    Get preprocessing function for ImageNet normalization.
    
    Args:
        framework: 'pytorch' or 'jax'
        
    Returns:
        Preprocessing function
    """
    if framework == 'pytorch':
        import torch
        from torchvision import transforms
        
        # ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        def preprocess(x):
            # x is expected to be in [0, 1] range
            # Convert to [0, 255] then normalize
            x = x * 255.0
            return normalize(x)
        
        return preprocess
    
    elif framework == 'jax':
        import jax.numpy as jnp
        
        # ImageNet mean and std
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        
        def preprocess(x):
            # x is expected to be in [0, 1] range, shape (H, W, C) or (B, H, W, C)
            # Convert to [0, 255] then normalize
            x = x * 255.0
            # Normalize: (x - mean) / std
            # Expand dims for broadcasting if needed
            if x.ndim == 3:
                mean_expanded = mean.reshape(1, 1, 3)
                std_expanded = std.reshape(1, 1, 3)
            else:  # ndim == 4
                mean_expanded = mean.reshape(1, 1, 1, 3)
                std_expanded = std.reshape(1, 1, 1, 3)
            
            return (x - mean_expanded) / std_expanded
        
        return preprocess
    
    else:
        raise ValueError(f"Unknown framework: {framework}")


def load_imagenet100(
    root: str,
    framework: str = 'pytorch',
    split: str = 'val',
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0
) -> Any:
    """
    Load ImageNet-100 dataset.
    
    Args:
        root: Root directory containing ImageNet-100 dataset
        framework: 'pytorch' or 'jax'
        split: 'train' or 'val'
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (PyTorch only)
        
    Returns:
        DataLoader or iterator
    """
    if framework == 'pytorch':
        from torchvision import datasets, transforms
        import torch
        
        # ImageNet preprocessing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # Converts to [0, 1] and CHW format
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        dataset = datasets.ImageFolder(
            root=os.path.join(root, split),
            transform=transform
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return dataloader
    
    elif framework == 'jax':
        # For JAX, we'll convert PyTorch dataset to JAX arrays
        # This is a simplified version - in practice, you might use a different approach
        from torchvision import datasets, transforms
        import torch
        import jax.numpy as jnp
        
        # Load using PyTorch first
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        dataset = datasets.ImageFolder(
            root=os.path.join(root, split),
            transform=transform
        )
        
        # Convert to JAX format
        def jax_iterator():
            for i in range(len(dataset)):
                img, label = dataset[i]
                # Convert to numpy then JAX, change CHW to HWC
                img_np = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
                img_jax = jnp.array(img_np)
                
                # Apply normalization
                mean = jnp.array([0.485, 0.456, 0.406])
                std = jnp.array([0.229, 0.224, 0.225])
                img_jax = (img_jax - mean) / std
                
                yield img_jax, label
        
        return jax_iterator()
    
    else:
        raise ValueError(f"Unknown framework: {framework}")


def create_dataloader(
    data_source: str,
    batch_size: int,
    input_shape: Tuple[int, ...],
    framework: str = 'pytorch',
    device: Optional[Any] = None,
    **kwargs
) -> Any:
    """
    Create a data loader/iterator for benchmarking.
    
    Args:
        data_source: 'synthetic' or path to ImageNet-100 dataset
        batch_size: Batch size
        input_shape: Input shape tuple (framework-specific format)
        framework: 'pytorch' or 'jax'
        device: Device object (for PyTorch)
        **kwargs: Additional arguments (seed for synthetic, etc.)
        
    Returns:
        DataLoader, iterator, or generator
    """
    if data_source == 'synthetic':
        # Create a generator that yields synthetic batches
        def synthetic_generator():
            seed = kwargs.get('seed', None)
            while True:
                yield get_synthetic_batch(
                    batch_size=batch_size,
                    input_shape=input_shape,
                    framework=framework,
                    dtype=kwargs.get('dtype', 'float32'),
                    device=device,
                    seed=seed
                )
        
        return synthetic_generator()
    
    else:
        # Assume it's a path to ImageNet-100
        return load_imagenet100(
            root=data_source,
            framework=framework,
            batch_size=batch_size,
            **kwargs
        )

