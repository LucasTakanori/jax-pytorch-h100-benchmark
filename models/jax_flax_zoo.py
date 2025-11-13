"""
JAX/Flax model implementations.

Native implementations of ResNet-50 and Vision Transformer Base matching PyTorch architectures.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable, Optional, Dict, Any
from functools import partial


# Model metadata (matching PyTorch)
MODEL_METADATA = {
    'resnet50': {
        'name': 'ResNet-50',
        'params': 25_557_032,  # ~25M parameters (will be computed)
        'input_shape': (224, 224, 3),  # HWC format for JAX
        'flops': 4.1e9,  # ~4.1 GFLOPs
    },
    'vit_b_16': {
        'name': 'Vision Transformer Base',
        'params': 86_567_656,  # ~86M parameters (will be computed)
        'input_shape': (224, 224, 3),  # HWC format for JAX
        'flops': 17.6e9,  # ~17.6 GFLOPs
    }
}


class ResNetBlock(nn.Module):
    """ResNet bottleneck block."""
    filters: int
    stride: int = 1
    use_projection: bool = False
    
    @nn.compact
    def __call__(self, x):
        residual = x
        
        # Main path
        x = nn.Conv(self.filters, (1, 1), strides=self.stride)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        
        x = nn.Conv(self.filters, (3, 3))(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        
        x = nn.Conv(self.filters * 4, (1, 1))(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        
        # Projection shortcut if needed
        if self.use_projection:
            residual = nn.Conv(self.filters * 4, (1, 1), strides=self.stride)(residual)
            residual = nn.BatchNorm(use_running_average=True)(residual)
        
        x = x + residual
        x = nn.relu(x)
        return x


class ResNet50(nn.Module):
    """ResNet-50 implementation in Flax."""
    
    num_classes: int = 1000
    
    @nn.compact
    def __call__(self, x, train: bool = False):
        # Initial conv layer
        x = nn.Conv(64, (7, 7), strides=2, padding=3)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        
        # ResNet blocks
        # Layer 1: 3 blocks, 256 filters
        for _ in range(3):
            x = ResNetBlock(64, stride=1 if _ > 0 else 1, use_projection=_ == 0)(x)
        
        # Layer 2: 4 blocks, 512 filters
        for i in range(4):
            x = ResNetBlock(128, stride=2 if i == 0 else 1, use_projection=i == 0)(x)
        
        # Layer 3: 6 blocks, 1024 filters
        for i in range(6):
            x = ResNetBlock(256, stride=2 if i == 0 else 1, use_projection=i == 0)(x)
        
        # Layer 4: 3 blocks, 2048 filters
        for i in range(3):
            x = ResNetBlock(512, stride=2 if i == 0 else 1, use_projection=i == 0)(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # Global average pool
        
        # Final classifier
        x = nn.Dense(self.num_classes)(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer Base implementation in Flax."""
    
    num_classes: int = 1000
    patch_size: int = 16
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_dim: int = 3072
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, train: bool = False):
        # Patch embedding
        B, H, W, C = x.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        
        # Linear projection of patches
        x = nn.Conv(self.hidden_dim, (self.patch_size, self.patch_size), 
                   strides=self.patch_size, padding='VALID')(x)
        x = x.reshape(B, num_patches, self.hidden_dim)
        
        # Add class token
        cls_token = self.param('cls_token', nn.initializers.normal(stddev=0.02),
                              (1, 1, self.hidden_dim))
        cls_tokens = jnp.tile(cls_token, (B, 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        
        # Add positional embeddings
        pos_embed = self.param('pos_embed', nn.initializers.normal(stddev=0.02),
                              (1, num_patches + 1, self.hidden_dim))
        x = x + pos_embed
        
        # Transformer blocks
        for _ in range(self.num_layers):
            # Self-attention
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not train
            )(x, x)
            x = x + residual
            
            # MLP
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.mlp_dim)(x)
            x = nn.gelu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            x = x + residual
        
        # Extract class token
        x = nn.LayerNorm()(x)
        x = x[:, 0]  # Class token
        
        # Classifier
        x = nn.Dense(self.num_classes)(x)
        return x


def get_imagenet_preprocessing_jax():
    """
    Get ImageNet preprocessing function for JAX.
    
    Returns:
        Preprocessing function
    """
    mean = jnp.array([0.485, 0.456, 0.406])
    std = jnp.array([0.229, 0.224, 0.225])
    
    def preprocess_fn(x: jnp.ndarray) -> jnp.ndarray:
        """
        Preprocess input array.
        
        Args:
            x: Input array in [0, 1] range, shape (B, H, W, C) or (H, W, C)
            
        Returns:
            Normalized array
        """
        # Convert from [0, 1] to [0, 255] then normalize
        x = x * 255.0
        
        # Normalize: (x - mean) / std
        if x.ndim == 3:
            mean_expanded = mean.reshape(1, 1, 3)
            std_expanded = std.reshape(1, 1, 3)
        else:  # ndim == 4
            mean_expanded = mean.reshape(1, 1, 1, 3)
            std_expanded = std.reshape(1, 1, 1, 3)
        
        return (x - mean_expanded) / std_expanded
    
    return preprocess_fn


def get_flax_model(
    model_name: str,
    input_shape: Tuple[int, ...],
    rng_key: jax.random.PRNGKey,
    num_classes: int = 1000
) -> Tuple[Callable, Any, Callable, Dict[str, Any]]:
    """
    Get JAX/Flax model with consistent interface.
    
    Args:
        model_name: 'resnet50' or 'vit_b_16'
        input_shape: Input shape tuple in HWC format (H, W, C)
        rng_key: JAX random key for initialization
        num_classes: Number of output classes
        
    Returns:
        Tuple of (apply_fn, params, preprocessing_fn, metadata)
    """
    if model_name not in MODEL_METADATA:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_METADATA.keys())}")
    
    metadata = MODEL_METADATA[model_name].copy()
    
    # Create model
    if model_name == 'resnet50':
        model = ResNet50(num_classes=num_classes)
    elif model_name == 'vit_b_16':
        model = VisionTransformer(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    # Initialize parameters
    dummy_input = jnp.ones((1, *input_shape), dtype=jnp.float32)
    params = model.init(rng_key, dummy_input, train=False)
    
    # Create apply function
    @partial(jax.jit, static_argnames=('train',))
    def apply_fn(params, x, train=False):
        return model.apply(params, x, train=train)
    
    # Preprocessing function
    preprocessing = get_imagenet_preprocessing_jax()
    
    return apply_fn, params, preprocessing, metadata


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get model metadata.
    
    Args:
        model_name: 'resnet50' or 'vit_b_16'
        
    Returns:
        Dictionary with model metadata
    """
    if model_name not in MODEL_METADATA:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_METADATA[model_name].copy()

