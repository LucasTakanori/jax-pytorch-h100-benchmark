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
        'implemented': True
    },
    'vit_b_16': {
        'name': 'Vision Transformer Base',
        'params': 86_567_656,  # ~86M parameters (will be computed)
        'input_shape': (224, 224, 3),  # HWC format for JAX
        'flops': 17.6e9,  # ~17.6 GFLOPs
        'implemented': True
    },
    'mobilenet_v3_small': {
        'name': 'MobileNetV3-Small',
        'params': 2_537_682,  # ~2.5M parameters
        'input_shape': (224, 224, 3),  # HWC format for JAX
        'flops': 0.056e9,  # ~56 MFLOPs
        'implemented': True
    },
    'efficientnet_b0': {
        'name': 'EfficientNet-B0',
        'params': 5_288_548,  # ~5.3M parameters
        'input_shape': (224, 224, 3),  # HWC format for JAX
        'flops': 0.39e9,  # ~390 MFLOPs
        'implemented': True
    }
}


def hard_swish(x):
    """Hard-swish activation function."""
    return x * jax.nn.relu6(x + 3.0) / 6.0


def hard_sigmoid(x):
    """Hard-sigmoid activation function."""
    return jax.nn.relu6(x + 3.0) / 6.0


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block."""
    se_ratio: float = 0.25

    @nn.compact
    def __call__(self, x):
        channels = x.shape[-1]
        squeezed_channels = max(1, int(channels * self.se_ratio))

        # Squeeze (global average pooling)
        se = jnp.mean(x, axis=(1, 2), keepdims=True)

        # Excitation
        se = nn.Conv(squeezed_channels, (1, 1))(se)
        se = nn.relu(se)
        se = nn.Conv(channels, (1, 1))(se)
        se = jax.nn.sigmoid(se)

        return x * se


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


class InvertedResidualBlock(nn.Module):
    """Inverted residual block for MobileNetV3."""
    expansion: int
    out_channels: int
    kernel_size: int
    stride: int
    use_se: bool
    activation: str  # 'relu' or 'hard_swish'

    @nn.compact
    def __call__(self, x):
        in_channels = x.shape[-1]
        residual = x

        # Expansion phase
        if self.expansion != 1:
            x = nn.Conv(in_channels * self.expansion, (1, 1))(x)
            x = nn.BatchNorm(use_running_average=True)(x)
            x = hard_swish(x) if self.activation == 'hard_swish' else nn.relu(x)

        # Depthwise convolution
        x = nn.Conv(
            in_channels * self.expansion,
            (self.kernel_size, self.kernel_size),
            strides=self.stride,
            feature_group_count=in_channels * self.expansion,
            padding='SAME'
        )(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = hard_swish(x) if self.activation == 'hard_swish' else nn.relu(x)

        # Squeeze-and-excitation
        if self.use_se:
            x = SqueezeExcitation()(x)

        # Projection phase
        x = nn.Conv(self.out_channels, (1, 1))(x)
        x = nn.BatchNorm(use_running_average=True)(x)

        # Skip connection
        if self.stride == 1 and in_channels == self.out_channels:
            x = x + residual

        return x


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution for EfficientNet."""
    expansion: int
    out_channels: int
    kernel_size: int
    stride: int
    se_ratio: float = 0.25

    @nn.compact
    def __call__(self, x):
        in_channels = x.shape[-1]
        residual = x
        expanded_channels = in_channels * self.expansion

        # Expansion phase
        if self.expansion != 1:
            x = nn.Conv(expanded_channels, (1, 1))(x)
            x = nn.BatchNorm(use_running_average=True)(x)
            x = jax.nn.swish(x)

        # Depthwise convolution
        x = nn.Conv(
            expanded_channels,
            (self.kernel_size, self.kernel_size),
            strides=self.stride,
            feature_group_count=expanded_channels,
            padding='SAME'
        )(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = jax.nn.swish(x)

        # Squeeze-and-excitation
        if self.se_ratio > 0:
            x = SqueezeExcitation(se_ratio=self.se_ratio)(x)

        # Projection phase
        x = nn.Conv(self.out_channels, (1, 1))(x)
        x = nn.BatchNorm(use_running_average=True)(x)

        # Skip connection
        if self.stride == 1 and in_channels == self.out_channels:
            x = x + residual

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


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small implementation in Flax."""

    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Initial conv layer
        x = nn.Conv(16, (3, 3), strides=2, padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = hard_swish(x)

        # MobileNetV3-Small architecture
        # Format: (expansion, out_channels, kernel_size, stride, use_se, activation)
        configs = [
            (1, 16, 3, 2, True, 'relu'),
            (4.5, 24, 3, 2, False, 'relu'),
            (3.67, 24, 3, 1, False, 'relu'),
            (4, 40, 5, 2, True, 'hard_swish'),
            (6, 40, 5, 1, True, 'hard_swish'),
            (6, 40, 5, 1, True, 'hard_swish'),
            (3, 48, 5, 1, True, 'hard_swish'),
            (3, 48, 5, 1, True, 'hard_swish'),
            (6, 96, 5, 2, True, 'hard_swish'),
            (6, 96, 5, 1, True, 'hard_swish'),
            (6, 96, 5, 1, True, 'hard_swish'),
        ]

        for exp, out_ch, k, s, se, act in configs:
            x = InvertedResidualBlock(
                expansion=int(exp),
                out_channels=out_ch,
                kernel_size=k,
                stride=s,
                use_se=se,
                activation=act
            )(x)

        # Final layers
        x = nn.Conv(576, (1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = hard_swish(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Classifier
        x = nn.Dense(1024)(x)
        x = hard_swish(x)
        x = nn.Dense(self.num_classes)(x)

        return x


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 implementation in Flax."""

    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Initial conv layer
        x = nn.Conv(32, (3, 3), strides=2, padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.swish(x)

        # EfficientNet-B0 architecture
        # Format: (expansion, out_channels, kernel_size, stride, num_repeats)
        configs = [
            (1, 16, 3, 1, 1),    # Stage 1
            (6, 24, 3, 2, 2),    # Stage 2
            (6, 40, 5, 2, 2),    # Stage 3
            (6, 80, 3, 2, 3),    # Stage 4
            (6, 112, 5, 1, 3),   # Stage 5
            (6, 192, 5, 2, 4),   # Stage 6
            (6, 320, 3, 1, 1),   # Stage 7
        ]

        for exp, out_ch, k, s, repeats in configs:
            for i in range(repeats):
                stride = s if i == 0 else 1
                x = MBConvBlock(
                    expansion=exp,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=stride,
                    se_ratio=0.25
                )(x)

        # Final conv layer
        x = nn.Conv(1280, (1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.swish(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

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
        model_name: 'resnet50', 'vit_b_16', 'mobilenet_v3_small', 'efficientnet_b0'
        input_shape: Input shape tuple in HWC format (H, W, C)
        rng_key: JAX random key for initialization
        num_classes: Number of output classes

    Returns:
        Tuple of (apply_fn, params, preprocessing_fn, metadata)

    Note:
        MobileNetV3-Small and EfficientNet-B0 are not yet implemented in JAX/Flax.
        Use PyTorch implementations for these models.
    """
    if model_name not in MODEL_METADATA:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_METADATA.keys())}")

    metadata = MODEL_METADATA[model_name].copy()

    # Check if model is implemented
    if not metadata.get('implemented', True):
        raise NotImplementedError(
            f"{metadata['name']} is not yet implemented in JAX/Flax. "
            f"Please use the PyTorch implementation for this model, or implement the JAX/Flax version."
        )

    # Create model
    if model_name == 'resnet50':
        model = ResNet50(num_classes=num_classes)
    elif model_name == 'vit_b_16':
        model = VisionTransformer(num_classes=num_classes)
    elif model_name == 'mobilenet_v3_small':
        model = MobileNetV3Small(num_classes=num_classes)
    elif model_name == 'efficientnet_b0':
        model = EfficientNetB0(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    # Initialize parameters
    # Initialize parameters
    # For ViT, we use a smaller batch size for initialization to avoid OOM/CUDA errors
    init_batch_size = 1
    dummy_input = jnp.ones((init_batch_size, *input_shape), dtype=jnp.float32)
    
    # Ensure caches are clear before heavy initialization
    if model_name == 'vit_b_16':
        try:
            jax.clear_caches()
        except:
            pass
            
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
        model_name: 'resnet50', 'vit_b_16', 'mobilenet_v3_small', or 'efficientnet_b0'

    Returns:
        Dictionary with model metadata
    """
    if model_name not in MODEL_METADATA:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_METADATA[model_name].copy()

