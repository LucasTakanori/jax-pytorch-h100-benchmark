"""
PyTorch model registry and implementations.

Provides ResNet-50 and Vision Transformer Base models with consistent interface.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Tuple, Callable, Optional, Dict, Any


# Model metadata
MODEL_METADATA = {
    'resnet50': {
        'name': 'ResNet-50',
        'params': 25_557_032,  # ~25M parameters
        'input_shape': (3, 224, 224),
        'flops': 4.1e9,  # ~4.1 GFLOPs
    },
    'vit_b_16': {
        'name': 'Vision Transformer Base',
        'params': 86_567_656,  # ~86M parameters
        'input_shape': (3, 224, 224),
        'flops': 17.6e9,  # ~17.6 GFLOPs
    }
}


def get_imagenet_preprocessing() -> transforms.Compose:
    """
    Get ImageNet preprocessing pipeline.
    
    Returns:
        transforms.Compose with normalization
    """
    return transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_torch_model(
    model_name: str,
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Callable, Tuple[int, ...], Dict[str, Any]]:
    """
    Get PyTorch model with consistent interface.
    
    Args:
        model_name: 'resnet50' or 'vit_b_16'
        pretrained: Whether to load pretrained weights
        device: Optional device to move model to
        
    Returns:
        Tuple of (model, preprocessing_fn, input_shape, metadata)
    """
    if model_name not in MODEL_METADATA:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_METADATA.keys())}")
    
    metadata = MODEL_METADATA[model_name]
    input_shape = metadata['input_shape']
    
    # Load model
    if model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(weights=None)
    
    elif model_name == 'vit_b_16':
        if pretrained:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_b_16(weights=None)
    
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    # Set to evaluation mode
    model.eval()
    
    # Move to device if provided
    if device is not None:
        model = model.to(device)
    
    # Preprocessing function
    preprocessing = get_imagenet_preprocessing()
    
    def preprocess_fn(x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input tensor.
        
        Args:
            x: Input tensor in [0, 1] range, shape (B, C, H, W)
            
        Returns:
            Normalized tensor
        """
        # Convert from [0, 1] to [0, 255] then normalize
        x = x * 255.0
        return preprocessing(x)
    
    return model, preprocess_fn, input_shape, metadata


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

