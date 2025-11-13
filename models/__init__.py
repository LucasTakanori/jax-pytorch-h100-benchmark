"""
Model implementations for PyTorch and JAX/Flax.
"""

from .torch_zoo import get_torch_model
from .jax_flax_zoo import get_flax_model

__all__ = [
    'get_torch_model',
    'get_flax_model',
]

