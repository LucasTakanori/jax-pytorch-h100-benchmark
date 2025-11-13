"""
Utilities package for 594Project benchmarking framework.
"""

from .device import (
    get_torch_device,
    get_jax_device,
    synchronize_torch,
    synchronize_jax,
    print_device_info,
    detect_all_devices,
    DeviceInfo,
)

__all__ = [
    'get_torch_device',
    'get_jax_device',
    'synchronize_torch',
    'synchronize_jax',
    'print_device_info',
    'detect_all_devices',
    'DeviceInfo',
]

