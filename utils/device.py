"""
Device detection and management utilities for PyTorch and JAX.

Supports:
- CPU (fallback)
- CUDA (NVIDIA GPUs, including H100)
- MPS (Apple Silicon GPU)
- TPU (Google Cloud TPU, if available)
"""

import sys
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Information about a detected device."""
    name: str
    device_type: str  # 'cpu', 'cuda', 'mps', 'tpu'
    available: bool
    memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    device_id: Optional[int] = None
    backend: Optional[str] = None  # 'pytorch' or 'jax'


def get_torch_device(prefer: Optional[str] = None) -> Tuple[Any, DeviceInfo]:
    """
    Get the best available PyTorch device.
    
    Args:
        prefer: Preferred device type ('cuda', 'mps', 'cpu'). If None, auto-selects best available.
    
    Returns:
        Tuple of (torch.device, DeviceInfo)
    """
    import torch
    
    device_info = None
    
    # Check CUDA first (for H100 and other NVIDIA GPUs)
    if (prefer is None or prefer == 'cuda') and torch.cuda.is_available():
        device = torch.device('cuda')
        device_id = torch.cuda.current_device()
        memory_gb = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        compute_cap = f"{torch.cuda.get_device_properties(device_id).major}.{torch.cuda.get_device_properties(device_id).minor}"
        device_name = torch.cuda.get_device_name(device_id)
        device_info = DeviceInfo(
            name=device_name,
            device_type='cuda',
            available=True,
            memory_gb=memory_gb,
            compute_capability=compute_cap,
            device_id=device_id,
            backend='pytorch'
        )
        return device, device_info
    
    # Check MPS (Apple Silicon)
    if (prefer is None or prefer == 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        # MPS doesn't expose memory info easily, so we'll leave it None
        device_info = DeviceInfo(
            name='Apple Silicon GPU (Metal)',
            device_type='mps',
            available=True,
            backend='pytorch'
        )
        return device, device_info
    
    # Fallback to CPU
    device = torch.device('cpu')
    device_info = DeviceInfo(
        name='CPU',
        device_type='cpu',
        available=True,
        backend='pytorch'
    )
    return device, device_info


def get_jax_device(prefer: Optional[str] = None) -> Tuple[Any, DeviceInfo]:
    """
    Get the best available JAX device.
    
    Args:
        prefer: Preferred device type ('cuda', 'tpu', 'cpu'). If None, auto-selects best available.
    
    Returns:
        Tuple of (jax.Device, DeviceInfo)
    """
    import jax
    
    devices = jax.devices()
    
    # Check for TPU
    if (prefer is None or prefer == 'tpu') and len(devices) > 0:
        first_device = devices[0]
        if first_device.platform == 'tpu':
            device_info = DeviceInfo(
                name=f'TPU {first_device.id}',
                device_type='tpu',
                available=True,
                device_id=first_device.id,
                backend='jax'
            )
            return first_device, device_info
    
    # Check for GPU (CUDA)
    if (prefer is None or prefer == 'cuda'):
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if gpu_devices:
            device = gpu_devices[0]
            # Try to get memory info if available
            try:
                memory_info = jax.devices()[0].memory_stats() if hasattr(jax.devices()[0], 'memory_stats') else None
                memory_gb = memory_info.get('bytes_limit', 0) / 1e9 if memory_info else None
            except:
                memory_gb = None
            
            device_info = DeviceInfo(
                name=f'GPU {device.id}',
                device_type='cuda',
                available=True,
                device_id=device.id,
                memory_gb=memory_gb,
                backend='jax'
            )
            return device, device_info
    
    # Fallback to CPU
    cpu_devices = [d for d in devices if d.platform == 'cpu']
    device = cpu_devices[0] if cpu_devices else devices[0]
    device_info = DeviceInfo(
        name='CPU',
        device_type='cpu',
        available=True,
        backend='jax'
    )
    return device, device_info


def synchronize_torch(device: Any) -> None:
    """Synchronize PyTorch device operations for accurate timing."""
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def synchronize_jax() -> None:
    """Synchronize JAX operations (block until ready)."""
    # JAX operations are already blocking by default when using .block_until_ready()
    # This is a placeholder for consistency
    pass


def print_device_info(device_info: DeviceInfo) -> None:
    """Print formatted device information."""
    print(f"Device: {device_info.name}")
    print(f"  Type: {device_info.device_type.upper()}")
    print(f"  Available: {device_info.available}")
    if device_info.memory_gb:
        print(f"  Memory: {device_info.memory_gb:.2f} GB")
    if device_info.compute_capability:
        print(f"  Compute Capability: {device_info.compute_capability}")
    if device_info.device_id is not None:
        print(f"  Device ID: {device_info.device_id}")


def detect_all_devices() -> Dict[str, Any]:
    """
    Detect all available devices for both PyTorch and JAX.
    
    Returns:
        Dictionary with 'pytorch' and 'jax' keys containing device info
    """
    results = {}
    
    # PyTorch devices
    try:
        torch_device, torch_info = get_torch_device()
        results['pytorch'] = {
            'device': torch_device,
            'info': torch_info
        }
    except Exception as e:
        results['pytorch'] = {'error': str(e)}
    
    # JAX devices
    try:
        jax_device, jax_info = get_jax_device()
        results['jax'] = {
            'device': jax_device,
            'info': jax_info
        }
    except Exception as e:
        results['jax'] = {'error': str(e)}
    
    return results

