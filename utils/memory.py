"""
Memory profiling utilities for tracking peak memory usage during inference.
"""

import psutil
import os
from typing import Optional, Any, ContextManager
from contextlib import contextmanager
from .device import DeviceInfo


class MemoryTracker:
    """Context manager for tracking memory usage during inference."""
    
    def __init__(self, framework: str, device: Any = None, device_info: Optional[DeviceInfo] = None):
        """
        Initialize memory tracker.
        
        Args:
            framework: 'pytorch' or 'jax'
            device: Device object (PyTorch device or JAX device)
            device_info: DeviceInfo object with device metadata
        """
        self.framework = framework
        self.device = device
        self.device_info = device_info
        self.initial_memory_mb = None
        self.peak_memory_mb = None
        
    def __enter__(self):
        """Enter context and record initial memory."""
        self.initial_memory_mb = get_current_memory(self.framework, self.device)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record peak memory."""
        self.peak_memory_mb = get_peak_memory(self.framework, self.device)
        return False
    
    def get_peak_usage_mb(self) -> Optional[float]:
        """Get peak memory usage in MB."""
        return self.peak_memory_mb
    
    def get_delta_mb(self) -> Optional[float]:
        """Get memory delta (peak - initial) in MB."""
        if self.initial_memory_mb is None or self.peak_memory_mb is None:
            return None
        return self.peak_memory_mb - self.initial_memory_mb


def get_current_memory(framework: str, device: Any = None) -> float:
    """
    Get current memory usage in MB.
    
    Args:
        framework: 'pytorch' or 'jax'
        device: Device object (for PyTorch CUDA/MPS)
        
    Returns:
        Memory usage in MB
    """
    if framework == 'pytorch':
        return get_pytorch_memory(device)
    elif framework == 'jax':
        return get_jax_memory()
    else:
        # Fallback to process memory
        return get_process_memory()
    

def get_peak_memory(framework: str, device: Any = None) -> float:
    """
    Get peak memory usage in MB.
    
    Args:
        framework: 'pytorch' or 'jax'
        device: Device object (for PyTorch CUDA/MPS)
        
    Returns:
        Peak memory usage in MB
    """
    if framework == 'pytorch':
        return get_pytorch_peak_memory(device)
    elif framework == 'jax':
        return get_jax_memory()  # JAX doesn't expose peak easily, use current
    else:
        return get_process_memory()


def get_pytorch_memory(device: Any) -> float:
    """
    Get PyTorch device memory usage in MB.
    
    Args:
        device: torch.device object
        
    Returns:
        Memory usage in MB
    """
    try:
        import torch
        
        if device.type == 'cuda':
            # CUDA memory
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(device.index) / 1e6
        elif device.type == 'mps':
            # MPS memory (Apple Silicon)
            if hasattr(torch.mps, 'current_allocated_memory'):
                return torch.mps.current_allocated_memory() / 1e6
            # Fallback: estimate from process memory
            return get_process_memory()
        
        # CPU: use process memory
        return get_process_memory()
    except Exception:
        # Fallback to process memory on error
        return get_process_memory()


def get_pytorch_peak_memory(device: Any) -> float:
    """
    Get PyTorch peak memory usage in MB.
    
    Args:
        device: torch.device object
        
    Returns:
        Peak memory usage in MB
    """
    try:
        import torch
        
        if device.type == 'cuda':
            # CUDA peak memory
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated(device.index) / 1e6
        elif device.type == 'mps':
            # MPS doesn't expose peak easily, use current
            return get_pytorch_memory(device)
        
        # CPU: use process memory
        return get_process_memory()
    except Exception:
        return get_process_memory()


def get_jax_memory() -> float:
    """
    Get JAX memory usage in MB.
    
    Note: JAX doesn't expose device memory easily on all backends,
    so we use process memory as a fallback.
    
    Returns:
        Memory usage in MB
    """
    try:
        import jax
        
        # Try to get device memory if available
        devices = jax.devices()
        if devices and devices[0].platform == 'gpu':
            # For GPU, try to get memory stats
            try:
                # This may not work on all JAX backends
                memory_info = jax.devices()[0].memory_stats()
                if memory_info and 'bytes_limit' in memory_info:
                    return memory_info['bytes_limit'] / 1e6
            except:
                pass
        
        # Fallback to process memory
        return get_process_memory()
    except Exception:
        return get_process_memory()


def get_process_memory() -> float:
    """
    Get process memory usage (RSS) in MB.
    
    Returns:
        Process memory usage in MB
    """
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1e6
    except Exception:
        return 0.0


def reset_memory_stats(framework: str, device: Any = None) -> None:
    """
    Reset memory statistics (primarily for PyTorch CUDA).
    
    Args:
        framework: 'pytorch' or 'jax'
        device: Device object (for PyTorch)
    """
    if framework == 'pytorch':
        try:
            import torch
            if device and device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device.index)
        except Exception:
            pass
    # JAX doesn't have a reset function


@contextmanager
def track_memory(framework: str, device: Any = None) -> ContextManager[MemoryTracker]:
    """
    Context manager for tracking memory usage.
    
    Usage:
        with track_memory('pytorch', device) as tracker:
            # Run inference
            model(input)
        peak_mb = tracker.get_peak_usage_mb()
    
    Args:
        framework: 'pytorch' or 'jax'
        device: Device object
        
    Yields:
        MemoryTracker instance
    """
    tracker = MemoryTracker(framework, device)
    with tracker:
        yield tracker

