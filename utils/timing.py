"""
Timing utilities for benchmarking inference performance.

Provides latency statistics, synchronization helpers, and throughput calculations.
"""

import time
import numpy as np
from typing import List, Optional, Any
from dataclasses import dataclass
from .device import synchronize_torch, synchronize_jax


@dataclass
class LatencyStats:
    """Statistics computed from a list of latency measurements."""
    p50_ms: float  # Median (50th percentile)
    p95_ms: float  # 95th percentile
    p99_ms: float  # 99th percentile
    mean_ms: float  # Mean
    std_ms: float   # Standard deviation
    min_ms: float   # Minimum
    max_ms: float   # Maximum
    count: int      # Number of measurements

    @classmethod
    def from_durations(cls, durations_s: List[float]) -> 'LatencyStats':
        """
        Compute statistics from a list of durations in seconds.
        
        Args:
            durations_s: List of latency measurements in seconds
            
        Returns:
            LatencyStats object with all statistics in milliseconds
        """
        if not durations_s:
            raise ValueError("Cannot compute statistics from empty list")
        
        durations_ms = np.array(durations_s) * 1000.0  # Convert to milliseconds
        sorted_durations = np.sort(durations_ms)
        n = len(sorted_durations)
        
        return cls(
            p50_ms=np.median(durations_ms),
            p95_ms=np.percentile(durations_ms, 95),
            p99_ms=np.percentile(durations_ms, 99),
            mean_ms=np.mean(durations_ms),
            std_ms=np.std(durations_ms),
            min_ms=np.min(durations_ms),
            max_ms=np.max(durations_ms),
            count=n
        )
    
    def __str__(self) -> str:
        """String representation of latency statistics."""
        return (
            f"Latency Stats (n={self.count}): "
            f"mean={self.mean_ms:.3f}ms, "
            f"p50={self.p50_ms:.3f}ms, "
            f"p95={self.p95_ms:.3f}ms, "
            f"p99={self.p99_ms:.3f}ms"
        )


def synchronize(framework: str, device: Any = None) -> None:
    """
    Synchronize device operations for accurate timing.
    
    Args:
        framework: 'pytorch' or 'jax'
        device: Device object (required for PyTorch, optional for JAX)
    """
    if framework == 'pytorch':
        if device is None:
            raise ValueError("Device required for PyTorch synchronization")
        synchronize_torch(device)
    elif framework == 'jax':
        # JAX operations are blocking by default when using block_until_ready()
        # This is a no-op for consistency
        synchronize_jax()
    else:
        raise ValueError(f"Unknown framework: {framework}")


def calculate_throughput(batch_size: int, latency_s: float) -> float:
    """
    Calculate throughput in images per second.
    
    Args:
        batch_size: Number of images in batch
        latency_s: Latency per batch in seconds
        
    Returns:
        Throughput in images per second
    """
    if latency_s <= 0:
        return 0.0
    return batch_size / latency_s


def measure_latency(
    inference_fn,
    warmup_iterations: int = 5,
    measurement_iterations: int = 50,
    framework: str = 'pytorch',
    device: Any = None,
    verbose: bool = False
) -> LatencyStats:
    """
    Measure inference latency with warmup and collect statistics.
    
    Args:
        inference_fn: Function that performs inference (no arguments)
        warmup_iterations: Number of warmup iterations
        measurement_iterations: Number of measurement iterations
        framework: 'pytorch' or 'jax'
        device: Device object (for PyTorch synchronization)
        verbose: Print progress if True
        
    Returns:
        LatencyStats object with computed statistics
    """
    # Warmup phase
    if verbose:
        print(f"Warming up ({warmup_iterations} iterations)...")
    
    for _ in range(warmup_iterations):
        inference_fn()
    
    # Synchronize before measurement
    synchronize(framework, device)
    
    # Measurement phase
    if verbose:
        print(f"Measuring latency ({measurement_iterations} iterations)...")
    
    durations = []
    for i in range(measurement_iterations):
        start_time = time.perf_counter()
        inference_fn()
        synchronize(framework, device)
        end_time = time.perf_counter()
        durations.append(end_time - start_time)
    
    return LatencyStats.from_durations(durations)


def measure_compilation_time(
    compile_fn,
    framework: str = 'jax',
    verbose: bool = False
) -> float:
    """
    Measure JIT/XLA compilation time (primarily for JAX).
    
    Args:
        compile_fn: Function that triggers compilation (first call)
        framework: 'pytorch' or 'jax' (JAX typically has longer compilation)
        verbose: Print progress if True
        
    Returns:
        Compilation time in milliseconds
    """
    if verbose:
        print(f"Measuring {framework.upper()} compilation time...")
    
    start_time = time.perf_counter()
    compile_fn()  # First call triggers compilation
    end_time = time.perf_counter()
    
    compilation_time_s = end_time - start_time
    return compilation_time_s * 1000.0  # Convert to milliseconds

