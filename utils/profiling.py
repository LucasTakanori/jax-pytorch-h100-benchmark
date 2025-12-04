"""
Advanced profiling utilities for tracking memory bandwidth, PCIe transfers,
cache behavior, and kernel-level metrics.

Provides detailed hardware profiling for PyTorch and JAX models using
framework profilers and NVML.
"""

import time
import os
import tempfile
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import json


@dataclass
class ProfileStats:
    """Comprehensive profiling statistics."""
    # Memory metrics
    memory_bandwidth_gb_s: Optional[float] = None  # Memory bandwidth utilization
    memory_read_gb: Optional[float] = None  # Total memory read
    memory_write_gb: Optional[float] = None  # Total memory written

    # PCIe metrics
    pcie_host_to_device_gb: Optional[float] = None  # H2D transfer volume
    pcie_device_to_host_gb: Optional[float] = None  # D2H transfer volume
    pcie_bandwidth_gb_s: Optional[float] = None  # PCIe bandwidth

    # Cache metrics
    l1_cache_hit_rate: Optional[float] = None  # L1 cache hit rate (%)
    l2_cache_hit_rate: Optional[float] = None  # L2 cache hit rate (%)
    l1_cache_requests: Optional[int] = None  # Total L1 requests
    l2_cache_requests: Optional[int] = None  # Total L2 requests

    # Kernel metrics
    kernel_count: Optional[int] = None  # Number of kernels launched
    avg_kernel_duration_ms: Optional[float] = None  # Average kernel duration
    total_kernel_time_ms: Optional[float] = None  # Total kernel execution time
    avg_occupancy: Optional[float] = None  # Average GPU occupancy (%)
    avg_warp_efficiency: Optional[float] = None  # Average warp efficiency (%)

    # GPU utilization
    gpu_utilization_pct: Optional[float] = None  # GPU compute utilization
    memory_utilization_pct: Optional[float] = None  # Memory controller utilization

    # Profiler metadata
    profiler_type: Optional[str] = None  # 'pytorch' or 'jax'
    duration_ms: Optional[float] = None  # Total profiling duration
    trace_file: Optional[str] = None  # Path to trace file (if saved)

    def __str__(self) -> str:
        """String representation of profile statistics."""
        parts = []
        if self.memory_bandwidth_gb_s is not None:
            parts.append(f"mem_bw={self.memory_bandwidth_gb_s:.2f}GB/s")
        if self.pcie_host_to_device_gb is not None:
            parts.append(f"H2D={self.pcie_host_to_device_gb:.3f}GB")
        if self.kernel_count is not None:
            parts.append(f"kernels={self.kernel_count}")
        if self.avg_occupancy is not None:
            parts.append(f"occupancy={self.avg_occupancy:.1f}%")

        return f"ProfileStats({', '.join(parts)})" if parts else "ProfileStats(no data)"


class PyTorchProfiler:
    """PyTorch-specific profiler for detailed CUDA metrics."""

    def __init__(
        self,
        with_stack: bool = False,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_flops: bool = False
    ):
        """
        Initialize PyTorch profiler.

        Args:
            with_stack: Record source code stack traces
            record_shapes: Record tensor shapes
            profile_memory: Profile memory usage
            with_flops: Calculate FLOPs
        """
        self.with_stack = with_stack
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_flops = with_flops
        self.prof = None
        self.trace_file = None

    def start(self, trace_dir: Optional[str] = None):
        """Start profiling."""
        import torch
        from torch.profiler import profile, ProfilerActivity, schedule

        # Setup trace directory
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            self.trace_file = trace_dir
        else:
            self.trace_file = None

        # Create profiler
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self.prof = profile(
            activities=activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir) if trace_dir else None
        )
        self.prof.__enter__()

    def stop(self) -> ProfileStats:
        """Stop profiling and return statistics."""
        if self.prof is None:
            return ProfileStats(profiler_type='pytorch')

        self.prof.__exit__(None, None, None)

        # Extract statistics from profiler
        stats = ProfileStats(profiler_type='pytorch')

        try:
            # Get key averages
            key_averages = self.prof.key_averages()

            # Count CUDA kernels
            cuda_kernels = [evt for evt in key_averages if evt.device_type.name == 'CUDA']
            stats.kernel_count = len(cuda_kernels)

            # Calculate total and average kernel time
            if cuda_kernels:
                total_cuda_time_us = sum(evt.cuda_time_total for evt in cuda_kernels)
                stats.total_kernel_time_ms = total_cuda_time_us / 1000.0
                stats.avg_kernel_duration_ms = total_cuda_time_us / len(cuda_kernels) / 1000.0

            # Memory metrics (if available)
            if self.profile_memory:
                try:
                    total_mem_events = [evt for evt in key_averages if evt.cpu_memory_usage > 0 or evt.cuda_memory_usage > 0]
                    if total_mem_events:
                        total_cuda_mem_mb = sum(evt.cuda_memory_usage for evt in total_mem_events) / 1e6
                        # Estimate memory bandwidth (rough approximation)
                        if stats.total_kernel_time_ms and stats.total_kernel_time_ms > 0:
                            time_s = stats.total_kernel_time_ms / 1000.0
                            stats.memory_bandwidth_gb_s = (total_cuda_mem_mb / 1000.0) / time_s
                except:
                    pass

        except Exception as e:
            print(f"Warning: Could not extract detailed profiling stats: {e}")

        return stats

    @contextmanager
    def profile(self, trace_dir: Optional[str] = None):
        """Context manager for profiling."""
        self.start(trace_dir)
        try:
            yield self
        finally:
            return self.stop()


class JAXProfiler:
    """JAX-specific profiler for XLA/CUDA metrics."""

    def __init__(self, create_perfetto_trace: bool = True):
        """
        Initialize JAX profiler.

        Args:
            create_perfetto_trace: Create Perfetto trace for TensorBoard
        """
        self.create_perfetto_trace = create_perfetto_trace
        self.trace_dir = None
        self.start_time = None

    def start(self, trace_dir: Optional[str] = None):
        """Start JAX profiling."""
        import jax

        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            self.trace_dir = trace_dir
        else:
            self.trace_dir = tempfile.mkdtemp(prefix='jax_trace_')

        self.start_time = time.time()

        try:
            # Start JAX profiler
            jax.profiler.start_trace(self.trace_dir)
        except Exception as e:
            print(f"Warning: Could not start JAX profiler: {e}")

    def stop(self) -> ProfileStats:
        """Stop JAX profiling and return statistics."""
        import jax

        stats = ProfileStats(profiler_type='jax')

        try:
            # Stop profiler
            jax.profiler.stop_trace()

            if self.start_time:
                duration_s = time.time() - self.start_time
                stats.duration_ms = duration_s * 1000.0

            stats.trace_file = self.trace_dir

        except Exception as e:
            print(f"Warning: Could not stop JAX profiler: {e}")

        return stats

    @contextmanager
    def profile(self, trace_dir: Optional[str] = None):
        """Context manager for profiling."""
        self.start(trace_dir)
        try:
            yield self
        finally:
            return self.stop()


class NVMLProfiler:
    """NVML-based profiler for GPU metrics (framework-agnostic)."""

    def __init__(self, device_id: int = 0, sample_interval_ms: float = 50.0):
        """
        Initialize NVML profiler.

        Args:
            device_id: CUDA device ID
            sample_interval_ms: Sampling interval in milliseconds
        """
        self.device_id = device_id
        self.sample_interval_s = sample_interval_ms / 1000.0
        self.samples = []
        self.nvml_available = False
        self.nvml_handle = None

        try:
            import pynvml
            self.pynvml = pynvml
            self.pynvml.nvmlInit()
            self.nvml_handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.nvml_available = True
        except Exception as e:
            print(f"Warning: NVML not available ({e}). GPU profiling disabled.")

    def sample(self) -> Optional[Dict[str, Any]]:
        """Take a single profiling sample."""
        if not self.nvml_available or self.nvml_handle is None:
            return None

        try:
            # Get utilization rates
            util = self.pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)

            # Get memory info
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)

            # Get PCIe throughput (if supported)
            try:
                pcie_tx = self.pynvml.nvmlDeviceGetPcieThroughput(
                    self.nvml_handle,
                    self.pynvml.NVML_PCIE_UTIL_TX_BYTES
                )  # KB/s
                pcie_rx = self.pynvml.nvmlDeviceGetPcieThroughput(
                    self.nvml_handle,
                    self.pynvml.NVML_PCIE_UTIL_RX_BYTES
                )  # KB/s
            except:
                pcie_tx, pcie_rx = None, None

            return {
                'timestamp': time.time(),
                'gpu_utilization_pct': util.gpu,
                'memory_utilization_pct': util.memory,
                'memory_used_gb': mem_info.used / 1e9,
                'memory_total_gb': mem_info.total / 1e9,
                'pcie_tx_kb_s': pcie_tx,
                'pcie_rx_kb_s': pcie_rx
            }
        except Exception as e:
            return None

    def start_sampling(self):
        """Start collecting samples."""
        self.samples.clear()
        sample = self.sample()
        if sample:
            self.samples.append(sample)

    def collect_sample(self):
        """Collect a single sample."""
        sample = self.sample()
        if sample:
            self.samples.append(sample)

    def stop_sampling(self) -> ProfileStats:
        """Stop sampling and calculate statistics."""
        # Final sample
        sample = self.sample()
        if sample:
            self.samples.append(sample)

        stats = ProfileStats()

        if not self.samples or len(self.samples) < 2:
            return stats

        # Calculate average utilization
        gpu_utils = [s['gpu_utilization_pct'] for s in self.samples]
        mem_utils = [s['memory_utilization_pct'] for s in self.samples]
        stats.gpu_utilization_pct = sum(gpu_utils) / len(gpu_utils)
        stats.memory_utilization_pct = sum(mem_utils) / len(mem_utils)

        # Calculate PCIe transfers
        pcie_tx_samples = [s['pcie_tx_kb_s'] for s in self.samples if s['pcie_tx_kb_s'] is not None]
        pcie_rx_samples = [s['pcie_rx_kb_s'] for s in self.samples if s['pcie_rx_kb_s'] is not None]

        if pcie_tx_samples and pcie_rx_samples:
            # Estimate total transfer (KB/s averaged over samples, multiplied by duration)
            duration_s = self.samples[-1]['timestamp'] - self.samples[0]['timestamp']
            avg_tx_kb_s = sum(pcie_tx_samples) / len(pcie_tx_samples)
            avg_rx_kb_s = sum(pcie_rx_samples) / len(pcie_rx_samples)

            stats.pcie_host_to_device_gb = (avg_rx_kb_s * duration_s) / 1e6  # RX is host to device
            stats.pcie_device_to_host_gb = (avg_tx_kb_s * duration_s) / 1e6  # TX is device to host
            stats.pcie_bandwidth_gb_s = (avg_tx_kb_s + avg_rx_kb_s) / 1e6

        # Calculate memory bandwidth (rough estimate from utilization)
        # Note: This is approximate. For accurate bandwidth, need hardware counters.
        if stats.memory_utilization_pct:
            # Assume peak bandwidth based on device (e.g., H100 has ~3TB/s)
            # This is a placeholder - actual peak bandwidth should be device-specific
            estimated_peak_bandwidth_gb_s = 2000.0  # Conservative estimate
            stats.memory_bandwidth_gb_s = (stats.memory_utilization_pct / 100.0) * estimated_peak_bandwidth_gb_s

        return stats

    def cleanup(self):
        """Cleanup NVML resources."""
        if self.nvml_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass


@contextmanager
def profile_pytorch(
    trace_dir: Optional[str] = None,
    with_nvml: bool = True,
    nvml_device_id: int = 0
):
    """
    Profile PyTorch execution with detailed metrics.

    Args:
        trace_dir: Directory to save profiling traces
        with_nvml: Also collect NVML metrics
        nvml_device_id: CUDA device ID for NVML

    Yields:
        Tuple of (PyTorchProfiler, Optional[NVMLProfiler])
    """
    pytorch_prof = PyTorchProfiler(record_shapes=True, profile_memory=True)
    nvml_prof = NVMLProfiler(nvml_device_id) if with_nvml else None

    # Start profilers
    pytorch_prof.start(trace_dir)
    if nvml_prof:
        nvml_prof.start_sampling()

    try:
        yield pytorch_prof, nvml_prof
    finally:
        # Stop profilers
        pytorch_stats = pytorch_prof.stop()
        nvml_stats = nvml_prof.stop_sampling() if nvml_prof else ProfileStats()

        # Merge stats
        merged_stats = pytorch_stats
        if nvml_stats:
            merged_stats.gpu_utilization_pct = nvml_stats.gpu_utilization_pct
            merged_stats.memory_utilization_pct = nvml_stats.memory_utilization_pct
            merged_stats.pcie_host_to_device_gb = nvml_stats.pcie_host_to_device_gb
            merged_stats.pcie_device_to_host_gb = nvml_stats.pcie_device_to_host_gb
            merged_stats.pcie_bandwidth_gb_s = nvml_stats.pcie_bandwidth_gb_s

        if nvml_prof:
            nvml_prof.cleanup()

        return merged_stats


@contextmanager
def profile_jax(
    trace_dir: Optional[str] = None,
    with_nvml: bool = True,
    nvml_device_id: int = 0
):
    """
    Profile JAX execution with detailed metrics.

    Args:
        trace_dir: Directory to save profiling traces
        with_nvml: Also collect NVML metrics
        nvml_device_id: CUDA device ID for NVML

    Yields:
        Tuple of (JAXProfiler, Optional[NVMLProfiler])
    """
    jax_prof = JAXProfiler()
    nvml_prof = NVMLProfiler(nvml_device_id) if with_nvml else None

    # Start profilers
    jax_prof.start(trace_dir)
    if nvml_prof:
        nvml_prof.start_sampling()

    try:
        yield jax_prof, nvml_prof
    finally:
        # Stop profilers
        jax_stats = jax_prof.stop()
        nvml_stats = nvml_prof.stop_sampling() if nvml_prof else ProfileStats()

        # Merge stats
        merged_stats = jax_stats
        if nvml_stats:
            merged_stats.gpu_utilization_pct = nvml_stats.gpu_utilization_pct
            merged_stats.memory_utilization_pct = nvml_stats.memory_utilization_pct
            merged_stats.pcie_host_to_device_gb = nvml_stats.pcie_host_to_device_gb
            merged_stats.pcie_device_to_host_gb = nvml_stats.pcie_device_to_host_gb
            merged_stats.pcie_bandwidth_gb_s = nvml_stats.pcie_bandwidth_gb_s
            merged_stats.memory_bandwidth_gb_s = nvml_stats.memory_bandwidth_gb_s

        if nvml_prof:
            nvml_prof.cleanup()

        return merged_stats


def profile_execution(
    fn: Callable,
    framework: str = 'pytorch',
    trace_dir: Optional[str] = None,
    device_id: int = 0,
    iterations: int = 1
) -> ProfileStats:
    """
    Profile execution of a function.

    Args:
        fn: Function to profile (callable with no arguments)
        framework: 'pytorch' or 'jax'
        trace_dir: Directory to save traces
        device_id: CUDA device ID
        iterations: Number of times to run function

    Returns:
        ProfileStats with collected metrics
    """
    if framework == 'pytorch':
        with profile_pytorch(trace_dir, with_nvml=True, nvml_device_id=device_id) as (prof, nvml):
            for _ in range(iterations):
                fn()
            return prof.stop()
    elif framework == 'jax':
        with profile_jax(trace_dir, with_nvml=True, nvml_device_id=device_id) as (prof, nvml):
            for _ in range(iterations):
                fn()
            return prof.stop()
    else:
        raise ValueError(f"Unknown framework: {framework}")
