"""
Energy monitoring utilities for tracking GPU power consumption and energy usage.

Provides real-time power tracking, energy consumption calculations, and temperature monitoring
using NVIDIA Management Library (NVML) for CUDA devices.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Any, ContextManager
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class EnergyStats:
    """Statistics for energy consumption during a benchmark run."""
    total_energy_j: float  # Total energy consumed in Joules
    avg_power_w: float  # Average power draw in Watts
    peak_power_w: float  # Peak power draw in Watts
    min_power_w: float  # Minimum power draw in Watts
    duration_s: float  # Duration of measurement in seconds
    samples: int  # Number of power samples collected
    avg_temperature_c: Optional[float] = None  # Average GPU temperature
    peak_temperature_c: Optional[float] = None  # Peak GPU temperature

    def __str__(self) -> str:
        """String representation of energy statistics."""
        temp_str = ""
        if self.avg_temperature_c is not None:
            temp_str = f", temp={self.avg_temperature_c:.1f}°C"
        return (
            f"Energy Stats: "
            f"total={self.total_energy_j:.2f}J, "
            f"avg_power={self.avg_power_w:.2f}W, "
            f"peak={self.peak_power_w:.2f}W"
            f"{temp_str}"
        )


@dataclass
class PowerSample:
    """Single power measurement sample."""
    timestamp: float  # Time of measurement (seconds since epoch)
    power_w: float  # Power draw in Watts
    temperature_c: Optional[float] = None  # GPU temperature in Celsius
    utilization_pct: Optional[float] = None  # GPU utilization percentage


class EnergyTracker:
    """Tracks energy consumption during model execution."""

    def __init__(self, device_id: int = 0, sample_interval_ms: float = 10.0):
        """
        Initialize energy tracker.

        Args:
            device_id: CUDA device ID to monitor
            sample_interval_ms: Sampling interval in milliseconds (default: 10ms)
        """
        self.device_id = device_id
        self.sample_interval_s = sample_interval_ms / 1000.0
        self.samples: List[PowerSample] = []
        self.nvml_available = False
        self.nvml_handle = None
        self._start_time = None
        self._end_time = None

        # Try to initialize NVML
        try:
            import pynvml
            self.pynvml = pynvml
            self.pynvml.nvmlInit()
            self.nvml_handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.nvml_available = True
        except Exception as e:
            print(f"Warning: NVML not available ({e}). Energy tracking disabled.")
            self.nvml_available = False

    def _get_power_sample(self) -> Optional[PowerSample]:
        """Get a single power measurement sample."""
        if not self.nvml_available or self.nvml_handle is None:
            return None

        try:
            # Get power draw (milliwatts -> watts)
            power_mw = self.pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
            power_w = power_mw / 1000.0

            # Get temperature
            try:
                temp_c = self.pynvml.nvmlDeviceGetTemperature(
                    self.nvml_handle,
                    self.pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                temp_c = None

            # Get utilization
            try:
                util = self.pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                utilization_pct = util.gpu
            except:
                utilization_pct = None

            return PowerSample(
                timestamp=time.time(),
                power_w=power_w,
                temperature_c=temp_c,
                utilization_pct=utilization_pct
            )
        except Exception as e:
            print(f"Warning: Failed to get power sample: {e}")
            return None

    def start(self) -> None:
        """Start energy tracking (single sample)."""
        self._start_time = time.time()
        sample = self._get_power_sample()
        if sample:
            self.samples.append(sample)

    def sample(self) -> None:
        """Take a power measurement sample."""
        sample = self._get_power_sample()
        if sample:
            self.samples.append(sample)

    def stop(self) -> None:
        """Stop energy tracking (final sample)."""
        self._end_time = time.time()
        sample = self._get_power_sample()
        if sample:
            self.samples.append(sample)

    def get_stats(self) -> Optional[EnergyStats]:
        """
        Calculate energy statistics from collected samples.

        Returns:
            EnergyStats object with energy consumption metrics, or None if no samples
        """
        if not self.samples or len(self.samples) < 2:
            return None

        # Extract power and temperature data
        powers = np.array([s.power_w for s in self.samples])
        temperatures = [s.temperature_c for s in self.samples if s.temperature_c is not None]

        # Calculate duration
        start_ts = self.samples[0].timestamp
        end_ts = self.samples[-1].timestamp
        duration_s = end_ts - start_ts

        if duration_s <= 0:
            duration_s = len(self.samples) * self.sample_interval_s

        # Calculate energy using trapezoidal integration
        # Energy (J) = integral of Power (W) over time (s)
        if len(powers) > 1:
            time_intervals = np.diff([s.timestamp for s in self.samples])
            avg_powers_per_interval = (powers[:-1] + powers[1:]) / 2
            total_energy_j = np.sum(avg_powers_per_interval * time_intervals)
        else:
            total_energy_j = powers[0] * duration_s

        # Calculate statistics
        avg_power_w = np.mean(powers)
        peak_power_w = np.max(powers)
        min_power_w = np.min(powers)

        avg_temp_c = np.mean(temperatures) if temperatures else None
        peak_temp_c = np.max(temperatures) if temperatures else None

        return EnergyStats(
            total_energy_j=total_energy_j,
            avg_power_w=avg_power_w,
            peak_power_w=peak_power_w,
            min_power_w=min_power_w,
            duration_s=duration_s,
            samples=len(self.samples),
            avg_temperature_c=avg_temp_c,
            peak_temperature_c=peak_temp_c
        )

    def reset(self) -> None:
        """Reset tracker and clear samples."""
        self.samples.clear()
        self._start_time = None
        self._end_time = None

    def cleanup(self) -> None:
        """Cleanup NVML resources."""
        if self.nvml_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass
            self.nvml_available = False


class ContinuousEnergyMonitor:
    """
    Continuously monitors energy consumption in the background.

    Useful for long-running training sessions.
    """

    def __init__(self, device_id: int = 0, sample_interval_ms: float = 100.0):
        """
        Initialize continuous energy monitor.

        Args:
            device_id: CUDA device ID to monitor
            sample_interval_ms: Sampling interval in milliseconds (default: 100ms)
        """
        self.tracker = EnergyTracker(device_id, sample_interval_ms)
        self.monitoring = False
        self.monitor_thread = None

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            self.tracker.sample()
            time.sleep(self.tracker.sample_interval_s)

    def start(self) -> None:
        """Start background monitoring."""
        import threading

        if not self.tracker.nvml_available:
            print("Warning: NVML not available. Cannot start monitoring.")
            return

        self.tracker.reset()
        self.tracker.start()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self) -> Optional[EnergyStats]:
        """
        Stop background monitoring and return statistics.

        Returns:
            EnergyStats object with collected metrics
        """
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        self.tracker.stop()
        return self.tracker.get_stats()

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop()
        self.tracker.cleanup()


@contextmanager
def track_energy(
    device_id: int = 0,
    sample_interval_ms: float = 10.0,
    continuous: bool = False
) -> ContextManager[EnergyTracker]:
    """
    Context manager for tracking energy consumption.

    Usage:
        with track_energy(device_id=0) as tracker:
            # Run inference or training
            model(input)
        stats = tracker.get_stats()
        print(f"Energy consumed: {stats.total_energy_j:.2f} J")

    Args:
        device_id: CUDA device ID to monitor
        sample_interval_ms: Sampling interval in milliseconds
        continuous: Use continuous background monitoring (for long operations)

    Yields:
        EnergyTracker or ContinuousEnergyMonitor instance
    """
    if continuous:
        monitor = ContinuousEnergyMonitor(device_id, sample_interval_ms)
        monitor.start()
        try:
            yield monitor
        finally:
            monitor.cleanup()
    else:
        tracker = EnergyTracker(device_id, sample_interval_ms)
        tracker.start()
        try:
            yield tracker
        finally:
            tracker.stop()
            tracker.cleanup()


def measure_energy(
    fn,
    device_id: int = 0,
    sample_interval_ms: float = 10.0,
    iterations: int = 1
) -> Optional[EnergyStats]:
    """
    Measure energy consumption of a function.

    Args:
        fn: Function to measure (should be callable with no arguments)
        device_id: CUDA device ID to monitor
        sample_interval_ms: Sampling interval in milliseconds
        iterations: Number of times to run the function

    Returns:
        EnergyStats object with energy consumption metrics
    """
    tracker = EnergyTracker(device_id, sample_interval_ms)

    if not tracker.nvml_available:
        tracker.cleanup()
        return None

    tracker.start()

    # Run function and sample power periodically
    for i in range(iterations):
        fn()
        tracker.sample()

    tracker.stop()
    stats = tracker.get_stats()
    tracker.cleanup()

    return stats


def get_device_power_limit(device_id: int = 0) -> Optional[float]:
    """
    Get the power limit of a CUDA device.

    Args:
        device_id: CUDA device ID

    Returns:
        Power limit in Watts, or None if unavailable
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        pynvml.nvmlShutdown()
        return power_limit_mw / 1000.0  # Convert to Watts
    except Exception:
        return None


def get_current_power(device_id: int = 0) -> Optional[float]:
    """
    Get current power draw of a CUDA device.

    Args:
        device_id: CUDA device ID

    Returns:
        Current power draw in Watts, or None if unavailable
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        pynvml.nvmlShutdown()
        return power_mw / 1000.0  # Convert to Watts
    except Exception:
        return None


def get_device_info(device_id: int = 0) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive device information including power capabilities.

    Args:
        device_id: CUDA device ID

    Returns:
        Dictionary with device information
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        # Get device name
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        # Get power info
        power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        current_power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)

        # Get temperature
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temperature = None

        # Get memory info
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total_gb = mem_info.total / 1e9
            memory_used_gb = mem_info.used / 1e9
        except:
            memory_total_gb = None
            memory_used_gb = None

        pynvml.nvmlShutdown()

        return {
            'name': name,
            'device_id': device_id,
            'power_limit_w': power_limit_mw / 1000.0,
            'current_power_w': current_power_mw / 1000.0,
            'temperature_c': temperature,
            'memory_total_gb': memory_total_gb,
            'memory_used_gb': memory_used_gb
        }
    except Exception as e:
        return None
