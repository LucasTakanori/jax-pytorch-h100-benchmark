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

from .timing import (
    LatencyStats,
    synchronize,
    calculate_throughput,
    measure_latency,
    measure_compilation_time,
)

from .memory import (
    MemoryTracker,
    get_current_memory,
    get_peak_memory,
    reset_memory_stats,
    track_memory,
)

from .logging import (
    BenchmarkLogger,
    create_result_dict,
)

from .data import (
    get_synthetic_batch,
    get_preprocessing,
    load_imagenet100,
    create_dataloader,
)

from .validation import (
    compare_outputs,
    validate_forward_pass,
    validate_model_architecture,
    compare_models,
)

__all__ = [
    # Device utilities
    'get_torch_device',
    'get_jax_device',
    'synchronize_torch',
    'synchronize_jax',
    'print_device_info',
    'detect_all_devices',
    'DeviceInfo',
    # Timing utilities
    'LatencyStats',
    'synchronize',
    'calculate_throughput',
    'measure_latency',
    'measure_compilation_time',
    # Memory utilities
    'MemoryTracker',
    'get_current_memory',
    'get_peak_memory',
    'reset_memory_stats',
    'track_memory',
    # Logging utilities
    'BenchmarkLogger',
    'create_result_dict',
    # Data utilities
    'get_synthetic_batch',
    'get_preprocessing',
    'load_imagenet100',
    'create_dataloader',
    # Validation utilities
    'compare_outputs',
    'validate_forward_pass',
    'validate_model_architecture',
    'compare_models',
]

