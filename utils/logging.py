"""
CSV logging utilities for benchmark results.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


class BenchmarkLogger:
    """Manages CSV file creation and writing for benchmark results."""
    
    # CSV field names (schema)
    FIELDNAMES = [
        # Metadata
        'timestamp',
        'framework',
        'model',
        'device',
        'device_type',
        'batch_size',
        'input_shape',
        'dtype',
        
        # Benchmark configuration
        'warmup_iterations',
        'measurement_iterations',
        
        # Latency metrics (milliseconds)
        'latency_p50_ms',
        'latency_p95_ms',
        'latency_p99_ms',
        'latency_mean_ms',
        'latency_std_ms',
        'latency_min_ms',
        'latency_max_ms',
        
        # Performance metrics
        'throughput_ips',  # Images per second
        'memory_mb',       # Peak memory in MB
        'compilation_time_ms',  # JIT/XLA compilation time
        
        # Optional metadata
        'git_commit',
        'notes'
    ]
    
    def __init__(self, output_dir: str = 'results/raw', filename: Optional[str] = None):
        """
        Initialize benchmark logger.
        
        Args:
            output_dir: Directory to save CSV files
            filename: Optional custom filename (default: auto-generated with timestamp)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename:
            self.filename = filename
        else:
            # Auto-generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.filename = f'benchmark_results_{timestamp}.csv'
        
        self.filepath = self.output_dir / self.filename
        self.file_exists = self.filepath.exists()
        
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash if available."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def append_result(self, result: Dict[str, Any]) -> None:
        """
        Append a single benchmark result to CSV file.
        
        Args:
            result: Dictionary with benchmark results (keys should match FIELDNAMES)
        """
        # Ensure all required fields are present
        row = {}
        for field in self.FIELDNAMES:
            row[field] = result.get(field, '')
        
        # Add timestamp if not present
        if not row['timestamp']:
            row['timestamp'] = datetime.now().isoformat()
        
        # Add git commit if not present
        if not row['git_commit']:
            row['git_commit'] = self._get_git_commit()
        
        # Write to CSV
        file_exists = self.filepath.exists()
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
    
    def append_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Append multiple benchmark results to CSV file.
        
        Args:
            results: List of result dictionaries
        """
        for result in results:
            self.append_result(result)

    def log_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Backwards-compatible alias for append_results used by older scripts.
        """
        self.append_results(results)
    
    def read_results(self) -> pd.DataFrame:
        """
        Read all results from CSV file into pandas DataFrame.
        
        Returns:
            DataFrame with all benchmark results
        """
        if not self.filepath.exists():
            return pd.DataFrame()
        
        return pd.read_csv(self.filepath)
    
    def get_filepath(self) -> Path:
        """Get the full filepath of the CSV file."""
        return self.filepath


def create_result_dict(
    framework: str,
    model: str,
    device: str,
    device_type: str,
    batch_size: int,
    input_shape: tuple,
    dtype: str,
    warmup_iterations: int,
    measurement_iterations: int,
    latency_stats: Any,  # LatencyStats object
    throughput_ips: float,
    memory_mb: float,
    compilation_time_ms: Optional[float] = None,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a standardized result dictionary for logging.
    
    Args:
        framework: 'pytorch' or 'jax'
        model: Model name (e.g., 'resnet50', 'vit_b_16')
        device: Device name/identifier
        device_type: 'cpu', 'cuda', 'mps', 'tpu'
        batch_size: Batch size used
        input_shape: Input shape tuple (e.g., (3, 224, 224))
        dtype: Data type (e.g., 'float32')
        warmup_iterations: Number of warmup iterations
        measurement_iterations: Number of measurement iterations
        latency_stats: LatencyStats object
        throughput_ips: Throughput in images per second
        memory_mb: Peak memory in MB
        compilation_time_ms: Optional compilation time in ms
        notes: Optional notes
        
    Returns:
        Dictionary ready for CSV logging
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'framework': framework,
        'model': model,
        'device': device,
        'device_type': device_type,
        'batch_size': batch_size,
        'input_shape': str(input_shape),
        'dtype': dtype,
        'warmup_iterations': warmup_iterations,
        'measurement_iterations': measurement_iterations,
        'latency_p50_ms': latency_stats.p50_ms,
        'latency_p95_ms': latency_stats.p95_ms,
        'latency_p99_ms': latency_stats.p99_ms,
        'latency_mean_ms': latency_stats.mean_ms,
        'latency_std_ms': latency_stats.std_ms,
        'latency_min_ms': latency_stats.min_ms,
        'latency_max_ms': latency_stats.max_ms,
        'throughput_ips': throughput_ips,
        'memory_mb': memory_mb,
        'compilation_time_ms': compilation_time_ms if compilation_time_ms is not None else '',
        'git_commit': '',
        'notes': notes if notes else ''
    }

