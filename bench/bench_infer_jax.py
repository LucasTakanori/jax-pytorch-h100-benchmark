import jax, jax.numpy as jnp
import flax.linen as nn
import time, numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.device import get_jax_device, print_device_info

class TinyNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3,3))(x)
        x = nn.relu(x)
        x = x.mean(axis=(1,2))
        return nn.Dense(10)(x)

def main():
    # Use device detection utility
    device, device_info = get_jax_device()
    
    print("=" * 60)
    print("JAX + Flax Inference Benchmark")
    print("=" * 60)
    print_device_info(device_info)
    print()
    
    # Move computation to detected device
    x = jnp.ones((16, 224, 224, 3))
    model = TinyNet()
    params = model.init(jax.random.PRNGKey(0), x)

    @jax.jit
    def f(p, x):
        return model.apply(p, x)

    # Warmup (JIT compile)
    print("Warming up (JIT compilation)...")
    _ = f(params, x).block_until_ready()

    # Benchmark
    print("Running benchmark (20 iterations)...")
    times = []
    for _ in range(20):
        t = time.perf_counter()
        _ = f(params, x).block_until_ready()
        times.append(time.perf_counter() - t)

    print("=" * 60)
    print("✅ JAX + Flax working!")
    print(f"Median Latency: {np.median(times):.4f}s")
    print(f"Mean Latency: {np.mean(times):.4f}s")
    print(f"Throughput: {16 * 20 / np.sum(times):.2f} images/sec")
    print("=" * 60)

if __name__ == "__main__":
    main()