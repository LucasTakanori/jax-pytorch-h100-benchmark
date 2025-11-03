import jax, jax.numpy as jnp
import flax.linen as nn
import time, numpy as np

class TinyNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3,3))(x)
        x = nn.relu(x)
        x = x.mean(axis=(1,2))
        return nn.Dense(10)(x)

def main():
    x = jnp.ones((16, 224, 224, 3))
    model = TinyNet()
    params = model.init(jax.random.PRNGKey(0), x)

    @jax.jit
    def f(p, x):
        return model.apply(p, x)

    # Warmup (JIT compile)
    _ = f(params, x).block_until_ready()

    times = []
    for _ in range(20):
        t = time.perf_counter()
        _ = f(params, x).block_until_ready()
        times.append(time.perf_counter() - t)

    print("✅ JAX + Flax working!")
    print(f"Median Latency: {np.median(times):.4f}s")

if __name__ == "__main__":
    main()