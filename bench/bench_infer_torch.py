import torch
import torchvision.models as models
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.device import get_torch_device, synchronize_torch, print_device_info

def main():
    # Use device detection utility
    device, device_info = get_torch_device()
    
    print("=" * 60)
    print("PyTorch Inference Benchmark")
    print("=" * 60)
    print_device_info(device_info)
    print()

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = model.to(device).eval()

    x = torch.randn(16, 3, 224, 224, device=device)

    # Warmup
    print("Warming up...")
    for _ in range(5):
        model(x)
    synchronize_torch(device)

    # Benchmark
    print("Running benchmark (20 iterations)...")
    t0 = time.perf_counter()
    for _ in range(20):
        model(x)
    synchronize_torch(device)
    elapsed = time.perf_counter() - t0

    print("=" * 60)
    print("✅ ResNet18 ran successfully!")
    print(f"Avg Latency/Batch: {elapsed/20:.4f}s")
    print(f"Throughput: {16 * 20 / elapsed:.2f} images/sec")
    print("=" * 60)

if __name__ == "__main__":
    main()