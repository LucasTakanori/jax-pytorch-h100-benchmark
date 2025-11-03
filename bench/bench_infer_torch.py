import torch
import torchvision.models as models
import time

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using:", device)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = model.to(device).eval()

    x = torch.randn(16, 3, 224, 224, device=device)

    # Warmup
    for _ in range(5):
        model(x)

    torch.mps.synchronize() if device.type == "mps" else None

    t0 = time.perf_counter()
    for _ in range(20):
        model(x)
    torch.mps.synchronize() if device.type == "mps" else None

    print("✅ ResNet18 ran successfully!")
    print(f"Avg Latency/Batch: {(time.perf_counter() - t0)/20:.4f}s")

if __name__ == "__main__":
    main()