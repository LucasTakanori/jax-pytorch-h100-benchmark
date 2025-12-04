import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torchvision.models as models
from datasets import Image as HFImage
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.device import get_torch_device, print_device_info, synchronize_torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_DATASET_ROOT = os.environ.get(
    "IMAGENET100_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'imagenet-100'))
)


def _resolve_data_files(dataset_root: str, split: str) -> dict:
    data_dir = Path(dataset_root).expanduser().resolve() / 'data'
    if not data_dir.exists():
        raise FileNotFoundError(f"ImageNet-100 parquet directory not found at {data_dir}")
    pattern = data_dir / f"{split}-*.parquet"
    matches = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet shards matching {pattern}")
    return {split: str(pattern)}


def _load_hf_dataset(dataset_root: str, split: str):
    data_files = _resolve_data_files(dataset_root, split)
    dataset = load_dataset('parquet', data_files=data_files, split=split)
    return dataset.cast_column('image', HFImage(decode=True))


class Imagenet100TorchDataset(Dataset):
    """Wrap Hugging Face dataset with torchvision preprocessing."""

    def __init__(self, hf_dataset, image_size: int = 224):
        self.dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.dataset[int(idx)]
        image = sample['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image)
        label = int(sample['label'])
        return tensor, label


def _build_dataloader(
    dataset_root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, int]:
    hf_dataset = _load_hf_dataset(dataset_root, split)
    torch_dataset = Imagenet100TorchDataset(hf_dataset)
    pin = device.type == 'cuda'
    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return dataloader, len(torch_dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet18 ImageNet-100 benchmark")
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT,
                        help="Path to the downloaded ImageNet-100 parquet folder")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                        help="Dataset split to benchmark on")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()

    # Use device detection utility
    device, device_info = get_torch_device()

    print("=" * 60)
    print("PyTorch Inference Benchmark")
    print("=" * 60)
    print_device_info(device_info)
    print(f"Dataset root: {os.path.abspath(args.dataset_root)}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print()

    dataloader, dataset_size = _build_dataloader(
        dataset_root=args.dataset_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"Samples in split: {dataset_size}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = model.to(device).eval()

    print("Running single pass over dataset...")
    total_images = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader, 1):
            inputs = inputs.to(device, non_blocking=device.type == 'cuda')
            _ = model(inputs)
            total_images += inputs.size(0)
            if batch_idx % 50 == 0:
                print(f"  Processed {total_images}/{dataset_size} samples...")
    synchronize_torch(device)
    elapsed = time.perf_counter() - start_time
    throughput = total_images / elapsed if elapsed > 0 else 0.0

    print("=" * 60)
    print("✅ ResNet18 ran successfully!")
    print(f"Samples processed: {total_images}/{dataset_size}")
    print(f"Total time: {elapsed:.4f}s")
    print(f"Throughput: {throughput:.2f} images/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()