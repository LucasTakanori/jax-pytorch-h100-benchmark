import argparse
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image as PILImage

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.device import get_jax_device, print_device_info

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_DATASET_ROOT = os.environ.get(
    "IMAGENET100_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'imagenet-100'))
)
RESAMPLE_BICUBIC = getattr(getattr(PILImage, "Resampling", PILImage), "BICUBIC")


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


def _preprocess_image(image: PILImage.Image) -> np.ndarray:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224), resample=RESAMPLE_BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - IMAGENET_MEAN) / IMAGENET_STD


def _create_numpy_batch_iterator(
    dataset_root: str,
    split: str,
    batch_size: int,
) -> Tuple[Iterator[np.ndarray], int]:
    hf_dataset = _load_hf_dataset(dataset_root, split)
    dataset_size = len(hf_dataset)

    def iterator():
        batch = []
        for example in hf_dataset:
            batch.append(_preprocess_image(example['image']))
            if len(batch) == batch_size:
                yield np.stack(batch, axis=0).astype(np.float32)
                batch = []
        if batch:
            yield np.stack(batch, axis=0).astype(np.float32)

    return iterator(), dataset_size


class TinyNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = x.mean(axis=(1, 2))
        return nn.Dense(10)(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JAX + Flax ImageNet-100 benchmark")
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT,
                        help="Path to the downloaded ImageNet-100 parquet folder")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                        help="Dataset split to benchmark on")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    return parser.parse_args()


def main():
    args = parse_args()

    # Use device detection utility
    device, device_info = get_jax_device()

    print("=" * 60)
    print("JAX + Flax Inference Benchmark")
    print("=" * 60)
    print_device_info(device_info)
    print(f"Dataset root: {os.path.abspath(args.dataset_root)}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print()

    batch_iter, dataset_size = _create_numpy_batch_iterator(
        dataset_root=args.dataset_root,
        split=args.split,
        batch_size=args.batch_size,
    )

    try:
        first_batch = next(batch_iter)
    except StopIteration:
        raise RuntimeError("Dataset is empty; nothing to benchmark.")

    batch_stream = itertools.chain([first_batch], batch_iter)

    model = TinyNet()
    params = model.init(jax.random.PRNGKey(0), jnp.array(first_batch))

    @jax.jit
    def f(p, x):
        return model.apply(p, x)

    # Warmup / compilation
    print("Compiling and warming up...")
    _ = f(params, jax.device_put(first_batch)).block_until_ready()

    print("Running single pass over dataset...")
    total_images = 0
    times = []
    start_time = time.perf_counter()
    for batch_idx, batch in enumerate(batch_stream, 1):
        device_batch = jax.device_put(batch)
        t = time.perf_counter()
        _ = f(params, device_batch).block_until_ready()
        step_time = time.perf_counter() - t
        times.append(step_time)
        total_images += batch.shape[0]
        if batch_idx % 50 == 0:
            print(f"  Processed {total_images}/{dataset_size} samples...")

    total_time = time.perf_counter() - start_time
    mean_step = np.mean(times) if times else 0.0

    print("=" * 60)
    print("✅ JAX + Flax working!")
    print(f"Samples processed: {total_images}/{dataset_size}")
    print(f"Total time: {total_time:.4f}s")
    if times:
        print(f"Median batch latency: {np.median(times):.4f}s")
        print(f"Mean batch latency: {mean_step:.4f}s")
    print(f"Throughput: {total_images / total_time:.2f} images/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()