"""
Training/fine-tuning benchmark runner for PyTorch and JAX models.

Provides comprehensive training benchmarks with energy monitoring,
profiling, and accuracy tracking.
"""

import sys
import os
import time
import csv
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from datasets import load_dataset, load_from_disk
from datasets import builder as datasets_builder
from fsspec.implementations.local import LocalFileSystem
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.device import get_torch_device, get_jax_device, DeviceInfo
from utils.timing import synchronize
from utils.memory import get_peak_memory, reset_memory_stats
from utils.energy import EnergyTracker, EnergyStats
from utils.profiling import ProfileStats, NVMLProfiler

# JAX/Flax imports
try:
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    from flax import linen as nn
    import orbax.checkpoint
except ImportError:
    pass  # Will fail later if JAX is selected but not installed
if not hasattr(datasets_builder, "_local_fs_patch"):
    _orig_is_remote = datasets_builder.is_remote_filesystem

    def _patched_is_remote(fs):
        if isinstance(fs, LocalFileSystem):
            return False
        return _orig_is_remote(fs)

    datasets_builder.is_remote_filesystem = _patched_is_remote
    datasets_builder._local_fs_patch = True


def _safe_cache_subdir(name: str) -> str:
    """Convert dataset identifiers to filesystem-friendly names."""
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)


@dataclass
class TrainingConfig:
    """Configuration for training benchmark."""
    framework: str  # 'pytorch' or 'jax'
    model_name: str  # 'resnet50', 'vit_b_16', etc.
    dataset_path: str  # Path to ImageNet-100 dataset or HF dataset id
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-4
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    scheduler: str = 'cosine'  # 'cosine', 'none'
    device_prefer: Optional[str] = None
    num_workers: int = 4
    save_checkpoints: bool = False
    checkpoint_dir: str = 'checkpoints'
    csv_output_dir: str = 'results/training'
    verbose: bool = True
    track_energy: bool = True
    track_profiling: bool = True
    eval_frequency: int = 1  # Evaluate every N epochs
    dataset_cache_dir: Optional[str] = None
    run_id: str = "default_run"
    session_dir: Optional[str] = None

    def __post_init__(self):
        if self.session_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(
                self.csv_output_dir,
                self.run_id,
                self.model_name,
                f"{self.framework}_bs{self.batch_size}_{timestamp}"
            )


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    train_accuracy_top1: float
    train_accuracy_top5: float
    val_loss: Optional[float] = None
    val_accuracy_top1: Optional[float] = None
    val_accuracy_top5: Optional[float] = None
    epoch_duration_s: float = 0.0
    samples_per_sec: float = 0.0
    energy_j: Optional[float] = None
    avg_power_w: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None


def create_dataloaders(config: TrainingConfig) -> Tuple[Any, Any, bool]:
    """
    Setup data loaders for ImageNet-100.
    
    Returns:
        Tuple of (train_loader, val_loader, is_hf_dataset)
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets as tv_datasets, transforms
    from PIL import Image

    if config.verbose:
        print(f"Loading ImageNet-100 from {config.dataset_path}")

    dataset_arg = config.dataset_path
    cache_dir = config.dataset_cache_dir

    # ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if os.path.isdir(dataset_arg):
        train_dir = os.path.join(dataset_arg, 'train')
        val_dir = os.path.join(dataset_arg, 'validation')
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise FileNotFoundError(
                f"Expected ImageNet-100 structure with 'train' and 'validation' folders under "
                f"{dataset_arg} (got train_dir={train_dir}, val_dir={val_dir})"
            )

        train_dataset = tv_datasets.ImageFolder(train_dir, transform=transform)
        val_dataset = tv_datasets.ImageFolder(val_dir, transform=transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        if config.verbose:
            print(f"  Train samples: {len(train_dataset)}")
            print(f"  Val samples: {len(val_dataset)}")
            print(f"  Batch size: {config.batch_size}")

        return train_loader, val_loader, False
    else:
        load_kwargs = {}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        hf_saved_dataset_dir = None
        hf_saved_dataset_uri = None
        if cache_dir:
            hf_saved_dataset_dir = os.path.join(
                cache_dir,
                f"hf_saved_{_safe_cache_subdir(dataset_arg)}"
            )
            hf_saved_dataset_uri = Path(hf_saved_dataset_dir).as_uri()

        if hf_saved_dataset_dir and os.path.isdir(hf_saved_dataset_dir):
            if config.verbose:
                print(f"  Loading dataset from disk cache: {hf_saved_dataset_dir}")
            hf_dataset = load_from_disk(hf_saved_dataset_uri or hf_saved_dataset_dir)
        else:
            if config.verbose:
                print(f"  Downloading dataset {dataset_arg} via Hugging Face Hub")
            hf_dataset = load_dataset(
                dataset_arg,
                download_mode="reuse_dataset_if_exists",
                **load_kwargs
            )
            if hf_saved_dataset_dir:
                if config.verbose:
                    print(f"  Saving dataset to disk cache: {hf_saved_dataset_dir}")
                hf_dataset.save_to_disk(hf_saved_dataset_dir)

        def _select_split(preferred_names, available):
            for name in preferred_names:
                if name in available:
                    return name
            return None

        train_split = _select_split(['train', 'training'], hf_dataset.keys())
        val_split = _select_split(['validation', 'val', 'test'], hf_dataset.keys())

        if train_split is None or val_split is None:
            raise ValueError(
                f"Dataset {dataset_arg} must have train and validation/test splits "
                f"(found splits: {list(hf_dataset.keys())})"
            )

        dataset = {
            'train': hf_dataset[train_split],
            'validation': hf_dataset[val_split]
        }

        def preprocess_fn(examples):
            images = []
            for img in examples['image']:
                if isinstance(img, Image.Image):
                    pil_img = img
                else:
                    pil_img = Image.fromarray(img)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                images.append(transform(pil_img))

            return {
                'pixel_values': torch.stack(images),
                'labels': torch.tensor(examples['label'], dtype=torch.long)
            }

        dataset['train'].set_transform(preprocess_fn)
        dataset['validation'].set_transform(preprocess_fn)

        train_loader = DataLoader(
            dataset['train'],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            dataset['validation'],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        if config.verbose:
            print(f"  HuggingFace dataset: {dataset_arg}")
            print(f"  Train split ({train_split}): {len(dataset['train'])} samples")
            print(f"  Val split ({val_split}): {len(dataset['validation'])} samples")
            print(f"  Batch size: {config.batch_size}")

        return train_loader, val_loader, True


class PyTorchTrainer:
    """PyTorch training implementation."""

    def __init__(self, config: TrainingConfig):
        """Initialize PyTorch trainer."""
        self.config = config
        self.device, self.device_info = get_torch_device(config.device_prefer)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self._hf_dataset_mode = False
        self.scheduler = None

    def setup_model(self):
        """Setup model and move to device."""
        import torch
        import torch.nn as nn
        from models.torch_zoo import get_torch_model

        # Load model (with pretrained weights for fine-tuning)
        self.model, _, input_shape, metadata = get_torch_model(
            self.config.model_name,
            pretrained=True,
            device=self.device
        )
        self.model.eval()  # Start in eval mode
        self.model.to(self.device)

        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()

        if self.config.verbose:
            print(f"Model: {metadata['name']}")
            print(f"  Parameters: {metadata['params']:,}")
            print(f"  Input shape: {input_shape}")

    def setup_optimizer(self):
        """Setup optimizer."""
        import torch.optim as optim

        params = self.model.parameters()

        if self.config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(params, lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(params, lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        scheduler_name = (self.config.scheduler or 'none').lower()
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=0.0
            )
        elif scheduler_name == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def setup_dataloaders(self):
        """Setup data loaders."""
        self.train_loader, self.val_loader, self._hf_dataset_mode = create_dataloaders(self.config)

    def train_epoch(self, epoch: int) -> EpochMetrics:
        """Train for one epoch."""
        import torch

        self.model.train()
        reset_memory_stats('pytorch', self.device)

        # Setup energy tracking
        energy_tracker = None
        nvml_profiler = None

        if self.config.track_energy and self.device.type == 'cuda':
            device_idx = self.device.index if self.device.index is not None else 0
            energy_tracker = EnergyTracker(device_id=device_idx)
            energy_tracker.start()

        if self.config.track_profiling and self.device.type == 'cuda':
            device_idx = self.device.index if self.device.index is not None else 0
            nvml_profiler = NVMLProfiler(device_id=device_idx)
            nvml_profiler.start_sampling()

        # Training loop
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0

        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            if self._hf_dataset_mode:
                images = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
            else:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_samples += labels.size(0)

            # Calculate top-1 and top-5 accuracy
            _, pred = outputs.topk(5, 1, True, True)
            correct = pred.eq(labels.view(-1, 1).expand_as(pred))
            correct_top1 += correct[:, :1].sum().item()
            correct_top5 += correct[:, :5].sum().item()

            # Sample profiling metrics periodically
            if nvml_profiler and batch_idx % 10 == 0:
                nvml_profiler.collect_sample()

            if self.config.verbose and batch_idx % 50 == 0:
                print(f"  Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        # Synchronize before measuring time
        synchronize('pytorch', self.device)
        epoch_duration = time.time() - epoch_start

        # Stop tracking
        energy_stats = None
        profile_stats = None

        if energy_tracker:
            energy_tracker.stop()
            energy_stats = energy_tracker.get_stats()
            energy_tracker.cleanup()

        if nvml_profiler:
            profile_stats = nvml_profiler.stop_sampling()
            nvml_profiler.cleanup()

        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        acc_top1 = 100.0 * correct_top1 / total_samples
        acc_top5 = 100.0 * correct_top5 / total_samples
        samples_per_sec = total_samples / epoch_duration
        peak_memory_mb = get_peak_memory('pytorch', self.device)

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=avg_loss,
            train_accuracy_top1=acc_top1,
            train_accuracy_top5=acc_top5,
            epoch_duration_s=epoch_duration,
            samples_per_sec=samples_per_sec,
            peak_memory_mb=peak_memory_mb
        )

        # Add energy stats
        if energy_stats:
            metrics.energy_j = energy_stats.total_energy_j
            metrics.avg_power_w = energy_stats.avg_power_w

        # Add profiling stats
        if profile_stats:
            metrics.gpu_utilization_pct = profile_stats.gpu_utilization_pct

        return metrics

    def validate(self, epoch: int) -> Tuple[float, float, float]:
        """Validate model on validation set."""
        import torch

        self.model.eval()

        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if self._hf_dataset_mode:
                    images = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                total_samples += labels.size(0)

                # Calculate top-1 and top-5 accuracy
                _, pred = outputs.topk(5, 1, True, True)
                correct = pred.eq(labels.view(-1, 1).expand_as(pred))
                correct_top1 += correct[:, :1].sum().item()
                correct_top5 += correct[:, :5].sum().item()

        avg_loss = total_loss / len(self.val_loader)
        acc_top1 = 100.0 * correct_top1 / total_samples
        acc_top5 = 100.0 * correct_top5 / total_samples

        return avg_loss, acc_top1, acc_top5

        return avg_loss, acc_top1, acc_top5

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        import torch
        
        if not self.config.save_checkpoints:
            return

        checkpoint_path = os.path.join(self.config.session_dir, f"checkpoint_epoch_{epoch}.pt")
        os.makedirs(self.config.session_dir, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, checkpoint_path)
        
        if self.config.verbose:
            print(f"  Saved checkpoint to {checkpoint_path}")

    def train(self) -> List[EpochMetrics]:
        """Run full training loop."""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Training Configuration:")
            print(f"  Framework: {self.config.framework.upper()}")
            print(f"  Model: {self.config.model_name}")
            print(f"  Epochs: {self.config.epochs}")
            print(f"  Batch Size: {self.config.batch_size}")
            print(f"  Learning Rate: {self.config.learning_rate}")
            print(f"  Optimizer: {self.config.optimizer.upper()}")
            print(f"  Device: {self.device_info.name}")
            print(f"{'='*60}\n")

        all_metrics = []

        for epoch in range(1, self.config.epochs + 1):
            if self.config.verbose:
                print(f"\nEpoch {epoch}/{self.config.epochs}")
                print("-" * 60)

            # Train epoch
            metrics = self.train_epoch(epoch)

            # Validate
            if epoch % self.config.eval_frequency == 0:
                val_loss, val_acc1, val_acc5 = self.validate(epoch)
                metrics.val_loss = val_loss
                metrics.val_accuracy_top1 = val_acc1
                metrics.val_accuracy_top5 = val_acc5

                if self.config.verbose:
                    print(f"  Train Loss: {metrics.train_loss:.4f}, "
                          f"Acc@1: {metrics.train_accuracy_top1:.2f}%, "
                          f"Acc@5: {metrics.train_accuracy_top5:.2f}%")
                    print(f"  Val Loss: {val_loss:.4f}, "
                          f"Acc@1: {val_acc1:.2f}%, "
                          f"Acc@5: {val_acc5:.2f}%")
                    print(f"  Time: {metrics.epoch_duration_s:.2f}s, "
                          f"Throughput: {metrics.samples_per_sec:.1f} samples/s")
                    if metrics.energy_j:
                        print(f"  Energy: {metrics.energy_j:.2f}J, "
                              f"Power: {metrics.avg_power_w:.1f}W")

            if self.scheduler:
                self.scheduler.step()
                if self.config.verbose:
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(f"  LR Scheduler -> lr: {current_lr:.6f}")

            if self.config.save_checkpoints:
                self.save_checkpoint(epoch)

            all_metrics.append(metrics)

        return all_metrics


class TrainState(train_state.TrainState):
    """Custom TrainState to include batch statistics."""
    batch_stats: Any


class JAXTrainer:
    """JAX/Flax training implementation."""

    def __init__(self, config: TrainingConfig):
        """Initialize JAX trainer."""
        self.config = config
        self.device, self.device_info = get_jax_device(config.device_prefer)
        self.state = None
        self.train_loader = None
        self.val_loader = None
        self._hf_dataset_mode = False
        self.apply_fn = None
        self.model = None  # Store Flax model instance
        
    def setup_model(self):
        """Setup model and optimizer."""
        from models import jax_flax_zoo
        
        # Initialize RNG
        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        
        # Determine input shape (HWC for JAX)
        input_shape = (224, 224, 3)
        
        # Get model and also store the model instance
        self.apply_fn, params, _, metadata = jax_flax_zoo.get_flax_model(
            self.config.model_name,
            input_shape=input_shape,
            rng_key=init_rng,
            num_classes=1000
        )
        
        # Store the actual model instance for training
        if self.config.model_name == 'resnet50':
            self.model = jax_flax_zoo.ResNet50(num_classes=1000)
        elif self.config.model_name == 'vit_b_16':
            self.model = jax_flax_zoo.VisionTransformer(num_classes=1000)
        elif self.config.model_name == 'mobilenet_v3_small':
            self.model = jax_flax_zoo.MobileNetV3Small(num_classes=1000)
        elif self.config.model_name == 'efficientnet_b0':
            self.model = jax_flax_zoo.EfficientNetB0(num_classes=1000)
        
        if self.config.verbose:
            print(f"Model: {metadata['name']}")
            print(f"  Parameters: {metadata['params']:,}")
            print(f"  Input shape: {input_shape}")
            
        # Setup optimizer
        if self.config.optimizer.lower() == 'adam':
            tx = optax.adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'adamw':
            tx = optax.adamw(learning_rate=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'sgd':
            tx = optax.sgd(learning_rate=self.config.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
        # Setup scheduler if needed
        if self.config.scheduler == 'cosine':
            # We need to wrap the optimizer with a schedule
            # But optax schedules are usually passed as learning_rate
            # Re-creating optimizer with schedule
            schedule = optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=self.config.epochs * 1000, # Approximate, will be updated in setup_dataloaders
                alpha=0.0
            )
            # Note: We'll need to update this after we know the number of steps
            if self.config.optimizer.lower() == 'adam':
                tx = optax.adam(learning_rate=schedule)
            elif self.config.optimizer.lower() == 'adamw':
                tx = optax.adamw(learning_rate=schedule)
            elif self.config.optimizer.lower() == 'sgd':
                tx = optax.sgd(learning_rate=schedule, momentum=0.9)
                
        # Create training state
        # Extract params and batch_stats correctly
        if 'batch_stats' in params:
            batch_stats = params['batch_stats']
            params_only = params['params']
        elif 'params' in params:
            # Some models might have {'params': {...}} structure
            params_only = params['params']
            batch_stats = {}
        else:
            # Direct params dict
            params_only = params
            batch_stats = {}
            
        self.state = TrainState.create(
            apply_fn=self.apply_fn,
            params=params_only,
            tx=tx,
            batch_stats=batch_stats
        )

    def setup_dataloaders(self):
        """Setup data loaders."""
        self.train_loader, self.val_loader, self._hf_dataset_mode = create_dataloaders(self.config)
        
        # Update scheduler if needed (now that we know dataset size)
        if self.config.scheduler == 'cosine':
            steps_per_epoch = len(self.train_loader)
            total_steps = self.config.epochs * steps_per_epoch
            
            schedule = optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=total_steps,
                alpha=0.0
            )
            
            if self.config.optimizer.lower() == 'adam':
                tx = optax.adam(learning_rate=schedule)
            elif self.config.optimizer.lower() == 'adamw':
                tx = optax.adamw(learning_rate=schedule)
            elif self.config.optimizer.lower() == 'sgd':
                tx = optax.sgd(learning_rate=schedule, momentum=0.9)
                
            # Re-create state with new optimizer
            self.state = TrainState.create(
                apply_fn=self.apply_fn,
                params=self.state.params,
                tx=tx,
                batch_stats=self.state.batch_stats
            )

    def train_epoch(self, epoch: int) -> EpochMetrics:
        """Train for one epoch."""
        
        # Store model reference for JIT
        model = self.model
        
        # Define training step
        def train_step_fn(state, images, labels):
            """Training step that handles mutable batch stats."""
            
            def loss_fn(params):
                # Use model.apply directly with mutable support
                # state.params is already extracted correctly in setup_model
                # We need to reconstruct the full variables dict for model.apply
                if state.batch_stats:
                    variables = {'params': params, 'batch_stats': state.batch_stats}
                else:
                    variables = {'params': params}
                    
                output = model.apply(
                    variables,
                    images,
                    train=True,
                    mutable=['batch_stats'] if state.batch_stats else False
                )
                # Unpack output - it returns (logits, mutated_variables) when mutable is used
                if isinstance(output, tuple):
                    logits, updates = output
                else:
                    logits = output
                    updates = {}
                    
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
                return loss, (logits, updates)
            
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (logits, updates)), grads = grad_fn(state.params)
            
            # Update state
            new_state = state.apply_gradients(grads=grads)
            if 'batch_stats' in updates:
                new_state = new_state.replace(batch_stats=updates['batch_stats'])
            
            # Compute accuracy
            accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
            
            # Top-5 accuracy
            top5_preds = jnp.argsort(logits, axis=-1)[:, -5:]
            labels_expanded = jnp.expand_dims(labels, -1)
            acc5 = jnp.mean(jnp.any(top5_preds == labels_expanded, axis=-1))
            
            return new_state, loss, accuracy, acc5
        
        # JIT compile
        train_step = jax.jit(train_step_fn)

        # Setup tracking
        reset_memory_stats('jax')
        energy_tracker = None
        nvml_profiler = None
        
        if self.config.track_energy and self.device_info.device_type == 'cuda':
            device_idx = self.device_info.device_id if self.device_info.device_id is not None else 0
            energy_tracker = EnergyTracker(device_id=device_idx)
            energy_tracker.start()
            
        if self.config.track_profiling and self.device_info.device_type == 'cuda':
            device_idx = self.device_info.device_id if self.device_info.device_id is not None else 0
            nvml_profiler = NVMLProfiler(device_id=device_idx)
            nvml_profiler.start_sampling()
            
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_samples = 0
        steps = 0
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Convert to numpy/JAX
            if self._hf_dataset_mode:
                images = batch['pixel_values'].numpy()
                labels = batch['labels'].numpy()
            else:
                images = batch[0].numpy()
                labels = batch[1].numpy()
                
            # Transpose NCHW -> NHWC for JAX
            images = images.transpose(0, 2, 3, 1)
            
            # Run step
            self.state, loss, acc1, acc5 = train_step(self.state, images, labels)
            
            # Wait for result (async dispatch)
            # We block periodically or at end, but for accurate timing per batch we might want to block?
            # For overall epoch time, we block at end.
            
            # Track metrics
            # We need to pull values to host to accumulate
            loss_val = loss.item()
            acc1_val = acc1.item()
            acc5_val = acc5.item()
            
            total_loss += loss_val
            total_acc1 += acc1_val
            total_acc5 += acc5_val
            total_samples += len(labels)
            steps += 1
            
            if nvml_profiler and batch_idx % 10 == 0:
                nvml_profiler.collect_sample()
                
            if self.config.verbose and batch_idx % 50 == 0:
                print(f"  Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss_val:.4f}")
                      
        # Block until all work is done
        jax.block_until_ready(self.state.params)
        epoch_duration = time.time() - epoch_start
        
        # Stop tracking
        energy_stats = None
        profile_stats = None
        
        if energy_tracker:
            energy_tracker.stop()
            energy_stats = energy_tracker.get_stats()
            energy_tracker.cleanup()
            
        if nvml_profiler:
            profile_stats = nvml_profiler.stop_sampling()
            nvml_profiler.cleanup()
            
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=total_loss / steps,
            train_accuracy_top1=total_acc1 / steps * 100,
            train_accuracy_top5=total_acc5 / steps * 100,
            epoch_duration_s=epoch_duration,
            samples_per_sec=total_samples / epoch_duration,
            peak_memory_mb=0.0 # JAX memory tracking is harder, leaving 0 for now
        )
        
        if energy_stats:
            metrics.energy_j = energy_stats.total_energy_j
            metrics.avg_power_w = energy_stats.avg_power_w
            
        if profile_stats:
            metrics.gpu_utilization_pct = profile_stats.gpu_utilization_pct
            
        return metrics

    def validate(self, epoch: int) -> Tuple[float, float, float]:
        """Validate model."""
        
        @jax.jit
        def eval_step(state, batch):
            images, labels = batch
            logits = state.apply_fn(
                {'params': state.params, 'batch_stats': state.batch_stats},
                images,
                train=False
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            
            accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
            
            top5_preds = jnp.argsort(logits, axis=-1)[:, -5:]
            labels_expanded = jnp.expand_dims(labels, -1)
            acc5 = jnp.mean(jnp.any(top5_preds == labels_expanded, axis=-1))
            
            return loss, accuracy, acc5
            
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        steps = 0
        
        for batch in self.val_loader:
            if self._hf_dataset_mode:
                images = batch['pixel_values'].numpy()
                labels = batch['labels'].numpy()
            else:
                images = batch[0].numpy()
                labels = batch[1].numpy()
                
            images = images.transpose(0, 2, 3, 1)
            
            loss, acc1, acc5 = eval_step(self.state, (images, labels))
            
            total_loss += loss.item()
            total_acc1 += acc1.item()
            total_acc5 += acc5.item()
            steps += 1
            
        return total_loss / steps, (total_acc1 / steps) * 100, (total_acc5 / steps) * 100

        return total_loss / steps, (total_acc1 / steps) * 100, (total_acc5 / steps) * 100

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        if not self.config.save_checkpoints:
            return

        checkpoint_dir = os.path.join(self.config.session_dir, f"checkpoint_epoch_{epoch}")
        # orbax requires absolute paths
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.config.session_dir, exist_ok=True)
        
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # Use the simpler save API without StandardSave
        checkpointer.save(checkpoint_dir, self.state)
        
        if self.config.verbose:
            print(f"  Saved checkpoint to {checkpoint_dir}")

    def train(self) -> List[EpochMetrics]:
        """Run full training loop."""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Training Configuration:")
            print(f"  Framework: {self.config.framework.upper()}")
            print(f"  Model: {self.config.model_name}")
            print(f"  Epochs: {self.config.epochs}")
            print(f"  Batch Size: {self.config.batch_size}")
            print(f"  Learning Rate: {self.config.learning_rate}")
            print(f"  Optimizer: {self.config.optimizer.upper()}")
            print(f"  Device: {self.device_info.name}")
            print(f"{'='*60}\n")

        all_metrics = []

        for epoch in range(1, self.config.epochs + 1):
            if self.config.verbose:
                print(f"\nEpoch {epoch}/{self.config.epochs}")
                print("-" * 60)

            # Train epoch
            metrics = self.train_epoch(epoch)

            # Validate
            if epoch % self.config.eval_frequency == 0:
                val_loss, val_acc1, val_acc5 = self.validate(epoch)
                metrics.val_loss = val_loss
                metrics.val_accuracy_top1 = val_acc1
                metrics.val_accuracy_top5 = val_acc5

                if self.config.verbose:
                    print(f"  Train Loss: {metrics.train_loss:.4f}, "
                          f"Acc@1: {metrics.train_accuracy_top1:.2f}%, "
                          f"Acc@5: {metrics.train_accuracy_top5:.2f}%")
                    print(f"  Val Loss: {val_loss:.4f}, "
                          f"Acc@1: {val_acc1:.2f}%, "
                          f"Acc@5: {val_acc5:.2f}%")
                    print(f"  Time: {metrics.epoch_duration_s:.2f}s, "
                          f"Throughput: {metrics.samples_per_sec:.1f} samples/s")
                    if metrics.energy_j:
                        print(f"  Energy: {metrics.energy_j:.2f}J, "
                              f"Power: {metrics.avg_power_w:.1f}W")

            if self.config.save_checkpoints:
                self.save_checkpoint(epoch)

            all_metrics.append(metrics)

        return all_metrics


def run_training_benchmark(config: TrainingConfig) -> List[EpochMetrics]:
    """
    Run training benchmark with given configuration.

    Args:
        config: TrainingConfig object

    Returns:
        List of EpochMetrics for each epoch
    """
    if config.framework == 'pytorch':
        trainer = PyTorchTrainer(config)
        trainer.setup_model()
        trainer.setup_optimizer()
        trainer.setup_dataloaders()
        return trainer.train()
    elif config.framework == 'jax':
        trainer = JAXTrainer(config)
        trainer.setup_model()  # Setup model first to initialize state
        trainer.setup_dataloaders()  # Then setup dataloaders (may update scheduler)
        return trainer.train()
    else:
        raise ValueError(f"Unknown framework: {config.framework}")


def save_training_results(
    metrics_list: List[EpochMetrics],
    config: TrainingConfig,
    output_dir: str = None # Legacy argument, ignored in favor of config.session_dir
):
    """Save training results to CSV."""
    # Use session dir for output
    output_dir = config.session_dir
    os.makedirs(output_dir, exist_ok=True)

    filename = "training_metrics.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', newline='') as f:
        # Build fieldnames
        fieldnames = [
            'framework', 'model', 'batch_size', 'learning_rate', 'optimizer',
            'epoch', 'train_loss', 'train_acc_top1', 'train_acc_top5',
            'val_loss', 'val_acc_top1', 'val_acc_top5',
            'epoch_duration_s', 'samples_per_sec',
            'energy_j', 'avg_power_w', 'peak_memory_mb', 'gpu_utilization_pct'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for metrics in metrics_list:
            row = {
                'framework': config.framework,
                'model': config.model_name,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'optimizer': config.optimizer,
                'epoch': metrics.epoch,
                'train_loss': metrics.train_loss,
                'train_acc_top1': metrics.train_accuracy_top1,
                'train_acc_top5': metrics.train_accuracy_top5,
                'val_loss': metrics.val_loss,
                'val_acc_top1': metrics.val_accuracy_top1,
                'val_acc_top5': metrics.val_accuracy_top5,
                'epoch_duration_s': metrics.epoch_duration_s,
                'samples_per_sec': metrics.samples_per_sec,
                'energy_j': metrics.energy_j,
                'avg_power_w': metrics.avg_power_w,
                'peak_memory_mb': metrics.peak_memory_mb,
                'gpu_utilization_pct': metrics.gpu_utilization_pct
            }
            writer.writerow(row)

    print(f"\nResults saved to: {filepath}")


def main():
    """Main entry point for training benchmark CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Training benchmark runner")
    parser.add_argument('--framework', type=str, default='pytorch',
                        choices=['pytorch', 'jax'],
                        help='Framework to use')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Model name')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to ImageNet-100 dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'none'],
                        help='LR scheduler')
    parser.add_argument('--output-dir', type=str, default='results/training',
                        help='Output directory for results')
    parser.add_argument('--no-energy', action='store_true',
                        help='Disable energy tracking')
    parser.add_argument('--no-profiling', action='store_true',
                        help='Disable profiling')
    parser.add_argument('--dataset-cache-dir', type=str,
                        default=os.environ.get('HF_DATASETS_CACHE'),
                        help='Optional cache directory for Hugging Face datasets')
    parser.add_argument('--run-id', type=str, default=time.strftime("%Y%m%d_%H%M%S"),
                        help='Run identifier for grouping results')
    parser.add_argument('--save-checkpoints', action='store_true',
                        help='Save model checkpoints')

    args = parser.parse_args()

    config = TrainingConfig(
        framework=args.framework,
        model_name=args.model,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        csv_output_dir=args.output_dir,
        track_energy=not args.no_energy,
        track_profiling=not args.no_profiling,
        dataset_cache_dir=args.dataset_cache_dir,
        run_id=args.run_id,
        save_checkpoints=args.save_checkpoints
    )

    metrics = run_training_benchmark(config)
    save_training_results(metrics, config)


if __name__ == '__main__':
    main()
