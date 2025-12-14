"""
PyTorch Lightning callbacks for LAQ training.

Includes:
- ValidationStrategyCallback: Flexible validation with multiple strategies
- EMACallback: Exponential moving average of model weights
"""

from typing import Optional, List, Dict, Any, Tuple

import torch
from torchvision.utils import make_grid
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from einops import rearrange

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ValidationStrategyCallback(Callback):
    """
    Flexible validation callback that runs multiple validation strategies.
    
    Features:
    - Fixed samples: diverse samples across datasets, same every validation
    - Random samples: different samples each time for diversity
    - Per-bucket visualization: separate grids for YouTube/Bridge etc.
    - Periodic heavy validation: latent transfer, clustering (configurable frequency)
    - Memory efficient: only caches limited samples, not entire val set
    
    Args:
        strategies: List of ValidationStrategy instances
        num_fixed_samples: Number of fixed samples to track (diverse across datasets)
        num_random_samples: Number of random samples to show
        max_cached_samples: Maximum samples to cache (prevents OOM)
    """
    
    def __init__(
        self,
        strategies: Optional[List] = None,
        num_fixed_samples: int = 8,
        num_random_samples: int = 8,
        max_cached_samples: int = 256,  # Limit to prevent OOM
    ):
        super().__init__()
        self.strategies = strategies or []
        self.num_fixed_samples = num_fixed_samples
        self.num_random_samples = num_random_samples
        self.max_cached_samples = max_cached_samples
        
        # Import here to avoid circular imports
        from laq.validation import ValidationCache
        self.cache = ValidationCache()
        
        # Track fixed sample indices (set on first validation with full data)
        self.fixed_indices: Optional[List[int]] = None
        self.validation_count = 0
        self._first_full_validation_done = False
        self._cached_sample_count = 0
    
    def _any_heavy_strategy_running(self) -> bool:
        """Check if any heavy strategy wants to run this validation."""
        for strategy in self.strategies:
            if strategy.needs_caching() and strategy.should_run():
                return True
        return False
    
    def _select_diverse_fixed_samples(
        self,
        all_frames: torch.Tensor,
        all_metadata: List[Dict[str, Any]],
        num_samples: int,
    ) -> Tuple[List[int], torch.Tensor, List[Dict[str, Any]]]:
        """
        Select diverse fixed samples across different datasets/scenes.
        
        Tries to get equal representation from each dataset type.
        """
        if not all_metadata:
            # No metadata - just use random sampling
            indices = torch.randperm(len(all_frames))[:num_samples].tolist()
            return indices, all_frames[indices], [{} for _ in indices]
        
        # Group indices by dataset_type
        by_dataset: Dict[str, List[int]] = {}
        for i, meta in enumerate(all_metadata):
            dtype = meta.get("dataset_type", "unknown")
            if dtype not in by_dataset:
                by_dataset[dtype] = []
            by_dataset[dtype].append(i)
        
        # Sample equally from each dataset type
        selected_indices = []
        dataset_types = list(by_dataset.keys())
        samples_per_dataset = max(1, num_samples // len(dataset_types))
        
        for dtype in dataset_types:
            indices = by_dataset[dtype]
            # Shuffle and take samples
            shuffled = torch.randperm(len(indices)).tolist()
            for j in shuffled[:samples_per_dataset]:
                if len(selected_indices) < num_samples:
                    selected_indices.append(indices[j])
        
        # If we need more samples, add randomly
        remaining_indices = [i for i in range(len(all_frames)) if i not in selected_indices]
        while len(selected_indices) < num_samples and remaining_indices:
            idx = remaining_indices.pop(torch.randint(len(remaining_indices), (1,)).item())
            selected_indices.append(idx)
        
        # Get frames and metadata for selected indices
        selected_frames = all_frames[selected_indices]
        selected_metadata = [all_metadata[i] for i in selected_indices]
        
        return selected_indices, selected_frames, selected_metadata
    
    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Initialize cache at start of validation."""
        self.cache.clear()
        self._cached_sample_count = 0
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Cache validation data with stratified sampling across datasets."""
        # Stop caching if we have enough samples
        if self._cached_sample_count >= self.max_cached_samples:
            return

        # Extract frames and metadata from batch
        if isinstance(batch, dict):
            frames = batch["frames"]
            # Extract ALL per-sample metadata dynamically (not just 4 hardcoded keys)
            batch_metadata = []
            batch_size = frames.shape[0]
            for i in range(batch_size):
                meta = {}
                for key in batch.keys():
                    if key == "frames":
                        continue
                    val = batch[key]
                    
                    # Special handling for 'action' and 'initial_state' which get transposed by default collate
                    # Original: [[dx0, dy0], ...] -> collated: [[dx0, ...], [dy0, ...]]
                    if (key == "action" or key == "initial_state") and isinstance(val, (list, tuple)) and len(val) > 0:
                        if isinstance(val[0], torch.Tensor) and val[0].ndim > 0:
                            # Reconstruct per-sample: [dx_i, dy_i, ...]
                            dims = [v[i].item() for v in val if i < len(v)]
                            meta[key] = dims
                        else:
                            meta[key] = val[i] if i < len(val) else None
                    elif isinstance(val, (list, tuple)):
                        meta[key] = val[i] if i < len(val) else None
                    elif isinstance(val, torch.Tensor):
                        if val.ndim > 0 and i < len(val):
                            # Scalar tensor element
                            meta[key] = val[i].item() if val[i].ndim == 0 else val[i].tolist()
                        elif val.ndim == 0:
                            meta[key] = val.item()
                    else:
                        meta[key] = val
                batch_metadata.append(meta)
        else:
            frames = batch
            batch_metadata = [{} for _ in range(frames.shape[0])]

        # Implement stratified caching: ensure proportional representation
        # Group samples by dataset_type to cache proportionally
        remaining_capacity = self.max_cached_samples - self._cached_sample_count
        if remaining_capacity <= 0:
            return

        samples_to_cache = min(frames.shape[0], remaining_capacity)

        # Check current cache distribution
        current_distribution = {}
        all_meta = self.cache.get_all_metadata()
        for m in all_meta:
            dtype = m.get("dataset_type", "unknown")
            current_distribution[dtype] = current_distribution.get(dtype, 0) + 1

        # Check batch distribution
        batch_distribution = {}
        for m in batch_metadata:
            dtype = m.get("dataset_type", "unknown")
            batch_distribution[dtype] = batch_distribution.get(dtype, 0) + 1

        # If we have multiple dataset types, try to balance
        if len(batch_distribution) > 1 or len(current_distribution) > 1:
            # Use stratified sampling: prioritize underrepresented datasets
            selected_indices = self._stratified_sample_indices(
                batch_metadata, current_distribution, samples_to_cache
            )
        else:
            # Simple case: just take first N samples
            selected_indices = list(range(samples_to_cache))

        # Extract selected samples
        frames = frames[selected_indices].detach().cpu()
        batch_metadata = [batch_metadata[i] for i in selected_indices]

        # Cache frames and metadata
        self.cache.frames.append(frames)
        self.cache.metadata.append(batch_metadata)
        self._cached_sample_count += len(selected_indices)

        # Cache latents and codes if heavy strategies are running
        if self._any_heavy_strategy_running():
            with torch.no_grad():
                device = pl_module.device
                frames_gpu = frames.to(device)

                # Get codebook indices
                indices = pl_module.model(frames_gpu, return_only_codebook_ids=True)
                self.cache.codes.append(indices.cpu())

                # Get quantized latents
                latents = pl_module.model.vq.codebooks[indices]
                self.cache.latents.append(latents.cpu())

    def _stratified_sample_indices(
        self,
        batch_metadata: List[Dict[str, Any]],
        current_distribution: Dict[str, int],
        num_samples: int,
    ) -> List[int]:
        """Select indices to balance dataset representation in cache."""
        # Group batch indices by dataset_type
        by_dataset: Dict[str, List[int]] = {}
        for i, meta in enumerate(batch_metadata):
            dtype = meta.get("dataset_type", "unknown")
            if dtype not in by_dataset:
                by_dataset[dtype] = []
            by_dataset[dtype].append(i)

        # Calculate how many samples we want from each dataset
        # Prioritize datasets that are underrepresented in current cache
        total_current = sum(current_distribution.values()) if current_distribution else 0

        selected_indices = []
        datasets = list(by_dataset.keys())

        if total_current == 0:
            # No samples yet - sample equally from each dataset type
            samples_per_dataset = max(1, num_samples // len(datasets))
            for dtype in datasets:
                indices = by_dataset[dtype]
                n = min(samples_per_dataset, len(indices))
                selected_indices.extend(indices[:n])
        else:
            # Prioritize underrepresented datasets
            # Target equal representation
            target_per_dataset = (total_current + num_samples) // max(len(datasets), len(current_distribution))

            for dtype in datasets:
                current_count = current_distribution.get(dtype, 0)
                needed = max(0, target_per_dataset - current_count)
                available = by_dataset[dtype]
                n = min(needed, len(available), num_samples - len(selected_indices))
                if n > 0:
                    selected_indices.extend(available[:n])

        # If we still need more samples, add randomly
        remaining = num_samples - len(selected_indices)
        if remaining > 0:
            all_indices = set(range(len(batch_metadata)))
            unused = list(all_indices - set(selected_indices))
            selected_indices.extend(unused[:remaining])

        return selected_indices[:num_samples]
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run all validation strategies."""
        self.validation_count += 1
        
        # Update strategy counters
        for strategy in self.strategies:
            strategy.increment_count()
        
        # Get all cached data
        all_frames = self.cache.get_all_frames()
        all_metadata = self.cache.get_all_metadata()
        
        # Select diverse fixed samples on first validation with enough data
        if all_frames is not None and not self._first_full_validation_done:
            if len(all_frames) >= self.num_fixed_samples:
                indices, fixed_frames, fixed_meta = self._select_diverse_fixed_samples(
                    all_frames, all_metadata, self.num_fixed_samples
                )
                self.fixed_indices = indices
                self.cache.fixed_indices = indices
                self.cache.fixed_frames = fixed_frames
                self.cache.fixed_metadata = fixed_meta
                self._first_full_validation_done = True
                print(f"✓ Selected {len(indices)} diverse fixed samples for visualization")
        
        # Run strategies
        for strategy in self.strategies:
            if strategy.should_run():
                try:
                    metrics = strategy.run(self.cache, pl_module, trainer)
                except Exception as e:
                    print(f"Warning: Strategy {strategy.name} failed: {e}")


class EMACallback(Callback):
    """
    Exponential Moving Average callback.

    Maintains EMA of model weights during training (LAPA style).

    Args:
        decay: EMA decay rate (default: 0.999)
        update_every: Update EMA every N steps (default: 1)
        update_after_step: Start EMA updates after N steps (default: 0)
    """

    def __init__(
        self,
        decay: float = 0.999,
        update_every: int = 1,
        update_after_step: int = 0,
    ):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.ema_model = None
        self.num_updates = 0

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Initialize EMA model."""
        # Clone model for EMA
        self.ema_model = type(pl_module.model)(
            **pl_module.model_config
        ).to(pl_module.device)
        self.ema_model.load_state_dict(pl_module.model.state_dict())
        self.ema_model.eval()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Update EMA weights after training step."""
        if trainer.global_step < self.update_after_step:
            return

        if trainer.global_step % self.update_every != 0:
            return

        # Update EMA weights
        self.num_updates += 1
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                pl_module.model.parameters(),
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1 - self.decay
                )

    def state_dict(self):
        """Save EMA state."""
        return {
            "ema_model": self.ema_model.state_dict() if self.ema_model else None,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict):
        """Load EMA state."""
        if state_dict["ema_model"] is not None and self.ema_model is not None:
            self.ema_model.load_state_dict(state_dict["ema_model"])
        self.num_updates = state_dict["num_updates"]
