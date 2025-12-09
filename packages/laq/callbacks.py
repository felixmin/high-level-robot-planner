"""
PyTorch Lightning callbacks for LAQ training.

Includes:
- ReconstructionVisualizationCallback: Logs reconstruction grids to WandB (legacy)
- ValidationStrategyCallback: Flexible validation with multiple strategies
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
        """Cache validation data if any strategy needs it (limited samples)."""
        # Stop caching if we have enough samples
        if self._cached_sample_count >= self.max_cached_samples:
            return
        
        # Extract frames and metadata from batch
        if isinstance(batch, dict):
            frames = batch["frames"]
            # Extract per-sample metadata
            batch_metadata = []
            batch_size = frames.shape[0]
            for i in range(batch_size):
                meta = {}
                for key in ["dataset_type", "scene_id", "video_id", "environment"]:
                    if key in batch:
                        val = batch[key]
                        if isinstance(val, (list, tuple)):
                            meta[key] = val[i] if i < len(val) else None
                        elif isinstance(val, torch.Tensor):
                            meta[key] = val[i].item() if val.ndim > 0 else val.item()
                        else:
                            meta[key] = val
                batch_metadata.append(meta)
        else:
            frames = batch
            batch_metadata = [{} for _ in range(frames.shape[0])]
        
        # Limit how many samples we cache from this batch
        remaining_capacity = self.max_cached_samples - self._cached_sample_count
        if remaining_capacity <= 0:
            return
        
        samples_to_cache = min(frames.shape[0], remaining_capacity)
        frames = frames[:samples_to_cache].detach().cpu()
        batch_metadata = batch_metadata[:samples_to_cache]
        
        # Cache frames and metadata
        self.cache.frames.append(frames)
        self.cache.metadata.append(batch_metadata)
        self._cached_sample_count += samples_to_cache
        
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


class ReconstructionVisualizationCallback(Callback):
    """
    Visualize LAQ reconstructions during training and/or validation.

    Matches LAPA visualization style:
    - Grid with 3 columns: [frame_t, frame_t+offset, reconstruction]
    - Logs to WandB as image

    Args:
        num_samples: Number of samples to visualize (default: 8)
        log_every_n_epochs: Visualization frequency (default: 1)
        visualize_train: Whether to visualize training reconstructions (default: False)
        visualize_val: Whether to visualize validation reconstructions (default: True)
    """

    def __init__(
        self,
        num_samples: int = 8,
        log_every_n_epochs: int = 1,
        visualize_train: bool = False,
        visualize_val: bool = True,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.visualize_train = visualize_train
        self.visualize_val = visualize_val

    def _get_wandb_logger(self, trainer: pl.Trainer):
        """Get WandB logger from trainer if available."""
        if not WANDB_AVAILABLE:
            return None

        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                return logger
        return None

    def _visualize_reconstructions(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_getter_method: str,
        log_key: str,
    ) -> None:
        """
        Common visualization logic for train and val.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            batch_getter_method: Name of method to get batch ('get_validation_batch' or 'get_training_batch')
            log_key: WandB key for logging ('val/reconstructions' or 'train/reconstructions')
        """
        # Check if we should visualize this epoch
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # Get WandB logger
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None:
            return

        # Get batch from module
        batch = getattr(pl_module, batch_getter_method)()
        if batch is None:
            return

        # Limit to num_samples
        batch = batch[:self.num_samples]

        # Generate reconstructions
        recons = pl_module.generate_reconstructions(batch)

        # Create visualization grid
        # Input: [B, C, 2, H, W] where 2 = [frame_t, frame_t+offset]
        # Output: grid with 3 columns per sample [frame_t, frame_t+offset, recons]
        frame_t = batch[:, :, 0]      # [B, C, H, W]
        frame_t_plus = batch[:, :, 1]  # [B, C, H, W]
        recons = recons.cpu()              # [B, C, H, W]

        # Stack: [frame_t, frame_t+offset, recons]
        imgs_and_recons = torch.stack([frame_t, frame_t_plus, recons], dim=0)  # [3, B, C, H, W]
        imgs_and_recons = rearrange(imgs_and_recons, 'r b c h w -> (b r) c h w')  # [B*3, C, H, W]

        # Clamp to valid range
        imgs_and_recons = imgs_and_recons.clamp(0.0, 1.0)

        # Create grid (3 columns per sample)
        grid = make_grid(
            imgs_and_recons,
            nrow=3,
            normalize=False,  # Already normalized
            value_range=(0, 1),
        )

        # Log to WandB
        wandb_logger.log_image(
            key=log_key,
            images=[grid],
            caption=[f"Epoch {trainer.current_epoch}"],
        )

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Called at the end of training epoch.

        Generates and logs training reconstruction visualizations.
        """
        if not self.visualize_train:
            return

        self._visualize_reconstructions(
            trainer=trainer,
            pl_module=pl_module,
            batch_getter_method='get_training_batch',
            log_key='train/reconstructions',
        )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Called at the end of validation epoch.

        Generates and logs validation reconstruction visualizations.
        """
        if not self.visualize_val:
            return

        self._visualize_reconstructions(
            trainer=trainer,
            pl_module=pl_module,
            batch_getter_method='get_validation_batch',
            log_key='val/reconstructions',
        )


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
