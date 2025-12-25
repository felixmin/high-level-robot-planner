"""
Visualization strategies for LAQ validation.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torchvision.utils import make_grid
from einops import rearrange
import lightning.pytorch as pl

from .core import ValidationStrategy, ValidationCache


class BasicVisualizationStrategy(ValidationStrategy):
    """
    Basic reconstruction visualization.

    Shows:
    - Fixed samples: diverse samples across datasets, same every validation
    - Random samples: different samples each time (diversity check)
    - Per-bucket samples: separate grids for each bucket (configurable filters)
    - Training samples: reconstructions from training data

    Buckets can be configured via val_buckets parameter:
    ```yaml
    val_buckets:
      youtube:
        dataset_type: "youtube"
      bridge_toykitchen:
        dataset_type: "bridge"
        environment: "toykitchen1"
      with_language:
        language: ["not_null", true]
    ```
    """

    def __init__(
        self,
        name: str = "basic_visualization",
        enabled: bool = True,
        num_fixed_samples: int = 8,
        num_random_samples: int = 8,
        num_train_samples: int = 8,
        visualize_train: bool = True,
        visualize_val: bool = True,
        visualize_per_bucket: bool = True,
        samples_per_bucket: int = 4,
        val_buckets: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=1,  # Always run
            **kwargs,  # Pass buckets, etc.
        )
        self.num_fixed_samples = num_fixed_samples
        self.num_random_samples = num_random_samples
        self.num_train_samples = num_train_samples
        self.visualize_train = visualize_train
        self.visualize_val = visualize_val
        self.visualize_per_bucket = visualize_per_bucket
        self.samples_per_bucket = samples_per_bucket
        # Default buckets: one per dataset_type if not specified
        self.val_buckets = val_buckets

    def needs_caching(self) -> bool:
        return True  # Need frames for visualization

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate reconstruction visualizations for both train and val."""
        metrics = {}
        wandb_logger = self._get_wandb_logger(trainer)

        # Use bucket name for prefixing if available
        bucket_name = cache.bucket_name or ""
        prefix = f"val/{bucket_name}" if bucket_name else "val"

        # === Training samples visualization ===
        if self.visualize_train and not bucket_name:  # Only for global cache
            self._visualize_training_samples(cache, pl_module, trainer, wandb_logger)

        # === Validation samples visualization ===
        if self.visualize_val:
            self._visualize_validation_samples(cache, pl_module, trainer, wandb_logger, prefix)

        return metrics

    def _visualize_training_samples(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        wandb_logger,
    ) -> None:
        """Visualize training samples by sampling from train dataloader."""
        if wandb_logger is None:
            return

        # Get train samples - either from cache or by sampling from dataloader
        train_frames = cache.train_frames
        train_metadata = cache.train_metadata

        if train_frames is None or len(train_frames) == 0:
            # Sample fresh from train dataloader
            train_frames, train_metadata = self._sample_from_train_dataloader(
                trainer, self.num_train_samples
            )
            if train_frames is not None:
                # Cache for bucket visualization
                cache.train_frames = train_frames
                cache.train_metadata = train_metadata

        if train_frames is None or len(train_frames) == 0:
            return

        # Create and log training reconstruction grid
        train_grid = self._create_recon_grid(train_frames, pl_module)
        if train_grid is not None:
            wandb_logger.log_image(
                key="train/reconstructions",
                images=[train_grid],
                caption=[f"Step {trainer.global_step} (training samples)"],
            )

        # === Per-bucket training visualization ===
        if self.visualize_per_bucket and train_metadata:
            self._visualize_buckets(
                train_frames, train_metadata, cache, pl_module, trainer,
                wandb_logger, prefix="train"
            )

    def _visualize_validation_samples(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        wandb_logger,
        prefix: str = "val",
    ) -> None:
        """Visualize validation samples from cache."""
        all_frames = cache.get_all_frames()
        all_metadata = cache.get_all_metadata()

        if all_frames is None or len(all_frames) == 0:
            return

        # Log cache distribution for debugging
        distribution = cache.get_dataset_distribution()
        bucket_name = cache.bucket_name or "global"
        if distribution:
            print(f"  [{bucket_name}] Cached validation samples per datasource: {distribution}")
            print(f"  [{bucket_name}] Total cached validation samples: {sum(distribution.values())}")

        # === Fixed samples (diverse across datasets) ===
        if cache.fixed_frames is not None and len(cache.fixed_frames) > 0:
            fixed_grid = self._create_recon_grid(cache.fixed_frames, pl_module)
            if wandb_logger and fixed_grid is not None:
                wandb_logger.log_image(
                    key=f"{prefix}/fixed_reconstructions",
                    images=[fixed_grid],
                    caption=[f"Step {trainer.global_step} (fixed diverse samples)"],
                )

        # === Random samples (different each time) ===
        n_random = min(self.num_random_samples, len(all_frames))
        if n_random > 0 and wandb_logger:
            random_indices = torch.randperm(len(all_frames))[:n_random]
            random_frames = all_frames[random_indices]
            random_grid = self._create_recon_grid(random_frames, pl_module)
            if random_grid is not None:
                wandb_logger.log_image(
                    key=f"{prefix}/random_reconstructions",
                    images=[random_grid],
                    caption=[f"Step {trainer.global_step} (random samples)"],
                )

        # === Per-bucket visualization ===
        if self.visualize_per_bucket and all_metadata:
            self._visualize_buckets(
                all_frames, all_metadata, cache, pl_module, trainer,
                wandb_logger, prefix="val"
            )

    def _visualize_buckets(
        self,
        all_frames: torch.Tensor,
        all_metadata: List[Dict[str, Any]],
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        wandb_logger,
        prefix: str = "val",
    ) -> None:
        """Visualize samples grouped by buckets using side dataloaders or cache."""
        if wandb_logger is None:
            return

        # Check if DataModule supports side dataloaders
        datamodule = getattr(trainer, "datamodule", None)
        use_dataloaders = datamodule is not None

        # Use configured buckets or fall back to dataset_type-based buckets
        if self.val_buckets:
            # Use config-defined buckets
            for bucket_name, filters in self.val_buckets.items():
                bucket_frames = None
                
                # Try to get data from side dataloader first (Targeted Evaluation)
                if use_dataloaders:
                    try:
                        # Determine if we need train or val loader
                        if prefix == "train" and hasattr(datamodule, "train_bucket_dataloader"):
                            loader = datamodule.train_bucket_dataloader(bucket_name)
                        elif prefix == "val" and hasattr(datamodule, "val_bucket_dataloader"):
                            loader = datamodule.val_bucket_dataloader(bucket_name)
                        else:
                            loader = None
                            
                        if loader:
                            # Fetch one batch
                            batch = next(iter(loader))
                            if isinstance(batch, dict):
                                bucket_frames = batch["frames"]
                            else:
                                bucket_frames = batch
                            # Move to device/cpu as needed (viz expects CPU usually)
                            bucket_frames = bucket_frames[:self.samples_per_bucket].detach().cpu()
                    except (ValueError, StopIteration):
                        # Dataloader might not exist for this bucket or be empty
                        pass

                # Fallback to cache filtering if dataloader failed
                if bucket_frames is None:
                    bucket_frames, _ = cache.get_frames_by_filter(
                        filters, frames=all_frames, metadata=all_metadata
                    )

                if bucket_frames is not None and len(bucket_frames) > 0:
                    # Randomly sample if we have more than needed (and came from cache)
                    if len(bucket_frames) > self.samples_per_bucket:
                        indices = torch.randperm(len(bucket_frames))[:self.samples_per_bucket]
                        samples = bucket_frames[indices]
                    else:
                        samples = bucket_frames

                    bucket_grid = self._create_recon_grid(samples, pl_module)
                    if bucket_grid is not None:
                        wandb_logger.log_image(
                            key=f"{prefix}/reconstructions_{bucket_name}",
                            images=[bucket_grid],
                            caption=[f"Step {trainer.global_step} ({bucket_name})"],
                        )
        else:
            # Fall back to auto-bucketing by dataset_type (Legacy/Default)
            dataset_types = set()
            for meta in all_metadata:
                dname = meta.get("dataset_name")
                if not dname:
                    dname = meta.get("dataset_type")
                if dname:
                    dataset_types.add(dname)

            for dname in dataset_types:
                bucket_frames, _ = cache.get_frames_by_filter(
                    {"dataset_name": dname}, frames=all_frames, metadata=all_metadata
                )
                if bucket_frames is not None and len(bucket_frames) > 0:
                    n_samples = min(self.samples_per_bucket, len(bucket_frames))
                    indices = torch.randperm(len(bucket_frames))[:n_samples]
                    samples = bucket_frames[indices]

                    bucket_grid = self._create_recon_grid(samples, pl_module)
                    if bucket_grid is not None:
                        wandb_logger.log_image(
                            key=f"{prefix}/reconstructions_{dname}",
                            images=[bucket_grid],
                            caption=[f"Step {trainer.global_step} ({dname})"],
                        )

    def _sample_from_train_dataloader(
        self,
        trainer: pl.Trainer,
        num_samples: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[List[Dict[str, Any]]]]:
        """Sample frames from training dataloader."""
        try:
            train_dataloader = trainer.train_dataloader
            if train_dataloader is None:
                return None, None

            # Get a batch from train dataloader
            frames_list = []
            metadata_list = []
            samples_collected = 0

            for batch in train_dataloader:
                if isinstance(batch, dict):
                    frames = batch["frames"]
                    # Extract metadata for each sample
                    batch_size = frames.shape[0]
                    for i in range(batch_size):
                        meta = {}
                        for key in batch.keys():
                            if key == "frames":
                                continue
                            val = batch[key]
                            if isinstance(val, (list, tuple)) and i < len(val):
                                meta[key] = val[i]
                            elif isinstance(val, torch.Tensor) and val.ndim > 0 and i < len(val):
                                meta[key] = val[i].item() if val[i].ndim == 0 else val[i]
                        metadata_list.append(meta)
                else:
                    frames = batch
                    batch_size = frames.shape[0]
                    metadata_list.extend([{} for _ in range(batch_size)])

                frames_list.append(frames.detach().cpu())
                samples_collected += frames.shape[0]

                if samples_collected >= num_samples:
                    break

            if not frames_list:
                return None, None

            all_frames = torch.cat(frames_list, dim=0)[:num_samples]
            all_metadata = metadata_list[:num_samples]

            return all_frames, all_metadata

        except Exception as e:
            print(f"Warning: Could not sample from train dataloader: {e}")
            return None, None

    def _create_recon_grid(
        self,
        frames: torch.Tensor,
        pl_module: pl.LightningModule,
    ) -> Optional[torch.Tensor]:
        """Create reconstruction grid."""
        if len(frames) == 0:
            return None

        # Generate reconstructions
        pl_module.eval()
        with torch.no_grad():
            recons = pl_module.model(
                frames.to(pl_module.device),
                return_recons_only=True,
            )
        pl_module.train()

        # Create grid: [frame_t, frame_t+offset, reconstruction]
        frame_t = frames[:, :, 0].cpu()
        frame_t_plus = frames[:, :, 1].cpu()
        recons = recons.cpu()

        # Stack and rearrange
        imgs = torch.stack([frame_t, frame_t_plus, recons], dim=0)
        imgs = rearrange(imgs, 'r b c h w -> (b r) c h w')
        imgs = imgs.clamp(0.0, 1.0)

        return make_grid(imgs, nrow=3, normalize=False)
