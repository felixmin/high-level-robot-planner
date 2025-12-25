"""
Validation strategies for LAQ training.

Implements flexible, configurable validation that can run:
- Light validation (always): reconstruction loss + visualizations
- Heavy validation (periodic): latent transfer analysis, clustering

Architecture (Composition Pattern):
- Buckets: Named data subsets with filters (e.g., "language_table", "bridge")
- Strategies: Self-contained validation instances with embedded bucket bindings
- Each strategy declares its metadata requirements and minimum sample counts
- Use 'type' field to create multiple instances of the same strategy type

Usage in config:
```yaml
validation:
  buckets:
    language_table:
      filters: {dataset_name: "language_table"}
      max_samples: 200
    bridge:
      filters: {dataset_name: "bridge"}
      max_samples: 200

  strategies:
    # Basic visualization (runs on global cache)
    basic:
      enabled: true
      visualize_train: true

    # Multiple instances of same strategy type with different buckets
    transfer_lt:
      type: latent_transfer
      buckets: ["language_table"]
      every_n_validations: 5

    transfer_bridge:
      type: latent_transfer
      buckets: ["bridge"]
      every_n_validations: 5

    # Action scatter only on language_table (has 2D actions)
    action_scatter_lt:
      type: action_token_scatter
      buckets: ["language_table"]
```

Key features:
- Instance names (e.g., "transfer_lt") become metric names in wandb
- 'type' field maps to strategy class (see STRATEGY_REGISTRY)
- 'buckets' list specifies which bucket caches to use
- Metadata is pruned to ESSENTIAL_METADATA_KEYS for RAM safety
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
from einops import rearrange
import lightning.pytorch as pl

# Configure matplotlib backend for headless/multi-threaded environments
# Must be done before importing pyplot anywhere
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Essential metadata keys to cache (RAM safety - prevents Bridge metadata bloat)
# Only these keys are retained when caching samples to buckets
ESSENTIAL_METADATA_KEYS = frozenset({
    "dataset_type",    # For bucketing/labeling (e.g., "youtube", "bridge", "oxe")
    "dataset_name",    # For OXE dataset filtering (e.g., "language_table", "bridge")
    "action",          # For action scatter strategies (only first 2 dims used)
    "initial_state",   # For state scatter strategies (only first 2 dims used)
})


def prune_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Prune metadata to essential keys only (RAM safety)."""
    return {k: v for k, v in metadata.items() if k in ESSENTIAL_METADATA_KEYS}


@dataclass
class BucketConfig:
    """Configuration for a validation data bucket."""
    name: str
    filters: Dict[str, Any] = field(default_factory=dict)
    max_samples: int = 100
    is_holdout: bool = False  # True if this is OOD/distribution shift data

    def matches(self, metadata: Dict[str, Any]) -> bool:
        """Check if sample metadata matches this bucket's filters."""
        if not self.filters:
            return True
        return _matches_filters(metadata, self.filters)


def _matches_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Check if metadata matches all filter criteria."""
    if not filters:
        return True

    for key, condition in filters.items():
        value = meta.get(key)

        # Handle operator-based conditions like ["!=", "static"] or [">", 10]
        if isinstance(condition, (list, tuple)) and len(condition) == 2:
            op, target = condition
            if op == "!=":
                if value == target:
                    return False
            elif op == "==":
                if value != target:
                    return False
            elif op == ">":
                if value is None or value <= target:
                    return False
            elif op == "<":
                if value is None or value >= target:
                    return False
            elif op == "in":
                if value not in target:
                    return False
            elif op == "not_null":
                if value is None:
                    return False
        # Handle list of allowed values
        elif isinstance(condition, (list, tuple)):
            if value not in condition:
                return False
        # Handle exact match
        else:
            if value != condition:
                return False

    return True


@dataclass
class ValidationCache:
    """Cache for validation data across batches."""
    frames: List[torch.Tensor] = field(default_factory=list)
    latents: List[torch.Tensor] = field(default_factory=list)
    codes: List[torch.Tensor] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)

    # Metadata for each sample (dataset_type, scene_id, etc.)
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    # Fixed samples for consistent visualization (set once, reused)
    fixed_frames: Optional[torch.Tensor] = None
    fixed_indices: Optional[List[int]] = None
    fixed_metadata: Optional[List[Dict[str, Any]]] = None

    # Training samples (cached separately)
    train_frames: Optional[torch.Tensor] = None
    train_metadata: Optional[List[Dict[str, Any]]] = None

    # Bucket info (set when this cache belongs to a bucket)
    bucket_name: Optional[str] = None
    is_holdout: bool = False

    # Sample count tracking
    _sample_count: int = 0
    max_samples: int = 256

    def clear(self):
        """Clear all cached data (but keep fixed samples and train samples)."""
        self.frames.clear()
        self.latents.clear()
        self.codes.clear()
        self.losses.clear()
        self.metadata.clear()
        self._sample_count = 0

    def is_full(self) -> bool:
        """Check if cache has reached max_samples."""
        return self._sample_count >= self.max_samples

    def add_sample(
        self,
        frame: torch.Tensor,
        meta: Dict[str, Any],
        code: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        prune: bool = True,
    ):
        """Add a single sample to the cache.

        Args:
            frame: Frame tensor to cache
            meta: Metadata dictionary
            code: Optional codebook indices
            latent: Optional latent representation
            prune: If True, prune metadata to essential keys only (RAM safety)
        """
        if self.is_full():
            return

        self.frames.append(frame.cpu() if frame.is_cuda else frame)
        cached_meta = prune_metadata(meta) if prune else meta
        self.metadata.append([cached_meta])  # Always store as list for consistency with add_batch
        self._sample_count += 1

        if code is not None:
            self.codes.append(code.cpu() if code.is_cuda else code)
        if latent is not None:
            self.latents.append(latent.cpu() if latent.is_cuda else latent)

    def add_batch(
        self,
        frames: torch.Tensor,
        metadata_list: List[Dict[str, Any]],
        codes: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        prune: bool = True,
    ):
        """Add a batch of samples to the cache, respecting max_samples.

        Args:
            frames: Batch of frame tensors
            metadata_list: List of metadata dictionaries
            codes: Optional batch of codebook indices
            latents: Optional batch of latent representations
            prune: If True, prune metadata to essential keys only (RAM safety)
        """
        remaining = self.max_samples - self._sample_count
        if remaining <= 0:
            return

        n_to_add = min(frames.shape[0], remaining)

        self.frames.append(frames[:n_to_add].cpu() if frames.is_cuda else frames[:n_to_add])
        if prune:
            cached_metadata = [prune_metadata(m) for m in metadata_list[:n_to_add]]
        else:
            cached_metadata = metadata_list[:n_to_add]
        self.metadata.append(cached_metadata)
        self._sample_count += n_to_add

        if codes is not None:
            self.codes.append(codes[:n_to_add].cpu() if codes.is_cuda else codes[:n_to_add])
        if latents is not None:
            self.latents.append(latents[:n_to_add].cpu() if latents.is_cuda else latents[:n_to_add])

    def sample_count(self) -> int:
        """Return current number of samples in cache."""
        return self._sample_count

    def get_all_frames(self) -> Optional[torch.Tensor]:
        """Concatenate all cached frames."""
        if not self.frames:
            return None
        return torch.cat(self.frames, dim=0)

    def get_all_latents(self) -> Optional[torch.Tensor]:
        """Concatenate all cached latents."""
        if not self.latents:
            return None
        return torch.cat(self.latents, dim=0)

    def get_all_codes(self) -> Optional[torch.Tensor]:
        """Concatenate all cached codes."""
        if not self.codes:
            return None
        return torch.cat(self.codes, dim=0)

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Flatten all metadata lists."""
        result = []
        for meta_batch in self.metadata:
            if isinstance(meta_batch, list):
                result.extend(meta_batch)
            else:
                result.append(meta_batch)
        return result

    def get_frames_by_filter(
        self,
        filters: Dict[str, Any],
        frames: Optional[torch.Tensor] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[torch.Tensor], List[Dict[str, Any]]]:
        """
        Get frames and metadata matching filter criteria.

        Args:
            filters: Dict of {key: value} or {key: [values]} to match.
                     Supports operators like ["!=", value] or [">", value].
            frames: Optional frames to filter (uses cached if None)
            metadata: Optional metadata to filter (uses cached if None)

        Returns:
            Tuple of (filtered_frames, filtered_metadata)
        """
        if frames is None:
            frames = self.get_all_frames()
        if metadata is None:
            metadata = self.get_all_metadata()

        if frames is None or not metadata:
            return None, []

        # Find matching indices
        indices = []
        for i, meta in enumerate(metadata):
            if _matches_filters(meta, filters):
                indices.append(i)

        if not indices:
            return None, []

        filtered_frames = frames[indices]
        filtered_metadata = [metadata[i] for i in indices]
        return filtered_frames, filtered_metadata

    def count_samples_with_metadata(self, required_keys: List[str]) -> int:
        """Count samples that have all required metadata keys with non-None values."""
        all_metadata = self.get_all_metadata()
        count = 0
        for meta in all_metadata:
            has_all = True
            for key in required_keys:
                val = meta.get(key)
                if val is None:
                    has_all = False
                    break
                # Check for 2D+ actions (need at least 2 dims for scatter plots)
                if key == "action" and isinstance(val, (list, tuple)) and len(val) < 2:
                    has_all = False
                    break
            if has_all:
                count += 1
        return count

    def get_frames_by_dataset_type(self, dataset_type: str) -> Optional[torch.Tensor]:
        """Get frames filtered by dataset type (convenience method)."""
        frames, _ = self.get_frames_by_filter({"dataset_type": dataset_type})
        return frames

    def get_dataset_distribution(self) -> Dict[str, int]:
        """Get count of samples per dataset type."""
        all_metadata = self.get_all_metadata()
        distribution: Dict[str, int] = {}
        for meta in all_metadata:
            dtype = meta.get("dataset_type", "unknown")
            distribution[dtype] = distribution.get(dtype, 0) + 1
        return distribution


class ValidationStrategy(ABC):
    """
    Base class for validation strategies.

    Each strategy decides:
    - When to run (via should_run)
    - What data it needs (via required_metadata, min_samples)
    - What to compute and log (via run)

    Strategies are self-contained with bucket bindings:
    - buckets: List of bucket names this strategy operates on

    Example config:
        transfer_lt:
            type: latent_transfer
            buckets: ["language_table"]
            every_n_validations: 2
    """

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        every_n_validations: int = 1,
        min_samples: int = 10,
        buckets: Optional[List[str]] = None,
        **kwargs,
    ):
        self.name = name
        self.enabled = enabled
        self.every_n_validations = every_n_validations
        self.min_samples = min_samples
        self.buckets = buckets or []  # Empty = use global cache
        self.validation_count = 0

    def should_run(self) -> bool:
        """Check if this strategy should run on current validation."""
        if not self.enabled:
            return False
        return (self.validation_count % self.every_n_validations) == 0

    def increment_count(self):
        """Increment validation counter."""
        self.validation_count += 1

    def required_metadata(self) -> List[str]:
        """
        Return list of metadata keys this strategy requires.

        Override in subclasses to declare requirements.
        Empty list means strategy works with any data.
        """
        return []

    def can_run(self, cache: ValidationCache) -> Tuple[bool, str]:
        """
        Check if strategy has sufficient applicable data in cache.

        Returns:
            Tuple of (can_run, reason_if_not)
        """
        # Check minimum sample count
        sample_count = cache.sample_count()
        if sample_count < self.min_samples:
            return False, f"Only {sample_count} samples (need {self.min_samples})"

        # Check required metadata
        required = self.required_metadata()
        if required:
            count_with_meta = cache.count_samples_with_metadata(required)
            if count_with_meta < self.min_samples:
                return False, f"Only {count_with_meta} samples with {required} (need {self.min_samples})"

        return True, ""

    @abstractmethod
    def needs_caching(self) -> bool:
        """Return True if this strategy needs data cached during validation."""
        pass

    def needs_codes(self) -> bool:
        """Return True if this strategy needs codebook indices cached."""
        return False

    @abstractmethod
    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """
        Run the validation strategy.

        Args:
            cache: Cached validation data
            pl_module: The Lightning module
            trainer: The trainer
            metric_suffix: Suffix for metric names (e.g., "_bridge_holdout" for bucket-specific logging)

        Returns:
            Dict of metrics to log
        """
        pass

    def _get_wandb_logger(self, trainer: pl.Trainer):
        """Get WandB logger from trainer."""
        if not WANDB_AVAILABLE:
            return None
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                return logger
        return None




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
                if isinstance(meta, dict) and "dataset_type" in meta:
                    dtype = meta.get("dataset_type")
                    if dtype:
                        dataset_types.add(dtype)

            for dtype in dataset_types:
                bucket_frames, _ = cache.get_frames_by_filter(
                    {"dataset_type": dtype}, frames=all_frames, metadata=all_metadata
                )
                if bucket_frames is not None and len(bucket_frames) > 0:
                    n_samples = min(self.samples_per_bucket, len(bucket_frames))
                    indices = torch.randperm(len(bucket_frames))[:n_samples]
                    samples = bucket_frames[indices]

                    bucket_grid = self._create_recon_grid(samples, pl_module)
                    if bucket_grid is not None:
                        wandb_logger.log_image(
                            key=f"{prefix}/reconstructions_{dtype}",
                            images=[bucket_grid],
                            caption=[f"Step {trainer.global_step} ({dtype})"],
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


class LatentTransferStrategy(ValidationStrategy):
    """
    Test if latent actions transfer between different scenes.

    For pairs (s_a, s_a') and (s_b, s_b'):
    1. Encode z_a = E(s_a, s_a')
    2. Apply z_a to s_b: s_b'_pred = D(s_b, z_a)
    3. Compare s_b'_pred with actual s_b' (should be different)
       and with s_a' (should be similar in "action" applied)

    This measures how "action-like" vs "state-specific" the latents are.
    Useful for comparing IID vs holdout buckets to see if actions generalize.
    """

    def __init__(
        self,
        name: str = "latent_transfer",
        enabled: bool = True,
        every_n_validations: int = 10,
        num_pairs: int = 256,
        min_samples: int = 4,  # Need at least 4 samples for 2 pairs
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )
        self.num_pairs = num_pairs

    def needs_caching(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Run latent transfer analysis."""
        metrics = {}

        all_frames = cache.get_all_frames()
        if all_frames is None or len(all_frames) < 4:
            return metrics

        # Sample pairs
        n = min(self.num_pairs, len(all_frames) // 2)
        indices = torch.randperm(len(all_frames))[:n * 2]

        # Split into source and target pairs
        source_frames = all_frames[indices[:n]]  # (s_a, s_a')
        target_frames = all_frames[indices[n:]]  # (s_b, s_b')

        pl_module.eval()
        with torch.no_grad():
            device = pl_module.device
            source_frames = source_frames.to(device)
            target_frames = target_frames.to(device)

            # Encode source pairs to get latent actions
            # Use task helper which handles raw pixels -> quantized latents
            source_latents, source_indices = pl_module.encode_latents(source_frames)  # z_a

            # Get target initial frames
            target_s0 = target_frames[:, :, 0:1]  # s_b (keep dim for concat)

            # Also encode target frames to get their true indices (for comparison)
            _, target_indices = pl_module.encode_latents(target_frames)

            # Decode: apply source latent to target initial frame
            # Use task helper which handles embedding and reshaping
            transferred_recons = pl_module.decode_with_latents(
                target_s0,
                source_latents,
            )  # s_b'_pred

            # Remove time dim if present [B, C, 1, H, W] -> [B, C, H, W]
            if transferred_recons.ndim == 5:
                transferred_recons = transferred_recons.squeeze(2)

            # Get ground truth
            target_s1_true = target_frames[:, :, 1]  # s_b' (true)
            source_s1_true = source_frames[:, :, 1]  # s_a' (true)

            # Compute transfer error (pred vs true target)
            transfer_mse = F.mse_loss(transferred_recons, target_s1_true)

            # Compute self-reconstruction error (for reference)
            self_recons = pl_module.model(target_frames, return_recons_only=True)
            self_mse = F.mse_loss(self_recons, target_s1_true)

        pl_module.train()

        # Use metric_suffix for bucket-specific logging
        metrics[f"val/latent_transfer_mse{metric_suffix}"] = transfer_mse.item()
        metrics[f"val/self_recon_mse{metric_suffix}"] = self_mse.item()
        metrics[f"val/transfer_ratio{metric_suffix}"] = transfer_mse.item() / (self_mse.item() + 1e-8)

        # Log to trainer
        pl_module.log_dict(metrics, sync_dist=True)
        
        # Visualize some transfers
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._visualize_transfers(
                source_frames[:4].cpu(),
                target_frames[:4].cpu(),
                transferred_recons[:4].cpu(),
                self_recons[:4].cpu(),  # True reconstruction for comparison
                source_indices[:4].cpu(),
                target_indices[:4].cpu(),
                wandb_logger,
                trainer.global_step,
            )

        return metrics
    
    def _visualize_transfers(
        self,
        source_frames: torch.Tensor,
        target_frames: torch.Tensor,
        transferred: torch.Tensor,
        true_recon: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        wandb_logger,
        global_step: int,
    ):
        """
        Visualize latent transfer results with true reconstruction comparison.

        Grid columns:
        1. s_a: Source first frame
        2. s_a': Source second frame (ground truth action result)
        3. s_b: Target first frame
        4. D(s_b, z_a): Transfer reconstruction (using source's latent)
        5. D(s_b, z_b): True reconstruction (using target's own latent)
        6. s_b': Target second frame (ground truth)
        """
        s_a = source_frames[:, :, 0]          # Source first frame
        s_a_prime = source_frames[:, :, 1]    # Source action result (GT)
        s_b = target_frames[:, :, 0]          # Target first frame
        s_b_recon_true = true_recon           # D(s_b, z_b) - true recon
        s_b_recon_transfer = transferred      # D(s_b, z_a) - transfer recon
        s_b_prime_true = target_frames[:, :, 1]  # Target GT

        num_samples = len(s_a)
        if num_samples == 0:
            return

        # Create figure
        # 6 columns of images
        fig, axes = plt.subplots(num_samples, 6, figsize=(20, 3.5 * num_samples), squeeze=False)
        
        # Column titles
        col_titles = ["s_a", "s_a'", "s_b", "D(s_b, z_a)\n(Transfer)", "D(s_b, z_b)\n(Self)", "s_b'"]
        
        for i in range(num_samples):
            # Row images (swapped 4 and 5 as requested)
            # 4: Transfer (z_a applied to s_b)
            # 5: Self (z_b applied to s_b)
            imgs = [
                s_a[i], 
                s_a_prime[i], 
                s_b[i], 
                s_b_recon_transfer[i], 
                s_b_recon_true[i], 
                s_b_prime_true[i]
            ]
            
            # Prepare token string
            tokens_a = str(source_indices[i].tolist())
            tokens_b = str(target_indices[i].tolist())
            
            row_text = f"Row {i}\nDet(a): {tokens_a}\nDet(b): {tokens_b}\nApp(b): {tokens_a}"
            
            for j, img in enumerate(imgs):
                ax = axes[i, j]
                
                # Convert [C, H, W] to [H, W, C] for imshow and clamp
                img_np = img.permute(1, 2, 0).clamp(0.0, 1.0).numpy()
                
                ax.imshow(img_np)
                ax.axis('off')
                
                if i == 0:
                    ax.set_title(col_titles[j], fontsize=12)
                
                if j == 0:
                    # Add text to the left of the first image of the row
                    ax.text(-0.2, 0.5, row_text, transform=ax.transAxes, 
                            va='center', ha='right', fontsize=10, rotation=0, family='monospace')

        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        
        wandb_logger.log_image(
            key="val/latent_transfer",
            images=[img],
            caption=[f"Step {global_step}"],
        )
        plt.close(fig)


class ClusteringStrategy(ValidationStrategy):
    """
    Analyze latent action distribution via clustering.

    - Collect latent codes from validation set
    - Run k-means clustering
    - Log cluster statistics and example frame pairs per cluster
    """

    def __init__(
        self,
        name: str = "clustering",
        enabled: bool = True,
        every_n_validations: int = 20,
        num_samples: int = 1000,
        num_clusters: int = 16,
        num_examples_per_cluster: int = 4,
        min_samples: int = 16,  # Need at least num_clusters samples
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.num_examples_per_cluster = num_examples_per_cluster

    def needs_caching(self) -> bool:
        return True

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Run clustering analysis."""
        metrics = {}

        all_frames = cache.get_all_frames()
        all_codes = cache.get_all_codes()

        if all_codes is None or len(all_codes) < self.num_clusters:
            return metrics

        # Limit samples
        n = min(self.num_samples, len(all_codes))
        indices = torch.randperm(len(all_codes))[:n]
        codes = all_codes[indices]
        frames = all_frames[indices] if all_frames is not None else None

        # Flatten codes for clustering (if multi-token)
        codes_flat = codes.reshape(len(codes), -1).float()

        # K-means clustering
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(codes_flat.cpu().numpy())
            cluster_labels = torch.from_numpy(cluster_labels)

            # Compute cluster sizes
            cluster_sizes = torch.bincount(cluster_labels, minlength=self.num_clusters)

            # Log metrics with suffix
            metrics[f"val/cluster_entropy{metric_suffix}"] = self._entropy(cluster_sizes.float())
            metrics[f"val/num_empty_clusters{metric_suffix}"] = (cluster_sizes == 0).sum().item()
            metrics[f"val/max_cluster_size{metric_suffix}"] = cluster_sizes.max().item()
            metrics[f"val/min_cluster_size{metric_suffix}"] = cluster_sizes[cluster_sizes > 0].min().item() if (cluster_sizes > 0).any() else 0

            pl_module.log_dict(metrics, sync_dist=True)
            
            # Visualize examples from each cluster
            if frames is not None:
                wandb_logger = self._get_wandb_logger(trainer)
                if wandb_logger is not None:
                    self._visualize_clusters(
                        frames, cluster_labels, wandb_logger, trainer.current_epoch
                    )
            
        except ImportError:
            print("sklearn not available for clustering analysis")
        
        return metrics
    
    def _entropy(self, counts: torch.Tensor) -> float:
        """Compute entropy of cluster distribution."""
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -(probs * probs.log()).sum().item()
    

    
    def _visualize_clusters(
        self,
        frames: torch.Tensor,
        cluster_labels: torch.Tensor,
        wandb_logger,
        epoch: int,
    ):
        """Visualize example frame pairs from each cluster."""
        grids = []
        
        for cluster_id in range(self.num_clusters):
            mask = cluster_labels == cluster_id
            cluster_frames = frames[mask]
            
            if len(cluster_frames) == 0:
                continue
            
            # Get examples
            n_examples = min(self.num_examples_per_cluster, len(cluster_frames))
            examples = cluster_frames[:n_examples]
            
            # Create mini-grid: [frame_t, frame_t+offset]
            frame_t = examples[:, :, 0]
            frame_t_plus = examples[:, :, 1]
            
            imgs = torch.stack([frame_t, frame_t_plus], dim=1)
            imgs = rearrange(imgs, 'b r c h w -> (b r) c h w')
            imgs = imgs.clamp(0.0, 1.0)
            
            grid = make_grid(imgs, nrow=2, normalize=False, padding=2)
            grids.append(grid)
        
        if grids:
            # Combine all cluster grids
            combined = torch.cat(grids, dim=2)  # Horizontal concat
            
            wandb_logger.log_image(
                key="val/cluster_examples",
                images=[combined],
                caption=[f"Epoch {epoch}: Examples from {len(grids)} clusters"],
            )


class CodebookHistogramStrategy(ValidationStrategy):
    """
    Visualize codebook usage distribution as a histogram.

    Shows which codebook entries are used most/least frequently,
    helping identify if the codebook is being utilized effectively.
    """

    def __init__(
        self,
        name: str = "codebook_histogram",
        enabled: bool = True,
        every_n_validations: int = 1,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )

    def needs_caching(self) -> bool:
        return True  # Need codes for histogram

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate codebook usage histogram."""
        metrics = {}

        all_codes = cache.get_all_codes()
        if all_codes is None or len(all_codes) == 0:
            return metrics

        # Flatten codes to get all codebook indices used
        # all_codes shape: [N, code_seq_len] where values are codebook indices
        codes_flat = all_codes.flatten()

        # Get codebook size from model (NSVQ uses num_embeddings)
        codebook_size = pl_module.model.vq.num_embeddings

        # Count usage per codebook entry
        counts = torch.bincount(codes_flat.long(), minlength=codebook_size)

        # Compute metrics with suffix
        total_codes = counts.sum().item()
        used_codes = (counts > 0).sum().item()
        metrics[f"val/codebook_utilization{metric_suffix}"] = used_codes / codebook_size
        metrics[f"val/codebook_entropy{metric_suffix}"] = self._entropy(counts.float())

        # Log to trainer
        pl_module.log_dict(metrics, sync_dist=True)

        # Create histogram visualization
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_histogram(counts, wandb_logger, trainer.global_step, codebook_size)

        return metrics

    def _entropy(self, counts: torch.Tensor) -> float:
        """Compute entropy of codebook distribution."""
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -(probs * probs.log()).sum().item()



    def _create_histogram(
        self,
        counts: torch.Tensor,
        wandb_logger,
        global_step: int,
        codebook_size: int,
    ):
        """Create and log histogram of codebook usage."""
        try:
            fig, ax = plt.subplots(figsize=(10, 4))

            x = range(codebook_size)
            ax.bar(x, counts.cpu().numpy(), color='steelblue', alpha=0.8)
            ax.set_xlabel('Codebook Index')
            ax.set_ylabel('Usage Count')
            ax.set_title(f'Codebook Usage Distribution (Step {global_step})')

            # Add statistics text
            used = (counts > 0).sum().item()
            ax.text(0.95, 0.95, f'Used: {used}/{codebook_size}',
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key="val/codebook_histogram",
                images=[img],
                caption=[f"Step {global_step}"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: codebook_histogram visualization failed: {e}")


class LatentSequenceHistogramStrategy(ValidationStrategy):
    """
    Visualize the distribution of latent token sequences (combinations).

    Since the combination space can be large (e.g., 8^4 = 4096), this strategy
    plots the top-N most frequent sequences to show if the model collapses
    to a few specific action patterns.
    """

    def __init__(
        self,
        name: str = "sequence_histogram",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_top_sequences: int = 50,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )
        self.num_top_sequences = num_top_sequences

    def needs_caching(self) -> bool:
        return True  # Need codes for sequence analysis

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate sequence usage histogram."""
        metrics = {}

        all_codes = cache.get_all_codes()
        if all_codes is None or len(all_codes) == 0:
            return metrics

        # all_codes shape: [N, code_seq_len]
        # Convert to list of tuples for counting
        sequences = [tuple(c.tolist()) for c in all_codes]

        counter = Counter(sequences)

        # Metrics
        unique_seqs = len(counter)

        # Calculate entropy of sequence distribution
        counts = torch.tensor(list(counter.values()), dtype=torch.float)
        metrics[f"val/sequence_entropy{metric_suffix}"] = self._entropy(counts)
        metrics[f"val/unique_sequences{metric_suffix}"] = unique_seqs

        pl_module.log_dict(metrics, sync_dist=True)

        # Visualize
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_histogram(counter, wandb_logger, trainer.global_step)

        return metrics

    def _entropy(self, counts: torch.Tensor) -> float:
        """Compute entropy of distribution."""
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -(probs * probs.log()).sum().item()



    def _create_histogram(
        self,
        counter,
        wandb_logger,
        global_step: int,
    ):
        """Create and log histogram of top sequence usage."""
        try:
            # Get top N most common
            most_common = counter.most_common(self.num_top_sequences)
            if not most_common:
                return

            labels, values = zip(*most_common)
            # Convert tuple labels to strings "1-2-3-4"
            str_labels = ["-".join(map(str, l)) for l in labels]

            fig, ax = plt.subplots(figsize=(12, 6))

            x = range(len(values))
            ax.bar(x, values, color='mediumpurple', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(str_labels, rotation=90, fontsize=8)
            ax.set_xlabel('Token Sequence')
            ax.set_ylabel('Count')
            ax.set_title(f'Top {len(values)} Latent Sequences (Step {global_step})')

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key="val/sequence_histogram",
                images=[img],
                caption=[f"Step {global_step}: Distribution of top {len(values)} sequences"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: sequence_histogram visualization failed: {e}")


class AllSequencesHistogramStrategy(ValidationStrategy):
    """
    Visualize the distribution of ALL latent token sequences (sorted frequency).

    This shows the "long tail" of the distribution.
    X-axis: Sequence rank (1 to N)
    Y-axis: Frequency count
    No labels on X-axis to avoid clutter.
    """

    def __init__(
        self,
        name: str = "all_sequences_histogram",
        enabled: bool = True,
        every_n_validations: int = 1,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )

    def needs_caching(self) -> bool:
        return True

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate all sequences histogram."""
        metrics = {}

        all_codes = cache.get_all_codes()
        if all_codes is None or len(all_codes) == 0:
            return metrics

        # all_codes shape: [N, code_seq_len]
        sequences = [tuple(c.tolist()) for c in all_codes]

        from collections import Counter
        counter = Counter(sequences)

        # Sort counts descending
        sorted_counts = sorted(counter.values(), reverse=True)

        # Visualize
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_plot(sorted_counts, wandb_logger, trainer.global_step)

        return metrics

    def _create_plot(
        self,
        counts: List[int],
        wandb_logger,
        global_step: int,
    ):
        """Create and log plot of all sequence counts."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            x = range(len(counts))
            ax.bar(x, counts, color='teal', width=1.0, alpha=0.8)
            # Alternatively use plot/fill_between for very dense data
            # ax.plot(x, counts, color='teal')
            # ax.fill_between(x, counts, color='teal', alpha=0.3)
            
            ax.set_xlabel('Sequence Rank')
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of All {len(counts)} Unique Sequences (Step {global_step})')
            ax.set_yscale('log') # Log scale helps see the tail
            
            # Add stats
            ax.text(0.95, 0.95, f'Total Unique: {len(counts)}',
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key="val/all_sequences_histogram",
                images=[img],
                caption=[f"Step {global_step}: Long tail distribution (log scale)"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: all_sequences_histogram visualization failed: {e}")


class MetadataScatterStrategy(ValidationStrategy):
    """
    Base class for scatter plot strategies that visualize metadata (actions, states)
    colored by codebook tokens or sequences.

    Provides common functionality:
    - Sample filtering by dataset and metadata requirements
    - Sample limiting with random selection
    - Matplotlib figure creation and cleanup
    """

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 1000,
        min_samples: int = 10,
        dataset_filter: Optional[str] = None,  # Changed: bucket filtering handles this now
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )
        self.num_samples = num_samples
        self.dataset_filter = dataset_filter

    def needs_caching(self) -> bool:
        return True

    def needs_codes(self) -> bool:
        return True

    def _filter_samples_with_metadata(
        self,
        cache: ValidationCache,
        required_keys: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[Any], List[int]]:
        """
        Filter samples that have required metadata keys.

        Args:
            cache: Validation cache
            required_keys: List of metadata keys required (e.g., ["action"])

        Returns:
            Tuple of (filtered_metadata, codes_list, valid_indices)
        """
        all_metadata = cache.get_all_metadata()
        all_codes = cache.get_all_codes()

        if not all_metadata or all_codes is None:
            return [], [], []

        filtered_meta = []
        codes_list = []
        valid_indices = []

        for i, meta in enumerate(all_metadata):
            # Dataset filter
            if self.dataset_filter and meta.get("dataset_name") != self.dataset_filter:
                continue

            # Check all required keys exist
            has_all = True
            for key in required_keys:
                val = meta.get(key)
                if val is None:
                    has_all = False
                    break
                # Need at least 2D for scatter plots
                if isinstance(val, (list, tuple)) and len(val) < 2:
                    has_all = False
                    break

            if has_all and i < len(all_codes):
                filtered_meta.append(meta)
                codes_list.append(all_codes[i])
                valid_indices.append(i)

        return filtered_meta, codes_list, valid_indices

    def _limit_samples(
        self,
        *arrays,
        n: Optional[int] = None,
    ) -> Tuple:
        """
        Randomly limit samples to n (or self.num_samples).

        Args:
            *arrays: Variable number of lists/tensors to sample from
            n: Number of samples (defaults to self.num_samples)

        Returns:
            Tuple of sampled arrays in same order as input
        """
        if not arrays or len(arrays[0]) == 0:
            return tuple([] for _ in arrays)

        n = n or self.num_samples
        total = len(arrays[0])
        n = min(n, total)

        indices = torch.randperm(total)[:n].tolist()

        result = []
        for arr in arrays:
            if isinstance(arr, torch.Tensor):
                result.append(arr[indices])
            else:
                result.append([arr[i] for i in indices])

        return tuple(result)

    def _create_scatter_figure(
        self,
        x: np.ndarray,
        y: np.ndarray,
        colors: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        colorbar_label: str = "Token ID",
        num_colors: int = 8,
        figsize: Tuple[int, int] = (8, 8),
        stats_text: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """
        Create a scatter plot figure.

        Args:
            x, y: Coordinates for scatter points
            colors: Values for coloring points
            title: Plot title
            xlabel, ylabel: Axis labels
            colorbar_label: Label for colorbar
            num_colors: Number of distinct colors in colormap
            figsize: Figure size tuple
            stats_text: Optional text to show in corner

        Returns:
            PIL Image of the figure, or None on error
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)

            cmap = plt.cm.get_cmap('tab20', num_colors)
            scatter = ax.scatter(
                x, y, c=colors, cmap=cmap, alpha=0.6, s=20,
                vmin=0, vmax=max(num_colors - 1, colors.max())
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            plt.colorbar(scatter, ax=ax, label=colorbar_label)

            if stats_text:
                ax.text(0.02, 0.98, stats_text,
                       transform=ax.transAxes, ha='left', va='top',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            plt.close(fig)
            return img
        except Exception as e:
            print(f"Warning: scatter plot creation failed: {e}")
            return None


class ActionTokenScatterStrategy(MetadataScatterStrategy):
    """
    Scatter plot of 2D actions colored by their assigned codebook tokens.

    Only runs when samples have 'action' metadata with 2D values.
    This helps visualize how the codebook discretizes continuous action space.
    """

    def __init__(
        self,
        name: str = "action_token_scatter",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 1000,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            num_samples=num_samples,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, dataset_filter, etc.
        )

    def required_metadata(self) -> List[str]:
        return ["action"]

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate action-token scatter plot."""
        metrics = {}

        # Filter samples with action metadata
        filtered_meta, codes_list, _ = self._filter_samples_with_metadata(cache, ["action"])

        if len(filtered_meta) < self.min_samples:
            return metrics

        # Extract actions and first token from codes
        actions = [meta["action"][:2] for meta in filtered_meta]
        tokens = [c[0].item() if c.ndim > 0 else c.item() for c in codes_list]

        # Limit samples
        actions, tokens = self._limit_samples(actions, tokens)

        if not actions:
            return metrics

        # Create scatter plot
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            actions_np = np.array(actions)
            tokens_np = np.array(tokens)
            codebook_size = pl_module.model.vq.num_embeddings

            img = self._create_scatter_figure(
                x=actions_np[:, 0],
                y=actions_np[:, 1],
                colors=tokens_np,
                title=f'2D Actions Colored by Codebook Token (Step {trainer.global_step})',
                xlabel='Action X (cumulative dx)',
                ylabel='Action Y (cumulative dy)',
                colorbar_label='Token ID',
                num_colors=codebook_size,
                stats_text=f'Samples: {len(actions)}\nUnique tokens: {len(set(tokens))}',
            )

            if img:
                wandb_logger.log_image(
                    key="val/action_token_scatter",
                    images=[img],
                    caption=[f"Step {trainer.global_step}: 2D actions colored by assigned token"],
                )

        return metrics


class ActionSequenceScatterStrategy(MetadataScatterStrategy):
    """
    Scatter plot of 2D actions colored by their assigned FULL token sequence.

    This visualizes if specific action trajectories (dx, dy) map consistently
    to specific latent code sequences.
    """

    def __init__(
        self,
        name: str = "action_sequence_scatter",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 1000,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            num_samples=num_samples,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, dataset_filter, etc.
        )

    def required_metadata(self) -> List[str]:
        return ["action"]

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate action-sequence scatter plot."""
        metrics = {}

        # Use base class filtering
        filtered_meta, codes_list, _ = self._filter_samples_with_metadata(cache, ["action"])

        if len(filtered_meta) < self.min_samples:
            return metrics

        # Map unique sequences to IDs
        sequences = [tuple(c.tolist()) for c in codes_list]
        unique_seqs = list(set(sequences))
        seq_to_id = {seq: i for i, seq in enumerate(unique_seqs)}
        num_unique_seqs = len(unique_seqs)

        # Extract actions and sequence IDs
        actions = [meta["action"][:2] for meta in filtered_meta]
        seq_ids = [seq_to_id[seq] for seq in sequences]

        # Use base class sample limiting
        actions, seq_ids = self._limit_samples(actions, seq_ids)

        if not actions:
            return metrics

        # Create scatter plot
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_sequence_scatter(
                actions, seq_ids, wandb_logger, trainer.global_step, num_unique_seqs
            )

        return metrics

    def _create_sequence_scatter(
        self,
        actions: List[List[float]],
        seq_ids: List[int],
        wandb_logger,
        global_step: int,
        num_unique: int,
    ):
        """Create scatter plot colored by sequence ID."""
        try:
            if num_unique > 100:
                warnings.warn(f"ActionSequenceScatterStrategy: {num_unique} unique sequences. Colors may be indistinguishable.")

            actions_np = np.array(actions)
            ids_np = np.array(seq_ids)

            fig, ax = plt.subplots(figsize=(10, 8))

            cmap = plt.cm.get_cmap('nipy_spectral', num_unique)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']

            # Plot each sequence ID with unique marker and color
            for uid in np.unique(ids_np):
                mask = ids_np == uid
                marker = markers[uid % len(markers)]
                color = cmap(uid / max(1, num_unique - 1))
                ax.scatter(actions_np[mask, 0], actions_np[mask, 1],
                          color=color, marker=marker, alpha=0.6, s=30)

            ax.set_xlabel('Action X (cumulative dx)')
            ax.set_ylabel('Action Y (cumulative dy)')
            ax.set_title(f'2D Actions Colored by Full Sequence (Step {global_step})')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            norm = plt.Normalize(vmin=0, vmax=num_unique-1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Sequence ID')

            ax.text(0.02, 0.98, f'Samples: {len(actions)}\nUnique Seqs: {num_unique}',
                   transform=ax.transAxes, ha='left', va='top',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key="val/action_sequence_scatter",
                images=[img],
                caption=[f"Step {global_step}: Colored by unique sequence ID"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: action_sequence_scatter visualization failed: {e}")


class TopSequencesScatterStrategy(MetadataScatterStrategy):
    """
    Scatter plot highlighting ONLY the top N most frequent latent sequences.

    Top sequences get distinct high-contrast colors.
    All other sequences are plotted in grey.
    This helps visualize if the most common modes correspond to specific
    actions (e.g., "move forward", "stop") or are scattered noise.
    """

    def __init__(
        self,
        name: str = "top_sequences_scatter",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 1000,
        num_top_sequences: int = 5,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            num_samples=num_samples,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, dataset_filter, etc.
        )
        self.num_top_sequences = num_top_sequences

    def required_metadata(self) -> List[str]:
        return ["action"]

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate top sequences scatter plot."""
        metrics = {}

        # Use base class filtering
        filtered_meta, codes_list, _ = self._filter_samples_with_metadata(cache, ["action"])

        if len(filtered_meta) < self.min_samples:
            return metrics

        # Extract actions and sequences
        actions = [meta["action"][:2] for meta in filtered_meta]
        sequences = [tuple(c.tolist()) for c in codes_list]

        # Find top N sequences
        counter = Counter(sequences)
        top_seqs_counts = counter.most_common(self.num_top_sequences)
        top_seqs = {seq for seq, _ in top_seqs_counts}
        seq_to_cat = {seq: i for i, (seq, _) in enumerate(top_seqs_counts)}

        # Map to categories: 0..N-1 for top, -1 for others
        categories = [seq_to_cat.get(seq, -1) for seq in sequences]

        # Use base class sample limiting
        actions, categories = self._limit_samples(actions, categories)

        if not actions:
            return metrics

        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_top_scatter(actions, categories, top_seqs_counts,
                                     wandb_logger, trainer.global_step)

        return metrics

    def _create_top_scatter(
        self,
        actions: List[List[float]],
        categories: List[int],
        top_seqs_counts: List[Tuple[Tuple[int, ...], int]],
        wandb_logger,
        global_step: int,
    ):
        """Create scatter plot with top sequences highlighted."""
        try:
            actions_np = np.array(actions)
            cats_np = np.array(categories)

            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot "Other" (Grey) first
            mask_other = cats_np == -1
            if np.any(mask_other):
                ax.scatter(actions_np[mask_other, 0], actions_np[mask_other, 1],
                          c='lightgrey', alpha=0.5, s=15,
                          label=f'Others ({np.sum(mask_other)})', zorder=1)

            # Plot Top N with distinct colors
            colors = plt.cm.tab10.colors
            for i, (seq, count) in enumerate(top_seqs_counts):
                mask = cats_np == i
                if np.any(mask):
                    ax.scatter(actions_np[mask, 0], actions_np[mask, 1],
                              c=[colors[i % len(colors)]], alpha=0.9, s=30,
                              label=f'{seq}: {count}', zorder=2)

            ax.set_xlabel('Action X (cumulative dx)')
            ax.set_ylabel('Action Y (cumulative dy)')
            ax.set_title(f'Top {self.num_top_sequences} Sequences vs Action (Step {global_step})')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key="val/top_sequences_scatter",
                images=[img],
                caption=[f"Step {global_step}: Top {self.num_top_sequences} sequences highlighted"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: top_sequences_scatter visualization failed: {e}")


class StateSequenceScatterStrategy(MetadataScatterStrategy):
    """
    Scatter plot of ROBOT STATE (x, y) colored by assigned token sequence.

    Visualizes how latent sequences distribute across the state space.
    Highlights the top N most frequent sequences to check for spatial clusters.
    Requires 'initial_state' in metadata.
    """

    def __init__(
        self,
        name: str = "state_sequence_scatter",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 1000,
        num_top_sequences: int = 20,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            num_samples=num_samples,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, dataset_filter, etc.
        )
        self.num_top_sequences = num_top_sequences

    def required_metadata(self) -> List[str]:
        return ["initial_state"]

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate state-sequence scatter plot."""
        metrics = {}

        # Use base class filtering
        filtered_meta, codes_list, _ = self._filter_samples_with_metadata(cache, ["initial_state"])

        if len(filtered_meta) < self.min_samples:
            return metrics

        # Extract states and sequences
        states = [meta["initial_state"][:2] for meta in filtered_meta]
        sequences = [tuple(c.tolist()) for c in codes_list]

        # Find top N sequences
        counter = Counter(sequences)
        top_seqs_counts = counter.most_common(self.num_top_sequences)
        seq_to_cat = {seq: i for i, (seq, _) in enumerate(top_seqs_counts)}

        # Map to categories: 0..N-1 for top, -1 for others
        categories = [seq_to_cat.get(seq, -1) for seq in sequences]

        # Use base class sample limiting
        states, categories = self._limit_samples(states, categories)

        if not states:
            return metrics

        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_state_scatter(states, categories, top_seqs_counts,
                                       wandb_logger, trainer.global_step)

        return metrics

    def _create_state_scatter(
        self,
        states: List[List[float]],
        categories: List[int],
        top_seqs_counts: List[Tuple[Tuple[int, ...], int]],
        wandb_logger,
        global_step: int,
    ):
        """Create scatter plot."""
        try:
            states_np = np.array(states)
            # Add jitter to reveal overlaps
            jitter = np.random.normal(0, 0.005, size=states_np.shape)
            states_np = states_np + jitter
            cats_np = np.array(categories)

            fig, ax = plt.subplots(figsize=(10, 8))

            cmap = plt.cm.get_cmap('tab20', self.num_top_sequences)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']

            # Plot "Other" (Grey) first
            mask_other = cats_np == -1
            if np.any(mask_other):
                ax.scatter(states_np[mask_other, 0], states_np[mask_other, 1],
                          c='lightgrey', marker='.', alpha=0.3, s=15,
                          label=f'Others ({np.sum(mask_other)})', zorder=1)

            # Plot Top N with markers
            for i, (seq, count) in enumerate(top_seqs_counts):
                mask = cats_np == i
                if np.any(mask):
                    ax.scatter(states_np[mask, 0], states_np[mask, 1],
                              color=cmap(i), marker=markers[i % len(markers)],
                              alpha=0.8, s=40, label=f'{seq}: {count}', zorder=2)

            ax.set_xlabel('State X')
            ax.set_ylabel('State Y')
            ax.set_title(f'State vs Top {self.num_top_sequences} Sequences (Step {global_step})')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key="val/state_sequence_scatter",
                images=[img],
                caption=[f"Step {global_step}: Top {self.num_top_sequences} sequences by state"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: state_sequence_scatter visualization failed: {e}")


# Strategy type registry - maps type names to classes
STRATEGY_REGISTRY: Dict[str, type] = {
    "basic": BasicVisualizationStrategy,
    "basic_visualization": BasicVisualizationStrategy,
    "latent_transfer": LatentTransferStrategy,
    "clustering": ClusteringStrategy,
    "codebook_histogram": CodebookHistogramStrategy,
    "sequence_histogram": LatentSequenceHistogramStrategy,
    "all_sequences_histogram": AllSequencesHistogramStrategy,
    "action_token_scatter": ActionTokenScatterStrategy,
    "action_sequence_scatter": ActionSequenceScatterStrategy,
    "top_sequences_scatter": TopSequencesScatterStrategy,
    "state_sequence_scatter": StateSequenceScatterStrategy,
}


def create_validation_strategies(
    config: Dict[str, Any],
    val_buckets: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[ValidationStrategy]:
    """
    Create validation strategies from config using composition pattern.

    Each config key is a strategy instance name, with optional 'type' field
    to specify which strategy class to use. This allows multiple instances
    of the same strategy type with different configurations.

    Args:
        config: validation.strategies config dict. Each key is an instance name.
            Example:
            {
                "basic": {"enabled": true, "visualize_train": true},
                "transfer_lt": {"type": "latent_transfer", "buckets": ["language_table"]},
                "transfer_bridge": {"type": "latent_transfer", "buckets": ["bridge"]},
            }
        val_buckets: Optional dict of bucket definitions for visualization

    Returns:
        List of ValidationStrategy instances
    """
    strategies = []

    if not config:
        return strategies

    for instance_name, instance_config in config.items():
        # Skip if not a dict (could be a top-level param like num_fixed_samples)
        if not isinstance(instance_config, dict):
            continue

        # Skip disabled strategies
        if not instance_config.get("enabled", True):
            continue

        # Get strategy type (defaults to instance_name for backwards compat)
        strategy_type = instance_config.get("type", instance_name)

        if strategy_type not in STRATEGY_REGISTRY:
            print(f"Warning: Unknown strategy type '{strategy_type}' for '{instance_name}', skipping")
            continue

        strategy_class = STRATEGY_REGISTRY[strategy_type]

        # Build kwargs, excluding 'type' which is just for class selection
        kwargs = {k: v for k, v in instance_config.items() if k != "type"}
        kwargs["name"] = instance_name  # Use instance name, not type

        # Pass val_buckets to basic visualization strategy
        if strategy_type in ("basic", "basic_visualization") and val_buckets:
            kwargs["val_buckets"] = val_buckets

        try:
            strategies.append(strategy_class(**kwargs))
        except TypeError as e:
            print(f"Warning: Failed to create strategy '{instance_name}': {e}")
            continue

    return strategies
