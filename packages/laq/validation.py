"""
Validation strategies for LAQ training.

Implements flexible, configurable validation that can run:
- Light validation (always): reconstruction loss + visualizations
- Heavy validation (periodic): latent transfer analysis, clustering

Usage in config:
```yaml
validation:
  strategies:
    basic:
      enabled: true
    latent_transfer:
      enabled: true
      every_n_validations: 10
    clustering:
      enabled: true
      every_n_validations: 20
```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
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

    def clear(self):
        """Clear all cached data (but keep fixed samples and train samples)."""
        self.frames.clear()
        self.latents.clear()
        self.codes.clear()
        self.losses.clear()
        self.metadata.clear()

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
            if self._matches_filters(meta, filters):
                indices.append(i)

        if not indices:
            return None, []

        filtered_frames = frames[indices]
        filtered_metadata = [metadata[i] for i in indices]
        return filtered_frames, filtered_metadata

    def _matches_filters(self, meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
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
    - What to compute and log (via run)
    """
    
    def __init__(
        self,
        name: str,
        enabled: bool = True,
        every_n_validations: int = 1,
        **kwargs,
    ):
        self.name = name
        self.enabled = enabled
        self.every_n_validations = every_n_validations
        self.validation_count = 0
    
    def should_run(self) -> bool:
        """Check if this strategy should run on current validation."""
        if not self.enabled:
            return False
        return (self.validation_count % self.every_n_validations) == 0
    
    def increment_count(self):
        """Increment validation counter."""
        self.validation_count += 1
    
    @abstractmethod
    def needs_caching(self) -> bool:
        """Return True if this strategy needs data cached during validation."""
        pass
    
    @abstractmethod
    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
    ) -> Dict[str, Any]:
        """
        Run the validation strategy.
        
        Args:
            cache: Cached validation data
            pl_module: The Lightning module
            trainer: The trainer
            
        Returns:
            Dict of metrics to log
        """
        pass


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
            name="basic_visualization",
            enabled=enabled,
            every_n_validations=1,  # Always run
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
    ) -> Dict[str, Any]:
        """Generate reconstruction visualizations for both train and val."""
        metrics = {}
        wandb_logger = self._get_wandb_logger(trainer)

        # === Training samples visualization ===
        if self.visualize_train:
            self._visualize_training_samples(cache, pl_module, trainer, wandb_logger)

        # === Validation samples visualization ===
        if self.visualize_val:
            self._visualize_validation_samples(cache, pl_module, trainer, wandb_logger)

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
    ) -> None:
        """Visualize validation samples from cache."""
        all_frames = cache.get_all_frames()
        all_metadata = cache.get_all_metadata()

        if all_frames is None or len(all_frames) == 0:
            return

        # Log cache distribution for debugging
        distribution = cache.get_dataset_distribution()
        if distribution:
            print(f"  Cached validation samples per datasource: {distribution}")
            print(f"  Total cached validation samples: {sum(distribution.values())}")

        # === Fixed samples (diverse across datasets) ===
        if cache.fixed_frames is not None and len(cache.fixed_frames) > 0:
            fixed_grid = self._create_recon_grid(cache.fixed_frames, pl_module)
            if wandb_logger and fixed_grid is not None:
                wandb_logger.log_image(
                    key="val/fixed_reconstructions",
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
                    key="val/random_reconstructions",
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

    def _get_wandb_logger(self, trainer: pl.Trainer):
        """Get WandB logger from trainer."""
        if not WANDB_AVAILABLE:
            return None
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                return logger
        return None

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
    """
    
    def __init__(
        self,
        enabled: bool = True,
        every_n_validations: int = 10,
        num_pairs: int = 256,
        **kwargs,
    ):
        super().__init__(
            name="latent_transfer",
            enabled=enabled,
            every_n_validations=every_n_validations,
        )
        self.num_pairs = num_pairs
    
    def needs_caching(self) -> bool:
        return True
    
    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
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
        
        metrics["val/latent_transfer_mse"] = transfer_mse.item()
        metrics["val/self_recon_mse"] = self_mse.item()
        metrics["val/transfer_ratio"] = transfer_mse.item() / (self_mse.item() + 1e-8)
        
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
    
    def _get_wandb_logger(self, trainer: pl.Trainer):
        if not WANDB_AVAILABLE:
            return None
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                return logger
        return None
    
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
        enabled: bool = True,
        every_n_validations: int = 20,
        num_samples: int = 1000,
        num_clusters: int = 16,
        num_examples_per_cluster: int = 4,
        **kwargs,
    ):
        super().__init__(
            name="clustering",
            enabled=enabled,
            every_n_validations=every_n_validations,
        )
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.num_examples_per_cluster = num_examples_per_cluster
    
    def needs_caching(self) -> bool:
        return True
    
    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
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
            
            # Log metrics
            metrics["val/cluster_entropy"] = self._entropy(cluster_sizes.float())
            metrics["val/num_empty_clusters"] = (cluster_sizes == 0).sum().item()
            metrics["val/max_cluster_size"] = cluster_sizes.max().item()
            metrics["val/min_cluster_size"] = cluster_sizes[cluster_sizes > 0].min().item() if (cluster_sizes > 0).any() else 0
            
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
    
    def _get_wandb_logger(self, trainer: pl.Trainer):
        if not WANDB_AVAILABLE:
            return None
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                return logger
        return None
    
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
        enabled: bool = True,
        every_n_validations: int = 1,
        **kwargs,
    ):
        super().__init__(
            name="codebook_histogram",
            enabled=enabled,
            every_n_validations=every_n_validations,
        )

    def needs_caching(self) -> bool:
        return True  # Need codes for histogram

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
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

        # Compute metrics
        total_codes = counts.sum().item()
        used_codes = (counts > 0).sum().item()
        metrics["val/codebook_utilization"] = used_codes / codebook_size
        metrics["val/codebook_entropy"] = self._entropy(counts.float())

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

    def _get_wandb_logger(self, trainer: pl.Trainer):
        if not WANDB_AVAILABLE:
            return None
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                return logger
        return None

    def _create_histogram(
        self,
        counts: torch.Tensor,
        wandb_logger,
        global_step: int,
        codebook_size: int,
    ):
        """Create and log histogram of codebook usage."""
        try:
            import matplotlib.pyplot as plt
            import io
            from PIL import Image

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
        except ImportError as e:
            print(f"Warning: codebook_histogram requires matplotlib: {e}")
        except Exception as e:
            print(f"Warning: codebook_histogram visualization failed: {e}")


class ActionTokenScatterStrategy(ValidationStrategy):
    """
    Scatter plot of 2D actions colored by their assigned codebook tokens.

    Only runs when samples have 'action' metadata with 2D values.
    This helps visualize how the codebook discretizes continuous action space.

    For 3D actions, could extend to show 2D projections or use different viz.
    """

    def __init__(
        self,
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 1000,
        **kwargs,
    ):
        super().__init__(
            name="action_token_scatter",
            enabled=enabled,
            every_n_validations=every_n_validations,
        )
        self.num_samples = num_samples

    def needs_caching(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
    ) -> Dict[str, Any]:
        """Generate action-token scatter plot."""
        metrics = {}

        all_frames = cache.get_all_frames()
        all_metadata = cache.get_all_metadata()
        all_codes = cache.get_all_codes()

        if all_frames is None or not all_metadata or all_codes is None:
            return metrics

        # Check if we have action data
        actions = []
        codes_list = []

        for i, meta in enumerate(all_metadata):
            if "action" not in meta:
                continue
            action = meta["action"]
            # Only support 2D actions for now
            if isinstance(action, (list, tuple)) and len(action) == 2:
                actions.append(action)
                # Get corresponding code (use first token if multi-token)
                if i < len(all_codes):
                    code = all_codes[i]
                    if code.ndim > 0:
                        code = code[0]  # Use first token
                    codes_list.append(code.item())

        if len(actions) < 10:
            # Not enough 2D action samples
            return metrics

        # Limit samples
        n = min(self.num_samples, len(actions))
        indices = torch.randperm(len(actions))[:n].tolist()
        actions = [actions[i] for i in indices]
        codes_list = [codes_list[i] for i in indices]

        # Create scatter plot
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_scatter(
                actions, codes_list, wandb_logger, trainer.global_step,
                pl_module.model.vq.num_embeddings
            )

        return metrics

    def _get_wandb_logger(self, trainer: pl.Trainer):
        if not WANDB_AVAILABLE:
            return None
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                return logger
        return None

    def _create_scatter(
        self,
        actions: List[List[float]],
        codes: List[int],
        wandb_logger,
        global_step: int,
        codebook_size: int,
    ):
        """Create and log scatter plot of actions colored by tokens."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            from PIL import Image

            actions_np = np.array(actions)
            codes_np = np.array(codes)

            fig, ax = plt.subplots(figsize=(8, 8))

            # Use a colormap with enough distinct colors
            cmap = plt.cm.get_cmap('tab20', codebook_size)

            scatter = ax.scatter(
                actions_np[:, 0], actions_np[:, 1],
                c=codes_np, cmap=cmap, alpha=0.6, s=20,
                vmin=0, vmax=codebook_size-1
            )

            ax.set_xlabel('Action X (cumulative dx)')
            ax.set_ylabel('Action Y (cumulative dy)')
            ax.set_title(f'2D Actions Colored by Codebook Token (Step {global_step})')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, label='Token ID')

            # Add statistics
            unique_codes = len(set(codes))
            ax.text(0.02, 0.98, f'Samples: {len(actions)}\nUnique tokens: {unique_codes}',
                   transform=ax.transAxes, ha='left', va='top',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key="val/action_token_scatter",
                images=[img],
                caption=[f"Step {global_step}: 2D actions colored by assigned token"],
            )

            plt.close(fig)
        except ImportError as e:
            print(f"Warning: action_token_scatter requires matplotlib: {e}")
        except Exception as e:
            print(f"Warning: action_token_scatter visualization failed: {e}")


def create_validation_strategies(
    config: Dict[str, Any],
    val_buckets: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[ValidationStrategy]:
    """
    Create validation strategies from config.

    Args:
        config: validation.strategies config dict
        val_buckets: Optional dict of bucket definitions for visualization
            Example:
            {
                "youtube": {"dataset_type": "youtube"},
                "bridge_toykitchen": {"dataset_type": "bridge", "environment": "toykitchen1"},
                "with_language": {"language": ["not_null", true]},
            }

    Returns:
        List of ValidationStrategy instances
    """
    strategies = []

    if not config:
        return strategies

    # Basic visualization (always-on by default)
    if config.get("basic", {}).get("enabled", True):
        # Convert OmegaConf to plain dict to allow adding keys
        basic_config = dict(config.get("basic", {}))
        # Add val_buckets if specified
        if val_buckets:
            basic_config["val_buckets"] = val_buckets
        strategies.append(BasicVisualizationStrategy(
            **basic_config,
            num_fixed_samples=config.get("num_fixed_samples", 4),
            num_random_samples=config.get("num_random_samples", 4),
        ))

    # Latent transfer analysis
    if config.get("latent_transfer", {}).get("enabled", False):
        strategies.append(LatentTransferStrategy(
            **config.get("latent_transfer", {}),
        ))

    # Clustering analysis
    if config.get("clustering", {}).get("enabled", False):
        strategies.append(ClusteringStrategy(
            **config.get("clustering", {}),
        ))

    # Codebook histogram
    if config.get("codebook_histogram", {}).get("enabled", False):
        strategies.append(CodebookHistogramStrategy(
            **config.get("codebook_histogram", {}),
        ))

    # Action-token scatter (for datasets with 2D actions like language_table)
    if config.get("action_token_scatter", {}).get("enabled", False):
        strategies.append(ActionTokenScatterStrategy(
            **config.get("action_token_scatter", {}),
        ))

    return strategies
