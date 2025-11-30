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
    
    def clear(self):
        """Clear all cached data (but keep fixed samples)."""
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
    
    def get_frames_by_dataset_type(self, dataset_type: str) -> Optional[torch.Tensor]:
        """Get frames filtered by dataset type."""
        all_frames = self.get_all_frames()
        all_metadata = self.get_all_metadata()
        
        if all_frames is None or not all_metadata:
            return None
        
        indices = [
            i for i, meta in enumerate(all_metadata)
            if meta.get("dataset_type") == dataset_type
        ]
        
        if not indices:
            return None
        
        return all_frames[indices]


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
    - Per-bucket samples: separate grids for each dataset type
    """
    
    def __init__(
        self,
        enabled: bool = True,
        num_fixed_samples: int = 8,
        num_random_samples: int = 8,
        visualize_train: bool = True,
        visualize_val: bool = True,
        visualize_per_bucket: bool = True,
        samples_per_bucket: int = 4,
        **kwargs,
    ):
        super().__init__(
            name="basic_visualization",
            enabled=enabled,
            every_n_validations=1,  # Always run
        )
        self.num_fixed_samples = num_fixed_samples
        self.num_random_samples = num_random_samples
        self.visualize_train = visualize_train
        self.visualize_val = visualize_val
        self.visualize_per_bucket = visualize_per_bucket
        self.samples_per_bucket = samples_per_bucket
    
    def needs_caching(self) -> bool:
        return True  # Need frames for visualization
    
    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
    ) -> Dict[str, Any]:
        """Generate reconstruction visualizations."""
        metrics = {}
        
        if not self.visualize_val:
            return metrics
        
        all_frames = cache.get_all_frames()
        all_metadata = cache.get_all_metadata()
        
        if all_frames is None or len(all_frames) == 0:
            return metrics
        
        wandb_logger = self._get_wandb_logger(trainer)
        
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
        if n_random > 0:
            random_indices = torch.randperm(len(all_frames))[:n_random]
            random_frames = all_frames[random_indices]
            random_grid = self._create_recon_grid(random_frames, pl_module)
            if wandb_logger and random_grid is not None:
                wandb_logger.log_image(
                    key="val/random_reconstructions",
                    images=[random_grid],
                    caption=[f"Step {trainer.global_step} (random samples)"],
                )
        
        # === Per-bucket visualization ===
        if self.visualize_per_bucket and all_metadata:
            # Get unique dataset types
            dataset_types = set()
            for meta in all_metadata:
                if isinstance(meta, dict) and "dataset_type" in meta:
                    dataset_types.add(meta["dataset_type"])
            
            for dtype in dataset_types:
                bucket_frames = cache.get_frames_by_dataset_type(dtype)
                if bucket_frames is not None and len(bucket_frames) > 0:
                    # Sample from this bucket
                    n_samples = min(self.samples_per_bucket, len(bucket_frames))
                    indices = torch.randperm(len(bucket_frames))[:n_samples]
                    samples = bucket_frames[indices]
                    
                    bucket_grid = self._create_recon_grid(samples, pl_module)
                    if wandb_logger and bucket_grid is not None:
                        wandb_logger.log_image(
                            key=f"val/reconstructions_{dtype}",
                            images=[bucket_grid],
                            caption=[f"Step {trainer.global_step} ({dtype} samples)"],
                        )
        
        return metrics
    
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
            source_latents = pl_module.model.encode(source_frames)  # z_a
            
            # Get target initial frames
            target_s0 = target_frames[:, :, 0:1]  # s_b (keep dim for concat)
            
            # Decode: apply source latent to target initial frame
            # This requires model to support cross-decoding
            transferred_recons = pl_module.model.decode(
                target_s0,
                source_latents,
            )  # s_b'_pred
            
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
                wandb_logger,
                trainer.current_epoch,
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
        wandb_logger,
        epoch: int,
    ):
        """Visualize latent transfer results."""
        # Grid: [s_a, s_a', s_b, s_b'_pred, s_b'_true]
        s_a = source_frames[:, :, 0]
        s_a_prime = source_frames[:, :, 1]
        s_b = target_frames[:, :, 0]
        s_b_prime_pred = transferred
        s_b_prime_true = target_frames[:, :, 1]
        
        imgs = torch.stack([s_a, s_a_prime, s_b, s_b_prime_pred, s_b_prime_true], dim=0)
        imgs = rearrange(imgs, 'r b c h w -> (b r) c h w')
        imgs = imgs.clamp(0.0, 1.0)
        
        grid = make_grid(imgs, nrow=5, normalize=False)
        
        wandb_logger.log_image(
            key="val/latent_transfer",
            images=[grid],
            caption=[f"Epoch {epoch}: s_a | s_a' | s_b | D(s_b,z_a) | s_b'"],
        )


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


def create_validation_strategies(config: Dict[str, Any]) -> List[ValidationStrategy]:
    """
    Create validation strategies from config.
    
    Args:
        config: validation.strategies config dict
        
    Returns:
        List of ValidationStrategy instances
    """
    strategies = []
    
    if not config:
        return strategies
    
    # Basic visualization (always-on by default)
    if config.get("basic", {}).get("enabled", True):
        strategies.append(BasicVisualizationStrategy(
            **config.get("basic", {}),
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
    
    return strategies
