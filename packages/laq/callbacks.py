"""
PyTorch Lightning callbacks for LAQ training.

Includes:
- ValidationStrategyCallback: Flexible validation with bucket-aware routing
- EMACallback: Exponential moving average of model weights
"""

import gc
from typing import Optional, List, Dict, Any

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
    Flexible validation callback with bucket-aware routing.

    Architecture (Composition Pattern):
    - Buckets: Named data subsets with filters (e.g., "language_table", "bridge")
    - Strategies: Self-contained validation logic with embedded bucket bindings

    Features:
    - Per-bucket caching: Each bucket has its own cache
    - Strategy-embedded binding: Strategies read from their own `buckets` property
    - Automatic applicability checks: Strategies check if they have enough valid data

    Args:
        strategies: List of ValidationStrategy instances (with buckets property)
        bucket_configs: Dict of bucket name -> BucketConfig or dict with filters
        num_fixed_samples: Number of fixed samples per bucket
        max_cached_samples: Maximum samples for global cache (fallback)
    """

    def __init__(
        self,
        strategies: Optional[List] = None,
        bucket_configs: Optional[Dict[str, Any]] = None,
        num_fixed_samples: int = 8,
        num_random_samples: int = 8,
        max_cached_samples: int = 256,
        run_gc_after_validation: bool = True,
    ):
        super().__init__()
        self.strategies = strategies or []
        self.num_fixed_samples = num_fixed_samples
        self.num_random_samples = num_random_samples
        self.max_cached_samples = max_cached_samples
        self.run_gc_after_validation = run_gc_after_validation

        # Import here to avoid circular imports
        from laq.validation import ValidationCache, BucketConfig

        # Create bucket configs
        self.bucket_configs: Dict[str, BucketConfig] = {}
        if bucket_configs:
            for name, cfg in bucket_configs.items():
                if isinstance(cfg, BucketConfig):
                    self.bucket_configs[name] = cfg
                else:
                    self.bucket_configs[name] = BucketConfig(
                        name=name,
                        filters=cfg.get("filters", {}),
                        max_samples=cfg.get("max_samples", 100),
                        is_holdout=cfg.get("is_holdout", False),
                    )

        # Create per-bucket caches
        self.bucket_caches: Dict[str, ValidationCache] = {}
        for name, cfg in self.bucket_configs.items():
            cache = ValidationCache()
            cache.bucket_name = name
            cache.is_holdout = cfg.is_holdout
            cache.max_samples = cfg.max_samples
            self.bucket_caches[name] = cache

        # Global cache (fallback for strategies without bucket bindings)
        self.global_cache = ValidationCache()
        self.global_cache.max_samples = max_cached_samples

        # Track fixed sample indices
        self.fixed_indices: Optional[List[int]] = None
        self.validation_count = 0
        self._first_full_validation_done = False

    def _any_strategy_needs_codes(self) -> bool:
        """Check if any running strategy needs codebook indices."""
        for strategy in self.strategies:
            if strategy.should_run() and strategy.needs_codes():
                return True
        return False

    def _extract_metadata(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract per-sample metadata from batch."""
        frames = batch["frames"]
        batch_metadata = []
        batch_size = frames.shape[0]

        for i in range(batch_size):
            meta = {}
            for key in batch.keys():
                if key == "frames":
                    continue
                val = batch[key]

                # Special handling for 'action' and 'initial_state' which get transposed
                if (key == "action" or key == "initial_state") and isinstance(val, (list, tuple)) and len(val) > 0:
                    if isinstance(val[0], torch.Tensor) and val[0].ndim > 0:
                        dims = [v[i].item() for v in val if i < len(v)]
                        meta[key] = dims
                    else:
                        meta[key] = val[i] if i < len(val) else None
                elif isinstance(val, (list, tuple)):
                    meta[key] = val[i] if i < len(val) else None
                elif isinstance(val, torch.Tensor):
                    if val.ndim > 0 and i < len(val):
                        meta[key] = val[i].item() if val[i].ndim == 0 else val[i].tolist()
                    elif val.ndim == 0:
                        meta[key] = val.item()
                else:
                    meta[key] = val
            batch_metadata.append(meta)

        return batch_metadata

    def _select_diverse_fixed_samples(
        self,
        cache,
        num_samples: int,
    ) -> None:
        """Select diverse fixed samples for a cache."""
        all_frames = cache.get_all_frames()
        all_metadata = cache.get_all_metadata()

        if all_frames is None or len(all_frames) < num_samples:
            return

        # Group by dataset_type for diversity
        by_dataset: Dict[str, List[int]] = {}
        for i, meta in enumerate(all_metadata):
            dtype = meta.get("dataset_type", "unknown")
            if dtype not in by_dataset:
                by_dataset[dtype] = []
            by_dataset[dtype].append(i)

        selected_indices = []
        dataset_types = list(by_dataset.keys())
        samples_per_dataset = max(1, num_samples // len(dataset_types)) if dataset_types else num_samples

        for dtype in dataset_types:
            indices = by_dataset[dtype]
            shuffled = torch.randperm(len(indices)).tolist()
            for j in shuffled[:samples_per_dataset]:
                if len(selected_indices) < num_samples:
                    selected_indices.append(indices[j])

        # Fill remaining with random
        remaining = [i for i in range(len(all_frames)) if i not in selected_indices]
        while len(selected_indices) < num_samples and remaining:
            idx = remaining.pop(torch.randint(len(remaining), (1,)).item())
            selected_indices.append(idx)

        cache.fixed_indices = selected_indices
        cache.fixed_frames = all_frames[selected_indices]
        cache.fixed_metadata = [all_metadata[i] for i in selected_indices]

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Clear all caches at start of validation."""
        self.global_cache.clear()
        for cache in self.bucket_caches.values():
            cache.clear()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Route samples to appropriate bucket caches."""
        # Extract frames and metadata
        if isinstance(batch, dict):
            frames = batch["frames"]
            metadata_list = self._extract_metadata(batch)
        else:
            frames = batch
            metadata_list = [{} for _ in range(frames.shape[0])]

        # Compute codes if any strategy needs them
        codes = None
        latents = None
        if self._any_strategy_needs_codes():
            with torch.no_grad():
                device = pl_module.device
                frames_gpu = frames.to(device)
                codes = pl_module.model(frames_gpu, return_only_codebook_ids=True).cpu()
                latents = pl_module.model.vq.codebooks[codes.to(device)].cpu()

        # Route each sample to matching bucket(s) AND global cache
        for i, meta in enumerate(metadata_list):
            frame = frames[i:i+1].detach().cpu()
            code = codes[i:i+1] if codes is not None else None
            latent = latents[i:i+1] if latents is not None else None

            # Add to global cache (with stratified balancing)
            if not self.global_cache.is_full():
                self.global_cache.add_sample(frame, meta, code, latent)

            # Route to matching buckets
            for bucket_name, bucket_cfg in self.bucket_configs.items():
                cache = self.bucket_caches[bucket_name]
                if not cache.is_full() and bucket_cfg.matches(meta):
                    cache.add_sample(frame, meta, code, latent)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run strategies on their assigned buckets."""
        self.validation_count += 1

        # Update strategy counters
        for strategy in self.strategies:
            strategy.increment_count()

        # Select fixed samples for global cache on first validation
        if not self._first_full_validation_done:
            self._select_diverse_fixed_samples(self.global_cache, self.num_fixed_samples)
            for cache in self.bucket_caches.values():
                self._select_diverse_fixed_samples(cache, min(4, self.num_fixed_samples))
            self._first_full_validation_done = True

            # Log cache stats
            global_count = self.global_cache.sample_count()
            print(f"✓ Global cache: {global_count} samples")
            for name, cache in self.bucket_caches.items():
                count = cache.sample_count()
                holdout_tag = " (holdout)" if cache.is_holdout else ""
                print(f"  [{name}]{holdout_tag}: {count} samples")

        # Run each strategy on its assigned buckets (read from strategy.buckets)
        for strategy in self.strategies:
            if not strategy.should_run():
                continue

            bucket_names = strategy.buckets  # Read directly from strategy

            # No bucket bindings -> use global cache
            if not bucket_names:
                can_run, reason = strategy.can_run(self.global_cache)
                if not can_run:
                    print(f"⚠️ Skipping {strategy.name}: {reason}")
                    continue
                try:
                    strategy.run(self.global_cache, pl_module, trainer)
                except Exception as e:
                    print(f"Warning: Strategy {strategy.name} failed: {e}")
                continue

            # Run on each bucket
            # If only one bucket is assigned, don't suffix the metric with the bucket name
            # (The strategy name itself, e.g. "transfer_bridge", provides enough context)
            use_bucket_suffix = len(bucket_names) > 1

            for bucket_name in bucket_names:
                if bucket_name not in self.bucket_caches:
                    print(f"⚠️ Bucket '{bucket_name}' not found for {strategy.name}")
                    continue
                
                cache = self.bucket_caches[bucket_name]
                can_run, reason = strategy.can_run(cache)
                
                if not can_run:
                    # Only warn if verbose or if it's a critical single-bucket strategy
                    if len(bucket_names) == 1:
                        print(f"⚠️ Skipping {strategy.name} on {bucket_name}: {reason}")
                    continue
                
                suffix = f"_{bucket_name}" if use_bucket_suffix else ""
                
                try:
                    strategy.run(cache, pl_module, trainer, metric_suffix=suffix)
                except Exception as e:
                    print(f"Warning: Strategy {strategy.name} on {bucket_name} failed: {e}")

        # Run garbage collection after validation to free memory
        # This helps prevent memory buildup when using tf.data pipelines
        # Can be disabled on high-memory systems (cluster) via config
        if self.run_gc_after_validation:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


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
        # Convert OmegaConf to dict if needed
        model_config = dict(pl_module.model_config) if hasattr(pl_module.model_config, 'items') else pl_module.model_config
        self.ema_model = type(pl_module.model)(
            **model_config
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
