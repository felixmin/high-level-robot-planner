"""
PyTorch Lightning task wrapper for LAQ training.

Wraps LatentActionQuantization in a LightningModule with:
- LAPA-style optimizer (separate weight decay groups)
- Loss logging and codebook usage tracking
- Hydra configuration integration
- Optional EMA
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
import lightning.pytorch as pl
from omegaconf import DictConfig

from laq.models.latent_action_quantization import LatentActionQuantization


def separate_weight_decayable_params(
    params: List[nn.Parameter],
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Separate parameters into two groups for weight decay.

    Following LAPA convention:
    - 2D+ parameters (weights): apply weight decay
    - <2D parameters (biases, layernorms): no weight decay

    Args:
        params: All model parameters

    Returns:
        (wd_params, no_wd_params) tuple
    """
    wd_params = []
    no_wd_params = []

    for param in params:
        if param.ndim >= 2:
            wd_params.append(param)
        else:
            no_wd_params.append(param)

    return wd_params, no_wd_params


class LAQTask(pl.LightningModule):
    """
    PyTorch Lightning task for LAQ training.

    Wraps LatentActionQuantization model with training logic matching LAPA:
    - AdamW optimizer with separated weight decay groups
    - Cosine annealing LR scheduler with warmup
    - Reconstruction loss + codebook usage tracking
    - Visualization support via callbacks

    Args:
        model_config: Model configuration (DictConfig or dict)
        training_config: Training configuration (DictConfig or dict)
        use_ema: Whether to use EMA (handled via callback if True)
    """

    def __init__(
        self,
        model_config: DictConfig,
        training_config: DictConfig,
        use_ema: bool = False,
    ):
        super().__init__()

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Store configs
        self.model_config = model_config
        self.training_config = training_config
        self.use_ema = use_ema

        # Initialize LAQ model
        self.model = LatentActionQuantization(
            dim=model_config.dim,
            quant_dim=model_config.quant_dim,
            codebook_size=model_config.codebook_size,
            image_size=model_config.image_size,
            patch_size=model_config.patch_size,
            spatial_depth=model_config.spatial_depth,
            temporal_depth=model_config.temporal_depth,
            dim_head=model_config.dim_head,
            heads=model_config.heads,
            code_seq_len=model_config.code_seq_len,
            channels=model_config.get("channels", 3),
            attn_dropout=model_config.get("attn_dropout", 0.0),
            ff_dropout=model_config.get("ff_dropout", 0.0),
        )

        # Storage for validation batch (for visualization)
        self.validation_batch = None

    def forward(
        self,
        video: torch.Tensor,
        step: int = 0,
        return_recons_only: bool = False,
    ) -> Any:
        """Forward pass through LAQ model."""
        return self.model(video, step=step, return_recons_only=return_recons_only)

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Frame pairs [B, C, 2, H, W] or metadata dict
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Handle metadata dict if present
        if isinstance(batch, dict):
            frames = batch["frames"]
        else:
            frames = batch

        # Forward pass
        loss, num_unique = self.model(frames, step=self.global_step)

        # Log metrics (skip if no trainer attached, e.g., in unit tests)
        if self._trainer is not None:
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)
            self.log("train/num_unique_codes", num_unique, prog_bar=True, sync_dist=True)
            self.log("train/lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch: Frame pairs [B, C, 2, H, W] or metadata dict
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Handle metadata dict if present
        if isinstance(batch, dict):
            frames = batch["frames"]
        else:
            frames = batch

        # Forward pass
        loss, num_unique = self.model(frames, step=self.global_step)

        # Log metrics (skip if no trainer attached, e.g., in unit tests)
        if self._trainer is not None:
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)
            self.log("val/num_unique_codes", num_unique, sync_dist=True)

        # Store first batch for visualization
        if batch_idx == 0 and self.validation_batch is None:
            self.validation_batch = frames[:8].detach().cpu()  # Store up to 8 samples

        return loss

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Reset validation batch storage
        self.validation_batch = None

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and LR scheduler.

        Uses LAPA-style optimizer:
        - Separate weight decay groups (2D+ params vs <2D params)
        - AdamW with cosine annealing LR
        - Optional warmup

        Returns:
            Dict with optimizer and lr_scheduler configs
        """
        opt_config = self.training_config.optimizer
        sched_config = self.training_config.scheduler

        # Separate parameters for weight decay
        all_params = list(self.model.parameters())
        wd_params, no_wd_params = separate_weight_decayable_params(all_params)

        # Create optimizer with parameter groups
        optimizer = AdamW(
            [
                {
                    "params": wd_params,
                    "weight_decay": opt_config.weight_decay,
                },
                {
                    "params": no_wd_params,
                    "weight_decay": 0.0,
                },
            ],
            lr=opt_config.lr,
            betas=tuple(opt_config.betas),
            eps=opt_config.eps,
        )

        # Create LR scheduler
        if sched_config.type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=sched_config.T_max,
                eta_min=sched_config.min_lr,
            )
        else:
            raise NotImplementedError(f"Scheduler type '{sched_config.type}' not implemented")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }

    def get_validation_batch(self) -> Optional[torch.Tensor]:
        """
        Get stored validation batch for visualization.

        Returns:
            Validation batch tensor or None
        """
        return self.validation_batch

    def generate_reconstructions(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate reconstructions for visualization.

        Args:
            batch: Frame pairs [B, C, 2, H, W]

        Returns:
            Reconstructions [B, C, H, W]
        """
        self.eval()
        with torch.no_grad():
            recons = self.model(batch.to(self.device), return_recons_only=True)
        self.train()
        return recons
