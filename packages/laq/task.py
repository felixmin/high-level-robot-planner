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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from laq.models.latent_action_quantization import LatentActionQuantization
from laq.models.flow import FlowConfig


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
        if not param.requires_grad:
            continue
            
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

        # Build flow config if specified
        flow_config = None
        if "flow" in model_config and model_config.flow is not None:
            flow_cfg = model_config.flow
            flow_config = FlowConfig(
                model=flow_cfg.model,
                loss_weight=flow_cfg.loss_weight,
                decoder_depth=flow_cfg.decoder_depth,
                warmup_steps=flow_cfg.get("warmup_steps", 0),
            )

        # Build codebook replacement schedule if specified
        codebook_replace_schedule = None
        if "codebook_replace_schedule" in model_config and model_config.codebook_replace_schedule is not None:
            # Convert from list of lists to list of tuples
            codebook_replace_schedule = [
                tuple(entry) for entry in model_config.codebook_replace_schedule
            ]

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
            use_dinov3_encoder=model_config.get("use_dinov3_encoder", False),
            dinov3_model_name=model_config.get("dinov3_model_name", "facebook/dinov3-vits16-pretrain-lvd1689m"),
            dinov3_pool_to_grid=model_config.get("dinov3_pool_to_grid", None),
            # Training decoder flags
            use_dino_decoder=model_config.get("use_dino_decoder", True),
            use_pixel_decoder=model_config.get("use_pixel_decoder", False),
            # Interpretability decoder flag
            use_aux_decoder=model_config.get("use_aux_decoder", True),
            flow_config=flow_config,
            codebook_replace_schedule=codebook_replace_schedule,
        )

        # Storage for validation and training batches (for visualization)
        self.validation_batch = None
        self.training_batch = None

        # Flag for one-time batch validation (to catch interface issues early)
        self._batch_validated = False

    def forward(
        self,
        video: torch.Tensor,
        step: int = 0,
        return_recons_only: bool = False,
        return_only_codebook_ids: bool = False,
    ) -> Any:
        """
        Forward pass through LAQ model.

        Returns:
            If return_recons_only: reconstructed frames [B, C, H, W]
            If return_only_codebook_ids: codebook indices [B, code_seq_len]
            Otherwise: (loss, metrics_dict)
        """
        return self.model(
            video,
            step=step,
            return_recons_only=return_recons_only,
            return_only_codebook_ids=return_only_codebook_ids,
        )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """
        Validate batch keys on first batch to catch interface issues early.

        This hook runs once per training to verify that the dataloader produces
        batches with standardized keys (frames, episode_id, frame_idx, etc.).
        Helps catch configuration errors (e.g., wrong collate function) early.

        Validates against STANDARD_BATCH_KEYS to ensure interface parity between
        LAQDataModule and OXEDataModule.
        """
        if self._batch_validated:
            return

        # Only validate when batch is a dict (metadata mode enabled)
        if isinstance(batch, dict):
            from common.data import validate_batch_keys, STANDARD_BATCH_KEYS
            import logging

            logger = logging.getLogger(__name__)

            # Validate all standard keys to ensure interface parity
            # This catches misconfigured dataloaders early
            validate_batch_keys(
                batch,
                required_keys=list(STANDARD_BATCH_KEYS),
                raise_on_missing=True,
            )

            # Log what keys we got (helpful for debugging)
            if self.trainer.is_global_zero:
                logger.info(f"Batch keys validated: {list(batch.keys())}")
                logger.info(f"Required standard keys present: {list(STANDARD_BATCH_KEYS)}")

        self._batch_validated = True

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

        # Forward pass - model returns (loss, metrics_dict)
        loss, metrics = self.model(frames, step=self.global_step)

        # Log metrics (skip if no trainer attached, e.g., in unit tests)
        if self._trainer is not None:
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)
            self.log("train/lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
            
            # Dynamic logging of model metrics
            for k, v in metrics.items():
                is_prog_bar = (k == "num_unique_codes")
                self.log(f"train/{k}", v, prog_bar=is_prog_bar, sync_dist=True)

        # Store first batch for visualization
        if batch_idx == 0 and self.training_batch is None:
            self.training_batch = frames[:8].detach().cpu()

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

        # Forward pass - use step=0 to avoid codebook replacement during validation
        loss, metrics = self.model(frames, step=0)

        # Log metrics (skip if no trainer attached, e.g., in unit tests)
        if self._trainer is not None:
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)
            
            # Dynamic logging of model metrics
            for k, v in metrics.items():
                self.log(f"val/{k}", v, sync_dist=True)

        # Store first batch for visualization
        if batch_idx == 0 and self.validation_batch is None:
            self.validation_batch = frames[:8].detach().cpu()

        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Reset training batch storage
        self.training_batch = None

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

        # Create LR scheduler (optional)
        if sched_config.get("type") == "none" or sched_config.get("type") is None:
            # No scheduler - return optimizer only
            return optimizer
        elif sched_config.type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=sched_config.T_max,
                eta_min=sched_config.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "lr",
                },
            }
        else:
            raise NotImplementedError(f"Scheduler type '{sched_config.type}' not implemented")

    def get_validation_batch(self) -> Optional[torch.Tensor]:
        """
        Get stored validation batch for visualization.

        Returns:
            Validation batch tensor or None
        """
        return self.validation_batch

    def get_training_batch(self) -> Optional[torch.Tensor]:
        """
        Get stored training batch for visualization.

        Returns:
            Training batch tensor or None
        """
        return self.training_batch

    def generate_reconstructions(
        self,
        batch: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Generate reconstructions for visualization.

        Args:
            batch: Frame pairs [B, C, 2, H, W]

        Returns:
            Reconstructions [B, C, H, W], or None if aux_decoder is disabled
        """
        self.eval()
        with torch.no_grad():
            recons = self.model(batch.to(self.device), return_recons_only=True)
        self.train()
        return recons

    def encode_latents(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode frame pairs to get latent actions and codebook indices.

        Args:
            batch: Frame pairs [B, C, 2, H, W]

        Returns:
            (latent_actions, codebook_indices)
            latent_actions: [B, code_seq_len, dim] projected to transformer dim
        """
        self.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            # Get codebook indices [B, code_seq_len]
            indices = self.model(batch, return_only_codebook_ids=True)
            # Get raw codebook vectors [B, code_seq_len, quant_dim]
            raw_latents = self.model.vq.codebooks[indices]
            # Project from quant_dim to transformer dim [B, code_seq_len, dim]
            latents = self.model.vq.project_out(raw_latents)
        self.train()
        return latents, indices

    def decode_with_latents(
        self,
        first_frames: torch.Tensor,
        latent_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode first frames with given latent actions.

        This enables latent transfer: apply action from one pair to another scene.

        Args:
            first_frames: First frames [B, C, 1, H, W] or [B, C, H, W]
            latent_actions: Latent actions from encoder

        Returns:
            Reconstructed next frames [B, C, 1, H, W]
        """
        import math
        from einops import rearrange

        self.eval()
        with torch.no_grad():
            first_frames = first_frames.to(self.device)
            latent_actions = latent_actions.to(self.device)

            # Ensure correct shape
            if first_frames.ndim == 4:
                first_frames = first_frames.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

            # Get first frame tokens
            # Use decoder_context_projection to match model forward pass (handles DINO vs learned embeddings)
            first_frame_tokens = self.model.decoder_context_projection(first_frames)

            # Reshape latents for decode
            code_seq_len = self.model.code_seq_len
            if math.sqrt(code_seq_len) % 1 == 0:
                action_h = int(math.sqrt(code_seq_len))
                action_w = int(math.sqrt(code_seq_len))
            elif code_seq_len == 2:
                action_h, action_w = 2, 1
            else:
                action_h, action_w = code_seq_len, 1

            # Reshape latents: [B, seq, dim] -> [B, t, h, w, d]
            if latent_actions.ndim == 2:
                latent_actions = latent_actions.unsqueeze(1)  # [B, dim] -> [B, 1, dim]
            latent_actions = rearrange(
                latent_actions, 'b (t h w) d -> b t h w d',
                t=1, h=action_h, w=action_w
            )

            # Decode
            recon = self.model.decode(first_frame_tokens, latent_actions)

        self.train()
        return recon
