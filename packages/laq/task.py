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
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
import lightning.pytorch as pl
from omegaconf import DictConfig

from laq.models.latent_action_quantization import LatentActionQuantization
from laq.models.dino import DINOFeatureExtractor


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
            use_dinov3_encoder=model_config.get("use_dinov3_encoder", False),
            dinov3_model_name=model_config.get("dinov3_model_name", "facebook/dinov3-vits16-pretrain-lvd1689m"),
            dinov3_pool_to_grid=model_config.get("dinov3_pool_to_grid", None),
        )
        
        # Perceptual Loss Setup
        self.perceptual_loss_config = training_config.get("perceptual_loss", {})
        self.use_perceptual_loss = self.perceptual_loss_config.get("enabled", False)
        
        if self.use_perceptual_loss:
            model_name = self.perceptual_loss_config.get("model_name", "facebook/dinov3-vits16-pretrain-lvd1689m")
            print(f"Initializing DINO Perceptual Loss with {model_name}")
            # Handle both int and tuple image_size
            img_size = model_config.image_size
            target_size = img_size[0] if isinstance(img_size, (list, tuple)) else img_size
            self.perceptual_loss_net = DINOFeatureExtractor(
                model_name=model_name,
                freeze=True,
                target_size=target_size
            )
            # Ensure it's in eval mode and frozen (handled by init but good to be safe)
            self.perceptual_loss_net.eval()
            for p in self.perceptual_loss_net.parameters():
                p.requires_grad = False

        # Storage for validation and training batches (for visualization)
        self.validation_batch = None
        self.training_batch = None

    def forward(
        self,
        video: torch.Tensor,
        step: int = 0,
        return_recons_only: bool = False,
        return_only_codebook_ids: bool = False,
    ) -> Any:
        """Forward pass through LAQ model."""
        return self.model(
            video, 
            step=step, 
            return_recons_only=return_recons_only,
            return_only_codebook_ids=return_only_codebook_ids
        )

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
        # model.forward returns (total_decoder_loss, num_unique, recon_video, aux_pixel_loss)
        combined_decoder_loss, num_unique, recon_video, aux_pixel_loss = self.model(frames, step=self.global_step)
        
        total_loss = combined_decoder_loss

        # Add Perceptual Loss
        if self.use_perceptual_loss:
            # Target is the second frame (index 1)
            # frames: [B, C, 2, H, W] -> target: [B, C, H, W]
            target_frame = frames[:, :, 1]
            
            # Extract features for target and recon
            layers = self.perceptual_loss_config.get("layers", [6, 11]) # Default for small
            
            # Using torch.no_grad() for target features
            with torch.no_grad():
                target_feats = self.perceptual_loss_net(
                    target_frame, 
                    output_hidden_states=True, 
                    layer_indices=layers
                )
            
            # Recon features (keep grad)
            # Clamp reconstruction to [0, 1] before passing to DINO (which expects valid images)
            # This is critical because DINO normalization (mean/std) assumes [0, 1] inputs.
            # Unbounded reconstruction can lead to exploded features/gradients.
            recon_video_clamped = torch.clamp(recon_video, 0.0, 1.0)
            
            recon_feats = self.perceptual_loss_net(
                recon_video_clamped, 
                output_hidden_states=True, 
                layer_indices=layers
            )
            
            # Compute L1 loss between features
            p_loss = 0.0
            for r_f, t_f in zip(recon_feats, target_feats):
                p_loss += F.l1_loss(r_f, t_f)
            
            weight = self.perceptual_loss_config.get("weight", 1.0)
            total_loss = total_loss + (weight * p_loss)
            
            if self._trainer is not None:
                self.log("train/perceptual_loss", p_loss, prog_bar=True, sync_dist=True)

        # Log metrics (skip if no trainer attached, e.g., in unit tests)
        if self._trainer is not None:
            self.log("train/loss", total_loss, prog_bar=True, sync_dist=True)
            self.log("train/combined_decoder_loss", combined_decoder_loss, prog_bar=True, sync_dist=True)
            self.log("train/aux_pixel_loss", aux_pixel_loss, prog_bar=True, sync_dist=True)
            # Main DINO loss is the remainder
            self.log("train/main_dino_loss", combined_decoder_loss - aux_pixel_loss, prog_bar=True, sync_dist=True)
            self.log("train/num_unique_codes", num_unique, prog_bar=True, sync_dist=True)
            self.log("train/lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        # Store first batch for visualization
        if batch_idx == 0 and self.training_batch is None:
            self.training_batch = frames[:8].detach().cpu()  # Store up to 8 samples

        return total_loss

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
        # (LAPA behavior: codebook replacement only during training via step != 0 check)
        combined_decoder_loss, num_unique, recon_video, aux_pixel_loss = self.model(frames, step=0)
        
        total_loss = combined_decoder_loss
        
        if self.use_perceptual_loss:
            target_frame = frames[:, :, 1]
            layers = self.perceptual_loss_config.get("layers", [6, 11])
            
            # No grad for validation
            with torch.no_grad():
                target_feats = self.perceptual_loss_net(
                    target_frame, output_hidden_states=True, layer_indices=layers
                )
                
                # Clamp for consistency with training
                recon_video_clamped = torch.clamp(recon_video, 0.0, 1.0)
                recon_feats = self.perceptual_loss_net(
                    recon_video_clamped, output_hidden_states=True, layer_indices=layers
                )
                
                p_loss = 0.0
                for r_f, t_f in zip(recon_feats, target_feats):
                    p_loss += F.l1_loss(r_f, t_f)
                
            weight = self.perceptual_loss_config.get("weight", 1.0)
            total_loss = total_loss + (weight * p_loss)
            
            if self._trainer is not None:
                self.log("val/perceptual_loss", p_loss, sync_dist=True)

        # Log metrics (skip if no trainer attached, e.g., in unit tests)
        if self._trainer is not None:
            self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)
            self.log("val/combined_decoder_loss", combined_decoder_loss, sync_dist=True)
            self.log("val/aux_pixel_loss", aux_pixel_loss, sync_dist=True)
            self.log("val/main_dino_loss", combined_decoder_loss - aux_pixel_loss, sync_dist=True)
            self.log("val/num_unique_codes", num_unique, sync_dist=True)

        # Store first batch for visualization
        if batch_idx == 0 and self.validation_batch is None:
            self.validation_batch = frames[:8].detach().cpu()  # Store up to 8 samples

        return total_loss

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
