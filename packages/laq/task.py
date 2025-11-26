"""
LAPA Lightning Module for training

Updated for LAPA transformer-based architecture with:
- MSE loss only (no VQ losses)
- Input format: [B, 3, 2, 256, 256]
- Output format: [B, 3, 1, 256, 256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from packages.laq.models.lapa import LAPA, create_lapa_from_config
from packages.common.logging import (
    log_reconstruction_images,
    log_training_metrics,
    log_codebook_heatmap
)


class LAQModule(pl.LightningModule):
    """
    Lightning module for LAPA (Latent Action Pretraining from Videos) training.
    
    This module implements the LAPA architecture with:
    - Transformer-based encoder (spatial + temporal)
    - NSVQ quantizer (delta quantization, single codebook)
    - Cross-attention decoder
    - MSE loss only (no VQ-specific losses)
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize LAPA model
        self.lapa = create_lapa_from_config({'model': model_config})
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # Loss function (MSE only for LAPA)
        self.reconstruction_loss = nn.MSELoss()
    
    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through LAPA pipeline.
        
        Args:
            frames: Input frame pairs [B, 3, 2, 256, 256]
            
        Returns:
            reconstructed: Reconstructed next frame [B, 3, 1, 256, 256]
            indices: Discrete latent action codes [B, 4]
            perplexity: Codebook usage metric
        """
        return self.lapa(frames)
    
    def compute_loss(
        self,
        frames: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss (MSE only for LAPA).
        
        Args:
            frames: Input frame pairs [B, 3, 2, 256, 256]
            reconstructed: Reconstructed frame [B, 3, 1, 256, 256]
            
        Returns:
            Dictionary with loss_recon key
        """
        # Target is the second frame: [B, 3, 2, 256, 256] → [B, 3, 1, 256, 256]
        target_frame = frames[:, :, 1:2, :, :]
        
        # Compute MSE loss
        recon_loss = self.reconstruction_loss(reconstructed, target_frame)
        
        return {
            'loss_recon': recon_loss
        }
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step with comprehensive logging."""
        # Handle batch from dataloader (may be list or tensor)
        if isinstance(batch, list):
            frames = batch[0]  # Extract tensor from list
        else:
            frames = batch
        
        # Forward pass
        reconstructed, indices, perplexity = self(frames)
        
        # Compute loss (MSE only)
        losses = self.compute_loss(frames, reconstructed)
        
        # Log metrics
        self.log('train/loss_recon', losses['loss_recon'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/perplexity', perplexity, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log codebook utilization
        codebook_util = self.lapa.get_codebook_utilization()
        self.log('train/codebook_utilization', codebook_util, on_step=True, on_epoch=True, prog_bar=True)
        
        # Compute additional metrics
        with torch.no_grad():
            # PSNR
            target_frame = frames[:, :, 1:2, :, :]
            mse = F.mse_loss(reconstructed, target_frame)
            psnr = 10 * torch.log10(4.0 / mse)  # Assuming values in [-1, 1], range = 2, range^2 = 4
            self.log('train/psnr', psnr, on_step=True, on_epoch=True, prog_bar=False)
        
        return losses['loss_recon']
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step with reconstruction visualization."""
        # Handle batch from dataloader (may be list or tensor)
        if isinstance(batch, list):
            frames = batch[0]  # Extract tensor from list
        else:
            frames = batch
        
        # Forward pass
        reconstructed, indices, perplexity = self(frames)
        
        # Compute loss
        losses = self.compute_loss(frames, reconstructed)
        
        # Log metrics
        self.log('val/loss_recon', losses['loss_recon'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/perplexity', perplexity.detach(), on_step=False, on_epoch=True, prog_bar=True)
        
        # Log codebook utilization
        codebook_util = self.lapa.get_codebook_utilization()
        self.log('val/codebook_utilization', codebook_util, on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute additional metrics
        with torch.no_grad():
            # PSNR
            target_frame = frames[:, :, 1:2, :, :]
            mse = F.mse_loss(reconstructed, target_frame)
            psnr = 10 * torch.log10(4.0 / mse)
            self.log('val/psnr', psnr, on_step=False, on_epoch=True, prog_bar=False)
        
        # Log reconstruction images (first batch only)
        if batch_idx == 0:
            target_frame = frames[:, :, 1:2, :, :].squeeze(2)  # [B, 3, 256, 256]
            recon_frame = reconstructed.squeeze(2)  # [B, 3, 256, 256]
            
            recon_images = log_reconstruction_images(
                target_frame, recon_frame, self.global_step, max_images=8
            )
            
            # Log to WandB (TensorBoard doesn't support .log() method)
            if hasattr(self.logger, 'experiment'):
                # Check if it's WandB (has .log method) or TensorBoard (has .add_image)
                if hasattr(self.logger.experiment, 'log'):
                    # WandB logger
                    self.logger.experiment.log({
                        "val/reconstruction_images": recon_images["reconstruction_images"],
                        "step": self.global_step
                    })
                # TensorBoard logging is handled via self.log() calls above
            
            # Log codebook usage heatmap
            usage_data = log_codebook_heatmap(
                indices,
                self.lapa.codebook_size,
                self.lapa.code_seq_len,
                self.global_step
            )
            
            # Log codebook usage to WandB (TensorBoard doesn't support tables)
            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'log'):
                    # WandB logger
                    self.logger.experiment.log({
                        "val/codebook_usage_table": usage_data["codebook_usage_table"],
                        "step": self.global_step
                    })
                # TensorBoard doesn't support table logging, skip
        
        return losses['loss_recon']
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Prediction step for latent label generation.
        
        Used by script 3_generate_latent_labels.py to extract discrete latent codes.
        """
        frames = batch
        
        # Get latent action indices
        indices = self.lapa.predict_latent_actions(frames)
        
        return indices
    
    def configure_optimizers(self):
        """Configure optimizer with warmup and cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine decay
                progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


def create_laq_module_from_config(config: Dict[str, Any]) -> LAQModule:
    """
    Create LAQ module from configuration dictionary.
    
    Args:
        config: Configuration dictionary with model and training parameters
        
    Returns:
        LAQModule instance
    """
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    return LAQModule(
        model_config=model_config,
        learning_rate=training_config.get('lr', 1e-4),
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_steps=training_config.get('warmup_steps', 1000)
    )
