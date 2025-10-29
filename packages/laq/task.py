"""
LAQ Lightning Module for training
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from packages.laq.models.encoder import Encoder
from packages.laq.models.quantizer import VectorQuantizer
from packages.laq.models.decoder import Decoder
from packages.common.logging import (
    log_reconstruction_images,
    log_training_metrics,
    log_codebook_heatmap
)


class LAQModule(pl.LightningModule):
    """
    Lightning module for LAQ (Latent Action Quantization) training.
    
    This module combines the encoder, quantizer, and decoder into a complete
    VQ-VAE system for learning discrete latent representations of video transitions.
    """
    
    def __init__(
        self,
        encoder_config: Dict[str, Any],
        quantizer_config: Dict[str, Any],
        decoder_config: Dict[str, Any],
        loss_weights: Dict[str, float] = None,
        learning_rate: float = 1e-4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize components
        self.encoder = Encoder(**encoder_config)
        self.quantizer = VectorQuantizer(**quantizer_config)
        self.decoder = Decoder(**decoder_config)
        
        # Loss weights (focus heavily on reconstruction)
        self.loss_weights = loss_weights or {
            'reconstruction': 10.0,  # Increased to prioritize reconstruction
            'codebook': 0.01,  # Further reduced
            'commitment': 0.01  # Further reduced
        }
        
        # Learning rate
        self.learning_rate = learning_rate
        
        # Loss function
        self.reconstruction_loss = nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the LAQ pipeline.
        
        Args:
            x: Input tensor of shape [B, 6, H, W] (concatenated frames)
            
        Returns:
            reconstructed: Reconstructed output [B, 3, H, W]
            quantized: Quantized latent features [B, num_tokens, embedding_dim]
            losses: Dictionary of VQ losses
        """
        # Encode
        encoded = self.encoder(x)
        
        # Quantize
        quantized, indices, vq_losses = self.quantizer(encoded)
        
        # Decode
        reconstructed = self.decoder(quantized)
        
        return reconstructed, quantized, vq_losses
    
    def compute_loss(self, x: torch.Tensor, reconstructed: torch.Tensor, vq_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss from reconstruction and VQ losses.
        
        Args:
            x: Input tensor
            reconstructed: Reconstructed tensor
            vq_losses: VQ losses from quantizer
            
        Returns:
            Dictionary of all losses
        """
        # Reconstruction loss (reconstruct frame_t+1 from concatenated input)
        target_frame = x[:, 3:]  # Take last 3 channels as target (frame_t+1)
        recon_loss = self.reconstruction_loss(reconstructed, target_frame)
        
        # VQ losses
        codebook_loss = vq_losses['codebook_loss']
        commitment_loss = vq_losses['commitment_loss']
        
        # Total loss
        total_loss = (
            self.loss_weights['reconstruction'] * recon_loss +
            self.loss_weights['codebook'] * codebook_loss +
            self.loss_weights['commitment'] * commitment_loss
        )
        
        return {
            'loss_total': total_loss,
            'loss_reconstruction': recon_loss,
            'loss_codebook': codebook_loss,
            'loss_commitment': commitment_loss
        }
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step with comprehensive logging."""
        x = batch
        
        # Forward pass
        reconstructed, quantized, vq_losses = self(x)
        
        # Compute losses
        losses = self.compute_loss(x, reconstructed, vq_losses)
        
        # Get quantizer indices for logging
        _, indices, _ = self.quantizer(self.encoder(x))
        
        # Log comprehensive metrics
        metrics = log_training_metrics(
            losses,
            indices,
            self.quantizer.vocab_size,
            self.quantizer.num_tokens,
            self.global_step
        )
        
        # Log all metrics
        for key, value in metrics.items():
            if key != 'step':
                self.log(f'train/{key}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        return losses['loss_total']
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step with reconstruction visualization."""
        x = batch
        
        # Forward pass
        reconstructed, quantized, vq_losses = self(x)
        
        # Compute losses
        losses = self.compute_loss(x, reconstructed, vq_losses)
        
        # Get quantizer indices for logging
        _, indices, _ = self.quantizer(self.encoder(x))
        
        # Log comprehensive metrics
        metrics = log_training_metrics(
            losses,
            indices,
            self.quantizer.vocab_size,
            self.quantizer.num_tokens,
            self.global_step
        )
        
        # Log all metrics
        for key, value in metrics.items():
            if key != 'step':
                self.log(f'val/{key}', value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log reconstruction images (as specified in PLAN.md)
        if batch_idx == 0:  # Only log on first batch to avoid spam
            target_frame = x[:, 3:]  # Take last 3 channels as target (frame_t+1)
            recon_images = log_reconstruction_images(
                target_frame, reconstructed, self.global_step, max_images=8
            )
            
            # Log reconstruction images
            self.logger.experiment.log({
                "val/reconstruction_images": recon_images["reconstruction_images"],
                "step": self.global_step
            })
            
            # Log codebook usage table
            usage_data = log_codebook_heatmap(
                indices, self.quantizer.vocab_size, self.quantizer.num_tokens, self.global_step
            )
            self.logger.experiment.log({
                "val/codebook_usage_table": usage_data["codebook_usage_table"],
                "step": self.global_step
            })
        
        return losses['loss_total']
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        return optimizer


def create_laq_module_from_config(config: Dict[str, Any]) -> LAQModule:
    """
    Create LAQ module from configuration dictionary.
    
    Args:
        config: Configuration dictionary with encoder, quantizer, decoder configs
        
    Returns:
        LAQModule instance
    """
    return LAQModule(
        encoder_config=config['encoder'],
        quantizer_config=config['quantizer'],
        decoder_config=config['decoder'],
        loss_weights=config.get('loss_weights', None),
        learning_rate=config.get('learning_rate', 1e-4)
    )
