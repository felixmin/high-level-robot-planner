"""
Integration tests for the complete LAPA model.

Tests the full encoder-NSVQ-decoder pipeline.
"""

import pytest
import torch
from packages.laq.models.lapa import LAPA, create_lapa_from_config


@pytest.mark.integration
class TestLAPAIntegration:
    """Integration tests for the full LAPA pipeline."""
    
    def test_lapa_full_forward(self, device, lapa_config):
        """Test full LAPA forward pass."""
        batch_size = 2
        
        model = LAPA(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            in_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            spatial_depth=lapa_config['spatial_depth'],
            temporal_depth=lapa_config['temporal_depth'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head'],
            mlp_ratio=lapa_config['mlp_ratio'],
            dropout=lapa_config['dropout'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        # Input: [B, 3, 2, 256, 256]
        frames = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device
        )
        
        reconstructed, indices, perplexity = model(frames)
        
        # Check output shapes (decoder adds temporal dimension)
        assert reconstructed.shape == (
            batch_size,
            lapa_config['channels'],
            1,
            lapa_config['image_size'],
            lapa_config['image_size']
        )
        assert indices.shape == (batch_size, 4)
        assert perplexity.shape == ()  # Scalar
        
        # Check validity
        assert not torch.isnan(reconstructed).any()
        assert not torch.isnan(perplexity).any()
        assert not torch.isnan(indices).any()
        
        # Indices should be in valid range
        assert (indices >= 0).all()
        assert (indices < lapa_config['codebook_size']).all()
    
    def test_lapa_end_to_end_gradient_flow(self, device, lapa_config):
        """Test gradient flow through entire LAPA pipeline."""
        batch_size = 2
        
        model = LAPA(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            in_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            spatial_depth=lapa_config['spatial_depth'],
            temporal_depth=lapa_config['temporal_depth'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        frames = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device,
            requires_grad=True
        )
        
        reconstructed, indices, perplexity = model(frames)
        
        # Compute MSE loss
        # Target is the second frame: [B, C, 2, H, W] → [B, C, 1, H, W]
        target = frames[:, :, 1:2, :, :]  # Last frame as target
        loss = torch.nn.functional.mse_loss(reconstructed, target)
        loss.backward()
        
        # Check gradients
        assert frames.grad is not None
        assert not torch.isnan(frames.grad).any()
        assert not torch.isinf(frames.grad).any()
    
    def test_lapa_predict_latent_actions(self, device, lapa_config):
        """Test latent action prediction."""
        batch_size = 2
        
        model = LAPA(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            in_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            spatial_depth=lapa_config['spatial_depth'],
            temporal_depth=lapa_config['temporal_depth'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        frames = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device
        )
        
        indices = model.predict_latent_actions(frames)
        
        # Check shape
        assert indices.shape == (batch_size, 4)
        
        # Check validity
        assert (indices >= 0).all()
        assert (indices < lapa_config['codebook_size']).all()
    
    def test_lapa_codebook_utilization(self, device, lapa_config):
        """Test codebook utilization metric."""
        batch_size = 4
        
        model = LAPA(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            in_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            spatial_depth=lapa_config['spatial_depth'],
            temporal_depth=lapa_config['temporal_depth'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        frames = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device
        )
        
        _, _, indices = model(frames)
        utilization = model.get_codebook_utilization()
        
        # Utilization should be between 0 and 1
        assert 0.0 <= utilization <= 1.0
    
    def test_lapa_perplexity(self, device, lapa_config):
        """Test perplexity calculation."""
        batch_size = 4
        
        model = LAPA(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            in_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            spatial_depth=lapa_config['spatial_depth'],
            temporal_depth=lapa_config['temporal_depth'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        frames = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device
        )
        
        # Perplexity is returned from forward pass
        _, _, perplexity = model(frames)
        
        # Perplexity should be between 1 and codebook_size
        assert 1.0 <= perplexity.item() <= lapa_config['codebook_size']
    
    def test_create_lapa_from_config(self, lapa_config):
        """Test LAPA creation from config."""
        config = {'model': lapa_config}
        
        model = create_lapa_from_config(config)
        
        assert isinstance(model, LAPA)
        assert model.encoder.dim == lapa_config['dim']
        assert model.nsvq.codebook_size == lapa_config['codebook_size']
        assert model.decoder.dim == lapa_config['dim']
    
    def test_lapa_reconstruction_quality(self, device, lapa_config):
        """Test that LAPA produces reasonable reconstructions."""
        batch_size = 2
        
        model = LAPA(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            in_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            spatial_depth=lapa_config['spatial_depth'],
            temporal_depth=lapa_config['temporal_depth'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        # Create frames with known structure
        frames = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device
        )
        
        reconstructed, _, _ = model(frames)
        
        # Reconstruction should be in valid range [-1, 1] (Tanh output)
        assert reconstructed.min() >= -1.0
        assert reconstructed.max() <= 1.0
        
        # MSE loss should be reasonable (not too large)
        # Target is the second frame: [B, C, 2, H, W] → [B, C, 1, H, W]
        target = frames[:, :, 1:2, :, :]
        mse = torch.nn.functional.mse_loss(reconstructed, target)
        assert mse.item() < 10.0  # Reasonable upper bound for random init


@pytest.mark.integration
@pytest.mark.slow
class TestLAPATraining:
    """Integration tests for LAPA training behavior."""
    
    def test_lapa_training_step(self, device, lapa_config):
        """Test a single training step."""
        batch_size = 2
        
        model = LAPA(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            in_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            spatial_depth=lapa_config['spatial_depth'],
            temporal_depth=lapa_config['temporal_depth'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        frames = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device
        )
        # Target is the second frame: [B, C, 2, H, W] → [B, C, 1, H, W]
        target = frames[:, :, 1:2, :, :]
        
        # Training step
        optimizer.zero_grad()
        reconstructed, _, _ = model(frames)
        loss = torch.nn.functional.mse_loss(reconstructed, target)
        loss.backward()
        optimizer.step()
        
        # Loss should be computed successfully
        assert not torch.isnan(loss)
        assert loss.item() > 0
    
    def test_lapa_multiple_training_steps(self, device, lapa_config):
        """Test multiple training steps show loss improvement."""
        batch_size = 2
        
        model = LAPA(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            in_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            spatial_depth=lapa_config['spatial_depth'],
            temporal_depth=lapa_config['temporal_depth'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Use same batch for overfitting test
        frames = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device
        )
        # Target is the second frame: [B, C, 2, H, W] → [B, C, 1, H, W]
        target = frames[:, :, 1:2, :, :]
        
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            reconstructed, _, _ = model(frames)
            loss = torch.nn.functional.mse_loss(reconstructed, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease (overfitting to single batch)
        assert losses[-1] < losses[0]

