"""
Unit tests for LAPA decoder.

Tests the cross-attention decoder that reconstructs frames from context and action tokens.
"""

import pytest
import torch
from packages.laq.models.decoder import LAPADecoder, create_decoder_from_config


@pytest.mark.unit
class TestLAPADecoder:
    """Test cases for LAPA decoder."""
    
    def test_decoder_forward(self, device, lapa_config):
        """Test decoder forward pass."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = LAPADecoder(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            out_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head'],
            mlp_ratio=lapa_config['mlp_ratio'],
            dropout=lapa_config['dropout']
        ).to(device)
        
        # Context: first frame tokens [B, 64, 1024]
        context = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        
        # Actions: quantized action tokens [B, 1, 2, 2, 1024]
        actions = torch.randn(batch_size, 1, 2, 2, lapa_config['dim'], device=device)
        
        reconstructed = model(context, actions)
        
        # Output should be [B, 3, 1, 256, 256] (decoder adds temporal dimension)
        assert reconstructed.shape == (
            batch_size,
            lapa_config['channels'],
            1,
            lapa_config['image_size'],
            lapa_config['image_size']
        )
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()
    
    def test_decoder_output_range(self, device, lapa_config):
        """Test that decoder output is in valid range [-1, 1] due to Tanh."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = LAPADecoder(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            out_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head']
        ).to(device)
        
        context = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        actions = torch.randn(batch_size, 1, 2, 2, lapa_config['dim'], device=device)
        
        reconstructed = model(context, actions)
        
        # Tanh bounds output to [-1, 1]
        assert reconstructed.min() >= -1.0
        assert reconstructed.max() <= 1.0
    
    def test_decoder_gradient_flow(self, device, lapa_config):
        """Test gradient flow through decoder."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = LAPADecoder(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            out_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head']
        ).to(device)
        
        context = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device, requires_grad=True)
        actions = torch.randn(batch_size, 1, 2, 2, lapa_config['dim'], device=device, requires_grad=True)
        
        reconstructed = model(context, actions)
        loss = reconstructed.sum()
        loss.backward()
        
        assert context.grad is not None
        assert actions.grad is not None
        assert not torch.isnan(context.grad).any()
        assert not torch.isnan(actions.grad).any()
    
    def test_decoder_cross_attention_sensitivity(self, device, lapa_config):
        """Test that decoder output changes with different action tokens."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = LAPADecoder(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            out_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head']
        ).to(device)
        
        context = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        actions1 = torch.randn(batch_size, 1, 2, 2, lapa_config['dim'], device=device)
        actions2 = torch.randn(batch_size, 1, 2, 2, lapa_config['dim'], device=device)
        
        reconstructed1 = model(context, actions1)
        reconstructed2 = model(context, actions2)
        
        # Different actions should produce different reconstructions
        assert not torch.allclose(reconstructed1, reconstructed2, atol=1e-4)
    
    def test_decoder_context_sensitivity(self, device, lapa_config):
        """Test that decoder output changes with different context."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = LAPADecoder(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            out_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head']
        ).to(device)
        
        context1 = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        context2 = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        actions = torch.randn(batch_size, 1, 2, 2, lapa_config['dim'], device=device)
        
        reconstructed1 = model(context1, actions)
        reconstructed2 = model(context2, actions)
        
        # Different context should produce different reconstructions
        assert not torch.allclose(reconstructed1, reconstructed2, atol=1e-4)
    
    def test_create_decoder_from_config(self):
        """Test decoder creation from config."""
        config = {
            'decoder': {
                'image_size': 256,
                'patch_size': 32,
                'channels': 3,
                'dim': 1024,
                'decoder_depth': 2,
                'decoder_heads': 16,
                'dim_head': 64,
                'mlp_ratio': 4,
                'dropout': 0.0
            }
        }
        
        decoder = create_decoder_from_config(config['decoder'])
        
        assert isinstance(decoder, LAPADecoder)
        assert decoder.patch_size == 32
        assert decoder.dim == 1024


@pytest.mark.unit
@pytest.mark.slow
class TestLAPADecoderPerformance:
    """Performance tests for LAPA decoder."""
    
    def test_decoder_different_batch_sizes(self, device, lapa_config):
        """Test decoder with different batch sizes."""
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = LAPADecoder(
            image_size=lapa_config['image_size'],
            patch_size=lapa_config['patch_size'],
            out_channels=lapa_config['channels'],
            dim=lapa_config['dim'],
            decoder_depth=lapa_config['decoder_depth'],
            heads=lapa_config['heads'],
            dim_head=lapa_config['dim_head']
        ).to(device)
        
        for batch_size in [1, 2, 4]:
            context = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
            actions = torch.randn(batch_size, 1, 2, 2, lapa_config['dim'], device=device)
            
            reconstructed = model(context, actions)
            
            assert reconstructed.shape == (
                batch_size,
                lapa_config['channels'],
                1,
                lapa_config['image_size'],
                lapa_config['image_size']
            )

