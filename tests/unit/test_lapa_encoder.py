"""
Unit tests for LAPA encoder.

Tests the PatchEmbed and LAPAEncoder modules.
"""

import pytest
import torch
from packages.laq.models.encoder import PatchEmbed, LAPAEncoder, create_encoder_from_config


@pytest.mark.unit
class TestPatchEmbed:
    """Test cases for patch embedding."""
    
    def test_patch_embed_forward(self, device, image_size, patch_size):
        """Test patch embedding forward pass."""
        batch_size, channels = 2, 3
        dim = 1024
        
        model = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=channels,
            embed_dim=dim
        ).to(device)
        
        # PatchEmbed expects [B, C, H, W], not [B, C, T, H, W]
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)
        output = model(x)
        
        num_patches = (image_size // patch_size) ** 2
        assert output.shape == (batch_size, num_patches, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_patch_embed_output_shape(self, device):
        """Test that patch embedding produces correct shape."""
        batch_size, channels = 2, 3
        image_size, patch_size, dim = 256, 32, 1024
        
        model = PatchEmbed(image_size, patch_size, channels, dim).to(device)
        # PatchEmbed expects [B, C, H, W], not [B, C, T, H, W]
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)
        
        output = model(x)
        expected_patches = (256 // 32) ** 2  # 64 patches
        
        assert output.shape == (batch_size, expected_patches, dim)


@pytest.mark.unit
class TestLAPAEncoder:
    """Test cases for LAPA encoder."""
    
    def test_encoder_forward(self, device, lapa_config):
        """Test encoder forward pass."""
        batch_size = 2
        config = lapa_config
        
        model = LAPAEncoder(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            in_channels=config['channels'],
            dim=config['dim'],
            spatial_depth=config['spatial_depth'],
            temporal_depth=config['temporal_depth'],
            heads=config['heads'],
            dim_head=config['dim_head'],
            mlp_ratio=config['mlp_ratio'],
            dropout=config['dropout']
        ).to(device)
        
        # Input: [B, C, T=2, H, W]
        x = torch.randn(
            batch_size, config['channels'], 2, 
            config['image_size'], config['image_size'],
            device=device
        )
        
        first_tokens, last_tokens = model(x)
        
        num_patches = (config['image_size'] // config['patch_size']) ** 2
        assert first_tokens.shape == (batch_size, num_patches, config['dim'])
        assert last_tokens.shape == (batch_size, num_patches, config['dim'])
        assert not torch.isnan(first_tokens).any()
        assert not torch.isnan(last_tokens).any()
    
    def test_encoder_output_difference(self, device, lapa_config):
        """Test that first and last tokens are different."""
        batch_size = 2
        config = lapa_config
        
        model = LAPAEncoder(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            in_channels=config['channels'],
            dim=config['dim'],
            spatial_depth=config['spatial_depth'],
            temporal_depth=config['temporal_depth'],
            heads=config['heads'],
            dim_head=config['dim_head'],
            mlp_ratio=config['mlp_ratio'],
            dropout=0.0  # No dropout for this test
        ).to(device)
        
        # Create frames where first and last are different
        x = torch.randn(
            batch_size, config['channels'], 2,
            config['image_size'], config['image_size'],
            device=device
        )
        
        first_tokens, last_tokens = model(x)
        
        # Tokens should be different (not identical)
        assert not torch.allclose(first_tokens, last_tokens, atol=1e-4)
    
    def test_encoder_gradient_flow(self, device, lapa_config):
        """Test gradient flow through encoder."""
        batch_size = 2
        config = lapa_config
        
        model = LAPAEncoder(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            in_channels=config['channels'],
            dim=config['dim'],
            spatial_depth=config['spatial_depth'],
            temporal_depth=config['temporal_depth'],
            heads=config['heads'],
            dim_head=config['dim_head']
        ).to(device)
        
        x = torch.randn(
            batch_size, config['channels'], 2,
            config['image_size'], config['image_size'],
            device=device,
            requires_grad=True
        )
        
        first_tokens, last_tokens = model(x)
        loss = first_tokens.sum() + last_tokens.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_create_encoder_from_config(self):
        """Test encoder creation from config."""
        config = {
            'encoder': {
                'image_size': 256,
                'patch_size': 32,
                'channels': 3,
                'dim': 1024,
                'spatial_depth': 2,
                'temporal_depth': 2,
                'heads': 16,
                'dim_head': 64,
                'mlp_ratio': 4,
                'dropout': 0.0
            }
        }
        
        encoder = create_encoder_from_config(config)
        
        assert isinstance(encoder, LAPAEncoder)
        assert encoder.patch_embed.patch_size == 32
        assert encoder.dim == 1024


@pytest.mark.unit
@pytest.mark.slow
class TestLAPAEncoderPerformance:
    """Performance tests for LAPA encoder."""
    
    def test_encoder_different_batch_sizes(self, device, lapa_config):
        """Test encoder with different batch sizes."""
        config = lapa_config
        
        model = LAPAEncoder(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            in_channels=config['channels'],
            dim=config['dim'],
            spatial_depth=config['spatial_depth'],
            temporal_depth=config['temporal_depth'],
            heads=config['heads'],
            dim_head=config['dim_head']
        ).to(device)
        
        for batch_size in [1, 2, 4]:
            x = torch.randn(
                batch_size, config['channels'], 2,
                config['image_size'], config['image_size'],
                device=device
            )
            
            first_tokens, last_tokens = model(x)
            
            num_patches = (config['image_size'] // config['patch_size']) ** 2
            assert first_tokens.shape == (batch_size, num_patches, config['dim'])
            assert last_tokens.shape == (batch_size, num_patches, config['dim'])

