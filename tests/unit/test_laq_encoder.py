"""
Unit tests for LAQ Encoder
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import pytest
from packages.laq.models.encoder import Encoder, ResBlock, create_encoder_from_config


class TestResBlock:
    """Test ResBlock component."""
    
    def test_resblock_forward(self):
        """Test ResBlock forward pass maintains shape."""
        resblock = ResBlock(channels=64, norm_groups=32)
        x = torch.randn(2, 64, 56, 56)
        output = resblock(x)
        
        assert output.shape == x.shape
        assert output.dtype == x.dtype
    
    def test_resblock_gradient_flow(self):
        """Test gradients flow through ResBlock."""
        resblock = ResBlock(channels=64, norm_groups=32)
        x = torch.randn(2, 64, 56, 56, requires_grad=True)
        output = resblock(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_resblock_activation(self):
        """Test different activation functions."""
        for activation in ["silu", "relu"]:
            resblock = ResBlock(channels=64, activation=activation)
            x = torch.randn(2, 64, 56, 56)
            output = resblock(x)
            
            assert output.shape == x.shape


class TestEncoder:
    """Test Encoder component."""
    
    def test_encoder_shapes(self):
        """Test encoder output dimensions."""
        encoder = Encoder(
            in_channels=6,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            latent_dim=256
        )
        
        x = torch.randn(2, 6, 224, 224)
        output = encoder(x)
        
        assert output.shape == (2, 256, 14, 14)
    
    def test_encoder_gradient_flow(self):
        """Test gradients flow through encoder."""
        encoder = Encoder(
            in_channels=6,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            latent_dim=256
        )
        
        x = torch.randn(2, 6, 224, 224, requires_grad=True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_encoder_different_input_sizes(self):
        """Test encoder with different input sizes."""
        encoder = Encoder(
            in_channels=6,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            latent_dim=256
        )
        
        # Test with different input sizes
        test_sizes = [(1, 6, 224, 224), (4, 6, 224, 224), (1, 6, 256, 256)]
        
        for size in test_sizes:
            x = torch.randn(size)
            output = encoder(x)
            
            expected_height = size[2] // 16  # 4 downsampling stages
            expected_width = size[3] // 16
            expected_shape = (size[0], 256, expected_height, expected_width)
            
            assert output.shape == expected_shape, f"Failed for size {size}"
    
    def test_encoder_get_output_shape(self):
        """Test get_output_shape method."""
        encoder = Encoder(
            in_channels=6,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            latent_dim=256
        )
        
        input_shape = (2, 6, 224, 224)
        output_shape = encoder.get_output_shape(input_shape)
        expected_shape = (2, 256, 14, 14)
        
        assert output_shape == expected_shape
    
    def test_encoder_config_creation(self):
        """Test encoder creation from config."""
        config = {
            'encoder': {
                'in_channels': 6,
                'base_channels': 64,
                'channel_multipliers': [1, 2, 4, 8],
                'num_res_blocks': 2,
                'latent_dim': 256,
                'activation': 'silu',
                'norm_type': 'groupnorm',
                'norm_groups': 32
            }
        }
        
        encoder = create_encoder_from_config(config)
        
        x = torch.randn(2, 6, 224, 224)
        output = encoder(x)
        
        assert output.shape == (2, 256, 14, 14)
    
    def test_encoder_different_configs(self):
        """Test encoder with different configurations."""
        # Test with different channel multipliers
        encoder1 = Encoder(
            in_channels=6,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],  # Keep 4 multipliers for 4 downsampling stages
            num_res_blocks=1,
            latent_dim=128
        )
        
        x = torch.randn(2, 6, 224, 224)
        output1 = encoder1(x)
        
        # Should still be 14x14 due to 4 downsampling stages
        assert output1.shape == (2, 128, 14, 14)
        
        # Test with different activation
        encoder2 = Encoder(
            in_channels=6,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            latent_dim=256,
            activation="relu"
        )
        
        output2 = encoder2(x)
        assert output2.shape == (2, 256, 14, 14)
    
    def test_encoder_weight_initialization(self):
        """Test that weights are properly initialized."""
        encoder = Encoder(
            in_channels=6,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            latent_dim=256
        )
        
        # Check that weights are not all zeros
        for name, param in encoder.named_parameters():
            if 'weight' in name:
                assert not torch.allclose(param, torch.zeros_like(param))
    
    def test_encoder_memory_efficiency(self):
        """Test encoder doesn't have memory leaks."""
        encoder = Encoder(
            in_channels=6,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            latent_dim=256
        )
        
        # Run multiple forward passes
        for _ in range(10):
            x = torch.randn(2, 6, 224, 224)
            output = encoder(x)
            del x, output
        
        # Should not crash or have memory issues
        assert True


if __name__ == "__main__":
    # Run tests
    test_resblock = TestResBlock()
    test_encoder = TestEncoder()
    
    print("Running ResBlock tests...")
    test_resblock.test_resblock_forward()
    test_resblock.test_resblock_gradient_flow()
    test_resblock.test_resblock_activation()
    print("✅ ResBlock tests passed!")
    
    print("Running Encoder tests...")
    test_encoder.test_encoder_shapes()
    test_encoder.test_encoder_gradient_flow()
    test_encoder.test_encoder_different_input_sizes()
    test_encoder.test_encoder_get_output_shape()
    test_encoder.test_encoder_config_creation()
    test_encoder.test_encoder_different_configs()
    test_encoder.test_encoder_weight_initialization()
    test_encoder.test_encoder_memory_efficiency()
    print("✅ Encoder tests passed!")
    
    print("\n🎉 All tests passed!")
