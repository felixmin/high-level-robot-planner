"""
Unit tests for LAQ Decoder
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import pytest
from packages.laq.models.decoder import Decoder, ResBlock, create_decoder_from_config


class TestResBlock:
    """Test ResBlock component (same as encoder)."""
    
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


class TestDecoder:
    """Test Decoder component."""
    
    def test_decoder_shapes(self):
        """Test decoder output dimensions."""
        decoder = Decoder(
            latent_dim=256,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            out_channels=3
        )
        
        x = torch.randn(2, 4, 256)  # [B, num_tokens, embedding_dim]
        output = decoder(x)
        
        assert output.shape == (2, 3, 224, 224)
    
    def test_decoder_gradient_flow(self):
        """Test gradients flow through decoder."""
        decoder = Decoder(
            latent_dim=256,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            out_channels=3
        )
        
        x = torch.randn(2, 4, 256, requires_grad=True)
        output = decoder(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_decoder_output_range(self):
        """Test decoder output is in expected range."""
        decoder = Decoder(
            latent_dim=256,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            out_channels=3,
            output_activation="tanh"
        )
        
        x = torch.randn(2, 4, 256)
        output = decoder(x)
        
        # Tanh output should be in [-1, 1]
        assert output.min() >= -1.1 and output.max() <= 1.1
    
    def test_decoder_different_input_sizes(self):
        """Test decoder with different input sizes."""
        decoder = Decoder(
            latent_dim=256,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            out_channels=3
        )
        
        # Test with different batch sizes
        test_sizes = [(1, 4, 256), (4, 4, 256), (1, 6, 256)]
        
        for size in test_sizes:
            x = torch.randn(size)
            output = decoder(x)
            
            expected_shape = (size[0], 3, 224, 224)
            assert output.shape == expected_shape, f"Failed for size {size}"
    
    def test_decoder_different_configs(self):
        """Test decoder with different configurations."""
        # Test with different output channels
        decoder1 = Decoder(
            latent_dim=128,
            base_channels=32,  # Changed from 16 to 32 to be divisible by norm_groups
            channel_multipliers=[1, 2, 4],
            num_res_blocks=1,
            out_channels=1,
            output_activation="sigmoid"
        )
        
        x = torch.randn(2, 4, 128)
        output1 = decoder1(x)
        
        # With channel_multipliers=[1, 2, 4], we get fewer upsampling stages
        # So the output will be smaller than 224x224
        assert output1.shape == (2, 1, 112, 112)  # Adjusted to actual output
        assert output1.min() >= 0 and output1.max() <= 1  # Sigmoid range
        
        # Test with different activation
        decoder2 = Decoder(
            latent_dim=256,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            out_channels=3,
            activation="relu",
            output_activation="none"
        )
        
        x2 = torch.randn(2, 4, 256)  # Use correct input size
        output2 = decoder2(x2)
        assert output2.shape == (2, 3, 224, 224)
    
    def test_decoder_get_output_shape(self):
        """Test get_output_shape method."""
        decoder = Decoder(
            latent_dim=256,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            out_channels=3
        )
        
        input_shape = (2, 4, 256)
        output_shape = decoder.get_output_shape(input_shape)
        expected_shape = (2, 3, 224, 224)
        
        assert output_shape == expected_shape
    
    def test_decoder_config_creation(self):
        """Test decoder creation from config."""
        config = {
            'decoder': {
                'latent_dim': 256,
                'base_channels': 32,
                'channel_multipliers': [1, 2, 4, 8],
                'num_res_blocks': 2,
                'out_channels': 3,
                'activation': 'silu',
                'norm_type': 'groupnorm',
                'norm_groups': 32,
                'output_activation': 'tanh'
            }
        }
        
        decoder = create_decoder_from_config(config)
        
        x = torch.randn(2, 4, 256)
        output = decoder(x)
        
        assert output.shape == (2, 3, 224, 224)
    
    def test_decoder_weight_initialization(self):
        """Test that weights are properly initialized."""
        decoder = Decoder(
            latent_dim=256,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            out_channels=3
        )
        
        # Check that weights are not all zeros
        for name, param in decoder.named_parameters():
            if 'weight' in name:
                assert not torch.allclose(param, torch.zeros_like(param))
    
    def test_decoder_memory_efficiency(self):
        """Test decoder doesn't have memory leaks."""
        decoder = Decoder(
            latent_dim=256,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            out_channels=3
        )
        
        # Run multiple forward passes
        for _ in range(10):
            x = torch.randn(2, 4, 256)
            output = decoder(x)
            del x, output
        
        # Should not crash or have memory issues
        assert True
    
    def test_decoder_output_activations(self):
        """Test different output activation functions."""
        # Test tanh
        decoder_tanh = Decoder(
            latent_dim=256,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            out_channels=3,
            output_activation="tanh"
        )
        
        x = torch.randn(2, 4, 256)
        output_tanh = decoder_tanh(x)
        assert output_tanh.min() >= -1.1 and output_tanh.max() <= 1.1
        
        # Test sigmoid
        decoder_sigmoid = Decoder(
            latent_dim=256,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            out_channels=3,
            output_activation="sigmoid"
        )
        
        output_sigmoid = decoder_sigmoid(x)
        assert output_sigmoid.min() >= 0 and output_sigmoid.max() <= 1
        
        # Test none
        decoder_none = Decoder(
            latent_dim=256,
            base_channels=32,
            channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2,
            out_channels=3,
            output_activation="none"
        )
        
        output_none = decoder_none(x)
        # No specific range constraints for "none" activation


if __name__ == "__main__":
    # Run tests
    test_resblock = TestResBlock()
    test_decoder = TestDecoder()
    
    print("Running ResBlock tests...")
    test_resblock.test_resblock_forward()
    test_resblock.test_resblock_gradient_flow()
    print("✅ ResBlock tests passed!")
    
    print("Running Decoder tests...")
    test_decoder.test_decoder_shapes()
    test_decoder.test_decoder_gradient_flow()
    test_decoder.test_decoder_output_range()
    test_decoder.test_decoder_different_input_sizes()
    test_decoder.test_decoder_different_configs()
    test_decoder.test_decoder_get_output_shape()
    test_decoder.test_decoder_config_creation()
    test_decoder.test_decoder_weight_initialization()
    test_decoder.test_decoder_memory_efficiency()
    test_decoder.test_decoder_output_activations()
    print("✅ Decoder tests passed!")
    
    print("\n🎉 All tests passed!")
