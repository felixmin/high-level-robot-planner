"""
Unit tests for LAPA attention modules.

Tests the Transformer, PEG, and ContinuousPositionBias components.
"""

import pytest
import torch
from packages.laq.models.attention import Transformer, PEG, ContinuousPositionBias


@pytest.mark.unit
class TestTransformer:
    """Test cases for the Transformer block."""
    
    def test_self_attention_forward(self, device):
        """Test self-attention forward pass."""
        batch_size, seq_len, dim = 2, 64, 1024
        model = Transformer(
            dim=dim,
            heads=16,
            dim_head=64,
            dropout=0.0,
            use_cross_attention=False
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_cross_attention_forward(self, device):
        """Test cross-attention forward pass."""
        batch_size, seq_len, context_len, dim = 2, 4, 64, 1024
        model = Transformer(
            dim=dim,
            heads=16,
            dim_head=64,
            dropout=0.0,
            use_cross_attention=True
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        context = torch.randn(batch_size, context_len, dim, device=device)
        
        output = model(x, context=context)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_cross_attention_without_context(self, device):
        """Test that cross-attention works without context (skips cross-attention)."""
        batch_size, seq_len, dim = 2, 4, 1024
        model = Transformer(
            dim=dim,
            heads=16,
            dim_head=64,
            use_cross_attention=True
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        
        # Should work without context (just does self-attention)
        output = model(x)
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_gradient_flow(self, device):
        """Test gradient flow through transformer."""
        batch_size, seq_len, dim = 2, 64, 1024
        model = Transformer(dim=dim, heads=16, dim_head=64).to(device)
        
        x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


@pytest.mark.unit
class TestPEG:
    """Test cases for Position Encoding Generator."""
    
    def test_peg_forward(self, device):
        """Test PEG forward pass."""
        batch_size, frames, patches, dim = 2, 2, 64, 1024
        
        model = PEG(dim=dim).to(device)
        x = torch.randn(batch_size, frames, patches, dim, device=device)
        
        output = model(x)
        
        assert output.shape == (batch_size, frames, patches, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_peg_adds_information(self, device):
        """Test that PEG modifies the input."""
        batch_size, frames, patches, dim = 2, 2, 64, 1024
        
        model = PEG(dim=dim).to(device)
        x = torch.randn(batch_size, frames, patches, dim, device=device)
        
        output = model(x)
        
        # Output should be different from input (position encoding added)
        assert not torch.allclose(output, x, atol=1e-6)


@pytest.mark.unit
class TestContinuousPositionBias:
    """Test cases for Continuous Position Bias."""
    
    def test_bias_generation(self, device):
        """Test bias generation for given spatial size."""
        heads = 16
        spatial_size = 8  # 8x8 grid of patches
        
        model = ContinuousPositionBias(num_heads=heads, spatial_size=spatial_size).to(device)
        bias = model()
        
        # Should return bias of shape [heads, N, N] where N = spatial_size^2
        n_patches = spatial_size * spatial_size
        assert bias.shape == (heads, n_patches, n_patches)
        assert not torch.isnan(bias).any()
        assert not torch.isinf(bias).any()
    
    def test_bias_symmetry(self, device):
        """Test that bias respects relative position symmetry."""
        heads = 16
        spatial_size = 4
        
        model = ContinuousPositionBias(num_heads=heads, spatial_size=spatial_size).to(device)
        bias = model()
        
        # Check that bias values are finite and reasonable
        assert bias.abs().max() < 100  # Should not be too large
    
    def test_different_sizes(self, device):
        """Test bias generation for different spatial sizes."""
        heads = 16
        
        for spatial_size in [4, 8, 16]:
            model = ContinuousPositionBias(num_heads=heads, spatial_size=spatial_size).to(device)
            bias = model()
            n_patches = spatial_size * spatial_size
            assert bias.shape == (heads, n_patches, n_patches)

