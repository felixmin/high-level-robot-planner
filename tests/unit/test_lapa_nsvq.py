"""
Unit tests for LAPA NSVQ (Noise-Substitution Vector Quantization).

Tests the NSVQ module and its noise-substitution mechanism.
"""

import pytest
import torch
from packages.laq.models.nsvq import NSVQ


@pytest.mark.unit
class TestNSVQ:
    """Test cases for NSVQ quantization."""
    
    def test_nsvq_forward(self, device, lapa_config):
        """Test NSVQ forward pass."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = NSVQ(
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        first_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        last_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        
        action_tokens, indices, perplexity = model(first_tokens, last_tokens)
        
        # Check shapes
        assert action_tokens.shape == (batch_size, 1, 2, 2, lapa_config['dim'])
        assert indices.shape == (batch_size, 4)
        assert perplexity.shape == ()  # Scalar tensor
        
        # Check indices are in valid range
        assert (indices >= 0).all()
        assert (indices < lapa_config['codebook_size']).all()
        
        # Check for NaN/Inf
        assert not torch.isnan(action_tokens).any()
        assert not torch.isnan(indices).any()
        assert not torch.isnan(perplexity).any()
    
    def test_nsvq_delta_computation(self, device, lapa_config):
        """Test that NSVQ computes delta correctly."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = NSVQ(
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        first_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        last_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        
        action_tokens, indices, perplexity = model(first_tokens, last_tokens)
        
        # Perplexity should be a positive scalar
        assert perplexity.item() > 0
        assert perplexity.item() <= lapa_config['codebook_size']
    
    def test_nsvq_noise_substitution(self, device, lapa_config):
        """Test that noise substitution STE allows gradient flow."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = NSVQ(
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        first_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device, requires_grad=True)
        last_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device, requires_grad=True)
        
        action_tokens, indices, perplexity = model(first_tokens, last_tokens)
        
        # Compute loss and backprop
        loss = action_tokens.sum()
        loss.backward()
        
        # Gradients should flow through
        assert first_tokens.grad is not None
        assert last_tokens.grad is not None
        assert not torch.isnan(first_tokens.grad).any()
        assert not torch.isnan(last_tokens.grad).any()
    
    def test_nsvq_without_noise_substitution(self, device, lapa_config):
        """Test NSVQ without noise substitution (standard STE)."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = NSVQ(
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        first_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device, requires_grad=True)
        last_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device, requires_grad=True)
        
        action_tokens, indices, perplexity = model(first_tokens, last_tokens)
        
        # Should still work and allow gradients
        loss = action_tokens.sum()
        loss.backward()
        
        assert first_tokens.grad is not None
        assert last_tokens.grad is not None
    
    def test_nsvq_codebook_usage(self, device, lapa_config):
        """Test that NSVQ uses the codebook."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = NSVQ(
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        first_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        last_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        
        action_tokens, indices, perplexity = model(first_tokens, last_tokens)
        
        # Indices should use different codebook entries over multiple samples
        all_indices = []
        for _ in range(10):
            first = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
            last = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
            _, idx, _ = model(first, last)
            all_indices.append(idx)
        
        all_indices = torch.cat(all_indices, dim=0)
        unique_codes = torch.unique(all_indices)
        
        # Should use more than one code across multiple samples
        assert len(unique_codes) > 1
    
    def test_nsvq_deterministic(self, device, lapa_config):
        """Test that NSVQ is deterministic for same input."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = NSVQ(
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        model.eval()  # Set to eval mode to disable noise
        
        first_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        last_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        
        # Run twice
        with torch.no_grad():
            action_tokens1, indices1, _ = model(first_tokens, last_tokens)
            action_tokens2, indices2, _ = model(first_tokens, last_tokens)
        
        # Should be identical in eval mode
        assert torch.allclose(action_tokens1, action_tokens2)
        assert torch.equal(indices1, indices2)


@pytest.mark.unit
class TestNSVQEdgeCases:
    """Edge case tests for NSVQ."""
    
    def test_nsvq_zero_delta(self, device, lapa_config):
        """Test NSVQ when frames are identical (zero delta)."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = NSVQ(
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        # Identical frames
        tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device)
        
        action_tokens, indices, perplexity = model(tokens, tokens)
        
        # Should still produce valid output
        assert not torch.isnan(action_tokens).any()
        assert not torch.isnan(indices).any()
        assert not torch.isnan(perplexity).any()
    
    def test_nsvq_large_delta(self, device, lapa_config):
        """Test NSVQ with large change between frames."""
        batch_size = 2
        num_patches = (lapa_config['image_size'] // lapa_config['patch_size']) ** 2
        
        model = NSVQ(
            dim=lapa_config['dim'],
            quant_dim=lapa_config['quant_dim'],
            codebook_size=lapa_config['codebook_size'],
            code_seq_len=lapa_config['code_seq_len']
        ).to(device)
        
        # Very different frames
        first_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device) * 10
        last_tokens = torch.randn(batch_size, num_patches, lapa_config['dim'], device=device) * 10
        
        action_tokens, indices, perplexity = model(first_tokens, last_tokens)
        
        # Should still produce valid output without NaN
        assert not torch.isnan(action_tokens).any()
        assert not torch.isnan(indices).any()
        assert not torch.isnan(perplexity).any()

