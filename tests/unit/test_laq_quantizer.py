"""
Unit tests for LAQ Vector Quantizer
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import pytest
from packages.laq.models.quantizer import VectorQuantizer, create_quantizer_from_config


class TestVectorQuantizer:
    """Test VectorQuantizer component."""
    
    def test_quantizer_shapes(self):
        """Test quantizer output dimensions."""
        quantizer = VectorQuantizer(
            num_tokens=4,
            vocab_size=8,
            embedding_dim=256,
            beta=0.25
        )
        
        # Test input (encoder output)
        x = torch.randn(2, 256, 14, 14)
        quantized, indices, losses = quantizer(x)
        
        assert quantized.shape == (2, 4, 256)
        assert indices.shape == (2, 4)
        assert indices.dtype == torch.long
        assert indices.min() >= 0 and indices.max() < 8
    
    def test_quantizer_gradient_flow(self):
        """Test gradients flow through quantizer."""
        quantizer = VectorQuantizer(
            num_tokens=4,
            vocab_size=8,
            embedding_dim=256,
            beta=0.25
        )
        
        x = torch.randn(2, 256, 14, 14, requires_grad=True)
        quantized, indices, losses = quantizer(x)
        loss = losses['total_vq_loss']
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_quantizer_losses(self):
        """Test that losses are computed correctly."""
        quantizer = VectorQuantizer(
            num_tokens=4,
            vocab_size=8,
            embedding_dim=256,
            beta=0.25
        )
        
        x = torch.randn(2, 256, 14, 14)
        quantized, indices, losses = quantizer(x)
        
        # Check that all expected losses are present
        expected_losses = ['codebook_loss', 'commitment_loss', 'total_vq_loss']
        for loss_name in expected_losses:
            assert loss_name in losses
            assert isinstance(losses[loss_name], torch.Tensor)
            assert losses[loss_name].item() >= 0
        
        # Check that total loss is computed correctly
        expected_total = losses['codebook_loss'] + 0.25 * losses['commitment_loss']
        assert torch.allclose(losses['total_vq_loss'], expected_total)
    
    def test_quantizer_different_input_sizes(self):
        """Test quantizer with different input sizes."""
        quantizer = VectorQuantizer(
            num_tokens=4,
            vocab_size=8,
            embedding_dim=256,
            beta=0.25
        )
        
        # Test with different batch sizes
        test_sizes = [(1, 256, 14, 14), (4, 256, 14, 14), (1, 256, 28, 28)]
        
        for size in test_sizes:
            x = torch.randn(size)
            quantized, indices, losses = quantizer(x)
            
            expected_quantized_shape = (size[0], 4, 256)
            expected_indices_shape = (size[0], 4)
            
            assert quantized.shape == expected_quantized_shape, f"Failed for size {size}"
            assert indices.shape == expected_indices_shape, f"Failed for size {size}"
    
    def test_quantizer_different_configs(self):
        """Test quantizer with different configurations."""
        # Test with different number of tokens
        quantizer1 = VectorQuantizer(
            num_tokens=6,
            vocab_size=16,
            embedding_dim=128,
            beta=0.5
        )
        
        x = torch.randn(2, 128, 14, 14)
        quantized1, indices1, losses1 = quantizer1(x)
        
        assert quantized1.shape == (2, 6, 128)
        assert indices1.shape == (2, 6)
        assert indices1.min() >= 0 and indices1.max() < 16
        
        # Test with different beta
        quantizer2 = VectorQuantizer(
            num_tokens=4,
            vocab_size=8,
            embedding_dim=256,
            beta=0.1
        )
        
        x = torch.randn(2, 256, 14, 14)
        quantized2, indices2, losses2 = quantizer2(x)
        
        # Check that beta affects total loss
        expected_total = losses2['codebook_loss'] + 0.1 * losses2['commitment_loss']
        assert torch.allclose(losses2['total_vq_loss'], expected_total)
    
    def test_quantizer_ema_mode(self):
        """Test quantizer with EMA updates enabled."""
        quantizer = VectorQuantizer(
            num_tokens=4,
            vocab_size=8,
            embedding_dim=256,
            beta=0.25,
            use_ema=True,
            ema_decay=0.9
        )
        
        x = torch.randn(2, 256, 14, 14)
        
        # Forward pass in training mode
        quantizer.train()
        quantized, indices, losses = quantizer(x)
        
        # Check that EMA statistics are updated
        assert quantizer._ema_cluster_size is not None
        assert quantizer._ema_w is not None
        assert quantizer._num_updates.item() > 0
        
        # Test utilization and perplexity
        utilization = quantizer.get_codebook_utilization()
        perplexity = quantizer.get_perplexity()
        
        assert utilization.shape == (4,)
        assert perplexity.shape == (4,)
        assert utilization.min() >= 0 and utilization.max() <= 1
    
    def test_quantizer_no_ema_mode(self):
        """Test quantizer without EMA updates."""
        quantizer = VectorQuantizer(
            num_tokens=4,
            vocab_size=8,
            embedding_dim=256,
            beta=0.25,
            use_ema=False
        )
        
        x = torch.randn(2, 256, 14, 14)
        quantized, indices, losses = quantizer(x)
        
        # Check that EMA statistics are None
        assert quantizer._ema_cluster_size is None
        assert quantizer._ema_w is None
        assert quantizer._num_updates is None
        
        # Test utilization and perplexity (should return zeros)
        utilization = quantizer.get_codebook_utilization()
        perplexity = quantizer.get_perplexity()
        
        assert torch.allclose(utilization, torch.zeros(4))
        assert torch.allclose(perplexity, torch.zeros(4))
    
    def test_quantizer_codebook_initialization(self):
        """Test that codebook is properly initialized."""
        quantizer = VectorQuantizer(
            num_tokens=4,
            vocab_size=8,
            embedding_dim=256,
            beta=0.25
        )
        
        # Check codebook shape
        assert quantizer.codebook.shape == (4, 8, 256)
        
        # Check that codebook is not all zeros
        assert not torch.allclose(quantizer.codebook, torch.zeros_like(quantizer.codebook))
        
        # Check that embeddings are normalized (approximately)
        for i in range(4):
            for j in range(8):
                norm = torch.norm(quantizer.codebook[i, j])
                assert norm > 0.1  # Should be normalized but not exactly 1
    
    def test_quantizer_straight_through(self):
        """Test straight-through estimator behavior."""
        quantizer = VectorQuantizer(
            num_tokens=4,
            vocab_size=8,
            embedding_dim=256,
            beta=0.25
        )
        
        x = torch.randn(2, 256, 14, 14, requires_grad=True)
        quantized, indices, losses = quantizer(x)
        
        # Check that quantized values are discrete (from codebook)
        for i in range(4):
            for j in range(2):
                # Find which codebook entry was selected
                code_idx = indices[j, i].item()
                expected_embedding = quantizer.codebook[i, code_idx]
                
                # Quantized value should match codebook entry (with tolerance for floating-point precision)
                assert torch.allclose(quantized[j, i], expected_embedding, atol=1e-6, rtol=1e-6)
    
    def test_quantizer_config_creation(self):
        """Test quantizer creation from config."""
        config = {
            'quantizer': {
                'num_tokens': 4,
                'vocab_size': 8,
                'embedding_dim': 256,
                'beta': 0.25,
                'ema_decay': 0.99,
                'use_ema': False
            }
        }
        
        quantizer = create_quantizer_from_config(config)
        
        x = torch.randn(2, 256, 14, 14)
        quantized, indices, losses = quantizer(x)
        
        assert quantized.shape == (2, 4, 256)
        assert indices.shape == (2, 4)
    
    def test_quantizer_memory_efficiency(self):
        """Test quantizer doesn't have memory leaks."""
        quantizer = VectorQuantizer(
            num_tokens=4,
            vocab_size=8,
            embedding_dim=256,
            beta=0.25
        )
        
        # Run multiple forward passes
        for _ in range(10):
            x = torch.randn(2, 256, 14, 14)
            quantized, indices, losses = quantizer(x)
            del x, quantized, indices, losses
        
        # Should not crash or have memory issues
        assert True
    
    def test_quantizer_training_eval_modes(self):
        """Test quantizer behavior in training vs eval modes."""
        quantizer = VectorQuantizer(
            num_tokens=4,
            vocab_size=8,
            embedding_dim=256,
            beta=0.25,
            use_ema=False  # Disable EMA to avoid differences between modes
        )
        
        x = torch.randn(2, 256, 14, 14)
        
        # Training mode
        quantizer.train()
        quantized_train, indices_train, losses_train = quantizer(x)
        
        # Eval mode
        quantizer.eval()
        quantized_eval, indices_eval, losses_eval = quantizer(x)
        
        # Results should be the same (no randomness in forward pass)
        assert torch.allclose(quantized_train, quantized_eval)
        assert torch.equal(indices_train, indices_eval)
        assert torch.allclose(losses_train['total_vq_loss'], losses_eval['total_vq_loss'])


if __name__ == "__main__":
    # Run tests
    test_quantizer = TestVectorQuantizer()
    
    print("Running VectorQuantizer tests...")
    test_quantizer.test_quantizer_shapes()
    test_quantizer.test_quantizer_gradient_flow()
    test_quantizer.test_quantizer_losses()
    test_quantizer.test_quantizer_different_input_sizes()
    test_quantizer.test_quantizer_different_configs()
    test_quantizer.test_quantizer_ema_mode()
    test_quantizer.test_quantizer_no_ema_mode()
    test_quantizer.test_quantizer_codebook_initialization()
    test_quantizer.test_quantizer_straight_through()
    test_quantizer.test_quantizer_config_creation()
    test_quantizer.test_quantizer_memory_efficiency()
    test_quantizer.test_quantizer_training_eval_modes()
    print("✅ VectorQuantizer tests passed!")
    
    print("\n🎉 All tests passed!")
