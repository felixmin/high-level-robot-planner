"""
Integration test for LAQ Vector Quantizer with Hydra configuration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import hydra
from omegaconf import DictConfig
from packages.laq.models.quantizer import create_quantizer_from_config


def test_quantizer_with_hydra_config():
    """Test quantizer creation using Hydra configuration."""
    
    # Load configuration
    with hydra.initialize(config_path="../../config", version_base=None):
        cfg = hydra.compose(config_name="config", overrides=["experiment=laq_debug"])
    
    print("Loaded configuration:")
    print(f"Quantizer config: {cfg.quantizer}")
    
    # Create quantizer from config
    quantizer = create_quantizer_from_config({'quantizer': cfg.quantizer})
    
    print(f"Created quantizer with {sum(p.numel() for p in quantizer.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 256, 14, 14)  # Encoder output
    
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        quantized, indices, losses = quantizer(input_tensor)
    
    print(f"Quantized shape: {quantized.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Indices range: [{indices.min().item()}, {indices.max().item()}]")
    print(f"Losses: {losses}")
    
    # Verify expected output shapes
    expected_quantized_shape = torch.Size([2, 4, 256])
    expected_indices_shape = torch.Size([2, 4])
    
    assert quantized.shape == expected_quantized_shape, f"Expected {expected_quantized_shape}, got {quantized.shape}"
    assert indices.shape == expected_indices_shape, f"Expected {expected_indices_shape}, got {indices.shape}"
    assert indices.min() >= 0 and indices.max() < 8, "Indices should be in range [0, 7]"
    
    print("✅ Quantizer integration test passed!")
    
    return quantizer


if __name__ == "__main__":
    test_quantizer_with_hydra_config()
