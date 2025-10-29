"""
Integration test for LAQ Decoder with Hydra configuration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import hydra
from omegaconf import DictConfig
from packages.laq.models.decoder import create_decoder_from_config


def test_decoder_with_hydra_config():
    """Test decoder creation using Hydra configuration."""
    
    # Load configuration
    with hydra.initialize(config_path="../../config", version_base=None):
        cfg = hydra.compose(config_name="config", overrides=["experiment=laq_debug"])
    
    print("Loaded configuration:")
    print(f"Decoder config: {cfg.decoder}")
    
    # Create decoder from config
    decoder = create_decoder_from_config({'decoder': cfg.decoder})
    
    print(f"Created decoder with {sum(p.numel() for p in decoder.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 4, 256)  # Quantizer output
    
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = decoder(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Verify expected output shapes
    expected_shape = torch.Size([2, 3, 224, 224])
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    # Verify output range (should be in [-1, 1] for tanh)
    assert output.min() >= -1.1 and output.max() <= 1.1, f"Output range: [{output.min():.3f}, {output.max():.3f}]"
    
    print("✅ Decoder integration test passed!")
    
    return decoder


if __name__ == "__main__":
    test_decoder_with_hydra_config()
