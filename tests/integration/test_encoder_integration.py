"""
Integration test for LAQ Encoder with Hydra configuration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import hydra
from omegaconf import DictConfig
from packages.laq.models.encoder import create_encoder_from_config


def test_encoder_with_hydra_config():
    """Test encoder creation using Hydra configuration."""
    
    # Load configuration
    with hydra.initialize(config_path="../../config", version_base=None):
        cfg = hydra.compose(config_name="config", overrides=["experiment=laq_debug"])
    
    print("Loaded configuration:")
    print(f"Encoder config: {cfg.encoder}")
    
    # Create encoder from config
    encoder = create_encoder_from_config({'encoder': cfg.encoder})
    
    print(f"Created encoder with {sum(p.numel() for p in encoder.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 6, 224, 224)
    
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = encoder(input_tensor)
    
    print(f"Output shape: {output.shape}")
    
    # Verify expected output shape
    expected_shape = torch.Size([2, 256, 14, 14])
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print("✅ Encoder integration test passed!")
    
    return encoder


if __name__ == "__main__":
    test_encoder_with_hydra_config()
