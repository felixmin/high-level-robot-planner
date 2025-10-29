"""
Integration test for LAQ Encoder + Quantizer pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import hydra
from packages.laq.models.encoder import create_encoder_from_config
from packages.laq.models.quantizer import create_quantizer_from_config


def test_encoder_quantizer_pipeline():
    """Test encoder + quantizer pipeline."""
    
    # Load configuration
    with hydra.initialize(config_path="../../config", version_base=None):
        cfg = hydra.compose(config_name="config", overrides=["experiment=laq_debug"])
    
    print("Testing Encoder + Quantizer Pipeline...")
    
    # Create encoder and quantizer
    encoder = create_encoder_from_config({'encoder': cfg.encoder})
    quantizer = create_quantizer_from_config({'quantizer': cfg.quantizer})
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
    print(f"Quantizer parameters: {sum(p.numel() for p in quantizer.parameters())}")
    
    # Test input (concatenated frames)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 6, 224, 224)  # frame_t | frame_{t+1}
    
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        # Encoder forward pass
        encoded = encoder(input_tensor)
        print(f"Encoded shape: {encoded.shape}")
        
        # Quantizer forward pass
        quantized, indices, losses = quantizer(encoded)
        print(f"Quantized shape: {quantized.shape}")
        print(f"Indices shape: {indices.shape}")
        print(f"Indices: {indices}")
        print(f"Losses: {losses}")
    
    # Verify pipeline shapes
    assert encoded.shape == (2, 256, 14, 14), f"Expected (2, 256, 14, 14), got {encoded.shape}"
    assert quantized.shape == (2, 4, 256), f"Expected (2, 4, 256), got {quantized.shape}"
    assert indices.shape == (2, 4), f"Expected (2, 4), got {indices.shape}"
    assert indices.min() >= 0 and indices.max() < 8, "Indices should be in range [0, 7]"
    
    print("✅ Encoder + Quantizer pipeline test passed!")
    
    # Test gradient flow through entire pipeline
    input_tensor.requires_grad_(True)
    encoded = encoder(input_tensor)
    quantized, indices, losses = quantizer(encoded)
    loss = losses['total_vq_loss']
    loss.backward()
    
    assert input_tensor.grad is not None, "Gradients should flow to input"
    print("✅ Gradient flow through pipeline test passed!")
    
    return encoder, quantizer


if __name__ == "__main__":
    test_encoder_quantizer_pipeline()
