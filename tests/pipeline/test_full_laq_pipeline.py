"""
Integration test for full LAQ pipeline: Encoder + Quantizer + Decoder
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import hydra
from packages.laq.models.encoder import create_encoder_from_config
from packages.laq.models.quantizer import create_quantizer_from_config
from packages.laq.models.decoder import create_decoder_from_config


def test_full_laq_pipeline():
    """Test full LAQ pipeline: Encoder → Quantizer → Decoder."""
    
    # Load configuration
    with hydra.initialize(config_path="../../config", version_base=None):
        cfg = hydra.compose(config_name="config", overrides=["experiment=laq_debug"])
    
    print("Testing Full LAQ Pipeline...")
    
    # Create all components
    encoder = create_encoder_from_config({'encoder': cfg.encoder})
    quantizer = create_quantizer_from_config({'quantizer': cfg.quantizer})
    decoder = create_decoder_from_config({'decoder': cfg.decoder})
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
    print(f"Quantizer parameters: {sum(p.numel() for p in quantizer.parameters())}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in quantizer.parameters()) + sum(p.numel() for p in decoder.parameters())}")
    
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
        print(f"VQ Losses: {losses}")
        
        # Decoder forward pass
        reconstructed = decoder(quantized)
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Reconstructed range: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")
    
    # Verify pipeline shapes
    assert encoded.shape == (2, 256, 14, 14), f"Expected (2, 256, 14, 14), got {encoded.shape}"
    assert quantized.shape == (2, 4, 256), f"Expected (2, 4, 256), got {quantized.shape}"
    assert indices.shape == (2, 4), f"Expected (2, 4), got {indices.shape}"
    assert reconstructed.shape == (2, 3, 224, 224), f"Expected (2, 3, 224, 224), got {reconstructed.shape}"
    assert indices.min() >= 0 and indices.max() < 8, "Indices should be in range [0, 7]"
    
    print("✅ Full LAQ pipeline test passed!")
    
    # Test gradient flow through entire pipeline
    input_tensor.requires_grad_(True)
    encoded = encoder(input_tensor)
    quantized, indices, losses = quantizer(encoded)
    reconstructed = decoder(quantized)
    
    # Compute reconstruction loss
    target = torch.randn_like(reconstructed)  # Dummy target
    reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, target)
    total_loss = reconstruction_loss + losses['total_vq_loss']
    
    total_loss.backward()
    
    assert input_tensor.grad is not None, "Gradients should flow to input"
    print("✅ Gradient flow through full pipeline test passed!")
    
    # Test that the pipeline can be used for training
    print(f"Reconstruction loss: {reconstruction_loss.item():.4f}")
    print(f"VQ loss: {losses['total_vq_loss'].item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    return encoder, quantizer, decoder


if __name__ == "__main__":
    test_full_laq_pipeline()
