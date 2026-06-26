"""
LAPA: Latent Action Pretraining from Videos

Main model that combines all LAPA components:
- Encoder: Spatial-temporal transformer
- NSVQ: Delta quantization with single codebook
- Decoder: Cross-attention reconstruction

Input: [B, 3, 2, 256, 256] (frame pairs)
Output: Reconstructed frame [B, 3, 1, 256, 256], indices [B, 4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any

# Handle both relative imports (when used as module) and direct execution
try:
    from .encoder import LAPAEncoder, create_encoder_from_config
    from .nsvq import NSVQ
    from .decoder import LAPADecoder, create_decoder_from_config
except ImportError:
    from encoder import LAPAEncoder, create_encoder_from_config
    from nsvq import NSVQ
    from decoder import LAPADecoder, create_decoder_from_config


class LAPA(nn.Module):
    """
    Complete LAPA model for latent action learning.
    
    Architecture:
    1. Encoder: Frame pairs → first_tokens, last_tokens
    2. NSVQ: Delta quantization → quantized_actions, indices
    3. Decoder: Cross-attention reconstruction → next_frame
    
    Loss: MSE only (no VQ-specific losses)
    """
    
    def __init__(
        self,
        # Image parameters
        image_size: int = 256,
        patch_size: int = 32,
        in_channels: int = 3,
        
        # Model dimensions
        dim: int = 1024,
        quant_dim: int = 32,
        
        # Encoder parameters
        spatial_depth: int = 8,
        temporal_depth: int = 8,
        
        # Decoder parameters
        decoder_depth: int = 8,
        
        # Attention parameters
        heads: int = 16,
        dim_head: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        
        # NSVQ parameters
        codebook_size: int = 8,
        code_seq_len: int = 4,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.codebook_size = codebook_size
        self.code_seq_len = code_seq_len
        
        # Encoder
        self.encoder = LAPAEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            dim=dim,
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # NSVQ quantizer
        self.nsvq = NSVQ(
            dim=dim,
            quant_dim=quant_dim,
            codebook_size=codebook_size,
            code_seq_len=code_seq_len
        )
        
        # Decoder
        self.decoder = LAPADecoder(
            dim=dim,
            image_size=image_size,
            patch_size=patch_size,
            out_channels=in_channels,
            decoder_depth=decoder_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
    def forward(
        self,
        frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through LAPA.
        
        Args:
            frames: Input frame pairs [B, C, 2, H, W]
        
        Returns:
            reconstructed: Reconstructed next frame [B, C, 1, H, W]
            indices: Discrete latent action codes [B, code_seq_len]
            perplexity: Codebook usage metric (scalar)
        """
        # 1. Encode frame pairs
        first_tokens, last_tokens = self.encoder(frames)  # [B, 64, dim] each
        
        # 2. Quantize delta with NSVQ
        quantized_actions, indices, perplexity = self.nsvq(first_tokens, last_tokens)
        # quantized_actions: [B, 1, 2, 2, dim]
        # indices: [B, 4]
        
        # 3. Decode with cross-attention
        reconstructed = self.decoder(first_tokens, quantized_actions)
        # reconstructed: [B, 3, 1, 256, 256]
        
        return reconstructed, indices, perplexity
    
    def encode(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode frames to tokens (useful for analysis).
        
        Args:
            frames: Input frame pairs [B, C, 2, H, W]
        
        Returns:
            first_tokens: [B, num_patches, dim]
            last_tokens: [B, num_patches, dim]
        """
        return self.encoder(frames)
    
    def quantize(
        self,
        first_tokens: torch.Tensor,
        last_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize token deltas (useful for label generation).
        
        Args:
            first_tokens: [B, num_patches, dim]
            last_tokens: [B, num_patches, dim]
        
        Returns:
            quantized_actions: [B, 1, 2, 2, dim]
            indices: [B, code_seq_len]
            perplexity: Codebook usage metric
        """
        return self.nsvq(first_tokens, last_tokens)
    
    def predict_latent_actions(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Predict discrete latent action codes (for foundation model training).
        
        Args:
            frames: Input frame pairs [B, C, 2, H, W]
        
        Returns:
            indices: Discrete latent action codes [B, code_seq_len]
        """
        with torch.no_grad():
            first_tokens, last_tokens = self.encoder(frames)
            _, indices, _ = self.nsvq(first_tokens, last_tokens)
        return indices
    
    def get_codebook_utilization(self) -> float:
        """Get current codebook utilization percentage."""
        return self.nsvq.get_codebook_utilization()


def create_lapa_from_config(config: Dict[str, Any]) -> LAPA:
    """
    Create LAPA model from configuration dictionary.
    
    Args:
        config: Configuration dictionary with model parameters
    
    Returns:
        Initialized LAPA model
    """
    model_config = config.get('model', config)
    
    return LAPA(
        # Image parameters
        image_size=model_config.get('image_size', 256),
        patch_size=model_config.get('patch_size', 32),
        in_channels=model_config.get('channels', 3),
        
        # Model dimensions
        dim=model_config.get('dim', 1024),
        quant_dim=model_config.get('quant_dim', 32),
        
        # Encoder parameters
        spatial_depth=model_config.get('spatial_depth', 8),
        temporal_depth=model_config.get('temporal_depth', 8),
        
        # Decoder parameters
        decoder_depth=model_config.get('decoder_depth', 8),
        
        # Attention parameters
        heads=model_config.get('heads', 16),
        dim_head=model_config.get('dim_head', 64),
        mlp_ratio=model_config.get('mlp_ratio', 4),
        dropout=model_config.get('dropout', 0.0),
        
        # NSVQ parameters
        codebook_size=model_config.get('codebook_size', 8),
        code_seq_len=model_config.get('code_seq_len', 4),
    )


# Test function
def test_lapa():
    """Test complete LAPA model."""
    print("Testing LAPA Model...")
    
    # Create model with reduced size for testing
    model = LAPA(
        image_size=256,
        patch_size=32,
        in_channels=3,
        dim=1024,
        quant_dim=32,
        spatial_depth=2,  # Reduced for testing
        temporal_depth=2,
        decoder_depth=2,
        heads=16,
        dim_head=64,
        codebook_size=8,
        code_seq_len=4
    )
    
    # Test input: [B, C, T, H, W]
    batch_size = 2
    frames = torch.randn(batch_size, 3, 2, 256, 256)
    
    print(f"Input shape: {frames.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Forward pass
    with torch.no_grad():
        reconstructed, indices, perplexity = model(frames)
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Indices range: [{indices.min().item()}, {indices.max().item()}]")
    print(f"Perplexity: {perplexity.item():.3f}")
    print(f"Codebook utilization: {model.get_codebook_utilization():.2%}")
    
    # Verify expected shapes
    assert reconstructed.shape == (2, 3, 1, 256, 256), \
        f"Expected (2, 3, 1, 256, 256), got {reconstructed.shape}"
    assert indices.shape == (2, 4), f"Expected (2, 4), got {indices.shape}"
    assert indices.min() >= 0 and indices.max() < 8, "Indices should be in range [0, 7]"
    
    print("✅ LAPA forward pass test passed!")
    
    # Test gradient flow
    frames.requires_grad_(True)
    reconstructed, indices, perplexity = model(frames)
    loss = F.mse_loss(reconstructed, frames[:, :, 1:2, :, :])  # Compare to second frame
    loss.backward()
    
    assert frames.grad is not None, "Gradients should flow to input"
    print("✅ Gradient flow test passed!")
    
    # Test helper methods
    print("\nTesting helper methods...")
    
    with torch.no_grad():
        # Test encode
        first_tokens, last_tokens = model.encode(frames)
        print(f"Encoded tokens - First: {first_tokens.shape}, Last: {last_tokens.shape}")
        
        # Test quantize
        quantized, indices, perp = model.quantize(first_tokens, last_tokens)
        print(f"Quantized: {quantized.shape}, Indices: {indices.shape}")
        
        # Test predict_latent_actions
        predicted_indices = model.predict_latent_actions(frames)
        print(f"Predicted indices: {predicted_indices.shape}")
    
    print("✅ All LAPA tests passed!")
    
    return model


if __name__ == "__main__":
    test_lapa()




