"""
LAPA Decoder Implementation

Cross-attention decoder that reconstructs the next frame conditioned on:
- Context: first_frame_tokens (what we observe at time t)
- Actions: quantized_action_tokens (what we intend to do)

Uses spatial transformer with cross-attention to condition reconstruction on learned actions.

Input:
  - context: [B, 1, 64, 1024] (first frame tokens)
  - actions: [B, 1, 4, 1024] (quantized action tokens)
Output: [B, 3, 1, 256, 256] (reconstructed next frame)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

# Handle both relative imports (when used as module) and direct execution
try:
    from .attention import Transformer
except ImportError:
    from attention import Transformer


class LAPADecoder(nn.Module):
    """
    LAPA Cross-Attention Decoder.
    
    Reconstructs next frame by attending to:
    1. Context patches from first frame
    2. Action tokens representing intended motion
    
    Architecture:
    1. Flatten action grid [B, 1, 2, 2, dim] → [B, 1, 4, dim]
    2. Spatial transformer with cross-attention (decoder_depth layers)
    3. Patch-to-pixel projection
    """
    
    def __init__(
        self,
        dim: int = 1024,
        image_size: int = 256,
        patch_size: int = 32,
        out_channels: int = 3,
        decoder_depth: int = 8,
        heads: int = 16,
        dim_head: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches = (image_size // patch_size) ** 2  # 64
        
        # Cross-attention transformer blocks
        self.transformers = nn.ModuleList([
            Transformer(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_cross_attention=True,  # Enable cross-attention
                use_position_bias=True,
                spatial_size=image_size // patch_size
            )
            for _ in range(decoder_depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Patch to pixel projection
        # Each patch becomes patch_size × patch_size × out_channels pixels
        pixels_per_patch = patch_size * patch_size * out_channels
        self.to_pixels = nn.Linear(dim, pixels_per_patch)
        
        # Output activation
        self.output_activation = nn.Tanh()
        
    def forward(
        self,
        context: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            context: First frame tokens [B, 1, 64, dim] or [B, 64, dim]
            actions: Quantized action tokens [B, 1, 2, 2, dim] or [B, 1, 4, dim]
        
        Returns:
            Reconstructed next frame [B, 3, 1, 256, 256]
        """
        # Handle different input shapes
        if context.dim() == 3:
            # [B, 64, dim] → [B, 1, 64, dim]
            context = context.unsqueeze(1)
        
        if actions.dim() == 5:
            # [B, 1, 2, 2, dim] → [B, 1, 4, dim]
            B, _, _, _, D = actions.shape
            actions = actions.reshape(B, 1, 4, D)
        
        B, T, N, D = context.shape
        assert T == 1, f"Expected T=1, got {T}"
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        
        # Flatten temporal dimension for transformer processing
        x = context.squeeze(1)  # [B, 64, dim]
        action_context = actions.squeeze(1)  # [B, 4, dim]
        
        # Apply cross-attention transformers
        # Context patches attend to action tokens
        for transformer in self.transformers:
            x = transformer(x, context=action_context)
        
        # Apply final norm
        x = self.norm(x)  # [B, 64, dim]
        
        # Project patches to pixels
        x = self.to_pixels(x)  # [B, 64, patch_size^2 * out_channels]
        
        # Reshape to image - ensure contiguous memory for MPS compatibility
        # [B, 64, patch_size^2 * out_channels] → [B, 3, 256, 256]
        H = W = int(math.sqrt(self.num_patches))
        
        # Reshape patches: [B, 64, patch_size^2 * C] → [B, H, W, patch_size, patch_size, C]
        x = x.reshape(B, H, W, self.patch_size, self.patch_size, self.out_channels)
        
        # Rearrange to interleave patches correctly
        # NOTE: 6D permute can crash on MPS backend (Apple Silicon). This is a known MPS limitation.
        # For training on CUDA, this works fine. For MPS, tests use CPU device as workaround.
        # Original permute (0, 5, 1, 3, 2, 4) interleaves: [B, C, H_grid, patch_H, W_grid, patch_W]
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, H, patch_H, W, patch_W]
        
        # Reshape to final image: [B, C, H, patch_H, W, patch_W] → [B, C, H*patch_H, W*patch_W]
        x = x.reshape(B, self.out_channels, self.image_size, self.image_size)
        
        # Add temporal dimension: [B, 3, 256, 256] → [B, 3, 1, 256, 256]
        x = x.unsqueeze(2)
        
        # Apply output activation
        x = self.output_activation(x)
        
        return x


def create_decoder_from_config(config: dict) -> LAPADecoder:
    """
    Create LAPA decoder from configuration dictionary.
    
    Args:
        config: Configuration dictionary with decoder parameters
    
    Returns:
        Initialized LAPADecoder instance
    """
    return LAPADecoder(
        dim=config.get('dim', 1024),
        image_size=config.get('image_size', 256),
        patch_size=config.get('patch_size', 32),
        out_channels=config.get('channels', 3),
        decoder_depth=config.get('decoder_depth', 8),
        heads=config.get('decoder_heads', 16),
        dim_head=config.get('dim_head', 64),
        mlp_ratio=config.get('mlp_ratio', 4),
        dropout=config.get('dropout', 0.0)
    )


# Test function
def test_decoder():
    """Test LAPA decoder implementation."""
    print("Testing LAPA Decoder...")
    
    decoder = LAPADecoder(
        dim=1024,
        image_size=256,
        patch_size=32,
        out_channels=3,
        decoder_depth=2,  # Reduced for testing
        heads=16,
        dim_head=64
    )
    
    # Test inputs
    batch_size = 2
    context = torch.randn(batch_size, 1, 64, 1024)  # First frame tokens
    actions = torch.randn(batch_size, 1, 2, 2, 1024)  # Quantized actions
    
    print(f"Context shape: {context.shape}")
    print(f"Actions shape: {actions.shape}")
    
    # Forward pass
    with torch.no_grad():
        reconstructed = decoder(context, actions)
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Verify expected output shape
    expected_shape = torch.Size([2, 3, 1, 256, 256])
    assert reconstructed.shape == expected_shape, f"Expected {expected_shape}, got {reconstructed.shape}"
    
    # Verify output range (Tanh: [-1, 1])
    assert reconstructed.min() >= -1.1 and reconstructed.max() <= 1.1, \
        f"Output range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]"
    
    print("✅ Decoder test passed!")
    
    # Test gradient flow
    context.requires_grad_(True)
    actions.requires_grad_(True)
    
    reconstructed = decoder(context, actions)
    loss = reconstructed.sum()
    loss.backward()
    
    assert context.grad is not None, "Gradients should flow to context"
    assert actions.grad is not None, "Gradients should flow to actions"
    print("✅ Gradient flow test passed!")
    
    return decoder


if __name__ == "__main__":
    test_decoder()
