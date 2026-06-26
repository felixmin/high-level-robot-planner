"""
LAPA Encoder Implementation

Transformer-based encoder that processes frame pairs with:
- Patch embedding for video frames
- Spatial transformer (8 layers) with 2D position bias
- Temporal transformer (8 layers) for frame dynamics
- Outputs separate first and last frame tokens

Input: [B, 3, 2, 256, 256] (frame pairs)
Output: first_tokens [B, 64, 1024], last_tokens [B, 64, 1024]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

# Handle both relative imports (when used as module) and direct execution
try:
    from .attention import Transformer, PEG, ContinuousPositionBias
except ImportError:
    from attention import Transformer, PEG, ContinuousPositionBias


class PatchEmbed(nn.Module):
    """
    Patch embedding for video frames.
    
    Converts image frames into patch tokens.
    """
    
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 32,
        in_channels: int = 3,
        embed_dim: int = 1024
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 64 patches for 256/32
        
        # Projection: conv with kernel=patch_size, stride=patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W]
        
        Returns:
            Patch tokens [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        
        # Apply patch projection
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        
        # Reshape to sequence: [B, embed_dim, h, w] → [B, embed_dim, N] → [B, N, embed_dim]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Apply norm
        x = self.norm(x)
        
        return x


class LAPAEncoder(nn.Module):
    """
    LAPA Transformer Encoder.
    
    Processes frame pairs with spatial and temporal transformers.
    
    Architecture:
    1. Patch embedding for each frame
    2. Spatial transformer: Self-attention across patches within each frame
    3. Temporal transformer: Self-attention across frames for each patch position
    4. Split into first and last frame tokens
    """
    
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 32,
        in_channels: int = 3,
        dim: int = 1024,
        spatial_depth: int = 8,
        temporal_depth: int = 8,
        heads: int = 16,
        dim_head: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.spatial_depth = spatial_depth
        self.temporal_depth = temporal_depth
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim
        )
        
        # Positional encoding generator
        self.peg = PEG(dim=dim, kernel_size=3)
        
        # Spatial transformer blocks (with position bias)
        spatial_size = image_size // patch_size  # 8 for 256/32
        self.spatial_transformers = nn.ModuleList([
            Transformer(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_position_bias=True,
                spatial_size=spatial_size
            )
            for _ in range(spatial_depth)
        ])
        
        # Temporal transformer blocks (no position bias needed)
        self.temporal_transformers = nn.ModuleList([
            Transformer(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_position_bias=False
            )
            for _ in range(temporal_depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input frames [B, C, T, H, W] where T=2 (frame_t and frame_t+1)
        
        Returns:
            first_tokens: Tokens for frame_t [B, num_patches, dim]
            last_tokens: Tokens for frame_t+1 [B, num_patches, dim]
        """
        B, C, T, H, W = x.shape
        assert T == 2, f"Expected 2 frames, got {T}"
        assert H == W == self.image_size, f"Expected {self.image_size}x{self.image_size}, got {H}x{W}"
        
        # Patch embedding for each frame
        # Process each frame independently
        frame_tokens = []
        for t in range(T):
            frame = x[:, :, t, :, :]  # [B, C, H, W]
            tokens = self.patch_embed(frame)  # [B, num_patches, dim]
            frame_tokens.append(tokens)
        
        # Stack: [B, T, num_patches, dim]
        x = torch.stack(frame_tokens, dim=1)  # [B, 2, num_patches, dim]
        
        # Apply PEG (positional encoding)
        x = self.peg(x)  # [B, 2, num_patches, dim]
        
        # Spatial transformer: process each frame independently
        # Reshape for spatial processing: [B*T, num_patches, dim]
        x_spatial = x.reshape(B * T, self.num_patches, self.dim)
        
        for spatial_transformer in self.spatial_transformers:
            x_spatial = spatial_transformer(x_spatial)
        
        # Reshape back: [B, T, num_patches, dim]
        x = x_spatial.reshape(B, T, self.num_patches, self.dim)
        
        # Temporal transformer: process each patch position across frames
        # Reshape for temporal processing: [B*num_patches, T, dim]
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * self.num_patches, T, self.dim)
        
        for temporal_transformer in self.temporal_transformers:
            x_temporal = temporal_transformer(x_temporal)
        
        # Reshape back: [B, num_patches, T, dim] → [B, T, num_patches, dim]
        x = x_temporal.reshape(B, self.num_patches, T, self.dim).permute(0, 2, 1, 3)
        
        # Apply final norm
        x = self.norm(x)
        
        # Split into first and last frame tokens
        first_tokens = x[:, 0, :, :]  # [B, num_patches, dim]
        last_tokens = x[:, 1, :, :]   # [B, num_patches, dim]
        
        return first_tokens, last_tokens


def create_encoder_from_config(config: dict) -> LAPAEncoder:
    """
    Create LAPA encoder from configuration dictionary.
    
    Args:
        config: Configuration dictionary with encoder parameters
    
    Returns:
        Initialized LAPAEncoder instance
    """
    return LAPAEncoder(
        image_size=config.get('image_size', 256),
        patch_size=config.get('patch_size', 32),
        in_channels=config.get('channels', 3),
        dim=config.get('dim', 1024),
        spatial_depth=config.get('spatial_depth', 8),
        temporal_depth=config.get('temporal_depth', 8),
        heads=config.get('heads', 16),
        dim_head=config.get('dim_head', 64),
        mlp_ratio=config.get('mlp_ratio', 4),
        dropout=config.get('dropout', 0.0)
    )


# Test function
def test_encoder():
    """Test LAPA encoder implementation."""
    print("Testing LAPA Encoder...")
    
    encoder = LAPAEncoder(
        image_size=256,
        patch_size=32,
        in_channels=3,
        dim=1024,
        spatial_depth=2,  # Reduced for testing
        temporal_depth=2,
        heads=16,
        dim_head=64
    )
    
    # Test input: [B, C, T, H, W]
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 2, 256, 256)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        first_tokens, last_tokens = encoder(input_tensor)
    
    print(f"First tokens shape: {first_tokens.shape}")
    print(f"Last tokens shape: {last_tokens.shape}")
    
    # Verify expected output shapes
    expected_shape = torch.Size([2, 64, 1024])  # 64 patches (8×8), 1024 dim
    assert first_tokens.shape == expected_shape, f"Expected {expected_shape}, got {first_tokens.shape}"
    assert last_tokens.shape == expected_shape, f"Expected {expected_shape}, got {last_tokens.shape}"
    
    print("✅ Encoder test passed!")
    
    # Test gradient flow
    input_tensor.requires_grad_(True)
    first_tokens, last_tokens = encoder(input_tensor)
    loss = first_tokens.sum() + last_tokens.sum()
    loss.backward()
    
    assert input_tensor.grad is not None, "Gradients should flow to input"
    print("✅ Gradient flow test passed!")
    
    return encoder


if __name__ == "__main__":
    test_encoder()
