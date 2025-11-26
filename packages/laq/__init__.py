"""
LAPA Package

Stage 1: Latent Action Pretraining from Videos
Transformer-based architecture with NSVQ that compresses frame-to-frame transitions 
into discrete latent codes.

Key Components:
- LAPAEncoder: Spatial-temporal transformer encoder
- NSVQ: Noise-substitution vector quantization with delta quantization
- LAPADecoder: Cross-attention decoder for reconstruction
- LAPA: Complete model combining all components

Architecture Highlights:
- Quantizes DELTA (frame_t+1 - frame_t) instead of absolute features
- Single shared codebook (not per-position)
- MSE loss only (no VQ-specific losses)
- Input: [B, 3, 2, 256, 256] (stacked frame pairs)
- Output: [B, 3, 1, 256, 256] (reconstructed next frame)
"""

__version__ = "0.2.0"  # Updated for LAPA architecture

# Import main components
from .models.lapa import LAPA, create_lapa_from_config
from .models.encoder import LAPAEncoder
from .models.nsvq import NSVQ
from .models.decoder import LAPADecoder
from .task import LAQModule, create_laq_module_from_config
from .data import LAQVideoDataset, create_dataloader

__all__ = [
    'LAPA',
    'create_lapa_from_config',
    'LAPAEncoder',
    'NSVQ',
    'LAPADecoder',
    'LAQModule',
    'create_laq_module_from_config',
    'LAQVideoDataset',
    'create_dataloader',
]
