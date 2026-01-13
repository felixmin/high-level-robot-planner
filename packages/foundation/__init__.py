"""
LAPA Foundation Package

Stage 2: Foundation Policy (Vision-Language-Action Model)

Current implementation:
- Qwen3-VL / Cosmos-Reason2 backbone
- Discrete latent actions represented as special tokens (Approach A)
"""

__version__ = "0.1.0"

from foundation.action_tokens import ActionTokenConfig
from foundation.constrained_decode import ActionTokenIds
from foundation.vla_module import VLATokenLightningModule, VLAOptimizerConfig

__all__ = [
    "ActionTokenConfig",
    "ActionTokenIds",
    "VLATokenLightningModule",
    "VLAOptimizerConfig",
]
