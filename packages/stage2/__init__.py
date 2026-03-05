"""
LAPA Stage2 Package

Stage 2 / Stage 3 shared stage2 components.

Canonical backend path:
- stage2.backends.smolvla_shared*

Legacy implementations are kept under:
- stage2.legacy.*
"""

__version__ = "0.1.0"

from stage2.action_tokens import ActionTokenConfig
from stage2.constrained_decode import ActionTokenIds

__all__ = [
    "ActionTokenConfig",
    "ActionTokenIds",
]
