"""
LAPA Foundation Package

Stage 2 / Stage 3 shared foundation components.

Canonical backend path:
- foundation.backends.smolvla_shared*

Legacy implementations are kept under:
- foundation.legacy.*
"""

__version__ = "0.1.0"

from foundation.action_tokens import ActionTokenConfig
from foundation.constrained_decode import ActionTokenIds

__all__ = [
    "ActionTokenConfig",
    "ActionTokenIds",
]
