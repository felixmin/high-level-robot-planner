"""
Backends for Stage 2 / Foundation VLA training and inference.

Goal: keep model-specific behavior (processor, chat templates, token setup,
prompt masking, constrained decoding, parsing) behind a stable interface so
we can compare implementations cleanly.
"""

from foundation.backends.interfaces import (
    BackendMode,
    FoundationBatch,
    LatentOutput,
    LossOutput,
    VLABackend,
)

__all__ = [
    "BackendMode",
    "FoundationBatch",
    "LatentOutput",
    "LossOutput",
    "VLABackend",
]
