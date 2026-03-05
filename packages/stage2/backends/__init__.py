"""
Backends for Stage 2 policy training and inference.

Goal: keep model-specific behavior (processor, chat templates, token setup,
prompt masking, constrained decoding, parsing) behind a stable interface so
we can compare implementations cleanly.
"""

from stage2.backends.interfaces import (
    BackendMode,
    Stage2Batch,
    LatentOutput,
    LossOutput,
    PolicyBackend,
)

__all__ = [
    "BackendMode",
    "Stage2Batch",
    "LatentOutput",
    "LossOutput",
    "PolicyBackend",
]

try:
    from stage2.backends.smolvla_shared.config import (
        SmolVLASharedBackendConfig,
        SmolVLASharedCoreConfig,
    )
    from stage2.backends.smolvla_shared.model import SmolVLASharedCore
    from stage2.backends.smolvla_shared_backend import SmolVLASharedBackend
except ModuleNotFoundError:
    # Keep lightweight interface imports usable in environments without full model deps.
    pass
else:
    __all__ += [
        "SmolVLASharedBackendConfig",
        "SmolVLASharedCoreConfig",
        "SmolVLASharedCore",
        "SmolVLASharedBackend",
    ]
