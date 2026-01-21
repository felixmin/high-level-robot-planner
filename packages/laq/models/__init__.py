"""
LAQ Model Components

Avoid eager imports so lightweight submodules (e.g. `laq.models.flow`) do not
require optional heavy dependencies (e.g. `transformers`).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "NSVQ",
    "Attention",
    "Transformer",
    "ContinuousPositionBias",
    "PEG",
    "LatentActionQuantization",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "NSVQ": ("laq.models.nsvq", "NSVQ"),
    "Attention": ("laq.models.attention", "Attention"),
    "Transformer": ("laq.models.attention", "Transformer"),
    "ContinuousPositionBias": ("laq.models.attention", "ContinuousPositionBias"),
    "PEG": ("laq.models.attention", "PEG"),
    "LatentActionQuantization": ("laq.models.latent_action_quantization", "LatentActionQuantization"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))

