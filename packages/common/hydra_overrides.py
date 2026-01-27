"""
Hydra override helpers.

Hydra overrides have grammar `key=value`. If the value itself contains `=`,
Hydra's parser requires quoting.
"""

from __future__ import annotations


def autoquote_override_value(override: str) -> str:
    """
    Quote override values that contain an unquoted `=` so Hydra can parse them.

    Example:
      model.vla.checkpoint=/path/vla-stepstep=079000.ckpt
    becomes:
      model.vla.checkpoint="/path/vla-stepstep=079000.ckpt"
    """
    if "=" not in override:
        return override
    key, value = override.split("=", 1)
    if not value:
        return override
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return override
    if "=" not in value:
        return override
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{key}="{escaped}"'


def normalize_overrides(overrides: list[str]) -> list[str]:
    return [autoquote_override_value(o) for o in overrides]

