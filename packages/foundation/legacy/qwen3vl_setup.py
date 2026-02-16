"""
Helpers for preparing a Qwen3-VL (Cosmos-Reason2) model for action-token SFT.

This is kept small and testable; the actual from_pretrained() loading happens
in the training script.
"""

from __future__ import annotations

from typing import Any, Dict

from foundation.action_tokens import ActionTokenConfig, add_action_tokens


def prepare_action_token_training(
    *, model: Any, processor: Any, action_tokens: ActionTokenConfig
) -> Dict[str, int]:
    """
    1) Add action tokens to processor.tokenizer
    2) Resize model embeddings
    """

    if not hasattr(processor, "tokenizer"):
        raise TypeError("processor must expose a .tokenizer attribute")

    token_id_map = add_action_tokens(processor.tokenizer, action_tokens)

    resize = getattr(model, "resize_token_embeddings", None)
    if resize is None:
        raise TypeError("model must implement resize_token_embeddings()")

    resize(len(processor.tokenizer))
    return token_id_map
