"""
Utilities for representing discrete latent actions as special tokens.

Approach A (token-based actions):
- Add a small set of action tokens to the tokenizer: <ACTION>, </ACTION>, <ACT_0>.. <ACT_{K-1}>
- Train a VLM to output a fixed-length sequence of these tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class ActionTokenConfig:
    """Configuration for an action-token vocabulary."""

    codebook_size: int = 8
    code_seq_len: int = 4
    action_start: str = "<ACTION>"
    action_end: str = "</ACTION>"
    token_fmt: str = "<ACT_{i}>"

    def all_tokens(self) -> List[str]:
        return [self.action_start, self.action_end] + [
            self.token_fmt.format(i=i) for i in range(self.codebook_size)
        ]

    def code_tokens(self) -> List[str]:
        return [self.token_fmt.format(i=i) for i in range(self.codebook_size)]

    def validate_codes(self, codes: Sequence[int]) -> None:
        if len(codes) != self.code_seq_len:
            raise ValueError(f"Expected {self.code_seq_len} codes, got {len(codes)}")
        for code in codes:
            if not isinstance(code, int):
                raise TypeError(f"Codes must be ints, got {type(code)}")
            if code < 0 or code >= self.codebook_size:
                raise ValueError(
                    f"Code {code} out of range [0, {self.codebook_size - 1}]"
                )

    def format_target(self, codes: Sequence[int]) -> str:
        """Format codes as a target completion string."""
        self.validate_codes(codes)
        code_strs = [self.token_fmt.format(i=int(c)) for c in codes]
        return f"{self.action_start} {' '.join(code_strs)} {self.action_end}"


def add_action_tokens(tokenizer: object, config: ActionTokenConfig) -> Dict[str, int]:
    """
    Add action tokens to a Hugging Face tokenizer.

    This function intentionally avoids importing `transformers` so it's cheap to
    unit-test. It relies on the standard tokenizer protocol:
      - tokenizer.add_special_tokens({"additional_special_tokens": [...]})
      - tokenizer.convert_tokens_to_ids(token_str)
    """

    tokens = config.all_tokens()
    add_special_tokens = getattr(tokenizer, "add_special_tokens", None)
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if add_special_tokens is None or convert_tokens_to_ids is None:
        raise TypeError(
            "tokenizer must implement add_special_tokens and convert_tokens_to_ids"
        )

    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    return {tok: int(convert_tokens_to_ids(tok)) for tok in tokens}


def get_action_token_ids(tokenizer: object, config: ActionTokenConfig) -> Dict[str, int]:
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert_tokens_to_ids is None:
        raise TypeError("tokenizer must implement convert_tokens_to_ids")
    return {tok: int(convert_tokens_to_ids(tok)) for tok in config.all_tokens()}


def allowed_action_token_ids(
    token_id_map: Mapping[str, int], config: ActionTokenConfig
) -> List[int]:
    """Return a stable ordered list of allowed action-related token ids."""
    tokens = config.all_tokens()
    missing = [t for t in tokens if t not in token_id_map]
    if missing:
        raise KeyError(f"Missing ids for tokens: {missing}")
    return [int(token_id_map[t]) for t in tokens]


def is_code_token_id(token_id: int, token_id_map: Mapping[str, int], config: ActionTokenConfig) -> bool:
    return token_id in {token_id_map[t] for t in config.code_tokens() if t in token_id_map}


def extract_code_token_ids(
    token_ids: Iterable[int], token_id_map: Mapping[str, int], config: ActionTokenConfig
) -> List[int]:
    """Filter a token-id sequence down to action code token ids (excludes wrappers)."""
    code_ids = {token_id_map[t] for t in config.code_tokens() if t in token_id_map}
    return [int(t) for t in token_ids if int(t) in code_ids]

