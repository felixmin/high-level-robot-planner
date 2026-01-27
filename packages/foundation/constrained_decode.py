"""
Constrained decoding helpers for action-token generation.

We constrain the VLM to generate:
  <ACTION> [sep] <ACT_*> ([sep] <ACT_*>) x code_seq_len [sep] </ACTION> <eos>

This is used during validation/inference to avoid the model producing free-form text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class ActionTokenIds:
    action_start_id: int
    action_end_id: int
    action_code_ids: List[int]
    # Token ids that may appear between action tokens (e.g. whitespace).
    # These must be included if the training targets contain separators, otherwise
    # constrained decoding will force a different tokenization than what was trained.
    between_token_ids: List[int]
    eos_token_id: int
    code_seq_len: int

    def next_allowed_ids(self, generated_ids: Sequence[int]) -> List[int]:
        """
        Determine allowed next-token ids from the full sequence so far.

        Assumption: decoding begins from a prompt that does NOT already contain
        action tokens, so any action tokens observed were generated.
        """

        # Not started: force <ACTION>
        if self.action_start_id not in generated_ids:
            return [self.action_start_id]

        between_set = set(self.between_token_ids)
        code_set = set(self.action_code_ids)

        # After starting: count codes after the last <ACTION>
        last_start = max(
            i for i, t in enumerate(generated_ids) if t == self.action_start_id
        )
        suffix = generated_ids[last_start + 1 :]

        # If we've already emitted </ACTION>, force EOS.
        if self.action_end_id in suffix:
            return [self.eos_token_id]

        num_codes = sum(1 for t in suffix if t in code_set)
        last_token = suffix[-1] if suffix else None
        last_is_between = last_token in between_set if last_token is not None else False

        if num_codes < self.code_seq_len:
            # If we just emitted a separator, force a code next to avoid
            # generating separators forever under greedy decoding.
            if last_is_between:
                return list(self.action_code_ids)
            # Otherwise allow either a separator or a code.
            return list(self.between_token_ids) + list(self.action_code_ids)

        # After the expected number of codes, allow optional separators before closing.
        if last_is_between:
            return [self.action_end_id]
        return list(self.between_token_ids) + [self.action_end_id]


def make_prefix_allowed_tokens_fn(token_ids: ActionTokenIds):
    """
    Build an HF `prefix_allowed_tokens_fn` callback.

    Signature expected by `generate()`:
      fn(batch_id: int, input_ids: Tensor) -> List[int]
    """

    def _fn(batch_id: int, input_ids) -> List[int]:
        # input_ids is a 1D tensor for this batch element
        return token_ids.next_allowed_ids(input_ids.tolist())

    return _fn
