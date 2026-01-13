"""
Constrained decoding helpers for action-token generation.

We constrain the VLM to generate:
  <ACTION> <ACT_*> x code_seq_len </ACTION> <eos>

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

        # After starting: count codes after the last <ACTION>
        last_start = max(
            i for i, t in enumerate(generated_ids) if t == self.action_start_id
        )
        suffix = generated_ids[last_start + 1 :]

        # If we've already emitted </ACTION>, force EOS.
        if self.action_end_id in suffix:
            return [self.eos_token_id]

        num_codes = sum(1 for t in suffix if t in set(self.action_code_ids))

        if num_codes < self.code_seq_len:
            return list(self.action_code_ids)

        return [self.action_end_id]


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
