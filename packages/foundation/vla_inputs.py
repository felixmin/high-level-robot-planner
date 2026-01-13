"""
Batch building utilities for token-based action prediction (Approach A).

Core responsibilities:
- Build chat messages (image + instruction -> assistant action-token completion)
- Use the *processor* (not tokenizer-only) to compute prompt lengths for VLMs
- Create `labels` with prompt + padding masking so loss is only on the completion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass(frozen=True)
class ChatConfig:
    system_prompt: Optional[str] = None


def build_messages(
    image: Any,
    instruction: str,
    target_text: Optional[str],
    *,
    chat: ChatConfig,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if chat.system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": chat.system_prompt}],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    )

    if target_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        )
    return messages


def build_inputs_with_prompt_mask(
    *,
    processor: Any,
    images: Sequence[Any],
    instructions: Sequence[str],
    targets: Sequence[str],
    chat: ChatConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build a VLM batch with labels masked to compute loss only on the completion.

    Important: prompt lengths are computed via the processor with images to avoid
    mismatches between tokenizer-only token counts and VLM multimodal tokenization.
    """

    if not (len(images) == len(instructions) == len(targets)):
        raise ValueError("images, instructions, and targets must have the same length")

    full_messages = [
        build_messages(img, instr, tgt, chat=chat)
        for img, instr, tgt in zip(images, instructions, targets, strict=True)
    ]
    prompt_messages = [
        build_messages(img, instr, None, chat=chat)
        for img, instr in zip(images, instructions, strict=True)
    ]

    full_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in full_messages
    ]
    prompt_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in prompt_messages
    ]

    full_inputs = processor(
        text=full_texts,
        images=list(images),
        return_tensors="pt",
        padding=True,
    )
    prompt_inputs = processor(
        text=prompt_texts,
        images=list(images),
        return_tensors="pt",
        padding=True,
    )

    full_inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in full_inputs.items()
    }
    prompt_inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in prompt_inputs.items()
    }

    if "input_ids" not in full_inputs or "attention_mask" not in full_inputs:
        raise KeyError("processor output must include input_ids and attention_mask")
    if "attention_mask" not in prompt_inputs:
        raise KeyError("prompt processor output must include attention_mask")

    labels = full_inputs["input_ids"].clone()
    full_attention_mask = full_inputs["attention_mask"]
    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).to(torch.long)

    for i in range(labels.shape[0]):
        prompt_len = int(prompt_lens[i].item())
        labels[i, :prompt_len] = -100

    # Always mask padding to avoid loss on padded positions.
    labels = labels.masked_fill(full_attention_mask == 0, -100)

    full_inputs["labels"] = labels
    return full_inputs


def build_prompt_inputs(
    *,
    processor: Any,
    images: Sequence[Any],
    instructions: Sequence[str],
    chat: ChatConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build prompt-only inputs (no labels, used for generation)."""
    if len(images) != len(instructions):
        raise ValueError("images and instructions must have the same length")

    prompt_messages = [
        build_messages(img, instr, None, chat=chat)
        for img, instr in zip(images, instructions, strict=True)
    ]
    prompt_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in prompt_messages
    ]

    inputs = processor(
        text=prompt_texts,
        images=list(images),
        return_tensors="pt",
        padding=True,
    )
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }
