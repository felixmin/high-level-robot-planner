from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TemporalFieldRequest:
    deltas_steps: tuple[int, ...]
    required: bool = True


@dataclass(frozen=True)
class DatasetRequest:
    image_requests: dict[str, TemporalFieldRequest]
    state_request: TemporalFieldRequest | None = None
    action_request: TemporalFieldRequest | None = None
    include_task_text: bool = False
    include_subtask_text: bool = False
    include_metadata: bool = True
    pad_missing_future: bool = True
    image_size: tuple[int, int] | None = None
    image_dtype: str = "uint8"


@dataclass
class DatasetSample:
    image_streams: dict[str, torch.Tensor] | None = None
    image_padding_masks: dict[str, torch.Tensor] | None = None
    state: torch.Tensor | None = None
    state_is_pad: torch.Tensor | None = None
    action: torch.Tensor | None = None
    action_is_pad: torch.Tensor | None = None
    task_text: str | None = None
    subtask_text: str | None = None
    meta: dict[str, Any] | None = None


@dataclass
class BatchedDatasetSample:
    image_streams: dict[str, torch.Tensor] | None = None
    image_padding_masks: dict[str, torch.Tensor] | None = None
    state: torch.Tensor | None = None
    state_is_pad: torch.Tensor | None = None
    action: torch.Tensor | None = None
    action_is_pad: torch.Tensor | None = None
    task_text: list[str] | None = None
    subtask_text: list[str] | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class Stage1Batch:
    image_streams: dict[str, torch.Tensor]
    image_padding_masks: dict[str, torch.Tensor] | None = None
    task_text: list[str] | None = None
    subtask_text: list[str] | None = None
    state: torch.Tensor | None = None
    state_is_pad: torch.Tensor | None = None
    action: torch.Tensor | None = None
    action_is_pad: torch.Tensor | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class SampleToken:
    source_id: int
    episode_id: int
    anchor_abs_index: int
