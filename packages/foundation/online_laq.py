"""
Online LAQ label generation helpers.

These utilities adapt OpenX/OXE frame-pair batches to the LAQ (Stage 1) encoder
to generate discrete latent action codes during Stage 2 training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import torch


def oxe_frames_to_laq_video(frames: torch.Tensor) -> torch.Tensor:
    """
    Convert OXE batch frames to LAQ input layout.

    Expected OXE layout (from `common.data.oxe_collate_fn`):
      - frames: [B, 2, H, W, 3] uint8

    LAQ expects:
      - video: [B, 3, 2, H, W] float32 in [0, 1]
    """

    if frames.ndim != 5:
        raise ValueError(f"Expected 5D frames tensor, got shape {tuple(frames.shape)}")

    # Accept either [B, 2, H, W, 3] or [B, 2, 3, H, W] or [B, 3, 2, H, W].
    if frames.shape[-1] == 3:
        # [B, 2, H, W, 3] -> [B, 2, 3, H, W]
        video = frames.permute(0, 1, 4, 2, 3)
        # [B, 2, 3, H, W] -> [B, 3, 2, H, W]
        video = video.permute(0, 2, 1, 3, 4)
    elif frames.shape[2] == 3:
        # [B, 2, 3, H, W] -> [B, 3, 2, H, W]
        video = frames.permute(0, 2, 1, 3, 4)
    elif frames.shape[1] == 3 and frames.shape[2] == 2:
        video = frames
    else:
        raise ValueError(
            "Unrecognized frames layout; expected last dim=3 (BHWC), or shape[2]=3, or shape[1]=3 & shape[2]=2. "
            f"Got {tuple(frames.shape)}"
        )

    if video.dtype == torch.uint8:
        video = video.to(torch.float32) / 255.0
    else:
        video = video.to(torch.float32)

    return video


class LatentCodeProvider(Protocol):
    codebook_size: int
    code_seq_len: int

    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        """Return codebook indices [B, code_seq_len] for a video batch."""


@dataclass
class OnlineLAQConfig:
    laq_checkpoint_path: str


class LAQTaskCodeProvider(torch.nn.Module):
    """
    Thin wrapper around `laq.task.LAQTask` to expose code indices for Stage 2.
    """

    def __init__(self, laq_task: Any):
        super().__init__()
        self._laq_task = laq_task
        self._laq_task.eval()
        self._laq_task.freeze()

        # Expect the underlying model to expose these attributes (true for LAQ in this repo).
        self.codebook_size = int(self._laq_task.model.codebook_size)
        self.code_seq_len = int(self._laq_task.model.code_seq_len)

    @property
    def device(self) -> torch.device:
        return next(self._laq_task.parameters()).device

    @torch.no_grad()
    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        video = video.to(self.device)
        indices = self._laq_task.model(video, return_only_codebook_ids=True)
        if indices.ndim != 2:
            raise ValueError(
                f"Expected indices [B, S], got shape {tuple(indices.shape)}"
            )
        return indices.to(torch.long)


def extract_oxe_language(batch: Dict[str, Any]) -> list[str]:
    language = batch.get("language")
    if language is None:
        raise KeyError("Expected OXE batch to include 'language' (list[str])")
    if not isinstance(language, list):
        raise TypeError(f"Expected 'language' to be a list[str], got {type(language)}")
    return [str(x) for x in language]
