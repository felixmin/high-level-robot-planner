"""
Adapters for converting dataset frame tensors into objects accepted by VLM processors.

For Qwen3-VL (Cosmos-Reason2) processors, images are typically provided as PIL Images.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image


def oxe_first_frames_to_pil(frames: torch.Tensor) -> List[Image.Image]:
    """
    Convert an OXE batch of frame pairs into PIL images for the first frame.

    Accepts:
      - [B, 2, H, W, 3] uint8 (standard OXE collate)
      - [B, 2, 3, H, W] uint8

    Returns:
      - List[PIL.Image.Image] length B (RGB)
    """

    if frames.ndim != 5:
        raise ValueError(f"Expected frames with ndim=5, got {frames.ndim}")

    if frames.shape[-1] == 3:
        first = frames[:, 0]  # [B, H, W, 3]
    elif frames.shape[2] == 3:
        first = frames[:, 0].permute(0, 2, 3, 1)  # [B, H, W, 3]
    else:
        raise ValueError(f"Unrecognized frame layout: {tuple(frames.shape)}")

    if first.dtype != torch.uint8:
        # Assume in [0,1] float and convert to uint8
        first = (first.clamp(0, 1) * 255.0).to(torch.uint8)

    first_np = first.cpu().numpy()
    images: List[Image.Image] = []
    for i in range(first_np.shape[0]):
        arr = np.asarray(first_np[i])
        images.append(Image.fromarray(arr, mode="RGB"))
    return images
