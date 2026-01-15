from __future__ import annotations

import torch

from foundation.image_adapters import oxe_first_frames_to_pil


def test_oxe_first_frames_to_pil_from_b2hwc3():
    frames = torch.randint(0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8)
    images = oxe_first_frames_to_pil(frames)
    assert len(images) == 2
    assert images[0].mode == "RGB"
    assert images[0].size == (16, 16)


def test_oxe_first_frames_to_pil_from_b23hw():
    frames = torch.randint(0, 256, (2, 2, 3, 16, 16), dtype=torch.uint8)
    images = oxe_first_frames_to_pil(frames)
    assert len(images) == 2
    assert images[1].mode == "RGB"
    assert images[1].size == (16, 16)


def test_oxe_first_frames_to_pil_from_b32hw():
    frames = torch.randint(0, 256, (2, 3, 2, 16, 16), dtype=torch.uint8)
    images = oxe_first_frames_to_pil(frames)
    assert len(images) == 2
    assert images[0].mode == "RGB"
    assert images[0].size == (16, 16)
