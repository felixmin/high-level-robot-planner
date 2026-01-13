from __future__ import annotations

import pytest
import torch

from foundation.online_laq import oxe_frames_to_laq_video


def test_oxe_frames_to_laq_video_from_b2hwc3_uint8():
    frames = torch.randint(0, 256, (2, 2, 32, 32, 3), dtype=torch.uint8)
    video = oxe_frames_to_laq_video(frames)
    assert video.shape == (2, 3, 2, 32, 32)
    assert video.dtype == torch.float32
    assert float(video.min()) >= 0.0
    assert float(video.max()) <= 1.0


def test_oxe_frames_to_laq_video_from_b23hw_uint8():
    frames = torch.randint(0, 256, (2, 2, 3, 16, 16), dtype=torch.uint8)
    video = oxe_frames_to_laq_video(frames)
    assert video.shape == (2, 3, 2, 16, 16)
    assert video.dtype == torch.float32


def test_oxe_frames_to_laq_video_from_b32hw_float():
    frames = torch.rand((2, 3, 2, 8, 8), dtype=torch.float32)
    video = oxe_frames_to_laq_video(frames)
    assert video.shape == (2, 3, 2, 8, 8)
    assert video.dtype == torch.float32


def test_oxe_frames_to_laq_video_rejects_bad_shape():
    with pytest.raises(ValueError):
        oxe_frames_to_laq_video(torch.zeros((2, 3, 4)))

