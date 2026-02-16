from __future__ import annotations

import pytest
import torch

from foundation.online_laq import LAQTaskCodeProvider, extract_oxe_actions, oxe_frames_to_laq_video


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


class _FakeLAQModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vq = torch.nn.Module()
        self.vq.num_embeddings = 3
        self.vq.codebooks = torch.nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.code_seq_len = 2

    def forward(self, video: torch.Tensor, return_only_codebook_ids: bool = False):
        assert return_only_codebook_ids is True
        batch_size = int(video.shape[0])
        return torch.tensor([[0, 2]] * batch_size, dtype=torch.long, device=video.device)


class _FakeLAQTask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeLAQModel()
        self._param = torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def freeze(self):
        return self


def test_laq_provider_codes_and_vectors():
    provider = LAQTaskCodeProvider(_FakeLAQTask())
    video = torch.rand((2, 3, 2, 8, 8), dtype=torch.float32)

    codes, vectors = provider.codes_and_vectors_from_video(video)
    assert codes.shape == (2, 2)
    assert vectors.shape == (2, 2, 2)
    assert torch.allclose(vectors[0, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(vectors[0, 1], torch.tensor([1.0, 1.0]))


def test_extract_oxe_actions_from_list():
    batch = {"action": [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]}
    actions = extract_oxe_actions(batch)
    assert actions.shape == (2, 3)
    assert actions.dtype == torch.float32


def test_extract_oxe_actions_from_list_of_tensors():
    batch = {"action": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]}
    actions = extract_oxe_actions(batch)
    assert actions.shape == (2, 2)
    assert actions.dtype == torch.float32
