from __future__ import annotations

import torch

from common.lerobot_v3_adapters import dataset_batch_to_foundation_batch
from common.lerobot_v3_types import BatchedDatasetSample


def _batched_sample() -> BatchedDatasetSample:
    return BatchedDatasetSample(
        image_streams={"primary": torch.randint(0, 255, (2, 2, 3, 8, 8), dtype=torch.uint8)},
        image_padding_masks={"primary": torch.tensor([[False, False], [False, True]], dtype=torch.bool)},
        state=torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.float32),
        state_is_pad=torch.zeros((2, 1), dtype=torch.bool),
        action=torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=torch.float32,
        ),
        action_is_pad=torch.tensor([[False, True], [False, False]], dtype=torch.bool),
        task_text=["pick", "place"],
        meta={"dataset_name": ["a", "b"]},
    )


def test_dataset_batch_to_foundation_batch_maps_core_fields() -> None:
    foundation = dataset_batch_to_foundation_batch(_batched_sample())
    assert tuple(foundation.image_streams["primary"].shape) == (2, 2, 3, 8, 8)
    assert foundation.task_text == ["pick", "place"]
    assert tuple(foundation.state.shape) == (2, 1, 2)


def test_dataset_batch_to_foundation_batch_preserves_action_is_pad() -> None:
    foundation = dataset_batch_to_foundation_batch(_batched_sample())
    assert torch.equal(
        foundation.action_is_pad,
        torch.tensor([[False, True], [False, False]], dtype=torch.bool),
    )


def test_dataset_batch_to_foundation_batch_keeps_raw_actions_before_backend_normalization() -> None:
    foundation = dataset_batch_to_foundation_batch(_batched_sample())
    assert torch.equal(
        foundation.target_actions,
        torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=torch.float32,
        ),
    )
