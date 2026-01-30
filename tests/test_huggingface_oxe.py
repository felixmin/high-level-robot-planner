"""
Tests for HuggingFace-based OXE data loading.

These tests verify that we can load Open X-Embodiment datasets from HuggingFace
(via jxu124/OpenX-Embodiment) and extract frame pairs for LAQ training.

Run with: pytest tests/test_huggingface_oxe.py -v
"""

import pytest

# HF OXE backend relied on a HF dataset repo with a loading script; `datasets` no longer supports this.
# We use TFDS OXE (`data.backend=oxe_tf`) for all training and benchmarking.
pytest.skip("HF OXE backend is deprecated; use TFDS OXE (oxe_tf).", allow_module_level=True)

import torch
from torch.utils.data import DataLoader


class TestHFOXEFramePairDataset:
    """Tests for HFOXEFramePairDataset."""

    @pytest.fixture
    def bridge_dataset(self):
        """Create a small bridge dataset for testing."""
        from common.adapters.huggingface_oxe import HFOXEFramePairDataset

        return HFOXEFramePairDataset(
            dataset_name="bridge",
            split="train",
            offset=5,
            image_size=256,
            shuffle_buffer=10,
            return_metadata=True,
            samples_per_episode=2,  # Limit for faster tests
        )

    def test_dataset_yields_samples(self, bridge_dataset):
        """Verify dataset yields samples."""
        sample = next(iter(bridge_dataset))
        assert sample is not None
        assert "frames" in sample
        assert "episode_id" in sample
        assert "language" in sample

    def test_frames_shape(self, bridge_dataset):
        """Verify frames have correct shape [C, 2, H, W]."""
        sample = next(iter(bridge_dataset))
        frames = sample["frames"]
        assert frames.shape == (3, 2, 256, 256), f"Got shape {frames.shape}"

    def test_frames_dtype(self, bridge_dataset):
        """Verify frames are float tensors in [0, 1]."""
        sample = next(iter(bridge_dataset))
        frames = sample["frames"]
        assert frames.dtype == torch.float32
        assert frames.min() >= 0.0
        assert frames.max() <= 1.0

    def test_metadata_present(self, bridge_dataset):
        """Verify metadata fields are present."""
        sample = next(iter(bridge_dataset))
        assert "action" in sample
        assert "initial_state" in sample
        assert sample["action"].shape == (3,)  # bridge has 3D action


class TestHFCollate:
    """Tests for the collate function."""

    def test_collate_batches_correctly(self):
        """Verify collate produces correct batch shapes."""
        from common.adapters.huggingface_oxe import (
            HFOXEFramePairDataset,
            hf_collate_fn,
        )

        ds = HFOXEFramePairDataset(
            dataset_name="bridge",
            split="train",
            offset=5,
            image_size=256,
            shuffle_buffer=10,
            return_metadata=True,
            samples_per_episode=2,
        )

        loader = DataLoader(ds, batch_size=4, collate_fn=hf_collate_fn)
        batch = next(iter(loader))

        # Check batch structure
        assert batch["frames"].shape == (4, 3, 2, 256, 256)
        assert len(batch["episode_id"]) == 4
        assert len(batch["language"]) == 4
        assert batch["action"].shape == (4, 3)


class TestHFOXEDataModule:
    """Tests for HFOXEDataModule."""

    def test_datamodule_setup(self):
        """Verify DataModule creates train/val datasets."""
        from common.adapters.huggingface_oxe import HFOXEDataModule

        dm = HFOXEDataModule(
            datasets=[
                {
                    "name": "bridge",
                    "train_split": "train",
                    "val_split": "train",
                    "pair_offset_steps": 5,
                    "weight": 1.0,
                    "approx_num_pairs": None,
                }
            ],
            preprocess={"image_size": 256, "return_metadata": True},
            loader={"batch_size": 4, "num_workers": 0, "pin_memory": True},
            adapter={"hf": {"train_shuffle_buffer": 10, "val_shuffle_buffer": 0, "samples_per_episode": 2}},
        )
        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_datamodule_train_dataloader(self):
        """Verify train dataloader produces correct batches."""
        from common.adapters.huggingface_oxe import HFOXEDataModule

        dm = HFOXEDataModule(
            datasets=[
                {
                    "name": "bridge",
                    "train_split": "train",
                    "val_split": "train",
                    "pair_offset_steps": 5,
                    "weight": 1.0,
                    "approx_num_pairs": None,
                }
            ],
            preprocess={"image_size": 256, "return_metadata": True},
            loader={"batch_size": 4, "num_workers": 0, "pin_memory": True},
            adapter={"hf": {"train_shuffle_buffer": 10, "val_shuffle_buffer": 0, "samples_per_episode": 2}},
        )
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        # Verify batch format matches STANDARD_BATCH_KEYS
        assert "frames" in batch
        assert "episode_id" in batch
        assert "frame_idx" in batch
        assert "dataset_name" in batch
        assert "language" in batch

        # Verify shapes
        assert batch["frames"].shape[1:] == (3, 2, 256, 256)


class TestDatasetConfigs:
    """Tests for different dataset configurations."""

    @pytest.mark.parametrize("dataset_name", ["bridge", "language_table"])
    def test_dataset_loads(self, dataset_name):
        """Verify each supported dataset can be loaded."""
        from common.adapters.huggingface_oxe import HFOXEFramePairDataset

        ds = HFOXEFramePairDataset(
            dataset_name=dataset_name,
            split="train",
            offset=5,
            image_size=256,
            shuffle_buffer=10,
            return_metadata=True,
            samples_per_episode=1,
        )

        sample = next(iter(ds))
        assert sample["frames"].shape == (3, 2, 256, 256)
        assert sample["dataset_name"] == dataset_name


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
