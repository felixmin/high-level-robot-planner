"""
Tests for OXE v2 adapter.

Tests:
- OXEFramePairDatasetV2: output contract, shapes, dtypes
- Multi-dataset synthetic pipeline
- pair_frames_mode variants
- DataModule lifecycle
"""

import gc
import pytest
import torch
import numpy as np

pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_datasets")


SYNTHETIC_KWARGS = dict(
    image_size=64,
    batch_size=4,
    pair_frames_mode="endpoints",
    pair_frames_stride=1,
    pair_frames_n=2,
    total_threads=2,
    ram_budget_gb=1,
    shuffle_sample_buffer=0,
    seed=42,
    train=True,
    use_synthetic_data=True,
    synthetic_num_samples=100,
)


class TestOXEFramePairDatasetV2:
    @pytest.fixture
    def dataset(self):
        from common.adapters.oxe_v2 import OXEFramePairDatasetV2

        ds = OXEFramePairDatasetV2(
            dataset_entries=[
                {"name": "language_table", "pair_offset_steps": 5, "weight": 1.0,
                 "train_split": "train[:90%]", "val_split": "train[90%:]"},
            ],
            **SYNTHETIC_KWARGS,
        )
        yield ds
        ds.cleanup()

    def test_output_contract(self, dataset):
        """Verify output dict has all required keys with correct types."""
        it = iter(dataset)
        batch = next(it)

        assert isinstance(batch, dict)
        required_keys = {
            "frames", "episode_id", "frame_idx", "offset",
            "language", "dataset_name", "action", "initial_state", "robot",
        }
        assert required_keys == set(batch.keys())

    def test_frames_shape(self, dataset):
        """Frames should be (B, 3, num_frames, H, W) uint8."""
        batch = next(iter(dataset))
        frames = batch["frames"]

        assert isinstance(frames, torch.Tensor)
        B = SYNTHETIC_KWARGS["batch_size"]
        H = W = SYNTHETIC_KWARGS["image_size"]
        # endpoints mode: 2 frames
        assert frames.shape == (B, 3, 2, H, W)
        assert frames.dtype == torch.uint8

    def test_metadata_types(self, dataset):
        """Verify metadata field types."""
        batch = next(iter(dataset))

        B = SYNTHETIC_KWARGS["batch_size"]
        assert isinstance(batch["episode_id"], list) and len(batch["episode_id"]) == B
        assert isinstance(batch["frame_idx"], list) and len(batch["frame_idx"]) == B
        assert isinstance(batch["offset"], int)
        assert isinstance(batch["language"], list) and len(batch["language"]) == B
        assert isinstance(batch["dataset_name"], list) and len(batch["dataset_name"]) == B
        assert isinstance(batch["action"], list) and len(batch["action"]) == B
        assert isinstance(batch["initial_state"], list) and len(batch["initial_state"]) == B
        assert isinstance(batch["robot"], list) and len(batch["robot"]) == B

    def test_multiple_batches(self, dataset):
        """Verify we can iterate multiple batches without error."""
        it = iter(dataset)
        for _ in range(5):
            batch = next(it)
            assert batch["frames"].shape[0] == SYNTHETIC_KWARGS["batch_size"]

    def test_cleanup(self, dataset):
        """Verify cleanup releases pipeline resources."""
        _ = next(iter(dataset))
        dataset.cleanup()
        assert dataset._pipeline is None
        assert dataset._iterator is None


class TestMultiDatasetV2:
    def test_two_datasets(self):
        """Verify mixing two datasets produces correct output."""
        from common.adapters.oxe_v2 import OXEFramePairDatasetV2

        ds = OXEFramePairDatasetV2(
            dataset_entries=[
                {"name": "language_table", "pair_offset_steps": 5, "weight": 1.0,
                 "train_split": "train[:90%]", "val_split": "train[90%:]"},
                {"name": "bridge", "pair_offset_steps": 5, "weight": 1.0,
                 "train_split": "train[:90%]", "val_split": "train[90%:]"},
            ],
            **SYNTHETIC_KWARGS,
        )
        try:
            batch = next(iter(ds))
            assert batch["frames"].shape == (4, 3, 2, 64, 64)
            assert len(batch["dataset_name"]) == 4
        finally:
            ds.cleanup()


class TestPairFramesModes:
    @pytest.mark.parametrize("mode,expected_nf", [
        ("endpoints", 2),
        ("all", 6),  # offset=5 -> [0,1,2,3,4,5] -> 6 frames
        ("fixed_n", 3),
    ])
    def test_mode_num_frames(self, mode, expected_nf):
        """Verify different pair_frames_mode produce correct num_frames."""
        from common.adapters.oxe_v2 import OXEFramePairDatasetV2

        kwargs = {**SYNTHETIC_KWARGS}
        kwargs["pair_frames_mode"] = mode
        kwargs["pair_frames_n"] = 3  # for fixed_n

        ds = OXEFramePairDatasetV2(
            dataset_entries=[
                {"name": "language_table", "pair_offset_steps": 5, "weight": 1.0,
                 "train_split": "train[:90%]", "val_split": "train[90%:]"},
            ],
            **kwargs,
        )
        try:
            batch = next(iter(ds))
            assert batch["frames"].shape == (4, 3, expected_nf, 64, 64)
        finally:
            ds.cleanup()


class TestSharedFunctions:
    def test_compute_pair_frame_indices_endpoints(self):
        from common.adapters.oxe_shared import compute_pair_frame_indices

        assert compute_pair_frame_indices(5, "endpoints") == [0, 5]

    def test_compute_pair_frame_indices_all(self):
        from common.adapters.oxe_shared import compute_pair_frame_indices

        assert compute_pair_frame_indices(3, "all") == [0, 1, 2, 3]

    def test_compute_pair_frame_indices_stride(self):
        from common.adapters.oxe_shared import compute_pair_frame_indices

        assert compute_pair_frame_indices(10, "stride", stride=3) == [0, 3, 6, 9, 10]

    def test_compute_pair_frame_indices_fixed_n(self):
        from common.adapters.oxe_shared import compute_pair_frame_indices

        result = compute_pair_frame_indices(10, "fixed_n", n=3)
        assert result[0] == 0
        assert result[-1] == 10
        assert len(result) == 3

    def test_allocate_threads(self):
        from common.adapters.oxe_v2 import _allocate_threads

        alloc = _allocate_threads(10, [1.0, 1.0, 1.0])
        assert sum(alloc) <= 10
        assert all(t >= 1 for t in alloc)

    def test_allocate_threads_weighted(self):
        from common.adapters.oxe_v2 import _allocate_threads

        alloc = _allocate_threads(20, [3.0, 1.0])
        assert alloc[0] > alloc[1]


class TestOXEDataModuleV2:
    def test_setup_and_dataloader(self):
        from common.data import OXEDataModuleV2

        dm = OXEDataModuleV2(
            datasets=[
                {"name": "language_table", "pair_offset_steps": 5, "weight": 1.0,
                 "train_split": "train[:90%]", "val_split": "train[90%:]"},
            ],
            preprocess={"image_size": 64, "return_metadata": True},
            loader={"batch_size": 4, "num_workers": 0, "pin_memory": False},
            adapter={
                "tf_v2": {
                    "pair_frames": {"mode": "endpoints", "stride": 1, "n": 2},
                    "sampling": {"seed": 42},
                    "pipeline": {"total_threads": 2, "ram_budget_gb": 1},
                    "shuffle": {"sample_buffer": 0},
                    "tfds": {"source": "auto", "local_root": None},
                    "debug": {"use_synthetic_data": True, "synthetic_num_samples": 50},
                }
            },
        )
        dm.setup()
        try:
            train_dl = dm.train_dataloader()
            batch = next(iter(train_dl))
            assert batch["frames"].shape == (4, 3, 2, 64, 64)

            val_dl = dm.val_dataloader()
            val_batch = next(iter(val_dl))
            assert val_batch["frames"].shape == (4, 3, 2, 64, 64)
        finally:
            dm.teardown()


class TestMemoryStabilityV2:
    @pytest.mark.slow
    def test_no_memory_growth(self):
        """Verify no memory growth over multiple batches."""
        import tracemalloc

        from common.adapters.oxe_v2 import OXEFramePairDatasetV2

        ds = OXEFramePairDatasetV2(
            dataset_entries=[
                {"name": "language_table", "pair_offset_steps": 5, "weight": 1.0,
                 "train_split": "train[:90%]", "val_split": "train[90%:]"},
            ],
            **SYNTHETIC_KWARGS,
        )

        try:
            it = iter(ds)
            # Warm up
            for _ in range(5):
                _ = next(it)

            tracemalloc.start()
            snap1 = tracemalloc.take_snapshot()

            for _ in range(20):
                _ = next(it)

            snap2 = tracemalloc.take_snapshot()
            tracemalloc.stop()

            stats = snap2.compare_to(snap1, "lineno")
            growth = sum(s.size_diff for s in stats if s.size_diff > 0)
            # Allow up to 5MB growth (generous margin for gc timing)
            assert growth < 5 * 1024 * 1024, f"Memory grew by {growth / 1024 / 1024:.1f} MB"
        finally:
            ds.cleanup()
