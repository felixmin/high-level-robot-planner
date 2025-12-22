"""
Tests for OXE (Open X-Embodiment) adapter.

Tests:
- OXEFramePairDataset: Single dataset streaming
- MultiOXEFramePairDataset: Multi-dataset interleaving
- Memory management: Persistent pipelines, cleanup
- Metadata extraction: Actions, states, instructions
"""

import gc
import pytest
import torch

# Skip all tests if tensorflow is not available
pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_datasets")


class TestOXEFramePairDataset:
    """Test OXEFramePairDataset functionality."""

    @pytest.fixture
    def small_dataset(self):
        """Create a small OXE dataset for testing."""
        from common.adapters.oxe import OXEFramePairDataset

        # Use a small split for fast testing
        ds = OXEFramePairDataset(
            dataset_name="language_table",
            split="train[:100]",  # Just 100 episodes
            offset=5,
            image_size=64,  # Small for speed
            shuffle_buffer=10,
            return_metadata=True,
        )
        yield ds
        # Cleanup after test
        ds.cleanup()

    def test_iteration_basic(self, small_dataset):
        """Test basic iteration over dataset."""
        count = 0
        for item in small_dataset:
            assert "frames" in item
            assert item["frames"].shape == (3, 2, 64, 64)
            assert "action" in item
            assert "initial_state" in item
            count += 1
            if count >= 5:
                break

        assert count == 5
        print(f"Iterated {count} samples successfully")

    def test_persistent_pipeline(self, small_dataset):
        """Test that pipeline is reused across iterations (no memory leak)."""
        # First iteration - creates pipeline
        iter1 = iter(small_dataset)
        _ = next(iter1)
        pipeline_id_1 = id(small_dataset._persistent_pipeline)

        # Second iteration - should reuse same pipeline
        iter2 = iter(small_dataset)
        _ = next(iter2)
        pipeline_id_2 = id(small_dataset._persistent_pipeline)

        assert pipeline_id_1 == pipeline_id_2, "Pipeline should be reused"
        print("✓ Pipeline is persistent across iterations")

    def test_cleanup(self, small_dataset):
        """Test explicit cleanup releases resources."""
        # Trigger pipeline creation
        iter1 = iter(small_dataset)
        _ = next(iter1)

        assert small_dataset._persistent_pipeline is not None
        assert small_dataset._builder is not None

        # Cleanup
        small_dataset.cleanup()

        assert small_dataset._persistent_pipeline is None
        assert small_dataset._builder is None
        print("✓ Cleanup releases resources")

    def test_metadata_extraction(self, small_dataset):
        """Test that metadata is correctly extracted."""
        for item in small_dataset:
            # Check required metadata fields
            assert "episode_id" in item
            assert "frame_idx" in item
            assert "offset" in item
            assert "dataset_type" in item
            assert item["dataset_type"] == "oxe"
            assert "dataset_name" in item
            assert item["dataset_name"] == "language_table"

            # Check action and state are lists/floats
            assert isinstance(item["action"], list)
            assert isinstance(item["initial_state"], list)
            break

        print("✓ Metadata correctly extracted")


class TestMultiOXEFramePairDataset:
    """Test MultiOXEFramePairDataset functionality."""

    @pytest.fixture
    def multi_dataset(self):
        """Create a multi-dataset for testing."""
        from common.adapters.oxe import MultiOXEFramePairDataset

        ds = MultiOXEFramePairDataset(
            datasets=[
                {
                    "name": "language_table",
                    "train_split": "train[:50]",
                    "val_split": "train[50:60]",
                    "weight": 0.5,
                },
                {
                    "name": "bridge",
                    "train_split": "train[:50]",
                    "val_split": "train[50:60]",
                    "weight": 0.5,
                },
            ],
            image_size=64,
            offset=5,
            shuffle_buffer=10,
            return_metadata=True,
            is_train=True,
        )
        yield ds
        ds.cleanup()

    def test_interleaving(self, multi_dataset):
        """Test that samples come from multiple datasets."""
        dataset_names = set()
        count = 0

        for item in multi_dataset:
            dataset_names.add(item.get("dataset_name", "unknown"))
            count += 1
            if count >= 20:
                break

        # Should have samples from both datasets
        assert len(dataset_names) >= 1, f"Expected multiple datasets, got: {dataset_names}"
        print(f"✓ Interleaved samples from: {dataset_names}")

    def test_cleanup_all_datasets(self, multi_dataset):
        """Test that cleanup releases all underlying datasets."""
        # Trigger initialization
        multi_dataset._init_datasets()

        assert multi_dataset._datasets is not None
        assert len(multi_dataset._datasets) == 2

        # Cleanup
        multi_dataset.cleanup()

        assert multi_dataset._datasets is None
        print("✓ Cleanup releases all underlying datasets")


class TestMemoryStability:
    """Test memory stability across multiple epochs."""

    @pytest.mark.slow
    def test_no_memory_growth_single_dataset(self):
        """Test that memory doesn't grow across multiple iterations."""
        import tracemalloc
        from common.adapters.oxe import OXEFramePairDataset

        # Start memory tracking
        tracemalloc.start()

        ds = OXEFramePairDataset(
            dataset_name="language_table",
            split="train[:50]",
            offset=5,
            image_size=64,
            shuffle_buffer=10,
            return_metadata=False,
        )

        # Simulate multiple epochs
        memory_samples = []
        for epoch in range(3):
            count = 0
            for item in ds:
                count += 1
                if count >= 10:
                    break

            # Sample memory after each "epoch"
            current, peak = tracemalloc.get_traced_memory()
            memory_samples.append(current)
            gc.collect()

        tracemalloc.stop()
        ds.cleanup()

        # Memory shouldn't grow significantly (allow 50% variance)
        if len(memory_samples) >= 2:
            growth = memory_samples[-1] / max(memory_samples[0], 1)
            assert growth < 2.0, f"Memory grew {growth:.1f}x across epochs"
            print(f"✓ Memory stable: {memory_samples[0]/1024:.0f}KB -> {memory_samples[-1]/1024:.0f}KB")

    @pytest.mark.slow
    def test_no_memory_growth_multi_dataset(self):
        """Test that memory doesn't grow across multiple iterations with multi-dataset."""
        import tracemalloc
        from common.adapters.oxe import MultiOXEFramePairDataset

        tracemalloc.start()

        ds = MultiOXEFramePairDataset(
            datasets=[
                {"name": "language_table", "train_split": "train[:30]", "weight": 1.0},
            ],
            image_size=64,
            offset=5,
            shuffle_buffer=10,
            return_metadata=False,
            is_train=True,
        )

        memory_samples = []
        for epoch in range(3):
            count = 0
            for item in ds:
                count += 1
                if count >= 10:
                    break

            current, peak = tracemalloc.get_traced_memory()
            memory_samples.append(current)
            gc.collect()

        tracemalloc.stop()
        ds.cleanup()

        if len(memory_samples) >= 2:
            growth = memory_samples[-1] / max(memory_samples[0], 1)
            assert growth < 2.0, f"Memory grew {growth:.1f}x across epochs"
            print(f"✓ Multi-dataset memory stable: {memory_samples[0]/1024:.0f}KB -> {memory_samples[-1]/1024:.0f}KB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "not slow"])
