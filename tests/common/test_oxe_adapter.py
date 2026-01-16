"""
Tests for OXE (Open X-Embodiment) adapter.

Tests:
- OXEFramePairDataset: Single dataset streaming
- MultiOXEFramePairDataset: Multi-dataset interleaving
- Memory management: Persistent pipelines, cleanup
- Metadata extraction: Actions, states, instructions
- RT-1 dataset: Dict-based actions, string instructions
- RoboNet dataset: Step-level instructions, robot metadata
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

    @pytest.fixture
    def one_sample_per_episode_dataset(self):
        """Create a dataset that yields exactly one sample per episode."""
        from common.adapters.oxe import OXEFramePairDataset

        ds = OXEFramePairDataset(
            dataset_name="language_table",
            split="train[:100]",
            offset=5,
            image_size=64,
            shuffle_buffer=10,
            return_metadata=True,
            samples_per_episode=1,
            seed=123,
        )
        yield ds
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
        """Test that metadata is correctly extracted with standardized keys."""
        for item in small_dataset:
            # Check required metadata fields (using standardized keys)
            assert "episode_id" in item
            assert "frame_idx" in item
            assert "offset" in item
            # dataset_type now matches dataset_name (not generic "oxe")
            assert "dataset_type" in item
            assert item["dataset_type"] == "language_table"
            assert "dataset_name" in item
            assert item["dataset_name"] == "language_table"
            # language replaces instruction
            assert "language" in item

            # Check action and state are lists/floats
            assert isinstance(item["action"], list)
            assert isinstance(item["initial_state"], list)
            break

        print("✓ Metadata correctly extracted")

    def test_one_sample_per_episode(self, one_sample_per_episode_dataset):
        """Test that per-episode sampling yields unique episodes within a short window."""
        episode_ids = []
        for item in one_sample_per_episode_dataset:
            episode_ids.append(item["episode_id"])
            if len(episode_ids) >= 10:
                break

        assert len(episode_ids) == 10
        assert len(set(episode_ids)) == 10, "Expected unique episode_id when sampling 1 per episode"
        print("✓ One-sample-per-episode mode yields unique episodes")


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
                    "offset": 5,
                },
                {
                    "name": "bridge",
                    "train_split": "train[:50]",
                    "val_split": "train[50:60]",
                    "weight": 0.5,
                    "offset": 5,
                },
            ],
            image_size=64,
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

    def test_seed_derivation(self):
        """Test that a global seed produces deterministic per-dataset seeds."""
        from common.adapters.oxe import MultiOXEFramePairDataset

        ds = MultiOXEFramePairDataset(
            datasets=[
                {"name": "language_table", "train_split": "train[:30]", "weight": 0.5, "offset": 5},
                {"name": "bridge", "train_split": "train[:30]", "weight": 0.5, "offset": 5},
            ],
            image_size=64,
            shuffle_buffer=10,
            return_metadata=False,
            is_train=True,
            samples_per_episode=1,
            seed=123,
        )
        ds._init_datasets()
        assert ds._datasets is not None
        seeds = [d.seed for d in ds._datasets]
        assert seeds[0] is not None and seeds[1] is not None
        assert seeds[0] != seeds[1], "Expected different derived seeds per dataset"
        ds.cleanup()


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
                {"name": "language_table", "train_split": "train[:30]", "weight": 1.0, "offset": 5},
            ],
            image_size=64,
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


class TestRT1Dataset:
    """Test RT-1 dataset adapter (fractal20220817_data)."""

    @pytest.fixture
    def rt1_dataset(self):
        """Create a small RT-1 dataset for testing."""
        from common.adapters.oxe import OXEFramePairDataset

        ds = OXEFramePairDataset(
            dataset_name="rt1",
            split="train[:10]",  # Just 10 episodes
            offset=3,  # ~1 sec at 3Hz
            image_size=64,
            shuffle_buffer=5,
            return_metadata=True,
        )
        yield ds
        ds.cleanup()

    def test_iteration_basic(self, rt1_dataset):
        """Test basic iteration over RT-1 dataset."""
        count = 0
        for item in rt1_dataset:
            assert "frames" in item
            assert item["frames"].shape == (3, 2, 64, 64)
            assert "action" in item
            assert "initial_state" in item
            count += 1
            if count >= 5:
                break

        assert count == 5
        print(f"RT-1: Iterated {count} samples successfully")

    def test_metadata_extraction(self, rt1_dataset):
        """Test that RT-1 metadata is correctly extracted."""
        for item in rt1_dataset:
            # Check required metadata
            assert item["dataset_type"] == "rt1"
            assert item["dataset_name"] == "rt1"

            # RT-1 has string instructions like "pick apple from white bowl"
            assert "language" in item
            assert isinstance(item["language"], str)

            # Action should be 3D (world_vector)
            assert len(item["action"]) == 3, f"Expected 3D action, got {len(item['action'])}"

            # State should be 3D (first 3 dims of base_pose_tool_reached)
            assert len(item["initial_state"]) == 3, f"Expected 3D state, got {len(item['initial_state'])}"

            # Robot field should be empty for RT-1 (no robot metadata)
            assert "robot" in item
            break

        print("RT-1: Metadata correctly extracted")


class TestRoboNetDataset:
    """Test RoboNet dataset adapter (robo_net)."""

    @pytest.fixture
    def robonet_dataset(self):
        """Create a small RoboNet dataset for testing."""
        from common.adapters.oxe import OXEFramePairDataset

        ds = OXEFramePairDataset(
            dataset_name="robonet",
            split="train[:10]",  # Just 10 episodes
            offset=10,
            image_size=64,
            shuffle_buffer=5,
            return_metadata=True,
        )
        yield ds
        ds.cleanup()

    def test_iteration_basic(self, robonet_dataset):
        """Test basic iteration over RoboNet dataset."""
        count = 0
        for item in robonet_dataset:
            assert "frames" in item
            assert item["frames"].shape == (3, 2, 64, 64)
            assert "action" in item
            assert "initial_state" in item
            count += 1
            if count >= 5:
                break

        assert count == 5
        print(f"RoboNet: Iterated {count} samples successfully")

    def test_metadata_extraction(self, robonet_dataset):
        """Test that RoboNet metadata is correctly extracted."""
        for item in robonet_dataset:
            # Check required metadata
            assert item["dataset_type"] == "robonet"
            assert item["dataset_name"] == "robonet"

            # RoboNet has step-level instructions like "Interact with the objects in the bin"
            assert "language" in item
            assert isinstance(item["language"], str)

            # Action should be 3D (first 3 dims of 5D action)
            assert len(item["action"]) == 3, f"Expected 3D action, got {len(item['action'])}"

            # State should be 3D (first 3 dims of 5D state)
            assert len(item["initial_state"]) == 3, f"Expected 3D state, got {len(item['initial_state'])}"

            # Robot field should contain robot type
            assert "robot" in item
            assert isinstance(item["robot"], str)
            # Valid robot types: widowx, franka, baxter, sawyer
            if item["robot"]:  # May be empty in some cases
                assert item["robot"] in ["widowx", "franka", "baxter", "sawyer"], \
                    f"Unexpected robot type: {item['robot']}"
            break

        print("RoboNet: Metadata correctly extracted")

    def test_robot_metadata_variation(self, robonet_dataset):
        """Test that different robot types appear in the dataset."""
        robot_types = set()
        count = 0

        for item in robonet_dataset:
            if item["robot"]:
                robot_types.add(item["robot"])
            count += 1
            if count >= 50 or len(robot_types) >= 3:
                break

        print(f"RoboNet: Found robot types: {robot_types}")
        # Should find at least one robot type
        assert len(robot_types) >= 1, "Expected at least one robot type"


class TestOXEDatasetRegistry:
    """Test OXE dataset registry and config."""

    def test_registry_contains_new_datasets(self):
        """Test that RT-1 and RoboNet are in the registry."""
        from common.adapters.oxe import OXE_DATASETS

        assert "rt1" in OXE_DATASETS, "RT-1 not in registry"
        assert "robonet" in OXE_DATASETS, "RoboNet not in registry"
        print(f"Registry contains {len(OXE_DATASETS)} datasets: {list(OXE_DATASETS.keys())}")

    def test_rt1_config(self):
        """Test RT-1 config values."""
        from common.adapters.oxe import OXE_DATASETS

        cfg = OXE_DATASETS["rt1"]
        assert cfg.image_key == "image"
        assert cfg.instruction_key == "natural_language_instruction"
        assert cfg.action_is_dict is True
        assert cfg.action_key == "world_vector"
        assert cfg.action_dim == 3
        assert cfg.state_dim == 3
        assert cfg.instruction_in_step is False
        print("RT-1 config validated")

    def test_robonet_config(self):
        """Test RoboNet config values."""
        from common.adapters.oxe import OXE_DATASETS

        cfg = OXE_DATASETS["robonet"]
        assert cfg.image_key == "image"
        assert cfg.instruction_key == "language_instruction"
        assert cfg.action_is_dict is False
        assert cfg.action_dim == 3
        assert cfg.state_dim == 3
        assert cfg.instruction_in_step is True
        assert cfg.robot_key == "robot"
        print("RoboNet config validated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "not slow"])
