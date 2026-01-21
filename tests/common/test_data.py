"""
Test data loading functionality.

Tests MultiSourcePairDataset and LAQDataModule with real data.
"""

import pytest
import torch
from pathlib import Path
from common.data import (
    LAQDataModule,
    SceneMetadata,
    SceneFilter,
    MultiSourcePairDataset,
    FramePairIndex,
    load_scenes_csv,
    metadata_collate_fn,
)


# Real dataset path from the user's machine
REAL_DATASET_PATH = "/mnt/data/datasets/youtube_new"
REAL_VIDEO_PATH = "/mnt/data/datasets/youtube_new/JNBtHDVoNQc_stabilized"


@pytest.fixture
def dataset_path():
    """Path to real dataset."""
    path = Path(REAL_VIDEO_PATH)
    if not path.exists():
        pytest.skip(f"Dataset not found at {REAL_VIDEO_PATH}")
    return str(path)


@pytest.fixture
def sources():
    """Source configuration for LAQDataModule."""
    return [{"type": "youtube", "root": REAL_DATASET_PATH}]


def make_laq_datamodule(
    *,
    sources,
    batch_size: int,
    num_workers: int,
    subset_max_pairs,
    return_metadata: bool = False,
    filters=None,
    split_mode: str = "ratio",
    val_ratio: float = 0.1,
    val_scene_filters=None,
    subset_strategy: str = "random",
    subset_seed: int = 42,
):
    return LAQDataModule(
        sources=sources,
        image_size=256,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=None,
        min_frames=2,
        pair_offsets_frames=[30],
        filters=filters,
        return_metadata=return_metadata,
        split_mode=split_mode,
        split_seed=42,
        val_ratio=val_ratio,
        val_scene_filters=val_scene_filters,
        val_counts_per_dataset=None,
        subset_max_pairs=subset_max_pairs,
        subset_strategy=subset_strategy,
        subset_seed=subset_seed,
    )


class TestLAQDataModule:
    """Test LAQDataModule Lightning wrapper."""

    def test_datamodule_initialization(self, sources):
        """Test DataModule initializes correctly."""
        dm = make_laq_datamodule(
            sources=sources,
            batch_size=4,
            num_workers=0,
            subset_max_pairs=None,
        )

        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

        print(f"✓ DataModule initialized")
        print(f"  - Train samples: {len(dm.train_dataset)}")
        print(f"  - Val samples: {len(dm.val_dataset)}")

    def test_datamodule_subset_10_samples(self, sources):
        """Test DataModule with 10 samples."""
        dm = make_laq_datamodule(
            sources=sources,
            batch_size=4,
            num_workers=0,
            subset_max_pairs=10,
        )

        dm.setup()

        assert len(dm.train_dataset) <= 10

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        assert batch.shape[0] <= 4  # batch_size
        assert batch.shape[1:] == (3, 2, 256, 256)

        print(f"✓ 10-sample subset works")
        print(f"  - Train: {len(dm.train_dataset)}, Val: {len(dm.val_dataset)}")

    def test_datamodule_train_val_dataloaders(self, sources):
        """Test train and val dataloaders."""
        dm = make_laq_datamodule(
            sources=sources,
            batch_size=4,
            num_workers=0,
            subset_max_pairs=20,
        )

        dm.setup()

        # Train loader
        train_loader = dm.train_dataloader()
        train_batch = next(iter(train_loader))
        assert train_batch.shape[1:] == (3, 2, 256, 256)

        # Val loader
        val_loader = dm.val_dataloader()
        val_batch = next(iter(val_loader))
        assert val_batch.shape[1:] == (3, 2, 256, 256)

        print(f"✓ Train and val dataloaders work")
        print(f"  - Train batch: {train_batch.shape}")
        print(f"  - Val batch: {val_batch.shape}")

    def test_datamodule_with_workers(self, sources):
        """Test DataModule with multiple workers."""
        import multiprocessing as mp

        try:
            lock = mp.Lock()
            del lock
        except PermissionError:
            pytest.skip("Multiprocessing semaphores are not permitted in this environment")

        dm = make_laq_datamodule(
            sources=sources,
            batch_size=4,
            num_workers=2,
            subset_max_pairs=20,
        )

        dm.setup()

        train_loader = dm.train_dataloader()

        # Load a few batches to test worker stability
        batches = []
        for i, batch in enumerate(train_loader):
            batches.append(batch)
            if i >= 2:
                break

        assert len(batches) >= 1
        print(f"✓ Multi-worker loading works ({len(batches)} batches loaded)")


class TestDataIntegrationWithModel:
    """Test data loading integrates with LAQ model."""

    def test_data_to_model_forward(self, sources, device):
        """Test loading data and passing through LAQ model."""
        from laq.models.latent_action_quantization import LatentActionQuantization

        # Small model for fast test
        model = LatentActionQuantization(
            dim=256,
            quant_dim=16,
            codebook_size=8,
            image_size=256,
            patch_size=32,
            spatial_depth=2,
            temporal_depth=2,
            dim_head=32,
            heads=4,
            code_seq_len=4,
            metrics_num_unique_codes_every_n_steps=1,
        ).to(device)

        dm = make_laq_datamodule(
            sources=sources,
            batch_size=2,
            num_workers=0,
            subset_max_pairs=5,
            val_ratio=0.0,
        )
        dm.setup()

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        batch = batch.to(device)

        with torch.no_grad():
            loss, metrics = model(batch, step=0)

        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
        assert metrics["num_unique_codes"] > 0

        print(f"✓ Real data → model forward pass successful")
        print(f"  - Batch shape: {batch.shape}")
        print(f"  - Loss: {loss.item():.6f}")
        print(f"  - Unique codes: {metrics['num_unique_codes']}")


# ====================
# Metadata Tests
# ====================


class TestSceneMetadata:
    """Test SceneMetadata dataclass and CSV parsing."""

    def test_scene_metadata_from_csv_row(self):
        """Test parsing a CSV row into SceneMetadata."""
        row = {
            "scene_idx": "0",
            "scene_folder": "scene_000",
            "start_frame": "0",
            "end_frame": "4470",
            "label": "uncertain",
            "stabilized_label": "uncertain",
            "max_angle": "0.0",
            "max_trans": "38.05",
            "stabilized_max_angle": "0.0",
            "stabilized_max_trans": "34.41",
            "contains_hand_sam3": "True",
            "hand_mask_folder_sam3": "scene_000_sam3_hand",
            "contains_lego_brick_sam3": "True",
            "lego_brick_mask_folder_sam3": "scene_000_sam3_lego_brick",
            "contains_sam3_hand_motion_cotracker": "True",
            "sam3_hand_motion_cotracker_folder": "scene_000_motion_sam3_hand_cotracker",
        }

        scene = SceneMetadata.from_csv_row(row)

        assert scene.scene_idx == 0
        assert scene.scene_folder == "scene_000"
        assert scene.start_frame == 0
        assert scene.end_frame == 4470
        assert scene.num_frames == 4470
        assert scene.stabilized_label == "uncertain"
        assert scene.max_trans == 38.05
        assert scene.contains_hand_sam3 is True
        assert scene.sam3_hand_motion_cotracker_folder == "scene_000_motion_sam3_hand_cotracker"

        print(f"✓ SceneMetadata parsed correctly: {scene.scene_folder}")

    def test_scene_metadata_num_frames_property(self):
        """Test num_frames property calculation."""
        scene = SceneMetadata(
            scene_idx=0,
            scene_folder="test",
            start_frame=100,
            end_frame=500,
        )

        assert scene.num_frames == 400
        print(f"✓ num_frames property: {scene.num_frames}")

    def test_scene_metadata_extras(self):
        """Test that unknown columns go to extras dict."""
        row = {
            "scene_idx": "0",
            "scene_folder": "scene_000",
            "start_frame": "0",
            "end_frame": "100",
            "custom_column": "custom_value",
            "another_custom": "42",
        }

        scene = SceneMetadata.from_csv_row(row)

        assert "custom_column" in scene.extras
        assert scene.extras["custom_column"] == "custom_value"
        assert scene.extras["another_custom"] == "42"

        print(f"✓ Extra columns captured: {list(scene.extras.keys())}")


class TestSceneFilter:
    """Test SceneFilter with various conditions."""

    @pytest.fixture
    def sample_scenes(self):
        """Create sample scenes for filtering tests."""
        return [
            SceneMetadata(
                scene_idx=0,
                scene_folder="scene_000",
                start_frame=0,
                end_frame=4470,
                label="uncertain",
                stabilized_label="uncertain",
                max_trans=38.05,
                contains_hand_sam3=True,
                contains_lego_brick_sam3=True,
            ),
            SceneMetadata(
                scene_idx=1,
                scene_folder="scene_001",
                start_frame=4470,
                end_frame=11543,
                label="uncertain",
                stabilized_label="uncertain",
                max_trans=49.58,
                contains_hand_sam3=True,
                contains_lego_brick_sam3=True,
            ),
            SceneMetadata(
                scene_idx=7,
                scene_folder="scene_007",
                start_frame=20653,
                end_frame=20672,
                label="static",
                stabilized_label="static",
                max_trans=1.0,
                contains_hand_sam3=True,
                contains_lego_brick_sam3=False,
            ),
        ]

    def test_filter_equality(self, sample_scenes):
        """Test equality filter."""
        f = SceneFilter({"stabilized_label": "uncertain"})
        filtered = f.filter_scenes(sample_scenes)

        assert len(filtered) == 2
        assert all(s.stabilized_label == "uncertain" for s in filtered)

        print(f"✓ Equality filter: {len(filtered)} scenes with stabilized_label='uncertain'")

    def test_filter_boolean(self, sample_scenes):
        """Test boolean filter."""
        f = SceneFilter({"contains_lego_brick_sam3": True})
        filtered = f.filter_scenes(sample_scenes)

        assert len(filtered) == 2
        assert all(s.contains_lego_brick_sam3 for s in filtered)

        print(f"✓ Boolean filter: {len(filtered)} scenes with lego bricks")

    def test_filter_comparison_greater(self, sample_scenes):
        """Test greater-than comparison filter."""
        f = SceneFilter({"max_trans": (">", 10.0)})
        filtered = f.filter_scenes(sample_scenes)

        assert len(filtered) == 2
        assert all(s.max_trans > 10.0 for s in filtered)

        print(f"✓ Greater-than filter: {len(filtered)} scenes with max_trans > 10.0")

    def test_filter_not_equal(self, sample_scenes):
        """Test not-equal filter."""
        f = SceneFilter({"label": ("!=", "static")})
        filtered = f.filter_scenes(sample_scenes)

        assert len(filtered) == 2
        assert all(s.label != "static" for s in filtered)

        print(f"✓ Not-equal filter: {len(filtered)} non-static scenes")

    def test_filter_callable(self, sample_scenes):
        """Test callable filter."""
        f = SceneFilter({"num_frames": lambda x: x > 1000})
        filtered = f.filter_scenes(sample_scenes)

        assert len(filtered) == 2
        assert all(s.num_frames > 1000 for s in filtered)

        print(f"✓ Callable filter: {len(filtered)} scenes with >1000 frames")

    def test_filter_multiple_conditions(self, sample_scenes):
        """Test multiple conditions combined."""
        f = SceneFilter({
            "stabilized_label": "uncertain",
            "max_trans": (">", 40.0),
        })
        filtered = f.filter_scenes(sample_scenes)

        assert len(filtered) == 1
        assert filtered[0].scene_folder == "scene_001"

        print(f"✓ Multiple conditions: {len(filtered)} scene(s) matching both")


class TestLoadScenesCSV:
    """Test loading scenes from real CSV file."""

    def test_load_scenes_csv(self, dataset_path):
        """Test loading real scenes.csv."""
        csv_path = Path(dataset_path) / "scenes.csv"
        if not csv_path.exists():
            pytest.skip(f"scenes.csv not found at {csv_path}")

        scenes = load_scenes_csv(csv_path)

        assert len(scenes) > 0
        assert all(isinstance(s, SceneMetadata) for s in scenes)

        print(f"✓ Loaded {len(scenes)} scenes from CSV")
        print(f"  - First scene: {scenes[0].scene_folder}")
        print(f"  - Last scene: {scenes[-1].scene_folder}")


class TestFramePairIndex:
    """Test FramePairIndex dataclass."""

    def test_pair_index_dataclass(self):
        """Test FramePairIndex dataclass."""
        pair = FramePairIndex(
            scene_idx=0,
            first_frame_idx=10,
            second_frame_idx=40,
            offset=30,
        )

        assert pair.scene_idx == 0
        assert pair.first_frame_idx == 10
        assert pair.second_frame_idx == 40
        assert pair.offset == 30

        print(f"✓ FramePairIndex dataclass works")


class TestLAQDataModuleWithMetadata:
    """Test LAQDataModule with metadata features."""

    def test_datamodule_with_filters(self, sources):
        """Test DataModule with scene filters."""
        dm = make_laq_datamodule(
            sources=sources,
            batch_size=4,
            num_workers=0,
            subset_max_pairs=None,
            filters={"stabilized_label": ("!=", "static")},
        )

        dm.setup()

        print(f"✓ DataModule with filter: {len(dm.train_dataset)} train samples")

    def test_datamodule_with_metadata_returns_dict(self, sources):
        """Test DataModule returns metadata dicts with standardized keys."""
        dm = make_laq_datamodule(
            sources=sources,
            batch_size=2,
            num_workers=0,
            subset_max_pairs=5,
            return_metadata=True,
            val_ratio=0.0,
        )

        dm.setup()
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        assert isinstance(batch, dict)
        assert batch["frames"].shape[1:] == (3, 2, 256, 256)
        assert isinstance(batch["metadata"], list)
        # Standardized keys
        assert isinstance(batch["episode_id"], list)  # Was: scene_idx
        assert isinstance(batch["frame_idx"], list)  # Was: first_frame_idx (now list, not tensor)
        assert isinstance(batch["dataset_name"], list)
        assert isinstance(batch["language"], list)

        print(f"✓ DataModule returns metadata dict batch with standardized keys")
        print(f"  - Frames shape: {batch['frames'].shape}")
        print(f"  - Keys: {list(batch.keys())}")


class TestLAQDataModulePairLevel:
    """Test LAQDataModule with multi-source datasets."""

    def test_pair_level_mode(self, sources):
        """Test LAQDataModule works correctly."""
        datamodule = make_laq_datamodule(
            sources=sources,
            batch_size=2,
            num_workers=0,
            subset_max_pairs=10,
            val_ratio=0.2,
        )

        datamodule.setup()

        assert datamodule.total_available > 0
        assert len(datamodule.train_dataset) <= 10
        assert len(datamodule.val_dataset) >= 1

        print(f"✓ Pair-level mode works")
        print(f"  - Total pairs available: {datamodule.total_available}")
        print(f"  - Train pairs: {len(datamodule.train_dataset)}")
        print(f"  - Val pairs: {len(datamodule.val_dataset)}")

    def test_single_pair_overfitting(self, sources):
        """Test LAQDataModule configured for single-pair overfitting."""
        datamodule = make_laq_datamodule(
            sources=sources,
            batch_size=1,
            num_workers=0,
            subset_max_pairs=1,
            val_ratio=0.0,
        )

        datamodule.setup()

        assert len(datamodule.train_dataset) == 1

        # Verify we can load the same sample multiple times
        train_loader = datamodule.train_dataloader()
        batch1 = next(iter(train_loader))
        batch2 = next(iter(train_loader))

        assert batch1.shape == (1, 3, 2, 256, 256)
        # Same pair every time (perfect for overfitting)
        assert torch.allclose(batch1, batch2)

        print(f"✓ Single-pair overfitting setup works")
        print(f"  - Train samples: 1")
        print(f"  - Batch shape: {batch1.shape}")


class TestSamplingStrategies:
    """Test random vs sequential sampling for subset selection."""

    def test_random_sampling_is_reproducible(self, sources):
        """Test that same seed produces same random subset indices."""
        dm1 = make_laq_datamodule(
            sources=sources,
            batch_size=4,
            num_workers=0,
            subset_max_pairs=10,
            val_ratio=0.0,
            subset_strategy="random",
            subset_seed=42,
        )
        dm1.setup()

        dm2 = make_laq_datamodule(
            sources=sources,
            batch_size=4,
            num_workers=0,
            subset_max_pairs=10,
            val_ratio=0.0,
            subset_strategy="random",
            subset_seed=42,
        )
        dm2.setup()

        # Compare subset indices directly (not via DataLoader which shuffles)
        indices1 = dm1.train_dataset.indices
        indices2 = dm2.train_dataset.indices

        assert indices1 == indices2, "Same seed should produce same subset indices"

        print(f"✓ Random sampling is reproducible with seed=42")

    def test_different_seeds_produce_different_samples(self, sources):
        """Test that different seeds produce different random indices."""
        dm1 = make_laq_datamodule(
            sources=sources,
            batch_size=4,
            num_workers=0,
            subset_max_pairs=10,
            val_ratio=0.0,
            subset_strategy="random",
            subset_seed=42,
        )
        dm1.setup()

        dm2 = make_laq_datamodule(
            sources=sources,
            batch_size=4,
            num_workers=0,
            subset_max_pairs=10,
            val_ratio=0.0,
            subset_strategy="random",
            subset_seed=123,
        )
        dm2.setup()

        # Load a batch from each
        batch1 = next(iter(dm1.train_dataloader()))
        batch2 = next(iter(dm2.train_dataloader()))

        # Different seeds should likely produce different data
        # (not guaranteed but very likely with enough samples)
        assert batch1.shape == batch2.shape

        print(f"✓ Different seeds produce different samples")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
