"""
Test data loading functionality.

Tests ImageVideoDataset, MetadataAwareDataset, and LAQDataModule with real data.
"""

import pytest
import torch
from pathlib import Path
from common.data import (
    ImageVideoDataset,
    LAQDataModule,
    SceneMetadata,
    SceneFilter,
    MetadataAwareDataset,
    load_scenes_csv,
    metadata_collate_fn,
)


# Real dataset path from the user's machine
REAL_DATASET_PATH = "/mnt/data/datasets/youtube_new/JNBtHDVoNQc_stabilized"


@pytest.fixture
def dataset_path():
    """Path to real dataset."""
    path = Path(REAL_DATASET_PATH)
    if not path.exists():
        pytest.skip(f"Dataset not found at {REAL_DATASET_PATH}")
    return str(path)


class TestImageVideoDataset:
    """Test ImageVideoDataset with real data."""

    def test_dataset_initialization(self, dataset_path):
        """Test dataset initializes correctly."""
        dataset = ImageVideoDataset(
            folder=dataset_path,
            image_size=256,
            offset=30,
        )

        assert len(dataset) > 0
        print(f"✓ Dataset initialized with {len(dataset)} scenes")

    def test_dataset_load_single_sample(self, dataset_path):
        """Test loading a single frame pair."""
        dataset = ImageVideoDataset(
            folder=dataset_path,
            image_size=256,
            offset=30,
        )

        # Load first sample
        sample = dataset[0]

        # Check shape: [C, T, H, W] where T=2
        assert sample.shape == (3, 2, 256, 256)
        assert sample.dtype == torch.float32

        # Check value range (normalized by ToTensor)
        assert sample.min() >= 0.0
        assert sample.max() <= 1.0

        print(f"✓ Loaded sample shape: {sample.shape}")
        print(f"  - Value range: [{sample.min():.3f}, {sample.max():.3f}]")

    def test_dataset_load_multiple_samples(self, dataset_path):
        """Test loading multiple samples."""
        dataset = ImageVideoDataset(
            folder=dataset_path,
            image_size=256,
            offset=30,
        )

        # Load 5 samples
        samples = [dataset[i] for i in range(min(5, len(dataset)))]

        for i, sample in enumerate(samples):
            assert sample.shape == (3, 2, 256, 256)
            print(f"✓ Sample {i}: shape={sample.shape}")

    def test_dataset_different_offsets(self, dataset_path):
        """Test dataset with different frame offsets."""
        for offset in [1, 10, 30, 60]:
            dataset = ImageVideoDataset(
                folder=dataset_path,
                image_size=256,
                offset=offset,
            )

            sample = dataset[0]
            assert sample.shape == (3, 2, 256, 256)
            print(f"✓ Offset {offset}: loaded successfully")

    def test_dataset_different_image_sizes(self, dataset_path):
        """Test dataset with different image sizes."""
        for size in [128, 224, 256]:
            dataset = ImageVideoDataset(
                folder=dataset_path,
                image_size=size,
                offset=30,
            )

            sample = dataset[0]
            assert sample.shape == (3, 2, size, size)
            print(f"✓ Image size {size}: loaded successfully")

    def test_dataset_iteration(self, dataset_path):
        """Test iterating through dataset."""
        dataset = ImageVideoDataset(
            folder=dataset_path,
            image_size=256,
            offset=30,
        )

        # Test iteration with DataLoader
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        batch = next(iter(loader))
        assert batch.shape == (2, 3, 2, 256, 256)

        print(f"✓ DataLoader batch shape: {batch.shape}")


class TestLAQDataModule:
    """Test LAQDataModule Lightning wrapper."""

    def test_datamodule_initialization(self, dataset_path):
        """Test DataModule initializes correctly."""
        dm = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            offset=30,
            batch_size=4,
            num_workers=0,
            max_samples=None,  # Full dataset
        )

        # Setup is called automatically by Lightning, but we can call manually
        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

        print(f"✓ DataModule initialized")
        print(f"  - Train samples: {len(dm.train_dataset)}")
        print(f"  - Val samples: {len(dm.val_dataset)}")

    def test_datamodule_subset_1_sample(self, dataset_path):
        """Test DataModule with 1 sample (overfit test)."""
        dm = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            offset=30,
            batch_size=1,
            num_workers=0,
            max_samples=1,  # Only 1 sample!
        )

        dm.setup()

        # With 1 sample and 10% val split, we get 0 train, 1 val
        # So let's use 0% val split for 1 sample
        dm.val_split = 0.0
        dm.setup()

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        assert batch.shape == (1, 3, 2, 256, 256)

        print(f"✓ 1-sample subset works")
        print(f"  - Batch shape: {batch.shape}")

    def test_datamodule_subset_10_samples(self, dataset_path):
        """Test DataModule with 10 samples."""
        dm = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            offset=30,
            batch_size=4,
            num_workers=0,
            max_samples=10,
        )

        dm.setup()

        assert len(dm.train_dataset) == 9  # 90% of 10
        assert len(dm.val_dataset) == 1    # 10% of 10

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        assert batch.shape[0] == 4  # batch_size
        assert batch.shape[1:] == (3, 2, 256, 256)

        print(f"✓ 10-sample subset works")
        print(f"  - Train: {len(dm.train_dataset)}, Val: {len(dm.val_dataset)}")

    def test_datamodule_subset_100_samples(self, dataset_path):
        """Test DataModule with 100 samples (legacy mode since CSV has fewer scenes)."""
        dm = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            offset=30,
            batch_size=8,
            num_workers=0,
            max_samples=100,
            use_metadata=False,  # Legacy mode for testing subset logic
        )

        dm.setup()

        # Legacy ImageVideoDataset finds 114 scene folders
        # With max_samples=100, we get 90 train + 10 val
        assert len(dm.train_dataset) == 90   # 90% of 100
        assert len(dm.val_dataset) == 10     # 10% of 100

        print(f"✓ 100-sample subset works")
        print(f"  - Train: {len(dm.train_dataset)}, Val: {len(dm.val_dataset)}")

    def test_datamodule_train_val_dataloaders(self, dataset_path):
        """Test train and val dataloaders."""
        dm = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            offset=30,
            batch_size=4,
            num_workers=0,
            max_samples=20,
        )

        dm.setup()

        # Train loader
        train_loader = dm.train_dataloader()
        train_batch = next(iter(train_loader))
        assert train_batch.shape == (4, 3, 2, 256, 256)

        # Val loader
        val_loader = dm.val_dataloader()
        val_batch = next(iter(val_loader))
        assert val_batch.shape[0] <= 4  # May be smaller if val set < batch_size
        assert val_batch.shape[1:] == (3, 2, 256, 256)

        print(f"✓ Train and val dataloaders work")
        print(f"  - Train batch: {train_batch.shape}")
        print(f"  - Val batch: {val_batch.shape}")

    def test_datamodule_with_workers(self, dataset_path):
        """Test DataModule with multiple workers."""
        dm = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            offset=30,
            batch_size=4,
            num_workers=2,  # Use 2 workers
            max_samples=20,
        )

        dm.setup()

        train_loader = dm.train_dataloader()

        # Load a few batches to test worker stability
        batches = []
        for i, batch in enumerate(train_loader):
            batches.append(batch)
            if i >= 2:  # Load 3 batches
                break

        assert len(batches) == 3
        print(f"✓ Multi-worker loading works ({len(batches)} batches loaded)")


class TestDataIntegrationWithModel:
    """Test data loading integrates with LAQ model."""

    def test_data_to_model_forward(self, dataset_path, device):
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
        ).to(device)

        # Load real data
        dm = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            offset=30,
            batch_size=2,
            num_workers=0,
            max_samples=5,
        )
        dm.val_split = 0.0  # No val split for this test
        dm.setup()

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # Move to device and reshape for model
        # Input is [B, C, 2, H, W], model expects [B, C, 2, H, W]
        batch = batch.to(device)

        # Forward pass
        with torch.no_grad():
            loss, num_unique = model(batch, step=0)

        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
        assert num_unique > 0

        print(f"✓ Real data → model forward pass successful")
        print(f"  - Batch shape: {batch.shape}")
        print(f"  - Loss: {loss.item():.6f}")
        print(f"  - Unique codes: {num_unique}")


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

    def test_scene_metadata_empty_optional_fields(self):
        """Test parsing with empty optional fields."""
        row = {
            "scene_idx": "5",
            "scene_folder": "scene_005",
            "start_frame": "0",
            "end_frame": "100",
            "label": "static",
            "stabilized_label": "static",
            "max_angle": "0.0",
            "max_trans": "1.0",
            "stabilized_max_angle": "0.0",
            "stabilized_max_trans": "3.16",
            "contains_hand_sam3": "True",
            "hand_mask_folder_sam3": "scene_005_sam3_hand",
            "contains_lego_brick_sam3": "False",
            "lego_brick_mask_folder_sam3": "",  # Empty!
            "contains_sam3_hand_motion_cotracker": "True",
            "sam3_hand_motion_cotracker_folder": "scene_005_motion",
        }

        scene = SceneMetadata.from_csv_row(row)

        assert scene.contains_lego_brick_sam3 is False
        assert scene.lego_brick_mask_folder_sam3 is None  # Empty string → None

        print(f"✓ Empty optional fields parsed as None")


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

        assert len(filtered) == 2  # scene_000 has 4470, scene_001 has 7073 frames
        assert all(s.num_frames > 1000 for s in filtered)

        print(f"✓ Callable filter: {len(filtered)} scenes with >1000 frames")

    def test_filter_multiple_conditions(self, sample_scenes):
        """Test multiple conditions combined."""
        f = SceneFilter({
            "stabilized_label": "uncertain",
            "max_trans": (">", 40.0),
        })
        filtered = f.filter_scenes(sample_scenes)

        assert len(filtered) == 1  # Only scene_001 with max_trans=49.58
        assert filtered[0].scene_folder == "scene_001"

        print(f"✓ Multiple conditions: {len(filtered)} scene(s) matching both")

    def test_filter_no_conditions(self, sample_scenes):
        """Test empty filter (should return all)."""
        f = SceneFilter({})
        filtered = f.filter_scenes(sample_scenes)

        assert len(filtered) == len(sample_scenes)

        print(f"✓ Empty filter: returns all {len(filtered)} scenes")


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

    def test_scenes_have_motion_tracks(self, dataset_path):
        """Test that scenes have motion track folders."""
        csv_path = Path(dataset_path) / "scenes.csv"
        if not csv_path.exists():
            pytest.skip(f"scenes.csv not found at {csv_path}")

        scenes = load_scenes_csv(csv_path)

        with_tracks = [s for s in scenes if s.contains_sam3_hand_motion_cotracker]
        print(f"✓ {len(with_tracks)}/{len(scenes)} scenes have motion tracks")


class TestMetadataAwareDataset:
    """Test MetadataAwareDataset with real data and filtering."""

    def test_dataset_initialization(self, dataset_path):
        """Test MetadataAwareDataset initializes from scenes.csv."""
        dataset = MetadataAwareDataset(
            folder=dataset_path,
            image_size=256,
            offset=30,
        )

        assert len(dataset) > 0
        print(f"✓ MetadataAwareDataset initialized with {len(dataset)} scenes")

    def test_dataset_with_filter(self, dataset_path):
        """Test dataset with stabilized_label filter."""
        # All scenes without filter
        full_dataset = MetadataAwareDataset(
            folder=dataset_path,
            filters=None,
        )

        # Only uncertain scenes
        filtered_dataset = MetadataAwareDataset(
            folder=dataset_path,
            filters={"stabilized_label": ("!=", "static")},
        )

        assert len(filtered_dataset) <= len(full_dataset)
        print(f"✓ Filtering: {len(full_dataset)} → {len(filtered_dataset)} scenes")

    def test_dataset_returns_tensor(self, dataset_path):
        """Test dataset returns tensor by default."""
        dataset = MetadataAwareDataset(
            folder=dataset_path,
            image_size=256,
            offset=30,
            return_metadata=False,
        )

        sample = dataset[0]

        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (3, 2, 256, 256)

        print(f"✓ Returns tensor: {sample.shape}")

    def test_dataset_returns_metadata(self, dataset_path):
        """Test dataset returns dict with metadata."""
        dataset = MetadataAwareDataset(
            folder=dataset_path,
            image_size=256,
            offset=30,
            return_metadata=True,
        )

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "frames" in sample
        assert "scene_idx" in sample
        assert "metadata" in sample
        assert "motion_track_path" in sample
        assert "first_frame_idx" in sample
        assert "second_frame_idx" in sample

        assert sample["frames"].shape == (3, 2, 256, 256)
        assert isinstance(sample["metadata"], SceneMetadata)

        print(f"✓ Returns metadata dict:")
        print(f"  - Scene: {sample['metadata'].scene_folder}")
        print(f"  - Motion tracks: {sample['motion_track_path'] is not None}")

    def test_dataset_filter_scenes_with_hands(self, dataset_path):
        """Test filtering for scenes containing hands."""
        dataset = MetadataAwareDataset(
            folder=dataset_path,
            filters={"contains_hand_sam3": True},
        )

        assert len(dataset) > 0
        # Verify all returned scenes have hands
        for i in range(min(3, len(dataset))):
            assert dataset.scenes[i].contains_hand_sam3

        print(f"✓ {len(dataset)} scenes contain hands")

    def test_dataset_filter_large_motion(self, dataset_path):
        """Test filtering for high-motion scenes."""
        dataset = MetadataAwareDataset(
            folder=dataset_path,
            filters={"max_trans": (">", 20.0)},
        )

        for scene in dataset.scenes:
            assert scene.max_trans > 20.0

        print(f"✓ {len(dataset)} scenes with max_trans > 20.0")


class TestLAQDataModuleWithMetadata:
    """Test LAQDataModule with metadata filtering."""

    def test_datamodule_uses_metadata_by_default(self, dataset_path):
        """Test that DataModule uses MetadataAwareDataset by default."""
        dm = LAQDataModule(
            folder=dataset_path,
            batch_size=4,
            num_workers=0,
            max_samples=10,
            use_metadata=True,  # Default
        )

        dm.setup()

        # Should use metadata-aware loading
        assert dm.train_dataset is not None

        print(f"✓ DataModule with metadata: {len(dm.train_dataset)} train samples")

    def test_datamodule_with_filters(self, dataset_path):
        """Test DataModule with scene filters."""
        dm = LAQDataModule(
            folder=dataset_path,
            batch_size=4,
            num_workers=0,
            use_metadata=True,
            filters={"stabilized_label": ("!=", "static")},
        )

        dm.setup()

        print(f"✓ DataModule with filter: {len(dm.train_dataset)} train samples")

    def test_datamodule_legacy_mode(self, dataset_path):
        """Test DataModule with legacy ImageVideoDataset."""
        dm = LAQDataModule(
            folder=dataset_path,
            batch_size=4,
            num_workers=0,
            use_metadata=False,  # Legacy mode
            max_samples=10,
        )

        dm.setup()

        assert dm.train_dataset is not None

        print(f"✓ DataModule legacy mode: {len(dm.train_dataset)} train samples")

    def test_datamodule_with_metadata_returns_dict(self, dataset_path):
        """Test DataModule returns metadata dicts when configured."""
        dm = LAQDataModule(
            folder=dataset_path,
            batch_size=2,
            num_workers=0,
            use_metadata=True,
            return_metadata=True,
            max_samples=5,
            val_split=0.0,
        )

        dm.setup()
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        assert isinstance(batch, dict)
        assert batch["frames"].shape == (2, 3, 2, 256, 256)
        assert isinstance(batch["metadata"], list)
        assert len(batch["metadata"]) == 2
        assert isinstance(batch["scene_idx"], list)
        assert isinstance(batch["first_frame_idx"], torch.Tensor)

        print(f"✓ DataModule returns metadata dict batch")
        print(f"  - Frames shape: {batch['frames'].shape}")
        print(f"  - Metadata entries: {len(batch['metadata'])}")


class TestMetadataAwarePairDataset:
    """Test MetadataAwarePairDataset (pair-level indexing)."""

    def test_pair_dataset_initialization(self, dataset_path):
        """Test pair dataset initializes and pre-computes pairs."""
        from common.data import MetadataAwarePairDataset

        dataset = MetadataAwarePairDataset(
            folder=dataset_path,
            image_size=256,
            offsets=[30],
            min_frames=2,
        )

        assert len(dataset) > 0
        assert hasattr(dataset, 'pairs')
        assert hasattr(dataset, 'scenes')
        assert len(dataset.pairs) > 0

        print(f"✓ Pair dataset initialized")
        print(f"  - Total pairs: {len(dataset)}")
        print(f"  - Scenes: {len(dataset.scenes)}")

    def test_pair_dataset_multiple_offsets(self, dataset_path):
        """Test pair dataset with multiple offsets."""
        from common.data import MetadataAwarePairDataset

        dataset = MetadataAwarePairDataset(
            folder=dataset_path,
            image_size=256,
            offsets=[10, 20, 30],
            min_frames=2,
        )

        # Should have 3x as many pairs (roughly) as single offset
        single_offset = MetadataAwarePairDataset(
            folder=dataset_path,
            image_size=256,
            offsets=[30],
            min_frames=2,
        )

        assert len(dataset) > len(single_offset)
        print(f"✓ Multiple offsets create more pairs")
        print(f"  - Single offset [30]: {len(single_offset)} pairs")
        print(f"  - Three offsets [10,20,30]: {len(dataset)} pairs")

    def test_pair_dataset_load_sample(self, dataset_path):
        """Test loading a specific pair by index."""
        from common.data import MetadataAwarePairDataset

        dataset = MetadataAwarePairDataset(
            folder=dataset_path,
            image_size=256,
            offsets=[30],
            return_metadata=False,
        )

        frames = dataset[0]
        assert isinstance(frames, torch.Tensor)
        assert frames.shape == (3, 2, 256, 256)

        print(f"✓ Loaded pair by index")
        print(f"  - Shape: {frames.shape}")

    def test_pair_dataset_with_metadata(self, dataset_path):
        """Test pair dataset returns metadata dict."""
        from common.data import MetadataAwarePairDataset

        dataset = MetadataAwarePairDataset(
            folder=dataset_path,
            image_size=256,
            offsets=[30],
            return_metadata=True,
        )

        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "frames" in sample
        assert "scene_idx" in sample
        assert "first_frame_idx" in sample
        assert "second_frame_idx" in sample
        assert "offset" in sample
        assert "metadata" in sample

        print(f"✓ Pair dataset returns metadata dict")
        print(f"  - Keys: {list(sample.keys())}")

    def test_pair_index_dataclass(self):
        """Test FramePairIndex dataclass."""
        from common.data import FramePairIndex

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


class TestLAQDataModulePairLevel:
    """Test LAQDataModule with pair_level mode."""

    def test_pair_level_mode(self, dataset_path):
        """Test LAQDataModule in pair-level mode."""
        datamodule = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            offset=30,
            batch_size=2,
            num_workers=0,
            pair_level=True,
            offsets=[30],
            max_samples=10,
            val_split=0.2,
        )

        datamodule.setup()

        assert datamodule.total_available > 0
        assert len(datamodule.train_dataset) == 8  # 10 * 0.8
        assert len(datamodule.val_dataset) == 2    # 10 * 0.2

        print(f"✓ Pair-level mode works")
        print(f"  - Total pairs available: {datamodule.total_available}")
        print(f"  - Train pairs: {len(datamodule.train_dataset)}")
        print(f"  - Val pairs: {len(datamodule.val_dataset)}")

    def test_single_pair_overfitting(self, dataset_path):
        """Test LAQDataModule configured for single-pair overfitting."""
        datamodule = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            batch_size=1,
            num_workers=0,
            pair_level=True,
            offsets=[30],
            max_samples=1,
            val_split=0.0,  # No validation
        )

        datamodule.setup()

        assert len(datamodule.train_dataset) == 1
        assert len(datamodule.val_dataset) == 0

        # Verify we can load the same sample multiple times
        train_loader = datamodule.train_dataloader()
        batch1 = next(iter(train_loader))
        batch2 = next(iter(train_loader))

        assert batch1.shape == (1, 3, 2, 256, 256)
        # Same pair every time (perfect for overfitting)
        assert torch.allclose(batch1, batch2)

        print(f"✓ Single-pair overfitting setup works")
        print(f"  - Train samples: 1")
        print(f"  - Val samples: 0")
        print(f"  - Batch shape: {batch1.shape}")

    def test_pair_level_vs_scene_level(self, dataset_path):
        """Compare pair-level and scene-level dataset sizes."""
        # Scene-level (on-the-fly sampling)
        scene_dm = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            offset=30,
            batch_size=2,
            num_workers=0,
            pair_level=False,
            max_samples=10,
            val_split=0.0,
        )
        scene_dm.setup()

        # Pair-level (pre-computed pairs)
        pair_dm = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            offset=30,
            batch_size=2,
            num_workers=0,
            pair_level=True,
            offsets=[30],
            max_samples=10,
            val_split=0.0,
        )
        pair_dm.setup()

        # Pair-level should have many more samples (all frame pairs vs scenes)
        assert len(pair_dm.train_dataset) == 10  # 10 pairs
        assert len(scene_dm.train_dataset) == 10  # 10 scenes

        # But total available should be much larger for pair-level
        assert pair_dm.total_available > scene_dm.total_available

        print(f"✓ Pair-level vs scene-level comparison")
        print(f"  - Scene-level total: {scene_dm.total_available} scenes")
        print(f"  - Pair-level total: {pair_dm.total_available} pairs")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
