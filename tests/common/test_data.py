"""
Test data loading functionality.

Tests ImageVideoDataset and LAQDataModule with real data.
"""

import pytest
import torch
from pathlib import Path
from common.data import ImageVideoDataset, LAQDataModule


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
        """Test DataModule with 100 samples."""
        dm = LAQDataModule(
            folder=dataset_path,
            image_size=256,
            offset=30,
            batch_size=8,
            num_workers=0,
            max_samples=100,
        )

        dm.setup()

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
