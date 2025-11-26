"""
Data loading utilities for LAQ training.

Includes:
- ImageVideoDataset: Loads frame pairs from folders
- LAQDataModule: Lightning DataModule with subset support
"""

import os
import random
import re
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
from PIL import Image
import lightning.pytorch as pl


class ImageVideoDataset(Dataset):
    """
    Dataset for loading frame pairs from video scene folders.

    Adapted from LAPA's ImageVideoDataset for our project structure.

    Directory structure:
        root_folder/
        ├── scene_000/
        │   ├── frame_0001.jpg
        │   ├── frame_0002.jpg
        │   └── ...
        ├── scene_001/
        │   └── ...
        └── ...

    Returns:
        Tensor [C, 2, H, W] - concatenated frame_t and frame_t+offset
    """

    def __init__(
        self,
        folder: str,
        image_size: int = 256,
        offset: int = 30,
    ):
        super().__init__()

        self.folder = Path(folder)
        self.folder_list = [
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        ]
        self.folder_list.sort()  # Deterministic ordering

        self.image_size = image_size
        self.offset = offset

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        """Number of scene folders."""
        return len(self.folder_list)

    def __getitem__(self, index):
        """Get a frame pair from a scene folder."""
        try:
            offset = self.offset

            folder = self.folder_list[index]
            folder_path = os.path.join(self.folder, folder)
            img_list = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            # Robust sort by trailing integer (supports names like frame_00010.jpg)
            def frame_index(name: str) -> int:
                stem = os.path.splitext(name)[0]
                # Prefer explicit pattern 'frame_XXXXX'
                m = re.search(r"frame_(\d+)$", stem)
                if m:
                    return int(m.group(1))
                # Fallback: last integer group in the stem
                m2 = re.findall(r"(\d+)", stem)
                return int(m2[-1]) if m2 else -1

            img_list = sorted(img_list, key=frame_index)

            # Pick random frame pair
            first_frame_idx = random.randint(0, len(img_list) - 1)
            first_frame_idx = min(first_frame_idx, len(img_list) - 1)
            second_frame_idx = min(first_frame_idx + offset, len(img_list) - 1)

            first_path = os.path.join(folder_path, img_list[first_frame_idx])
            second_path = os.path.join(folder_path, img_list[second_frame_idx])

            img = Image.open(first_path)
            next_img = Image.open(second_path)

            transform_img = self.transform(img).unsqueeze(1)  # [C, 1, H, W]
            next_transform_img = self.transform(next_img).unsqueeze(1)  # [C, 1, H, W]

            cat_img = torch.cat([transform_img, next_transform_img], dim=1)  # [C, 2, H, W]
            return cat_img

        except Exception as e:
            print(f"Error loading index {index}: {e}")
            # Fallback to another random sample
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


class LAQDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for LAQ training.

    Features:
    - Subset support for incremental testing (1, 10, 100, ... samples)
    - Deterministic splits for reproducibility
    - Configurable via Hydra
    """

    def __init__(
        self,
        folder: str,
        image_size: int = 256,
        offset: int = 30,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        max_samples: Optional[int] = None,
        val_split: float = 0.1,
    ):
        super().__init__()

        self.folder = folder
        self.image_size = image_size
        self.offset = offset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.max_samples = max_samples
        self.val_split = val_split

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup train/val datasets."""
        # Create full dataset
        full_dataset = ImageVideoDataset(
            folder=self.folder,
            image_size=self.image_size,
            offset=self.offset,
        )

        # Apply subset if specified
        if self.max_samples is not None:
            indices = list(range(min(self.max_samples, len(full_dataset))))
            full_dataset = Subset(full_dataset, indices)

        # Split into train/val
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        # Deterministic split
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_size))

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )
