"""
Dataset classes for LAPA training

Updated for LAPA architecture:
- Returns stacked frame pairs [B, 3, 2, 256, 256]
- Image size 256×256 (not 224×224)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json


class LAQVideoDataset(Dataset):
    """
    Dataset for LAPA training using video files.
    
    Loads video frames and creates frame pairs for training.
    Returns stacked frames in format [C, T, H, W] where T=2.
    """
    
    def __init__(
        self,
        video_dir: str,
        max_frames_per_video: int = 5,
        frame_spacing: int = 1,
        image_size: Tuple[int, int] = (256, 256),  # Updated to 256×256 for LAPA
        normalize: bool = True,
        cache_frames: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            video_dir: Directory containing video files
            max_frames_per_video: Maximum frames to load per video
            frame_spacing: Number of frames between frame_t and frame_t+1
            image_size: Target image size (height, width) - 256×256 for LAPA
            normalize: Whether to normalize to [-1, 1]
            cache_frames: Whether to cache loaded frames in memory
        """
        self.video_dir = Path(video_dir)
        self.max_frames_per_video = max_frames_per_video
        self.frame_spacing = frame_spacing
        self.image_size = image_size
        self.normalize = normalize
        self.cache_frames = cache_frames
        self._frame_cache: dict[Path, List[np.ndarray]] = {}
        
        # Find all video files (support common extensions)
        supported_patterns = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm")
        self.video_files = []
        for pattern in supported_patterns:
            self.video_files.extend(self.video_dir.glob(pattern))
        self.video_files = sorted(self.video_files)
        if not self.video_files:
            raise ValueError(f"No video files found in {video_dir}")
    
    def load_video_frames(self, video_path: Path) -> List[np.ndarray]:
        """Load frames from a video file, with optional in-memory caching."""
        if self.cache_frames and video_path in self._frame_cache:
            return self._frame_cache[video_path]
        
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Load enough frames to support the requested spacing
        frames_needed = max(self.max_frames_per_video, 1 + self.frame_spacing)
        
        frame_count = 0
        while cap.isOpened() and frame_count < frames_needed:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.image_size)
            frames.append(frame_resized)
            frame_count += 1
        
        cap.release()
        
        if self.cache_frames:
            self._frame_cache[video_path] = frames
        
        return frames
    
    def preprocess_frames(self, frame_t: np.ndarray, frame_t1: np.ndarray) -> torch.Tensor:
        """
        Preprocess two frames for LAPA training.
        
        Returns stacked frames in format [C, T, H, W] where T=2.
        
        Args:
            frame_t: First frame [H, W, C]
            frame_t1: Second frame [H, W, C]
            
        Returns:
            Stacked tensor [C, 2, H, W] = [3, 2, 256, 256]
        """
        # Convert to tensor and normalize to [0, 1]
        frame_t_tensor = torch.from_numpy(frame_t).float() / 255.0
        frame_t1_tensor = torch.from_numpy(frame_t1).float() / 255.0
        
        if self.normalize:
            # Normalize to [-1, 1] range
            frame_t_tensor = frame_t_tensor * 2.0 - 1.0
            frame_t1_tensor = frame_t1_tensor * 2.0 - 1.0
        
        # Convert from [H, W, C] to [C, H, W]
        frame_t_tensor = frame_t_tensor.permute(2, 0, 1)  # [3, H, W]
        frame_t1_tensor = frame_t1_tensor.permute(2, 0, 1)  # [3, H, W]
        
        # Stack along temporal dimension: [C, 2, H, W]
        stacked = torch.stack([frame_t_tensor, frame_t1_tensor], dim=1)  # [3, 2, H, W]
        
        return stacked
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.video_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample."""
        video_path = self.video_files[idx]
        frames = self.load_video_frames(video_path)
        
        # Ensure we have enough frames for the spacing
        min_frames_needed = 1 + self.frame_spacing
        if len(frames) < min_frames_needed:
            # Use the last available frame as frame_t1
            frame_t = frames[0]
            frame_t1 = frames[-1] if len(frames) > 1 else frames[0]
        else:
            # Take frames with specified spacing
            frame_t = frames[0]
            frame_t1 = frames[self.frame_spacing]
        
        # Preprocess to [C, T, H, W] format
        sample = self.preprocess_frames(frame_t, frame_t1)
        
        return sample


def create_dataloader(
    video_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    drop_last: bool = True,
    frame_spacing: int = 1,
    image_size: Tuple[int, int] = (256, 256),  # Updated default for LAPA
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for LAPA training.
    
    Args:
        video_dir: Directory containing video files
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        drop_last: Whether to drop the last incomplete batch
        frame_spacing: Number of frames between frame_t and frame_t+1
        image_size: Target image size (H, W), default 256×256 for LAPA
        **dataset_kwargs: Additional arguments for LAQVideoDataset
        
    Returns:
        DataLoader instance
    """
    dataset = LAQVideoDataset(
        video_dir,
        frame_spacing=frame_spacing,
        image_size=image_size,
        **dataset_kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    
    return dataloader


# Test function
def test_dataset():
    """Test dataset implementation."""
    print("Testing LAPA Dataset...")
    
    # This requires actual video files, so we'll create a mock test
    # In practice, you would point this to a directory with videos
    
    # Expected output shape after preprocessing
    frame_t = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    frame_t1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Create a temporary dataset instance to test preprocessing
    class MockDataset(LAQVideoDataset):
        def __init__(self):
            self.normalize = True
            self.image_size = (256, 256)
    
    mock_dataset = MockDataset()
    sample = mock_dataset.preprocess_frames(frame_t, frame_t1)
    
    print(f"Preprocessed shape: {sample.shape}")
    print(f"Value range: [{sample.min():.3f}, {sample.max():.3f}]")
    
    # Verify expected shape
    expected_shape = torch.Size([3, 2, 256, 256])
    assert sample.shape == expected_shape, f"Expected {expected_shape}, got {sample.shape}"
    
    # Verify normalization range
    assert sample.min() >= -1.1 and sample.max() <= 1.1, \
        f"Values should be in [-1, 1], got [{sample.min():.3f}, {sample.max():.3f}]"
    
    print("✅ Dataset preprocessing test passed!")


if __name__ == "__main__":
    test_dataset()
