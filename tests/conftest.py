"""
Pytest configuration and fixtures for LAPA tests.

This file provides shared fixtures and configuration for all tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def device():
    """Get the best available device (CUDA or CPU).
    
    Note: MPS is excluded due to known limitations with complex operations
    (e.g., 6D permute in decoder). Tests run on CPU for local development
    and CUDA for CI/production environments.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture
def device_cpu():
    """Force CPU device for tests that crash on MPS.
    
    Note: Some operations (e.g., 6D permute in decoder) have known MPS backend
    limitations that cause crashes. These tests use CPU as a workaround.
    For production training on CUDA, these operations work correctly.
    """
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def image_size():
    """LAPA image size (256×256)."""
    return 256


@pytest.fixture
def patch_size():
    """LAPA patch size (32×32)."""
    return 32


@pytest.fixture
def model_dim():
    """LAPA model dimension."""
    return 1024


@pytest.fixture
def num_patches(image_size, patch_size):
    """Number of patches per frame."""
    return (image_size // patch_size) ** 2


@pytest.fixture
def sample_frame_pair(batch_size, image_size):
    """Generate sample frame pairs for testing.
    
    Returns:
        Tensor of shape [B, 3, 2, 256, 256]
    """
    return torch.randn(batch_size, 3, 2, image_size, image_size)


@pytest.fixture
def sample_frame_tokens(batch_size, num_patches, model_dim):
    """Generate sample frame tokens (encoder output).
    
    Returns:
        Tuple of (first_tokens, last_tokens), each [B, 64, 1024]
    """
    first_tokens = torch.randn(batch_size, num_patches, model_dim)
    last_tokens = torch.randn(batch_size, num_patches, model_dim)
    return first_tokens, last_tokens


@pytest.fixture
def sample_action_tokens(batch_size, model_dim):
    """Generate sample action tokens (NSVQ output).
    
    Returns:
        Tensor of shape [B, 1, 2, 2, 1024]
    """
    return torch.randn(batch_size, 1, 2, 2, model_dim)


@pytest.fixture
def sample_indices(batch_size):
    """Generate sample latent action indices.
    
    Returns:
        Tensor of shape [B, 4] with values in [0, 7]
    """
    return torch.randint(0, 8, (batch_size, 4))


@pytest.fixture
def lapa_config():
    """Configuration for LAPA model (reduced size for testing)."""
    return {
        'image_size': 256,
        'patch_size': 32,
        'channels': 3,
        'dim': 1024,
        'quant_dim': 32,
        'spatial_depth': 2,  # Reduced for testing
        'temporal_depth': 2,  # Reduced for testing
        'decoder_depth': 2,  # Reduced for testing
        'heads': 16,
        'dim_head': 64,
        'mlp_ratio': 4,
        'dropout': 0.0,
        'codebook_size': 8,
        'code_seq_len': 4,
    }


@pytest.fixture
def temp_video_dir():
    """Create a temporary directory for video test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_video_file(temp_video_dir):
    """Create a mock video file for testing.
    
    Note: Requires opencv-python to create actual video files.
    For now, returns the path where a video would be.
    """
    video_path = temp_video_dir / "test_video.mp4"
    return video_path


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def model_checkpoint_path(tmp_path):
    """Path for saving/loading model checkpoints in tests."""
    return tmp_path / "test_checkpoint.ckpt"


# Markers for skipping tests based on availability
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA GPU"
    )
    config.addinivalue_line(
        "markers", "mps: mark test as requiring Apple MPS"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and availability."""
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_mps = pytest.mark.skip(reason="MPS not available")
    
    for item in items:
        if "cuda" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_cuda)
        if "mps" in item.keywords and not torch.backends.mps.is_available():
            item.add_marker(skip_mps)

