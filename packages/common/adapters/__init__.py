"""
Dataset adapters for loading different dataset formats.

Each adapter implements the DatasetAdapter interface to convert
dataset-specific structures into unified SceneMetadata lists.

For streaming datasets (OXE), use OXEFramePairDataset directly
as it bypasses the file-based adapter pattern.
"""

from .base import DatasetAdapter
from .youtube import YoutubeAdapter
from .bridge import BridgeAdapter
from .oxe import OXEFramePairDataset, OXE_DATASETS, get_oxe_dataset_info

__all__ = [
    "DatasetAdapter",
    "YoutubeAdapter",
    "BridgeAdapter",
    "OXEFramePairDataset",
    "OXE_DATASETS",
    "get_oxe_dataset_info",
]
