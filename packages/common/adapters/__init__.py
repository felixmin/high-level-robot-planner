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
from .oxe_shared import OXE_DATASETS, get_oxe_dataset_info
from .oxe import OXEFramePairDataset
from .oxe_v2 import OXEFramePairDatasetV2
from .openx_local import OpenXLocalIndexedPairIterable, OpenXLocalFullPairDataset
from .openx_local_indexed_full import (
    OpenXLocalIndexedPairMapDataset,
    OpenXLocalIndexedEpisodePairSampler,
    prepare_openx_local_episode_index,
)

__all__ = [
    "DatasetAdapter",
    "YoutubeAdapter",
    "BridgeAdapter",
    "OXEFramePairDataset",
    "OXEFramePairDatasetV2",
    "OpenXLocalIndexedPairIterable",
    "OpenXLocalFullPairDataset",
    "OpenXLocalIndexedPairMapDataset",
    "OpenXLocalIndexedEpisodePairSampler",
    "prepare_openx_local_episode_index",
    "OXE_DATASETS",
    "get_oxe_dataset_info",
]
