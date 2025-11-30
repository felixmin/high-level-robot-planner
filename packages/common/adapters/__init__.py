"""
Dataset adapters for loading different dataset formats.

Each adapter implements the DatasetAdapter interface to convert
dataset-specific structures into unified SceneMetadata lists.
"""

from .base import DatasetAdapter
from .youtube import YoutubeAdapter
from .bridge import BridgeAdapter

__all__ = ["DatasetAdapter", "YoutubeAdapter", "BridgeAdapter"]
