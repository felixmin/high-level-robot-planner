"""
Base class for dataset adapters.

Dataset adapters convert dataset-specific structures into unified
SceneMetadata format for LAQ training.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data import SceneMetadata


class DatasetAdapter(ABC):
    """
    Base class for dataset adapters.

    Each adapter knows how to:
    1. Scan a dataset directory for scenes/trajectories
    2. Extract metadata from dataset-specific formats
    3. Return unified SceneMetadata objects

    Subclasses must implement:
    - collect_scenes(): Scan directory and return SceneMetadata list
    - get_frame_files(): Get frame file paths for a scene
    """

    @abstractmethod
    def collect_scenes(
        self,
        root: Path,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[SceneMetadata]:
        """
        Scan dataset directory and return list of SceneMetadata.

        Args:
            root: Root directory of dataset
            filters: Optional per-source filters to apply via SceneFilter
            **kwargs: Adapter-specific options

        Returns:
            List of SceneMetadata with all scenes in dataset (after filtering)
        """
        pass

    @abstractmethod
    def get_frame_files(self, scene: SceneMetadata, root: Path) -> List[Path]:
        """
        Get sorted list of frame file paths for a scene.

        Args:
            scene: Scene metadata
            root: Dataset root directory

        Returns:
            Sorted list of image file paths
        """
        pass
