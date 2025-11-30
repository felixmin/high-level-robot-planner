"""
YouTube dataset adapter.

Handles the YouTube video dataset structure with scenes.csv files.

Structure:
    youtube_new/
      ├── video1_stabilized/
      │   ├── scene_000_part_000/
      │   ├── scene_000_part_001/
      │   └── scenes.csv
      └── video2_stabilized/
          └── scenes.csv
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import DatasetAdapter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data import SceneMetadata, SceneFilter, load_scenes_csv


class YoutubeAdapter(DatasetAdapter):
    """
    Adapter for YouTube video dataset.

    Scans video folders for scenes.csv files and loads scene metadata.
    Supports both single-video folders and multi-video directories.
    """

    def collect_scenes(
        self,
        root: Path,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[SceneMetadata]:
        """
        Scan all video folders and load scenes from scenes.csv files.

        Args:
            root: Root directory (can be single video folder or parent of multiple)
            filters: Optional per-source filters to apply
            **kwargs: Unused

        Returns:
            List of SceneMetadata with video_id and dataset_type in extras
        """
        root = Path(root)
        all_scenes = []

        # Check if root itself has scenes.csv (single video folder)
        if (root / "scenes.csv").exists():
            video_dirs = [root]
        else:
            # Multi-video: scan for subdirectories with scenes.csv
            video_dirs = sorted([
                d for d in root.iterdir()
                if d.is_dir() and (d / "scenes.csv").exists()
            ])

        for video_dir in video_dirs:
            scenes_csv = video_dir / "scenes.csv"
            video_scenes = load_scenes_csv(scenes_csv)

            # Determine video_id and whether to prefix paths
            if video_dir == root:
                # Single video folder - no prefix needed
                video_id = root.name
                prefix = ""
            else:
                # Multi-video - prefix with video folder name
                video_id = video_dir.name
                prefix = f"{video_dir.name}/"

            for scene in video_scenes:
                # Prefix scene_folder for multi-video mode
                if prefix:
                    scene.scene_folder = f"{prefix}{scene.scene_folder}"

                # Add dataset metadata to extras
                scene.extras["dataset_type"] = "youtube"
                scene.extras["dataset_name"] = "youtube_new"
                scene.extras["video_id"] = video_id

                all_scenes.append(scene)

        # Apply per-source filters if specified
        if filters:
            filter_obj = SceneFilter(filters)
            all_scenes = filter_obj.filter_scenes(all_scenes)

        return all_scenes

    def get_frame_files(self, scene: SceneMetadata, root: Path) -> List[Path]:
        """Get sorted frame files from scene folder."""
        scene_path = root / scene.scene_folder
        frame_files = sorted(scene_path.glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(scene_path.glob("*.png"))
        return frame_files
