"""
BridgeV2 dataset adapter.

Handles the BridgeV2 dataset structure where each trajectory is a scene.

Structure:
    bridgev2/raw/bridge_data_v2/
      └── datacol1_toykitchen1/           ← {robot}_{environment}
          └── many_skills/                ← {task_category}
              └── 12/                     ← {task_id}
                  └── 2023-04-04_11-47-48/  ← {timestamp} = trajectory
                      ├── collection_metadata.json
                      ├── config.json
                      └── raw/traj_group0/traj4/images0/
                          ├── im_0.jpg
                          └── ...
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import DatasetAdapter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data import SceneMetadata, SceneFilter


class BridgeAdapter(DatasetAdapter):
    """
    Adapter for BridgeV2 dataset.

    Each trajectory folder is treated as a scene.
    Extracts metadata from folder hierarchy and JSON config files.
    """

    def collect_scenes(
        self,
        root: Path,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[SceneMetadata]:
        """
        Scan Bridge directory structure and create SceneMetadata for each trajectory.

        Args:
            root: Root directory of Bridge dataset
            filters: Optional per-source filters to apply
            **kwargs: Unused

        Returns:
            List of SceneMetadata with Bridge-specific metadata in extras
        """
        root = Path(root)
        all_scenes = []
        scene_idx = 0

        # Find all trajectory directories
        for traj_dir in self._find_trajectory_dirs(root):
            # Get image files
            image_files = self._get_trajectory_images(traj_dir)

            if len(image_files) < 2:  # Skip empty trajectories
                continue

            # Relative path from root
            rel_path = traj_dir.relative_to(root)

            # Parse metadata from JSON files
            metadata = self._parse_metadata(traj_dir)

            # Parse folder hierarchy for additional metadata
            path_parts = rel_path.parts
            # Format: {robot}_{environment}/{task_category}/{...}/{timestamp}
            robot_env = path_parts[0] if len(path_parts) > 0 else "unknown"
            task_category = path_parts[1] if len(path_parts) > 1 else "unknown"

            # Split robot_env into robot and environment
            robot_env_split = robot_env.split("_", 1)
            robot = robot_env_split[0] if len(robot_env_split) > 0 else "unknown"
            environment = robot_env_split[1] if len(robot_env_split) > 1 else "unknown"

            # Get timestamp (last folder)
            timestamp = path_parts[-1] if len(path_parts) > 0 else "unknown"

            # Create SceneMetadata
            scene = SceneMetadata(
                scene_idx=scene_idx,
                scene_folder=str(rel_path),
                start_frame=0,
                end_frame=len(image_files),
                extras={
                    "dataset_type": "bridge",
                    "dataset_name": "bridge_v2",
                    "robot": metadata.get("robot", robot),
                    "environment": metadata.get("environment", environment),
                    "task_category": task_category,
                    "timestamp": timestamp,
                    "trajectory_path": str(rel_path),
                    # From collection_metadata.json
                    "camera_type": metadata.get("camera_type", "unknown"),
                    "policy_desc": metadata.get("policy_desc", "unknown"),
                    "gripper": metadata.get("gripper", "unknown"),
                    # From config.json
                    "sequence_length": metadata.get("T", len(image_files)),
                    "image_width": metadata.get("image_width", 640),
                    "image_height": metadata.get("image_height", 480),
                },
            )
            all_scenes.append(scene)
            scene_idx += 1

        # Apply per-source filters if specified
        if filters:
            filter_obj = SceneFilter(filters)
            all_scenes = filter_obj.filter_scenes(all_scenes)

        return all_scenes

    def _find_trajectory_dirs(self, root: Path) -> List[Path]:
        """Find all directories containing trajectory images."""
        traj_dirs = set()

        # Pattern: any folder with raw/traj_group*/traj*/images*/
        for path in root.rglob("raw/traj_group*/traj*/images*"):
            if path.is_dir():
                # Trajectory folder is 4 levels up from images folder
                traj_dir = path.parent.parent.parent.parent
                traj_dirs.add(traj_dir)

        return sorted(traj_dirs)

    def _get_trajectory_images(self, traj_dir: Path) -> List[Path]:
        """Get all images from a trajectory folder."""
        images = []
        raw_dir = traj_dir / "raw"
        if raw_dir.exists():
            for images_dir in raw_dir.rglob("images*"):
                if images_dir.is_dir():
                    images.extend(images_dir.glob("im_*.jpg"))
        return sorted(images)

    def _parse_metadata(self, traj_dir: Path) -> Dict[str, Any]:
        """Parse metadata from JSON files in trajectory folder."""
        metadata = {}

        # Parse collection_metadata.json
        collection_meta_path = traj_dir / "collection_metadata.json"
        if collection_meta_path.exists():
            try:
                with open(collection_meta_path) as f:
                    collection_meta = json.load(f)
                    metadata["camera_type"] = collection_meta.get("camera_type")
                    metadata["policy_desc"] = collection_meta.get("policy_desc")
                    metadata["robot"] = collection_meta.get("robot")
                    metadata["gripper"] = collection_meta.get("gripper")
                    metadata["environment"] = collection_meta.get("environment")
            except (json.JSONDecodeError, IOError):
                pass

        # Parse config.json
        config_path = traj_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    if "agent" in config:
                        metadata["T"] = config["agent"].get("T")
                        metadata["image_height"] = config["agent"].get("image_height")
                        metadata["image_width"] = config["agent"].get("image_width")
            except (json.JSONDecodeError, IOError):
                pass

        return metadata

    def get_frame_files(self, scene: SceneMetadata, root: Path) -> List[Path]:
        """Get frame files from trajectory."""
        traj_dir = root / scene.scene_folder
        return self._get_trajectory_images(traj_dir)
