"""
BridgeV2 dataset adapter.

Handles the BridgeV2 dataset structure where EACH TRAJECTORY is a scene.

Dataset structure (multiple variants):
    
    bridgev2/raw/
    ├── bridge_data_v1/
    │   └── berkeley/                        ← {institution}
    │       └── toykitchen1/                 ← {environment}
    │           └── close_large4fbox_flaps/  ← {task}
    │               └── 2021-07-30_14-36-57/ ← {dated_folder}
    │                   ├── collection_metadata.json
    │                   ├── config.json
    │                   └── raw/traj_group0/
    │                       ├── traj0/       ← SCENE (trajectory)
    │                       │   ├── images0/
    │                       │   │   ├── im_0.jpg
    │                       │   │   └── ...
    │                       │   └── lang.txt
    │                       ├── traj1/
    │                       └── ...
    │
    ├── bridge_data_v2/
    │   └── datacol2_toykitchen2/            ← {robot}_{environment}
    │       └── many_skills/                 ← {task}
    │           └── 00/                      ← {collection_id}
    │               └── 2023-03-08_12-45-22/ ← {dated_folder}
    │                   └── raw/traj_group0/
    │                       └── traj*/       ← SCENE
    │
    ├── rss/
    │   └── toykitchen2/                     ← {environment}
    │       └── pnp_push_sweep/              ← {task}
    │           └── 00/                      ← {collection_id}
    │               └── 2022-12-07_14-55-30/ ← {dated_folder}
    │                   └── raw/traj_group0/
    │                       └── traj*/       ← SCENE
    │
    └── icra/, flap/                         ← similar structure

Key insight: Each traj* folder is ONE scene with its own images.
The dated folder contains metadata (collection_metadata.json) shared by all its trajectories.
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

    Each trajectory folder (traj0, traj1, etc.) is treated as one scene.
    Extracts metadata from folder hierarchy and JSON config files.
    
    Expected ~50,000+ trajectories across all subsets.
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
            root: Root directory of Bridge dataset (e.g., bridgev2/raw/)
            filters: Optional per-source filters to apply
            **kwargs: Unused

        Returns:
            List of SceneMetadata with Bridge-specific metadata in extras
        """
        root = Path(root)
        all_scenes = []
        scene_idx = 0
        
        # Cache for dated folder metadata (avoid re-reading JSON for each trajectory)
        metadata_cache: Dict[str, Dict[str, Any]] = {}

        # Find all trajectory directories (traj0, traj1, etc.)
        print(f"  Scanning for trajectories in {root}...")
        traj_dirs = self._find_all_trajectories(root)
        print(f"  Found {len(traj_dirs)} trajectory folders")

        for traj_dir in traj_dirs:
            # Get image files for this trajectory
            image_files = self._get_trajectory_images(traj_dir)

            if len(image_files) < 2:  # Skip trajectories with too few frames
                continue

            # Relative path from root to the trajectory folder
            rel_path = traj_dir.relative_to(root)
            
            # Get the dated folder (parent of raw/traj_group*/traj*)
            # traj_dir is like: .../dated_folder/raw/traj_group0/traj5
            dated_folder = traj_dir.parent.parent.parent
            
            # Load or get cached metadata from dated folder
            dated_folder_str = str(dated_folder)
            if dated_folder_str not in metadata_cache:
                metadata_cache[dated_folder_str] = self._parse_dated_folder_metadata(dated_folder)
            metadata = metadata_cache[dated_folder_str]
            
            # Parse language annotation from trajectory
            lang_annotation = self._get_language_annotation(traj_dir)

            # Parse hierarchy for additional context
            hierarchy = self._parse_path_hierarchy(rel_path, root)

            # Create SceneMetadata
            scene = SceneMetadata(
                scene_idx=scene_idx,
                scene_folder=str(rel_path),
                start_frame=0,
                end_frame=len(image_files),
                extras={
                    "dataset_type": "bridge",
                    "dataset_name": hierarchy.get("dataset_subset", "bridge_v2"),
                    # From collection_metadata.json (most reliable)
                    "robot": metadata.get("robot", "widowx"),
                    "environment": metadata.get("environment", hierarchy.get("environment", "unknown")),
                    "gripper": metadata.get("gripper", "default"),
                    "camera_type": metadata.get("camera_type", "unknown"),
                    "policy_desc": metadata.get("policy_desc", "unknown"),
                    # From path hierarchy
                    "task": hierarchy.get("task", "unknown"),
                    "institution": hierarchy.get("institution", "unknown"),
                    "dated_folder": hierarchy.get("dated_folder", "unknown"),
                    "traj_id": traj_dir.name,  # e.g., "traj0", "traj5"
                    # Language annotation
                    "language": lang_annotation,
                    # Stats
                    "num_frames": len(image_files),
                },
            )
            all_scenes.append(scene)
            scene_idx += 1

        # Apply per-source filters if specified
        if filters:
            filter_obj = SceneFilter(filters)
            all_scenes = filter_obj.filter_scenes(all_scenes)

        return all_scenes

    def _find_all_trajectories(self, root: Path) -> List[Path]:
        """
        Find all trajectory directories (traj0, traj1, etc.) containing images.
        
        Each traj* folder with an images0/ subfolder is a valid trajectory.
        """
        traj_dirs = []
        
        # Pattern: */raw/traj_group*/traj* where traj* contains images0/
        for images_dir in root.rglob("raw/traj_group*/traj*/images0"):
            if images_dir.is_dir():
                # traj_dir is parent of images0
                traj_dir = images_dir.parent
                traj_dirs.append(traj_dir)
        
        return sorted(set(traj_dirs))

    def _get_trajectory_images(self, traj_dir: Path) -> List[Path]:
        """Get sorted image files from a single trajectory folder."""
        images = []
        
        # Primary camera is images0 (over-the-shoulder view)
        images0_dir = traj_dir / "images0"
        if images0_dir.exists():
            images = sorted(images0_dir.glob("im_*.jpg"))
        
        return images

    def _parse_dated_folder_metadata(self, dated_folder: Path) -> Dict[str, Any]:
        """
        Parse metadata from JSON files in the dated folder.
        
        The dated folder (e.g., 2023-03-08_12-45-22/) contains:
        - collection_metadata.json: environment, robot, gripper, etc.
        - config.json: agent config (image size, etc.)
        """
        metadata = {}

        # Parse collection_metadata.json
        collection_meta_path = dated_folder / "collection_metadata.json"
        if collection_meta_path.exists():   
            try:
                with open(collection_meta_path) as f:
                    cm = json.load(f)
                    metadata["camera_type"] = cm.get("camera_type")
                    metadata["policy_desc"] = cm.get("policy_desc")
                    metadata["robot"] = cm.get("robot")
                    metadata["gripper"] = cm.get("gripper")
                    metadata["environment"] = cm.get("environment")
                    metadata["background"] = cm.get("background")
                    metadata["object_classes"] = cm.get("object_classes")
            except (json.JSONDecodeError, IOError):
                pass

        # Parse config.json
        config_path = dated_folder / "config.json"
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
    
    def _get_language_annotation(self, traj_dir: Path) -> str:
        """Get language annotation from lang.txt if it exists."""
        lang_path = traj_dir / "lang.txt"
        if lang_path.exists():
            try:
                with open(lang_path) as f:
                    lines = f.readlines()
                    # First line is the instruction, second line is confidence
                    if lines:
                        return lines[0].strip()
            except IOError:
                pass
        return ""

    def _parse_path_hierarchy(self, rel_path: Path, root: Path) -> Dict[str, str]:
        """
        Parse folder hierarchy to extract dataset structure info.
        
        Different subsets have different structures:
        - bridge_data_v1/berkeley/toykitchen1/task/dated/raw/...
        - bridge_data_v2/datacol2_toykitchen2/task/00/dated/raw/...
        - rss/toykitchen2/task/00/dated/raw/...
        - icra/subset/env/task/dated/raw/...
        """
        parts = rel_path.parts
        result = {
            "dataset_subset": parts[0] if len(parts) > 0 else "unknown",
            "dated_folder": "unknown",
            "task": "unknown",
            "environment": "unknown",
            "institution": "unknown",
        }
        
        # Find the dated folder (format: YYYY-MM-DD_HH-MM-SS)
        for i, part in enumerate(parts):
            if len(part) == 19 and part[4] == '-' and part[10] == '_':
                result["dated_folder"] = part
                # Task is typically 2 levels before dated folder
                if i >= 2:
                    result["task"] = parts[i-2] if i >= 2 else "unknown"
                break
        
        # Parse based on dataset subset
        if parts[0] == "bridge_data_v1":
            # bridge_data_v1/berkeley/toykitchen1/task/dated/...
            result["institution"] = parts[1] if len(parts) > 1 else "unknown"
            result["environment"] = parts[2] if len(parts) > 2 else "unknown"
            result["task"] = parts[3] if len(parts) > 3 else "unknown"
            
        elif parts[0] == "bridge_data_v2":
            # bridge_data_v2/datacol2_toykitchen2/task/00/dated/...
            if len(parts) > 1:
                # Parse robot_environment format (e.g., datacol2_toykitchen2)
                robot_env = parts[1]
                if "_" in robot_env:
                    # Find last underscore that separates robot from environment
                    # datacol2_toykitchen2 -> robot=datacol2, env=toykitchen2
                    # minsky_folding_table_white_tray -> robot=minsky, env=folding_table_white_tray
                    first_underscore = robot_env.find("_")
                    result["institution"] = robot_env[:first_underscore]
                    result["environment"] = robot_env[first_underscore+1:]
            result["task"] = parts[2] if len(parts) > 2 else "unknown"
            
        elif parts[0] in ["rss", "icra", "flap"]:
            # rss/toykitchen2/task/00/dated/...
            result["institution"] = parts[0]
            result["environment"] = parts[1] if len(parts) > 1 else "unknown"
            result["task"] = parts[2] if len(parts) > 2 else "unknown"
        
        return result

    def get_frame_files(self, scene: SceneMetadata, root: Path) -> List[Path]:
        """Get frame files from trajectory folder."""
        traj_dir = root / scene.scene_folder
        return self._get_trajectory_images(traj_dir)
