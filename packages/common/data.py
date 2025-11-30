"""
Data loading utilities for LAQ training.

Includes:
- SceneMetadata: Dataclass for scene metadata from CSV
- SceneFilter: Flexible filtering for scenes by any column
- ImageVideoDataset: Loads frame pairs from folders (legacy)
- MetadataAwareDataset: Enhanced dataset using scenes.csv (scene-level)
- FramePairIndex: Dataclass for explicit frame pair indexing
- MetadataAwarePairDataset: Pair-level dataset with pre-computed pairs
- LAQDataModule: Lightning DataModule with metadata filtering and pair-level support
"""

import csv
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
from PIL import Image
import lightning.pytorch as pl


@dataclass
class SceneMetadata:
    """
    Metadata for a single scene from scenes.csv.

    Designed to be extensible - stores all CSV columns as attributes.
    Core fields are typed; additional fields accessible via `extras` dict.
    """

    # Core identification
    scene_idx: int
    scene_folder: str
    start_frame: int
    end_frame: int

    # Motion labels (for filtering)
    label: str = "uncertain"
    stabilized_label: str = "uncertain"

    # Motion metrics
    max_angle: float = 0.0
    max_trans: float = 0.0
    stabilized_max_angle: float = 0.0
    stabilized_max_trans: float = 0.0

    # Hand detection
    contains_hand_sam3: bool = False
    hand_mask_folder_sam3: Optional[str] = None

    # Lego brick detection
    contains_lego_brick_sam3: bool = False
    lego_brick_mask_folder_sam3: Optional[str] = None

    # Motion tracks (for future decoder experiments)
    contains_sam3_hand_motion_cotracker: bool = False
    sam3_hand_motion_cotracker_folder: Optional[str] = None

    # Extra columns (for future extensibility)
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        """Number of frames in this scene."""
        return self.end_frame - self.start_frame

    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> "SceneMetadata":
        """Create SceneMetadata from a CSV row dictionary."""

        def parse_bool(val: str) -> bool:
            return val.lower() in ("true", "1", "yes")

        def parse_optional_str(val: str) -> Optional[str]:
            return val if val.strip() else None

        # Known fields
        known_fields = {
            "scene_idx",
            "scene_folder",
            "start_frame",
            "end_frame",
            "label",
            "stabilized_label",
            "max_angle",
            "max_trans",
            "stabilized_max_angle",
            "stabilized_max_trans",
            "contains_hand_sam3",
            "hand_mask_folder_sam3",
            "contains_lego_brick_sam3",
            "lego_brick_mask_folder_sam3",
            "contains_sam3_hand_motion_cotracker",
            "sam3_hand_motion_cotracker_folder",
        }

        # Collect extra columns
        extras = {k: v for k, v in row.items() if k not in known_fields}

        return cls(
            scene_idx=int(row["scene_idx"]),
            scene_folder=row["scene_folder"],
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            label=row.get("label", "uncertain"),
            stabilized_label=row.get("stabilized_label", "uncertain"),
            max_angle=float(row.get("max_angle", 0.0)),
            max_trans=float(row.get("max_trans", 0.0)),
            stabilized_max_angle=float(row.get("stabilized_max_angle", 0.0)),
            stabilized_max_trans=float(row.get("stabilized_max_trans", 0.0)),
            contains_hand_sam3=parse_bool(row.get("contains_hand_sam3", "false")),
            hand_mask_folder_sam3=parse_optional_str(
                row.get("hand_mask_folder_sam3", "")
            ),
            contains_lego_brick_sam3=parse_bool(
                row.get("contains_lego_brick_sam3", "false")
            ),
            lego_brick_mask_folder_sam3=parse_optional_str(
                row.get("lego_brick_mask_folder_sam3", "")
            ),
            contains_sam3_hand_motion_cotracker=parse_bool(
                row.get("contains_sam3_hand_motion_cotracker", "false")
            ),
            sam3_hand_motion_cotracker_folder=parse_optional_str(
                row.get("sam3_hand_motion_cotracker_folder", "")
            ),
            extras=extras,
        )


class SceneFilter:
    """
    Flexible scene filtering based on metadata columns.

    Supports (with YAML examples):
    - Equality: `stabilized_label: "uncertain"` or `has_hands: true`
    - Comparison: `max_trans: [">", 10.0]` or `num_frames: [">=", 100]`
    - Exclusion: `label: ["!=", "static"]`
    - Multiple values (membership): `task_category: ["pnp_push_sweep", "stack_blocks"]`
    - Callable (Python only): `{"num_frames": lambda x: x > 100}`

    Operators: ">", ">=", "<", "<=", "!=", "=="

    Missing key behavior:
        If a filter refers to a field not present in a scene (neither as an
        attribute nor in scene.extras), that scene is EXCLUDED from results.
        This is intentional: filtering by dataset-specific keys like "robot"
        will exclude scenes from datasets that don't have that field.

        Recommendation: Use per-source filters for dataset-specific keys:
        ```yaml
        sources:
          - type: youtube
            root: /path/to/youtube
            filters:
              has_hands: true  # YouTube-specific

          - type: bridge
            root: /path/to/bridge
            filters:
              environment: toykitchen1  # Bridge-specific
        ```
    """

    def __init__(self, filters: Optional[Dict[str, Any]] = None):
        self.filters = filters or {}

    def matches(self, scene: SceneMetadata) -> bool:
        """Check if a scene matches all filter criteria."""
        for key, condition in self.filters.items():
            # Get value from scene (check extras if not a direct attribute)
            if hasattr(scene, key):
                value = getattr(scene, key)
            elif key in scene.extras:
                value = scene.extras[key]
            else:
                return False  # Unknown field - exclude

            # Apply condition
            if callable(condition):
                if not condition(value):
                    return False
            elif isinstance(condition, (tuple, list)) and len(condition) == 2:
                # YAML gives lists, not tuples: [">", 0.05]
                # Check if first element is an operator string
                first_elem = condition[0]
                if isinstance(first_elem, str) and first_elem in (">", ">=", "<", "<=", "!=", "=="):
                    # Treat as (operator, threshold)
                    op, threshold = condition
                    if op == ">" and not (value > threshold):
                        return False
                    elif op == ">=" and not (value >= threshold):
                        return False
                    elif op == "<" and not (value < threshold):
                        return False
                    elif op == "<=" and not (value <= threshold):
                        return False
                    elif op == "!=" and not (value != threshold):
                        return False
                    elif op == "==" and not (value == threshold):
                        return False
                else:
                    # Treat as "value in list" (multiple allowed values)
                    if value not in condition:
                        return False
            elif isinstance(condition, list):
                # List of allowed values: ["pnp_push_sweep", "stack_blocks"]
                if value not in condition:
                    return False
            else:
                # Direct equality
                if value != condition:
                    return False

        return True

    def filter_scenes(self, scenes: List[SceneMetadata]) -> List[SceneMetadata]:
        """Filter a list of scenes."""
        return [s for s in scenes if self.matches(s)]


def load_scenes_csv(csv_path: Union[str, Path]) -> List[SceneMetadata]:
    """Load all scenes from a scenes.csv file."""
    csv_path = Path(csv_path)
    scenes = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenes.append(SceneMetadata.from_csv_row(row))

    return scenes


def metadata_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for MetadataAwareDataset.

    Stacks tensors but keeps metadata as lists (can't be collated).

    Args:
        batch: List of dicts from MetadataAwareDataset

    Returns:
        Dict with:
        - 'frames': Stacked tensor [B, C, 2, H, W]
        - 'scene_idx': List of ints
        - 'metadata': List of SceneMetadata (not collated)
        - 'motion_track_path': List of paths or Nones
        - 'first_frame_idx': Tensor of ints
        - 'second_frame_idx': Tensor of ints
        - 'dataset_type': List of strings (for per-bucket visualization)
    """
    # Extract dataset_type from metadata for easy access
    dataset_types = []
    for item in batch:
        meta = item.get("metadata")
        if meta is not None and hasattr(meta, "extras"):
            dataset_types.append(meta.extras.get("dataset_type", "unknown"))
        else:
            dataset_types.append("unknown")
    
    return {
        "frames": torch.stack([item["frames"] for item in batch]),
        "scene_idx": [item["scene_idx"] for item in batch],
        "metadata": [item.get("metadata") for item in batch],
        "motion_track_path": [item.get("motion_track_path") for item in batch],
        "first_frame_idx": torch.tensor([item["first_frame_idx"] for item in batch]),
        "second_frame_idx": torch.tensor([item["second_frame_idx"] for item in batch]),
        "dataset_type": dataset_types,
    }


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


class MetadataAwareDataset(Dataset):
    """
    Enhanced dataset that uses scenes.csv for metadata-based filtering.

    Features:
    - Filter scenes by any metadata column
    - Use frame ranges from CSV (start_frame, end_frame)
    - Access auxiliary data folders (motion tracks, masks)
    - Extensible for future decoder experiments

    Directory structure:
        root_folder/
        ├── scenes.csv
        ├── scene_000/
        │   ├── frame_0001.jpg
        │   └── ...
        ├── scene_000_motion_sam3_hand_cotracker/  (motion tracks)
        │   └── ...
        └── ...

    Returns:
        Dict with:
        - 'frames': Tensor [C, 2, H, W] - frame pair
        - 'scene_idx': int - scene index
        - 'metadata': SceneMetadata - full scene metadata
        - 'motion_track_path': Optional[str] - path to motion track folder
    """

    def __init__(
        self,
        folder: str,
        image_size: int = 256,
        offset: int = 30,
        filters: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
        min_frames: int = 2,
    ):
        """
        Args:
            folder: Root folder containing scenes.csv and scene folders
            image_size: Size to resize images to
            offset: Frame offset for second frame
            filters: Dict of filter conditions for SceneFilter
            return_metadata: If True, return dict with metadata; else just tensor
            min_frames: Minimum frames required in a scene (skips smaller)
        """
        super().__init__()

        self.folder = Path(folder)
        self.image_size = image_size
        self.offset = offset
        self.return_metadata = return_metadata
        self.min_frames = min_frames

        # Load and filter scenes
        csv_path = self.folder / "scenes.csv"
        if csv_path.exists():
            all_scenes = load_scenes_csv(csv_path)
            scene_filter = SceneFilter(filters)
            self.scenes = scene_filter.filter_scenes(all_scenes)
            # Filter by min_frames
            self.scenes = [s for s in self.scenes if s.num_frames >= min_frames]
        else:
            # Fallback to legacy behavior (scan folders)
            self.scenes = self._create_scenes_from_folders()

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])

    def _create_scenes_from_folders(self) -> List[SceneMetadata]:
        """Fallback: create SceneMetadata from folder structure."""
        scenes = []
        folders = sorted([
            d for d in os.listdir(self.folder)
            if os.path.isdir(self.folder / d) and d.startswith("scene_")
        ])

        for idx, folder_name in enumerate(folders):
            folder_path = self.folder / folder_name
            img_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if len(img_files) >= self.min_frames:
                scenes.append(SceneMetadata(
                    scene_idx=idx,
                    scene_folder=folder_name,
                    start_frame=0,
                    end_frame=len(img_files),
                ))

        return scenes

    def _get_frame_files(self, scene: SceneMetadata) -> List[str]:
        """Get sorted list of frame files for a scene."""
        folder_path = self.folder / scene.scene_folder
        img_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Sort by frame number
        def frame_index(name: str) -> int:
            stem = os.path.splitext(name)[0]
            m = re.search(r"frame_(\d+)$", stem)
            if m:
                return int(m.group(1))
            m2 = re.findall(r"(\d+)", stem)
            return int(m2[-1]) if m2 else -1

        return sorted(img_files, key=frame_index)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        try:
            scene = self.scenes[index]
            folder_path = self.folder / scene.scene_folder
            img_files = self._get_frame_files(scene)

            # Pick random frame pair within valid range
            max_start = max(0, len(img_files) - 1 - self.offset)
            first_idx = random.randint(0, max_start)
            second_idx = min(first_idx + self.offset, len(img_files) - 1)

            # Load frames
            first_path = folder_path / img_files[first_idx]
            second_path = folder_path / img_files[second_idx]

            first_img = Image.open(first_path)
            second_img = Image.open(second_path)

            first_tensor = self.transform(first_img).unsqueeze(1)
            second_tensor = self.transform(second_img).unsqueeze(1)
            frames = torch.cat([first_tensor, second_tensor], dim=1)

            if self.return_metadata:
                # Get motion track path if available
                motion_path = None
                if scene.sam3_hand_motion_cotracker_folder:
                    motion_path = str(self.folder / scene.sam3_hand_motion_cotracker_folder)

                return {
                    "frames": frames,
                    "scene_idx": scene.scene_idx,
                    "metadata": scene,
                    "motion_track_path": motion_path,
                    "first_frame_idx": first_idx,
                    "second_frame_idx": second_idx,
                }
            else:
                return frames

        except Exception as e:
            print(f"Error loading scene {index}: {e}")
            if index < len(self) - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, len(self) - 1))


@dataclass
class FramePairIndex:
    """
    Index for a specific frame pair in a scene.

    Used by MetadataAwarePairDataset to identify concrete (frame_t, frame_t+offset) pairs.
    """
    scene_idx: int
    first_frame_idx: int
    second_frame_idx: int
    offset: int


class MetadataAwarePairDataset(Dataset):
    """
    Pair-level dataset: each index corresponds to a concrete (frame_t, frame_t+offset) pair.

    This is the basis for:
    - Overfitting on a single sample
    - Future pair-level filtering (e.g., by motion flow)
    - Explicit control over frame sampling

    Unlike MetadataAwareDataset which samples pairs on-the-fly, this dataset
    pre-computes all valid pairs and indexes them directly.

    Args:
        folder: Root folder containing scenes
        image_size: Size to resize images to
        offsets: List of frame offsets to use (default [30])
        filters: Dict of filters to apply to scenes
        min_frames: Minimum frames required per scene
        return_metadata: If True, return dict with frames and metadata
    """

    def __init__(
        self,
        folder: str,
        image_size: int = 256,
        offsets: Optional[List[int]] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_frames: int = 2,
        return_metadata: bool = False,
    ):
        super().__init__()

        self.folder = Path(folder)
        self.image_size = image_size
        self.offsets = offsets or [30]
        self.return_metadata = return_metadata
        self.min_frames = min_frames

        # 1) Load & filter scenes
        csv_path = self.folder / "scenes.csv"
        if csv_path.exists():
            all_scenes = load_scenes_csv(csv_path)
            scene_filter = SceneFilter(filters)
            self.scenes = [
                s for s in scene_filter.filter_scenes(all_scenes)
                if s.num_frames >= min_frames
            ]
        else:
            self.scenes = self._create_scenes_from_folders()

        # 2) Build explicit list of all frame pairs
        self.pairs = self._build_pairs(self.scenes, self.offsets)

        # 3) Transform pipeline
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])

    def _create_scenes_from_folders(self) -> List[SceneMetadata]:
        """Create scenes from folder structure (fallback when scenes.csv doesn't exist)."""
        scenes = []
        folders = sorted([
            d for d in os.listdir(self.folder)
            if os.path.isdir(self.folder / d) and d.startswith("scene_")
        ])

        for idx, folder_name in enumerate(folders):
            folder_path = self.folder / folder_name
            img_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if len(img_files) >= self.min_frames:
                scenes.append(SceneMetadata(
                    scene_idx=idx,
                    scene_folder=folder_name,
                    start_frame=0,
                    end_frame=len(img_files),
                ))
        return scenes

    def _get_frame_files(self, scene: SceneMetadata) -> List[str]:
        """Get sorted list of frame files for a scene."""
        folder_path = self.folder / scene.scene_folder
        img_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        def frame_index(name: str) -> int:
            stem = os.path.splitext(name)[0]
            m = re.search(r"frame_(\d+)$", stem)
            if m:
                return int(m.group(1))
            m2 = re.findall(r"(\d+)", stem)
            return int(m2[-1]) if m2 else -1

        return sorted(img_files, key=frame_index)

    def _build_pairs(self, scenes: List[SceneMetadata], offsets: List[int]) -> List[FramePairIndex]:
        """Build explicit list of all valid frame pairs."""
        pairs: List[FramePairIndex] = []
        for scene_idx, scene in enumerate(scenes):
            img_files = self._get_frame_files(scene)
            n = len(img_files)
            if n < 2:
                continue

            for offset in offsets:
                max_start = max(0, n - 1 - offset)
                for first_idx in range(max_start + 1):
                    second_idx = min(first_idx + offset, n - 1)
                    pairs.append(FramePairIndex(
                        scene_idx=scene_idx,
                        first_frame_idx=first_idx,
                        second_frame_idx=second_idx,
                        offset=offset,
                    ))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        scene = self.scenes[pair.scene_idx]
        folder_path = self.folder / scene.scene_folder
        img_files = self._get_frame_files(scene)

        first_path = folder_path / img_files[pair.first_frame_idx]
        second_path = folder_path / img_files[pair.second_frame_idx]

        img1 = Image.open(first_path)
        img2 = Image.open(second_path)

        x1 = self.transform(img1).unsqueeze(1)  # [C,1,H,W]
        x2 = self.transform(img2).unsqueeze(1)  # [C,1,H,W]
        frames = torch.cat([x1, x2], dim=1)     # [C,2,H,W]

        if not self.return_metadata:
            return frames

        return {
            "frames": frames,
            "scene_idx": scene.scene_idx,
            "first_frame_idx": pair.first_frame_idx,
            "second_frame_idx": pair.second_frame_idx,
            "offset": pair.offset,
            "metadata": scene,
        }


class MultiSourceSceneDataset(Dataset):
    """
    Scene-level dataset for multi-source data.

    Takes pre-collected SceneMetadata from multiple sources and loads frame pairs
    on-the-fly. Used by LAQDataModule for multi-source training.
    """

    def __init__(
        self,
        scenes: List[SceneMetadata],
        sources: List[Dict[str, Any]],
        image_size: int = 256,
        offset: int = 30,
        return_metadata: bool = False,
    ):
        super().__init__()
        self.scenes = scenes
        self.sources = sources
        self.image_size = image_size
        self.offset = offset
        self.return_metadata = return_metadata

        # Build source root lookup
        self._source_roots = {s["type"]: Path(s["root"]) for s in sources}

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            # Resize to fit within image_size while maintaining aspect ratio
            T.Resize(image_size, interpolation=T.InterpolationMode.LANCZOS),
            # Center crop to get square image
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def _get_root_for_scene(self, scene: SceneMetadata) -> Path:
        """Get the root directory for a scene based on its dataset_type."""
        dataset_type = scene.extras.get("dataset_type", "youtube")
        return self._source_roots.get(dataset_type, list(self._source_roots.values())[0])

    def _get_frame_files(self, scene: SceneMetadata, root: Path) -> List[str]:
        """Get sorted list of frame files for a scene."""
        folder_path = root / scene.scene_folder
        img_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        def frame_index(name: str) -> int:
            stem = os.path.splitext(name)[0]
            m = re.search(r"frame_(\d+)$", stem)
            if m:
                return int(m.group(1))
            # Try im_X pattern for Bridge
            m2 = re.search(r"im_(\d+)$", stem)
            if m2:
                return int(m2.group(1))
            m3 = re.findall(r"(\d+)", stem)
            return int(m3[-1]) if m3 else -1

        return sorted(img_files, key=frame_index)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        scene = self.scenes[index]
        root = self._get_root_for_scene(scene)
        folder_path = root / scene.scene_folder

        # Handle Bridge dataset (images in raw/traj_group/traj/images subfolder)
        if scene.extras.get("dataset_type") == "bridge":
            from common.adapters import BridgeAdapter
            adapter = BridgeAdapter()
            img_files = [p.name for p in adapter.get_frame_files(scene, root)]
            # Get actual image folder
            raw_path = folder_path / "raw"
            if raw_path.exists():
                for images_dir in raw_path.rglob("images*"):
                    if images_dir.is_dir():
                        folder_path = images_dir
                        break
        else:
            img_files = self._get_frame_files(scene, root)

        # Pick random frame pair
        max_start = max(0, len(img_files) - 1 - self.offset)
        first_idx = random.randint(0, max_start)
        second_idx = min(first_idx + self.offset, len(img_files) - 1)

        first_path = folder_path / img_files[first_idx]
        second_path = folder_path / img_files[second_idx]

        first_img = Image.open(first_path)
        second_img = Image.open(second_path)

        first_tensor = self.transform(first_img).unsqueeze(1)
        second_tensor = self.transform(second_img).unsqueeze(1)
        frames = torch.cat([first_tensor, second_tensor], dim=1)

        if not self.return_metadata:
            return frames

        return {
            "frames": frames,
            "scene_idx": scene.scene_idx,
            "metadata": scene,
            "motion_track_path": None,
            "first_frame_idx": first_idx,
            "second_frame_idx": second_idx,
        }


class MultiSourcePairDataset(Dataset):
    """
    Pair-level dataset for multi-source data.

    Takes pre-collected SceneMetadata and pre-computes all valid frame pairs.
    Used by LAQDataModule for deterministic multi-source training.
    """

    def __init__(
        self,
        scenes: List[SceneMetadata],
        sources: List[Dict[str, Any]],
        image_size: int = 256,
        offsets: Optional[List[int]] = None,
        return_metadata: bool = False,
    ):
        super().__init__()
        self.scenes = scenes
        self.sources = sources
        self.image_size = image_size
        self.offsets = offsets or [30]
        self.return_metadata = return_metadata

        # Build source root lookup
        self._source_roots = {s["type"]: Path(s["root"]) for s in sources}

        # Build all frame pairs
        self.pairs = self._build_pairs()

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            # Resize to fit within image_size while maintaining aspect ratio
            T.Resize(image_size, interpolation=T.InterpolationMode.LANCZOS),
            # Center crop to get square image
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def _get_root_for_scene(self, scene: SceneMetadata) -> Path:
        """Get the root directory for a scene based on its dataset_type."""
        dataset_type = scene.extras.get("dataset_type", "youtube")
        return self._source_roots.get(dataset_type, list(self._source_roots.values())[0])

    def _get_frame_files_for_scene(self, scene: SceneMetadata) -> List[Path]:
        """Get frame files for a scene, handling different dataset types."""
        root = self._get_root_for_scene(scene)

        if scene.extras.get("dataset_type") == "bridge":
            from common.adapters import BridgeAdapter
            adapter = BridgeAdapter()
            return adapter.get_frame_files(scene, root)
        else:
            from common.adapters import YoutubeAdapter
            adapter = YoutubeAdapter()
            return adapter.get_frame_files(scene, root)

    def _build_pairs(self) -> List[FramePairIndex]:
        """Build explicit list of all valid frame pairs."""
        pairs = []
        for scene_idx, scene in enumerate(self.scenes):
            frame_files = self._get_frame_files_for_scene(scene)
            n = len(frame_files)
            if n < 2:
                continue

            for offset in self.offsets:
                max_start = max(0, n - 1 - offset)
                for first_idx in range(max_start + 1):
                    second_idx = min(first_idx + offset, n - 1)
                    pairs.append(FramePairIndex(
                        scene_idx=scene_idx,
                        first_frame_idx=first_idx,
                        second_frame_idx=second_idx,
                        offset=offset,
                    ))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        scene = self.scenes[pair.scene_idx]

        frame_files = self._get_frame_files_for_scene(scene)

        first_path = frame_files[pair.first_frame_idx]
        second_path = frame_files[pair.second_frame_idx]

        img1 = Image.open(first_path)
        img2 = Image.open(second_path)

        x1 = self.transform(img1).unsqueeze(1)
        x2 = self.transform(img2).unsqueeze(1)
        frames = torch.cat([x1, x2], dim=1)

        if not self.return_metadata:
            return frames

        return {
            "frames": frames,
            "scene_idx": scene.scene_idx,
            "first_frame_idx": pair.first_frame_idx,
            "second_frame_idx": pair.second_frame_idx,
            "offset": pair.offset,
            "metadata": scene,
        }


class LAQDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for LAQ training.

    Features:
    - Multi-source support: Load from multiple datasets via adapters
    - Metadata-based filtering via SceneFilter
    - Metadata-based train/val splits (hold out specific videos/datasets)
    - Validation buckets for distribution shift analysis
    - Subset support for incremental testing
    - Configurable via Hydra

    Example config for multi-source:
    ```yaml
    sources:
      - type: youtube
        root: /path/to/youtube
        filters:
          has_hands: true
      - type: bridge
        root: /path/to/bridge
        filters:
          environment: toykitchen1
    ```

    Example config for metadata-based val split:
    ```yaml
    split_mode: metadata
    val_scene_filters:
      video_id: "held_out_video"
    ```
    """

    def __init__(
        self,
        # Legacy single-folder mode
        folder: Optional[str] = None,
        # Multi-source mode
        sources: Optional[List[Dict[str, Any]]] = None,
        # Common settings
        image_size: int = 256,
        offset: int = 30,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        max_samples: Optional[int] = None,
        val_split: float = 0.1,
        use_metadata: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
        min_frames: int = 2,
        pair_level: bool = False,
        offsets: Optional[List[int]] = None,
        sampling_strategy: str = "random",
        sampling_seed: int = 42,
        # Metadata-based split settings
        split_mode: str = "ratio",
        val_scene_filters: Optional[Dict[str, Any]] = None,
        # Validation buckets for distribution shift analysis
        val_buckets: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Args:
            folder: Root folder containing scenes (legacy, use sources for multi-dataset)
            sources: List of dataset sources, each with type, root, and optional filters
            image_size: Size to resize images to
            offset: Frame offset for second frame (legacy, used when not pair_level)
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            pin_memory: Pin memory for faster GPU transfer
            prefetch_factor: Prefetch factor for dataloaders
            max_samples: Maximum samples (for subset testing). When pair_level=True, this is pair count.
            val_split: Fraction of data for validation (used when split_mode="ratio")
            use_metadata: If True, use MetadataAwareDataset with scenes.csv
            filters: Global filter conditions applied after per-source filters
            return_metadata: If True, return dict with metadata
            min_frames: Minimum frames required per scene
            pair_level: If True, use MetadataAwarePairDataset (pre-computed pairs)
            offsets: List of frame offsets for pair-level mode (default [offset])
            sampling_strategy: 'random' for diverse samples, 'sequential' for neighboring
            sampling_seed: Random seed for reproducible random sampling
            split_mode: "ratio" for percentage-based, "metadata" for filter-based split
            val_scene_filters: Filters to select validation scenes (when split_mode="metadata")
            val_buckets: Dict of named validation buckets, each with filters for analysis
        """
        super().__init__()

        # Handle backward compatibility: convert folder to sources
        if sources is None and folder is not None:
            sources = [{"type": "youtube", "root": folder}]
        elif sources is None and folder is None:
            raise ValueError("Must provide either 'folder' or 'sources'")

        self.sources = sources
        self.folder = folder  # Keep for backward compat in dataset creation
        self.image_size = image_size
        self.offset = offset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.max_samples = max_samples
        self.val_split = val_split
        self.use_metadata = use_metadata
        self.filters = filters
        self.return_metadata = return_metadata
        self.min_frames = min_frames
        self.pair_level = pair_level
        self.offsets = offsets or [offset]
        self.sampling_strategy = sampling_strategy
        self.sampling_seed = sampling_seed
        self.split_mode = split_mode
        self.val_scene_filters = val_scene_filters
        self.val_buckets = val_buckets

        self.train_dataset = None
        self.val_dataset = None
        self.val_bucket_datasets: Dict[str, Dataset] = {}
        self.scenes: List[SceneMetadata] = []

    def _get_adapter(self, source_type: str):
        """Get adapter for dataset type."""
        from common.adapters import YoutubeAdapter, BridgeAdapter

        adapters = {
            "youtube": YoutubeAdapter(),
            "bridge": BridgeAdapter(),
        }

        if source_type not in adapters:
            raise ValueError(
                f"Unknown dataset type: {source_type}. "
                f"Available: {list(adapters.keys())}"
            )

        return adapters[source_type]

    def _collect_all_scenes(self) -> List[SceneMetadata]:
        """Collect scenes from all sources using adapters."""
        all_scenes = []

        for source_config in self.sources:
            source_type = source_config["type"]
            source_root = Path(source_config["root"])
            source_filters = source_config.get("filters", None)

            adapter = self._get_adapter(source_type)
            scenes = adapter.collect_scenes(source_root, filters=source_filters)

            print(f"✓ Loaded {len(scenes)} scenes from {source_type} ({source_root})")
            all_scenes.extend(scenes)

        # Apply global filters if specified
        if self.filters:
            filter_obj = SceneFilter(self.filters)
            all_scenes = filter_obj.filter_scenes(all_scenes)
            print(f"✓ After global filtering: {len(all_scenes)} scenes")

        # Filter by min_frames
        all_scenes = [s for s in all_scenes if s.num_frames >= self.min_frames]

        print(f"✓ Total scenes: {len(all_scenes)}")
        return all_scenes

    def _split_scenes_by_metadata(
        self, scenes: List[SceneMetadata]
    ) -> tuple[List[SceneMetadata], List[SceneMetadata]]:
        """Split scenes into train/val based on metadata filters."""
        if not self.val_scene_filters:
            raise ValueError("val_scene_filters required when split_mode='metadata'")

        val_filter = SceneFilter(self.val_scene_filters)
        val_scenes = val_filter.filter_scenes(scenes)

        # Build set of val scene identifiers for exclusion
        val_ids = {(s.scene_folder, s.scene_idx) for s in val_scenes}
        train_scenes = [
            s for s in scenes
            if (s.scene_folder, s.scene_idx) not in val_ids
        ]

        print(f"✓ Metadata split: {len(train_scenes)} train, {len(val_scenes)} val scenes")
        return train_scenes, val_scenes

    def _split_scenes_by_ratio(
        self, scenes: List[SceneMetadata]
    ) -> tuple[List[SceneMetadata], List[SceneMetadata]]:
        """Split scenes into train/val based on ratio."""
        total = len(scenes)
        val_size = int(total * self.val_split)
        train_size = total - val_size

        train_scenes = scenes[:train_size]
        val_scenes = scenes[train_size:]

        return train_scenes, val_scenes

    def _build_validation_buckets(self, scenes: List[SceneMetadata]) -> Dict[str, List[SceneMetadata]]:
        """Build validation buckets for distribution shift analysis."""
        if not self.val_buckets:
            return {}

        buckets = {}
        for bucket_name, bucket_filters in self.val_buckets.items():
            bucket_filter = SceneFilter(bucket_filters)
            bucket_scenes = bucket_filter.filter_scenes(scenes)
            buckets[bucket_name] = bucket_scenes
            print(f"  - Val bucket '{bucket_name}': {len(bucket_scenes)} scenes")

        return buckets

    def setup(self, stage: Optional[str] = None):
        """Setup train/val datasets."""
        # Use legacy path for backward compatibility when folder is provided directly
        # New multi-source path is used when:
        # 1. Multiple sources are configured OR
        # 2. metadata-based split is requested OR
        # 3. val_buckets are configured
        use_multi_source = (
            self.folder is None or  # explicit sources provided
            self.split_mode == "metadata" or
            self.val_buckets is not None or
            (self.sources and len(self.sources) > 1)
        )

        if self.use_metadata and use_multi_source:
            all_scenes = self._collect_all_scenes()
            self.scenes = all_scenes

            # Split scenes into train/val
            if self.split_mode == "metadata":
                train_scenes, val_scenes = self._split_scenes_by_metadata(all_scenes)
            else:  # ratio
                train_scenes, val_scenes = self._split_scenes_by_ratio(all_scenes)

            # Build validation buckets for distribution shift analysis
            if self.val_buckets:
                print("Building validation buckets:")
                val_bucket_scenes = self._build_validation_buckets(val_scenes)

            # Get root folder (for single-source backward compat)
            root_folder = self.sources[0]["root"]

            # Create datasets from scene lists
            if self.pair_level:
                full_train = MultiSourcePairDataset(
                    scenes=train_scenes,
                    sources=self.sources,
                    image_size=self.image_size,
                    offsets=self.offsets,
                    return_metadata=self.return_metadata,
                )
                full_val = MultiSourcePairDataset(
                    scenes=val_scenes,
                    sources=self.sources,
                    image_size=self.image_size,
                    offsets=self.offsets,
                    return_metadata=self.return_metadata,
                )
                # Create bucket datasets
                for bucket_name, bucket_scenes in val_bucket_scenes.items() if self.val_buckets else []:
                    self.val_bucket_datasets[bucket_name] = MultiSourcePairDataset(
                        scenes=bucket_scenes,
                        sources=self.sources,
                        image_size=self.image_size,
                        offsets=self.offsets,
                        return_metadata=self.return_metadata,
                    )
            else:
                full_train = MultiSourceSceneDataset(
                    scenes=train_scenes,
                    sources=self.sources,
                    image_size=self.image_size,
                    offset=self.offset,
                    return_metadata=self.return_metadata,
                )
                full_val = MultiSourceSceneDataset(
                    scenes=val_scenes,
                    sources=self.sources,
                    image_size=self.image_size,
                    offset=self.offset,
                    return_metadata=self.return_metadata,
                )
                # Create bucket datasets
                for bucket_name, bucket_scenes in (val_bucket_scenes.items() if self.val_buckets else []):
                    self.val_bucket_datasets[bucket_name] = MultiSourceSceneDataset(
                        scenes=bucket_scenes,
                        sources=self.sources,
                        image_size=self.image_size,
                        offset=self.offset,
                        return_metadata=self.return_metadata,
                    )

            self.total_available = len(all_scenes)

            # Apply max_samples subset
            if self.max_samples is not None:
                num_train = min(self.max_samples, len(full_train))
                num_val = min(max(1, int(num_train * self.val_split)), len(full_val))

                if self.sampling_strategy == "random":
                    rng = random.Random(self.sampling_seed)
                    train_indices = rng.sample(range(len(full_train)), num_train)
                    val_indices = rng.sample(range(len(full_val)), num_val) if len(full_val) > 0 else []
                else:
                    train_indices = list(range(num_train))
                    val_indices = list(range(num_val))

                self.train_dataset = Subset(full_train, train_indices)
                self.val_dataset = Subset(full_val, val_indices) if val_indices else full_val
            else:
                self.train_dataset = full_train
                self.val_dataset = full_val

        else:
            # Legacy single-folder mode
            self._setup_legacy(stage)

    def _setup_legacy(self, stage: Optional[str] = None):
        """Legacy setup for single-folder mode."""
        folder = self.folder or self.sources[0]["root"]

        if self.use_metadata:
            if self.pair_level:
                full_dataset = MetadataAwarePairDataset(
                    folder=folder,
                    image_size=self.image_size,
                    offsets=self.offsets,
                    filters=self.filters,
                    min_frames=self.min_frames,
                    return_metadata=self.return_metadata,
                )
            else:
                full_dataset = MetadataAwareDataset(
                    folder=folder,
                    image_size=self.image_size,
                    offset=self.offset,
                    filters=self.filters,
                    return_metadata=self.return_metadata,
                    min_frames=self.min_frames,
                )
        else:
            full_dataset = ImageVideoDataset(
                folder=folder,
                image_size=self.image_size,
                offset=self.offset,
            )

        self.total_available = len(full_dataset)

        # Apply subset if specified
        if self.max_samples is not None:
            num_samples = min(self.max_samples, len(full_dataset))

            if self.sampling_strategy == "random":
                rng = random.Random(self.sampling_seed)
                indices = rng.sample(range(len(full_dataset)), num_samples)
            elif self.sampling_strategy == "sequential":
                indices = list(range(num_samples))
            else:
                raise ValueError(
                    f"Invalid sampling_strategy: {self.sampling_strategy}. "
                    "Must be 'random' or 'sequential'."
                )

            full_dataset = Subset(full_dataset, indices)

        # Split into train/val
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_size))

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

    def train_dataloader(self):
        """Create training dataloader."""
        collate_fn = metadata_collate_fn if self.return_metadata else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        collate_fn = metadata_collate_fn if self.return_metadata else None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
        )

    def val_bucket_dataloader(self, bucket_name: str):
        """Create dataloader for a specific validation bucket."""
        if bucket_name not in self.val_bucket_datasets:
            raise ValueError(f"Unknown validation bucket: {bucket_name}")

        collate_fn = metadata_collate_fn if self.return_metadata else None
        return DataLoader(
            self.val_bucket_datasets[bucket_name],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_fn,
        )
