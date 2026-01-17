"""
Data loading utilities for LAQ training.

Includes:
- SceneMetadata: Dataclass for scene metadata from CSV
- SceneFilter: Flexible filtering for scenes by any column
- FramePairIndex: Dataclass for explicit frame pair indexing
- MultiSourcePairDataset: Pair-level dataset for multi-source data
- LAQDataModule: Lightning DataModule with multi-source support
"""

import csv
import logging
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

# Module-level logger
logger = logging.getLogger(__name__)


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
    - Not null: `action: ["not_null", true]`
    - Membership: `dataset_name: ["in", ["youtube", "bridge"]]`
    - Multiple values: `task_category: ["pnp_push_sweep", "stack_blocks"]`
    - Callable (Python only): `{"num_frames": lambda x: x > 100}`

    Operators: ">", ">=", "<", "<=", "!=", "==", "in", "not_null"

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

    def _get_scene_value(self, scene: SceneMetadata, key: str) -> Any:
        """Get value from scene, checking attributes then extras."""
        if hasattr(scene, key):
            return getattr(scene, key)
        elif key in scene.extras:
            return scene.extras[key]
        return None  # Key not found

    def matches(self, scene: SceneMetadata) -> bool:
        """Check if a scene matches all filter criteria."""
        from common.filters import matches_filters
        return matches_filters(
            data=scene,  # type: ignore (SceneMetadata is dict-like for our purposes)
            filters=self.filters,
            get_value=lambda s, k: self._get_scene_value(s, k),
        )

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


def oxe_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for OXE datasets.
    
    Stacks 'frames' into a tensor but keeps other metadata fields as lists.
    This prevents errors when collating heterogeneous data (e.g., variable length actions).
    """
    if not batch:
        return {}

    # Separate frames and metadata
    frames_list = []
    metadata = {}
    
    # Initialize metadata lists
    for key in batch[0].keys():
        if key != "frames":
            metadata[key] = []

    for item in batch:
        frames_list.append(item["frames"])
        for key, value in item.items():
            if key != "frames":
                # Handle cases where keys might be missing in some items (though unlikely with consistent dataset)
                if key in metadata:
                    metadata[key].append(value)
    
    # Stack frames
    result = {
        "frames": torch.stack(frames_list),
    }
    
    # Add metadata lists
    result.update(metadata)
    
    return result


def metadata_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for MultiSourcePairDataset.

    Stacks tensors but keeps metadata as lists (can't be collated).
    Uses standardized keys for interface consistency with OXE datasets.

    Args:
        batch: List of dicts from MultiSourcePairDataset

    Returns:
        Dict with standardized keys:
        - 'frames': Stacked tensor [B, C, 2, H, W]
        - 'episode_id': List of strings (was: scene_idx)
        - 'frame_idx': List of ints (was: first_frame_idx)
        - 'dataset_name': List of strings (source identifier)
        - 'dataset_type': List of strings (alias for dataset_name, OXE compatibility)
        - 'language': List of strings (task descriptions)
        - 'metadata': List of SceneMetadata (raw, not collated)
        - 'second_frame_idx': Tensor of ints (kept for compatibility)
        - 'environment': List of strings (for analysis)
        - 'task': List of strings (for analysis)
    """
    # Extract useful fields from metadata for easy access
    dataset_names = []
    environments = []
    tasks = []
    languages = []
    episode_ids = []
    frame_indices = []

    for item in batch:
        meta = item.get("metadata")
        # Convert scene_idx to episode_id (string format for consistency)
        episode_ids.append(str(item.get("scene_idx", "")))
        frame_indices.append(item.get("first_frame_idx", 0))

        if meta is not None and hasattr(meta, "extras"):
            extras = meta.extras
            # Use dataset_type as dataset_name for consistency
            dataset_names.append(extras.get("dataset_type", "unknown"))
            environments.append(extras.get("environment", "unknown"))
            tasks.append(extras.get("task", "unknown"))
            languages.append(extras.get("language", ""))
        else:
            dataset_names.append("unknown")
            environments.append("unknown")
            tasks.append("unknown")
            languages.append("")

    return {
        # Core standardized keys (match OXE interface exactly)
        "frames": torch.stack([item["frames"] for item in batch]),
        "episode_id": episode_ids,
        "frame_idx": frame_indices,
        "dataset_name": dataset_names,
        "dataset_type": dataset_names,  # Alias for OXE compatibility
        "language": languages,
        # Additional metadata
        "metadata": [item.get("metadata") for item in batch],
        "second_frame_idx": torch.tensor([item["second_frame_idx"] for item in batch]),
        "environment": environments,
        "task": tasks,
    }


# Standard batch keys expected in LAQ training batches
STANDARD_BATCH_KEYS = frozenset({
    "frames",       # [B, C, 2, H, W] frame pairs tensor
    "episode_id",   # List[str] unique episode/scene identifiers
    "frame_idx",    # List[int] start frame indices
    "dataset_name", # List[str] source dataset names
    "language",     # List[str] task descriptions/instructions
})


def validate_batch_keys(
    batch: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
    raise_on_missing: bool = True,
) -> bool:
    """
    Validate that a batch dictionary contains required keys.

    Used to ensure interface consistency between LAQDataModule and OXEDataModule.
    Call this once at the start of training to catch configuration errors early.

    Args:
        batch: Batch dictionary from dataloader
        required_keys: List of required keys (defaults to STANDARD_BATCH_KEYS)
        raise_on_missing: If True, raise ValueError on missing keys; else return False

    Returns:
        True if all required keys present, False otherwise (when raise_on_missing=False)

    Raises:
        ValueError: If required keys are missing and raise_on_missing=True

    Example:
        >>> batch = next(iter(dataloader))
        >>> validate_batch_keys(batch)  # Raises if missing keys
        >>> validate_batch_keys(batch, required_keys=["frames", "language"])
    """
    import logging

    logger = logging.getLogger(__name__)

    if required_keys is None:
        required_keys = list(STANDARD_BATCH_KEYS)

    missing_keys = [key for key in required_keys if key not in batch]

    if missing_keys:
        msg = f"Batch missing required keys: {missing_keys}. Present keys: {list(batch.keys())}"
        if raise_on_missing:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    # Optional: warn about unexpected keys (for debugging)
    extra_keys = set(batch.keys()) - STANDARD_BATCH_KEYS - {"metadata", "second_frame_idx", "environment", "task", "offset", "action", "initial_state"}
    if extra_keys:
        logger.debug(f"Batch contains extra keys (not an error): {extra_keys}")

    return True


@dataclass
class FramePairIndex:
    """
    Index for a specific frame pair in a scene.

    Used by MultiSourcePairDataset to identify concrete (frame_t, frame_t+offset) pairs.
    """
    scene_idx: int
    first_frame_idx: int
    second_frame_idx: int
    offset: int


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

        # Cache for frame files per scene
        self._scene_frame_files: List[List[Path]] = []

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
        self._scene_frame_files = []
        
        for scene_idx, scene in enumerate(self.scenes):
            frame_files = self._get_frame_files_for_scene(scene)
            self._scene_frame_files.append(frame_files)
            
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

        # Use cached frame files instead of scanning disk
        frame_files = self._scene_frame_files[pair.scene_idx]

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

    Example config:
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
        sources: List[Dict[str, Any]],
        image_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        max_pairs: Optional[int] = None,
        val_split: float = 0.1,
        filters: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
        min_frames: int = 2,
        offsets: Optional[List[int]] = None,
        sampling_strategy: str = "random",
        sampling_seed: int = 42,
        split_mode: str = "ratio",
        val_scene_filters: Optional[Dict[str, Any]] = None,
        val_buckets: Optional[Dict[str, Dict[str, Any]]] = None,
        val_counts_per_dataset: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            sources: List of dataset sources, each with type, root, and optional filters
            image_size: Size to resize images to
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            pin_memory: Pin memory for faster GPU transfer
            prefetch_factor: Prefetch factor for dataloaders
            max_pairs: Maximum frame pairs (for subset testing/debugging)
            val_split: Fraction of data for validation (used when split_mode="ratio")
            filters: Global filter conditions applied after per-source filters
            return_metadata: If True, return dict with metadata
            min_frames: Minimum frames required per scene
            offsets: List of frame offsets for pair generation (default [30])
            sampling_strategy: 'random' for diverse samples, 'sequential' for neighboring
            sampling_seed: Random seed for reproducible random sampling
            split_mode: "ratio" for percentage-based, "metadata" for filter-based split
            val_scene_filters: Filters to select validation scenes (when split_mode="metadata")
            val_buckets: Dict of named validation buckets, each with filters for analysis
        """
        super().__init__()

        if not sources:
            raise ValueError("Must provide 'sources' configuration")

        self.sources = sources
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.max_pairs = max_pairs
        self.val_split = val_split
        self.filters = filters
        self.return_metadata = return_metadata
        self.min_frames = min_frames
        self.offsets = offsets or [30]
        self.sampling_strategy = sampling_strategy
        self.sampling_seed = sampling_seed
        self.split_mode = split_mode
        self.val_scene_filters = val_scene_filters
        self.val_buckets = val_buckets
        self.val_counts_per_dataset = val_counts_per_dataset

        self.train_dataset = None
        self.val_dataset = None
        self.val_bucket_datasets: Dict[str, Dataset] = {}
        self.train_bucket_datasets: Dict[str, Dataset] = {}
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

            logger.info(f"✓ Loaded {len(scenes)} scenes from {source_type} ({source_root})")
            all_scenes.extend(scenes)

        # Apply global filters if specified
        if self.filters:
            filter_obj = SceneFilter(self.filters)
            all_scenes = filter_obj.filter_scenes(all_scenes)
            logger.info(f"✓ After global filtering: {len(all_scenes)} scenes")

        # Filter by min_frames
        all_scenes = [s for s in all_scenes if s.num_frames >= self.min_frames]

        logger.info(f"✓ Total scenes: {len(all_scenes)}")
        return all_scenes

    def _split_scenes_by_fixed_count(
        self, scenes: List[SceneMetadata]
    ) -> tuple[List[SceneMetadata], List[SceneMetadata]]:
        """
        Split scenes into train/val by targeting a fixed FRAME PAIR count from each dataset.

        Accumulates scenes until reaching the target frame pair count per dataset.
        This ensures balanced validation regardless of scene length variations
        (important for long YouTube videos vs short Bridge trajectories).

        Requires val_counts_per_dataset configuration.
        Example: {'youtube': 1000, 'bridge': 1000} means ~1000 frame pairs from each
        """
        import random

        if not self.val_counts_per_dataset:
            raise ValueError("val_counts_per_dataset required when split_mode='fixed_count'")

        # Get max offset for frame pair calculation
        max_offset = max(self.offsets) if self.offsets else 30

        def pairs_per_scene(scene: SceneMetadata) -> int:
            """Calculate number of frame pairs from a scene."""
            return max(0, scene.num_frames - max_offset)

        # Group scenes by dataset_type
        by_dataset: Dict[str, List[SceneMetadata]] = {}
        for scene in scenes:
            dtype = scene.extras.get("dataset_type", "unknown")
            if dtype not in by_dataset:
                by_dataset[dtype] = []
            by_dataset[dtype].append(scene)

        train_scenes = []
        val_scenes = []

        # Accumulate scenes until reaching target frame pair count
        for dtype, dtype_scenes in by_dataset.items():
            target_pairs = self.val_counts_per_dataset.get(dtype, 0)

            if target_pairs == 0:
                # No val pairs requested for this dataset
                train_scenes.extend(dtype_scenes)
                logger.info(f"  - {dtype}: {len(dtype_scenes)} train scenes, 0 val scenes (no val requested)")
                continue

            # Shuffle deterministically
            shuffled = dtype_scenes.copy()
            random.Random(42).shuffle(shuffled)

            # Accumulate scenes until we reach target frame pair count
            val_subset = []
            val_pair_count = 0

            for scene in shuffled:
                scene_pairs = pairs_per_scene(scene)
                if val_pair_count < target_pairs:
                    val_subset.append(scene)
                    val_pair_count += scene_pairs

            # Rest go to train
            val_set = set(id(s) for s in val_subset)
            train_subset = [s for s in shuffled if id(s) not in val_set]

            val_scenes.extend(val_subset)
            train_scenes.extend(train_subset)

            # Calculate train frame pair count for logging
            train_pair_count = sum(pairs_per_scene(s) for s in train_subset)

            logger.info(f"  - {dtype}: {len(train_subset)} train scenes ({train_pair_count} pairs), "
                  f"{len(val_subset)} val scenes ({val_pair_count} pairs, target: {target_pairs})")

        # Shuffle final lists
        random.Random(42).shuffle(train_scenes)
        random.Random(42).shuffle(val_scenes)

        total_train_pairs = sum(pairs_per_scene(s) for s in train_scenes)
        total_val_pairs = sum(pairs_per_scene(s) for s in val_scenes)

        logger.info(f"✓ Fixed count split: {len(train_scenes)} train scenes ({total_train_pairs} pairs), "
              f"{len(val_scenes)} val scenes ({total_val_pairs} pairs)")
        return train_scenes, val_scenes

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

        logger.info(f"✓ Metadata split: {len(train_scenes)} train, {len(val_scenes)} val scenes")
        return train_scenes, val_scenes

    def _split_scenes_by_ratio(
        self, scenes: List[SceneMetadata]
    ) -> tuple[List[SceneMetadata], List[SceneMetadata]]:
        """
        Split scenes into train/val with stratification by dataset_type.
        
        This ensures both train and val have proportional representation
        from each dataset type (youtube, bridge, etc.).
        """
        import random
        
        # Group scenes by dataset_type
        by_dataset: Dict[str, List[SceneMetadata]] = {}
        for scene in scenes:
            dtype = scene.extras.get("dataset_type", "unknown")
            if dtype not in by_dataset:
                by_dataset[dtype] = []
            by_dataset[dtype].append(scene)
        
        train_scenes = []
        val_scenes = []
        
        # Stratified split: take val_split proportion from each dataset type
        for dtype, dtype_scenes in by_dataset.items():
            # Shuffle within each dataset type for randomness
            shuffled = dtype_scenes.copy()
            random.Random(42).shuffle(shuffled)  # Deterministic shuffle
            
            val_size = int(len(shuffled) * self.val_split)
            val_size = max(1, val_size)  # At least 1 val sample per dataset
            
            train_scenes.extend(shuffled[:-val_size] if val_size > 0 else shuffled)
            val_scenes.extend(shuffled[-val_size:] if val_size > 0 else [])
        
        # Shuffle final lists to interleave dataset types
        random.Random(42).shuffle(train_scenes)
        random.Random(42).shuffle(val_scenes)
        
        # Log distribution
        train_by_type = {}
        val_by_type = {}
        for s in train_scenes:
            dt = s.extras.get("dataset_type", "unknown")
            train_by_type[dt] = train_by_type.get(dt, 0) + 1
        for s in val_scenes:
            dt = s.extras.get("dataset_type", "unknown")
            val_by_type[dt] = val_by_type.get(dt, 0) + 1
        
        logger.info(f"✓ Stratified split: {len(train_scenes)} train, {len(val_scenes)} val scenes")
        logger.info(f"  Train by dataset: {train_by_type}")
        logger.info(f"  Val by dataset: {val_by_type}")
        
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
            logger.info(f"  - Val bucket '{bucket_name}': {len(bucket_scenes)} scenes")

        return buckets

    def _build_training_buckets(self, scenes: List[SceneMetadata]) -> Dict[str, List[SceneMetadata]]:
        """Build training buckets for visualization/analysis."""
        if not self.val_buckets:
            return {}

        buckets = {}
        for bucket_name, bucket_filters in self.val_buckets.items():
            bucket_filter = SceneFilter(bucket_filters)
            bucket_scenes = bucket_filter.filter_scenes(scenes)
            buckets[bucket_name] = bucket_scenes
            logger.info(f"  - Train bucket '{bucket_name}': {len(bucket_scenes)} scenes")

        return buckets

    def setup(self, stage: Optional[str] = None):
        """Setup train/val datasets."""
        all_scenes = self._collect_all_scenes()
        self.scenes = all_scenes

        # Split scenes into train/val
        if self.split_mode == "metadata":
            train_scenes, val_scenes = self._split_scenes_by_metadata(all_scenes)
        elif self.split_mode == "fixed_count":
            train_scenes, val_scenes = self._split_scenes_by_fixed_count(all_scenes)
        else:  # ratio
            train_scenes, val_scenes = self._split_scenes_by_ratio(all_scenes)

        # Build validation buckets for distribution shift analysis
        val_bucket_scenes = {}
        train_bucket_scenes = {}
        if self.val_buckets:
            logger.info("Building buckets:")
            val_bucket_scenes = self._build_validation_buckets(val_scenes)
            train_bucket_scenes = self._build_training_buckets(train_scenes)

        # Create datasets from scene lists
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
        for bucket_name, bucket_scenes in val_bucket_scenes.items():
            base_dataset = MultiSourcePairDataset(
                scenes=bucket_scenes,
                sources=self.sources,
                image_size=self.image_size,
                offsets=self.offsets,
                return_metadata=self.return_metadata,
            )
            # Shuffle val bucket once during init for diverse visualization samples
            if len(base_dataset) > 0:
                shuffled_indices = torch.randperm(len(base_dataset)).tolist()
                self.val_bucket_datasets[bucket_name] = Subset(base_dataset, shuffled_indices)
            else:
                self.val_bucket_datasets[bucket_name] = base_dataset

        for bucket_name, bucket_scenes in train_bucket_scenes.items():
            self.train_bucket_datasets[bucket_name] = MultiSourcePairDataset(
                scenes=bucket_scenes,
                sources=self.sources,
                image_size=self.image_size,
                offsets=self.offsets,
                return_metadata=self.return_metadata,
            )

        self.total_available = len(all_scenes)

        # Apply max_pairs subset (for debugging/testing)
        if self.max_pairs is not None:
            num_train = min(self.max_pairs, len(full_train))
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

    def train_bucket_dataloader(self, bucket_name: str):
        """Create dataloader for a specific training bucket."""
        if bucket_name not in self.train_bucket_datasets:
            raise ValueError(f"Unknown training bucket: {bucket_name}")

        collate_fn = metadata_collate_fn if self.return_metadata else None
        return DataLoader(
            self.train_bucket_datasets[bucket_name],
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training buckets
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
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

    def get_pairs_per_dataset(self) -> Dict[str, Dict[str, int]]:
        """Get frame pair counts per dataset type for both train and val."""
        result = {"train": {}, "val": {}}

        def count_pairs(dataset) -> Dict[str, int]:
            """Count frame pairs per dataset type, handling Subsets."""
            counts = {}
            if dataset is None:
                return counts

            # Handle Subset wrapper
            if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
                # It's a Subset - count only the subset indices
                base_ds = dataset.dataset
                if hasattr(base_ds, 'pairs') and hasattr(base_ds, 'scenes'):
                    for idx in dataset.indices:
                        pair = base_ds.pairs[idx]
                        scene = base_ds.scenes[pair.scene_idx]
                        dtype = scene.extras.get("dataset_type", "unknown")
                        counts[dtype] = counts.get(dtype, 0) + 1
            elif hasattr(dataset, 'pairs') and hasattr(dataset, 'scenes'):
                # Full dataset (MultiSourcePairDataset)
                for pair in dataset.pairs:
                    scene = dataset.scenes[pair.scene_idx]
                    dtype = scene.extras.get("dataset_type", "unknown")
                    counts[dtype] = counts.get(dtype, 0) + 1

            return counts

        result["train"] = count_pairs(self.train_dataset)
        result["val"] = count_pairs(self.val_dataset)
        return result


class OXEDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Open X-Embodiment datasets.

    Streams data directly from Google Cloud Storage using tf.data pipelines,
    without caching to local disk. Designed for large-scale OXE datasets
    like language_table (442k episodes, multi-TB).

    Supports both single-dataset and multi-dataset configurations:

    Single dataset (legacy format):
    ```yaml
    dataset_name: bridge
    train_split: "train[:90%]"
    val_split: "train[90%:]"
    offset: 5
    ```

    Multi-dataset (new format):
    ```yaml
    datasets:
      - name: language_table
        train_split: "train[:90%]"
        val_split: "train[90%:]"
        weight: 0.6
      - name: bridge
        train_split: "train[:90%]"
        val_split: "train[90%:]"
        weight: 0.4
    offset: 5  # Shared default
    ```
    """

    def __init__(
        self,
        offset: int,
        prefetch_buffer: int,
        # New multi-dataset format
        datasets: Optional[list] = None,
        # Legacy single-dataset format (still supported)
        dataset_name: Optional[str] = None,
        gcs_path: Optional[str] = None,
        train_split: str = "train[:90%]",
        val_split: str = "train[90%:]",
        samples_per_episode: int = 0,
        sampling_seed: Optional[int] = None,
        image_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 0,  # IterableDataset + tf.data handles parallelism
        shuffle_buffer: int = 200,
        val_shuffle_buffer: int = 0,
        return_metadata: bool = True,
        persistent_iterator: bool = True,
        # Legacy parameters (ignored but kept for config compatibility)
        name: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datasets: List of dataset configs (new format), each with:
                - name: Dataset name (e.g., "bridge", "language_table")
                - train_split: TFDS split for training
                - val_split: TFDS split for validation
                - weight: Sampling weight (default 1.0)
                - offset: Optional per-dataset offset override
            dataset_name: Name of OXE dataset (legacy single-dataset format)
            gcs_path: Override GCS path (optional, uses registry default)
            train_split: TFDS split for training (legacy format)
            val_split: TFDS split for validation (legacy format)
            offset: Default frame offset for pairs (in steps)
            image_size: Target image size (will resize)
            batch_size: Batch size for dataloaders
            num_workers: DataLoader workers (0 recommended for IterableDataset)
            shuffle_buffer: tf.data shuffle buffer size
            prefetch_buffer: tf.data prefetch buffer size
            return_metadata: If True, return dict with metadata
        """
        super().__init__()

        # Normalize config: convert legacy format to list-based format
        if datasets is not None:
            # New multi-dataset format
            self.dataset_configs = list(datasets)
        elif dataset_name is not None:
            # Legacy single-dataset format -> convert to list
            self.dataset_configs = [{
                "name": dataset_name,
                "train_split": train_split,
                "val_split": val_split,
                "weight": 1.0,
                "gcs_path": gcs_path,
            }]
        else:
            raise ValueError("Must provide either 'datasets' list or 'dataset_name'")

        self.offset = offset
        self.samples_per_episode = samples_per_episode
        self.sampling_seed = sampling_seed
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer = shuffle_buffer
        self.val_shuffle_buffer = val_shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.return_metadata = return_metadata
        self.persistent_iterator = persistent_iterator

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Create train and val datasets."""
        from common.adapters.oxe import OXEFramePairDataset, MultiOXEFramePairDataset

        if len(self.dataset_configs) == 1:
            # Single dataset - use simple implementation
            cfg = self.dataset_configs[0]
            self.train_dataset = OXEFramePairDataset(
                dataset_name=cfg["name"],
                gcs_path=cfg.get("gcs_path"),
                split=cfg.get("train_split", "train[:90%]"),
                offset=cfg.get("offset", self.offset),
                image_size=self.image_size,
                shuffle_buffer=self.shuffle_buffer,
                prefetch_buffer=self.prefetch_buffer,
                return_metadata=self.return_metadata,
                persistent_iterator=self.persistent_iterator,
                samples_per_episode=cfg.get("samples_per_episode", self.samples_per_episode),
                seed=cfg.get("seed", self.sampling_seed),
            )
            self.val_dataset = OXEFramePairDataset(
                dataset_name=cfg["name"],
                gcs_path=cfg.get("gcs_path"),
                split=cfg.get("val_split", "train[90%:]"),
                offset=cfg.get("offset", self.offset),
                image_size=self.image_size,
                shuffle_buffer=self.val_shuffle_buffer,
                prefetch_buffer=self.prefetch_buffer,
                return_metadata=self.return_metadata,
                persistent_iterator=self.persistent_iterator,
                samples_per_episode=cfg.get("samples_per_episode", self.samples_per_episode),
                seed=cfg.get("seed", self.sampling_seed),
            )
            logger.info(f"✓ OXE DataModule initialized (single dataset)")
            logger.info(f"  - Dataset: {cfg['name']}")
            logger.info(f"  - Train split: {cfg.get('train_split', 'train[:90%]')}")
            logger.info(f"  - Val split: {cfg.get('val_split', 'train[90%:]')}")
        else:
            # Multiple datasets - use interleaving
            self.train_dataset = MultiOXEFramePairDataset(
                datasets=self.dataset_configs,
                image_size=self.image_size,
                shuffle_buffer=self.shuffle_buffer,
                prefetch_buffer=self.prefetch_buffer,
                return_metadata=self.return_metadata,
                is_train=True,
                persistent_iterator=self.persistent_iterator,
                samples_per_episode=self.samples_per_episode,
                seed=self.sampling_seed,
            )
            self.val_dataset = MultiOXEFramePairDataset(
                datasets=self.dataset_configs,
                image_size=self.image_size,
                shuffle_buffer=self.val_shuffle_buffer,
                prefetch_buffer=self.prefetch_buffer,
                return_metadata=self.return_metadata,
                is_train=False,
                persistent_iterator=self.persistent_iterator,
                samples_per_episode=self.samples_per_episode,
                seed=self.sampling_seed,
            )
            dataset_names = [cfg["name"] for cfg in self.dataset_configs]
            logger.info(f"✓ OXE DataModule initialized (multi-dataset)")
            logger.info(f"  - Datasets: {', '.join(dataset_names)}")

        logger.info(f"  - Offset: {self.offset} steps")
        logger.info(f"  - Samples per episode: {self.samples_per_episode if self.samples_per_episode else 'all'}")
        logger.info(f"  - Sampling seed: {self.sampling_seed}")
        logger.info(f"  - Image size: {self.image_size}")
        logger.info(f"  - Shuffle buffer: {self.shuffle_buffer}")
        logger.info(f"  - Val shuffle buffer: {self.val_shuffle_buffer}")

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=oxe_collate_fn if self.return_metadata else None,
            # Note: shuffle=False because IterableDataset handles shuffling internally
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=oxe_collate_fn if self.return_metadata else None,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Clean up resources when training/validation ends.

        Explicitly releases TensorFlow resources from streaming datasets
        to prevent memory/file handle leaks.

        Uses extremely permissive error handling to ensure clean process exit
        even if TensorFlow/GCS is in a bad state (e.g., killed mid-step).

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict', or None)
        """
        for attr_name in ("train_dataset", "val_dataset"):
            try:
                dataset = getattr(self, attr_name, None)
                if dataset is not None:
                    dataset.cleanup()
            except BaseException:
                pass  # Ignore all errors during shutdown
            finally:
                try:
                    setattr(self, attr_name, None)
                except BaseException:
                    pass
