# Dataset Adapter Architecture Plan

Plan for unifying multiple datasets (YouTube, BridgeV2, OpenX, S2S) through adapter pattern.

## Current State

### YouTube Structure
```
/mnt/data/datasets/youtube_new/
├── H6Yts-blLTk_stabilized/
│   ├── scene_000_part_000/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── scene_000_part_001/
│   └── scenes.csv
├── JNBtHDVoNQc_stabilized/
│   ├── scene_000_part_000/
│   └── scenes.csv
└── ...
```

**Current limitation**: Only loads single video folder
**Target**: Load all videos in `/mnt/data/datasets/youtube_new/`

### BridgeV2 Structure
```
/mnt/data/datasets/bridgev2/raw/
├── bridge_data_v1/
├── bridge_data_v2/
│   ├── datacol1_toykitchen1/
│   │   └── many_skills/
│   │       └── 12/
│   │           └── 2023-04-04_11-47-48/  ← trajectory (= scene)
│   │               ├── raw/traj_group0/traj4/images0/
│   │               │   ├── im_0.jpg
│   │               │   ├── im_1.jpg
│   │               │   └── ...
│   │               ├── config.json
│   │               └── collection_metadata.json
│   └── ...
├── icra/
└── rss/
```

**Challenge**: Deep nested hierarchy, no metadata CSV

---

## Proposed Architecture

### High-Level Flow

```
┌─────────────────┐
│ Config          │
│ sources:        │
│  - type: youtube│
│    root: ...    │
│  - type: bridge │
│    root: ...    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ LAQDataModule                   │
│ - Instantiates adapters         │
│ - Collects scenes from all      │
│ - Builds unified FramePairIndex │
└────────┬────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ Dataset Adapters (per dataset type)     │
│                                          │
│ YoutubeAdapter.collect_scenes()         │
│   → List[SceneMetadata]                 │
│                                          │
│ BridgeAdapter.collect_scenes()          │
│   → List[SceneMetadata]                 │
│                                          │
│ OpenXAdapter.collect_scenes()           │
│   → List[SceneMetadata]                 │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Unified SceneMetadata List   │
│ [SceneMetadata, ...]         │
│  - scene_folder (relative)   │
│  - num_frames               │
│  - has_hands                │
│  - motion_mean              │
│  - extras (dataset_type)    │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ FramePairIndex Builder       │
│ (existing _build_pairs)      │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ MetadataAwarePairDataset     │
│ __getitem__(idx)             │
│  → loads actual images       │
└──────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Core Adapter Infrastructure

**Files to create:**
- `packages/common/adapters/__init__.py`
- `packages/common/adapters/base.py`
- `packages/common/adapters/youtube.py`
- `packages/common/adapters/bridge.py`

**1. Base Adapter Interface** (`adapters/base.py`)

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from common.data import SceneMetadata

class DatasetAdapter(ABC):
    """Base class for dataset adapters."""

    @abstractmethod
    def collect_scenes(
        self,
        root: Path,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SceneMetadata]:
        """
        Scan dataset directory and return list of SceneMetadata.

        Args:
            root: Root directory of dataset
            filters: Optional per-source filters to apply
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
```

**2. YouTube Adapter** (`adapters/youtube.py`)

```python
class YoutubeAdapter(DatasetAdapter):
    """
    Adapter for YouTube video dataset.

    Structure:
        youtube_new/
          ├── video1_stabilized/
          │   ├── scene_000_part_000/
          │   ├── scene_000_part_001/
          │   └── scenes.csv
          └── video2_stabilized/
              └── scenes.csv
    """

    def collect_scenes(
        self,
        root: Path,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SceneMetadata]:
        """
        Scans all video folders, reads scenes.csv from each,
        and returns unified scene list with prefixed paths.
        """
        all_scenes = []

        # Find all video folders (containing scenes.csv)
        for video_dir in sorted(root.iterdir()):
            if not video_dir.is_dir():
                continue

            scenes_csv = video_dir / "scenes.csv"
            if not scenes_csv.exists():
                continue

            # Load scenes from this video
            video_scenes = load_scenes_csv(scenes_csv)

            # Prefix scene_folder with video directory name
            for scene in video_scenes:
                scene.scene_folder = f"{video_dir.name}/{scene.scene_folder}"
                scene.extras["video_id"] = video_dir.name
                scene.extras["dataset_type"] = "youtube"
                scene.extras["dataset_name"] = "youtube_new"
                all_scenes.append(scene)

        # Apply per-source filters if specified
        if filters:
            from common.data import SceneFilter
            filter_obj = SceneFilter(filters)
            all_scenes = filter_obj.filter_scenes(all_scenes)

        return all_scenes

    def get_frame_files(self, scene: SceneMetadata, root: Path) -> List[Path]:
        """Get frame files from scene folder."""
        scene_path = root / scene.scene_folder
        return sorted(scene_path.glob("*.jpg"))
```

**3. Bridge Adapter** (`adapters/bridge.py`)

```python
class BridgeAdapter(DatasetAdapter):
    """
    Adapter for BridgeV2 dataset.

    Structure:
        bridgev2/raw/
          ├── bridge_data_v1/
          ├── bridge_data_v2/
          │   └── datacol1_toykitchen1/
          │       └── many_skills/
          │           └── 12/
          │               └── 2023-04-04_11-47-48/  ← trajectory
          │                   └── raw/traj_group0/traj4/images0/
          │                       ├── im_0.jpg
          │                       └── ...
          └── ...

    Each trajectory folder is treated as a "scene".
    """

    def collect_scenes(
        self,
        root: Path,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SceneMetadata]:
        """
        Recursively scans Bridge directory structure and
        creates SceneMetadata for each trajectory.
        """
        all_scenes = []

        # Find all trajectory folders (those with images)
        for traj_dir in self._find_trajectory_dirs(root):
            # Get image files
            image_files = self._get_trajectory_images(traj_dir)

            if len(image_files) < 2:  # Skip empty trajectories
                continue

            # Relative path from root
            rel_path = traj_dir.relative_to(root)

            # Parse metadata from JSON files
            metadata = self._parse_metadata(traj_dir)

            # Parse folder hierarchy
            path_parts = rel_path.parts
            # Format: {robot}_{environment}/{task_category}/{task_id}/{timestamp}
            robot_env = path_parts[0]  # e.g., datacol1_toykitchen1
            task_category = path_parts[1] if len(path_parts) > 1 else "unknown"
            task_id = path_parts[2] if len(path_parts) > 2 else "unknown"
            timestamp = path_parts[3] if len(path_parts) > 3 else "unknown"

            # Split robot and environment
            robot_env_split = robot_env.split("_", 1)
            robot = robot_env_split[0] if len(robot_env_split) > 0 else "unknown"
            environment = robot_env_split[1] if len(robot_env_split) > 1 else "unknown"

            # Create SceneMetadata
            scene = SceneMetadata(
                scene_folder=str(rel_path),
                start_frame=0,
                end_frame=len(image_files) - 1,
                num_frames=len(image_files),
                has_hands=False,  # Unknown from images, could detect later
                motion_mean=0.0,  # Unknown, could compute from actions
                extras={
                    "dataset_type": "bridge",
                    "dataset_name": "bridge_v2",
                    "robot": robot,
                    "environment": environment,
                    "task_category": task_category,
                    "task_id": task_id,
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
                }
            )
            all_scenes.append(scene)

        # Apply per-source filters if specified
        if filters:
            from common.data import SceneFilter
            filter_obj = SceneFilter(filters)
            all_scenes = filter_obj.filter_scenes(all_scenes)

        return all_scenes

    def _find_trajectory_dirs(self, root: Path) -> List[Path]:
        """Find all directories containing trajectory images."""
        traj_dirs = []

        # Pattern: any folder with raw/traj_group*/traj*/images*/
        for path in root.rglob("raw/traj_group*/traj*/images*"):
            if path.is_dir():
                # Store parent (trajectory folder)
                traj_dir = path.parent.parent.parent.parent
                if traj_dir not in traj_dirs:
                    traj_dirs.append(traj_dir)

        return sorted(traj_dirs)

    def _get_trajectory_images(self, traj_dir: Path) -> List[Path]:
        """Get all images from a trajectory folder."""
        images = []
        for images_dir in (traj_dir / "raw").rglob("images*"):
            images.extend(images_dir.glob("im_*.jpg"))
        return sorted(images)

    def _parse_metadata(self, traj_dir: Path) -> Dict[str, Any]:
        """Parse metadata from JSON files in trajectory folder."""
        metadata = {}

        # Parse collection_metadata.json
        collection_meta_path = traj_dir / "collection_metadata.json"
        if collection_meta_path.exists():
            import json
            with open(collection_meta_path) as f:
                collection_meta = json.load(f)
                metadata["camera_type"] = collection_meta.get("camera_type")
                metadata["policy_desc"] = collection_meta.get("policy_desc")
                metadata["robot"] = collection_meta.get("robot")
                metadata["gripper"] = collection_meta.get("gripper")
                metadata["environment"] = collection_meta.get("environment")

        # Parse config.json
        config_path = traj_dir / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)
                if "agent" in config:
                    metadata["T"] = config["agent"].get("T")
                    metadata["image_height"] = config["agent"].get("image_height")
                    metadata["image_width"] = config["agent"].get("image_width")

        return metadata

    def get_frame_files(self, scene: SceneMetadata, root: Path) -> List[Path]:
        """Get frame files from trajectory."""
        traj_dir = root / scene.scene_folder
        return self._get_trajectory_images(traj_dir)
```

---

### Phase 2: Update LAQDataModule

**Modify `packages/common/data.py`:**

```python
from common.adapters import YoutubeAdapter, BridgeAdapter, OpenXAdapter

class LAQDataModule(pl.LightningDataModule):
    def __init__(
        self,
        # ... existing params ...

        # NEW: Multi-dataset support
        sources: Optional[List[Dict[str, Any]]] = None,
        # sources = [
        #     {"type": "youtube", "root": "/path/to/youtube"},
        #     {"type": "bridge", "root": "/path/to/bridge"},
        # ]

        # DEPRECATED (for backward compatibility)
        folder: Optional[str] = None,
        use_metadata: bool = True,

        # ... rest of params ...
    ):
        # Handle backward compatibility
        if sources is None and folder is not None:
            # Old API: single folder
            sources = [{"type": "youtube", "root": folder}]

        self.sources = sources
        # ... rest of init ...

    def setup(self, stage: Optional[str] = None):
        """Setup datasets from multiple sources."""

        # Collect scenes from all sources
        all_scenes = []

        for source_config in self.sources:
            source_type = source_config["type"]
            source_root = Path(source_config["root"])
            source_filters = source_config.get("filters", None)  # Per-source filters

            # Get appropriate adapter
            adapter = self._get_adapter(source_type)

            # Collect scenes (with per-source filtering)
            scenes = adapter.collect_scenes(source_root, filters=source_filters)

            print(f"✓ Loaded {len(scenes)} scenes from {source_type} ({source_root})")
            if source_filters:
                print(f"  - Applied {len(source_filters)} per-source filter(s)")
            all_scenes.extend(scenes)

        print(f"✓ Total scenes across all sources: {len(all_scenes)}")

        # Apply global filters if specified (in addition to per-source filters)
        if self.filters:
            filter_obj = SceneFilter(self.filters)
            all_scenes = filter_obj.filter_scenes(all_scenes)
            print(f"✓ After global filtering: {len(all_scenes)} scenes")

        # Rest is same as before: build pairs, split train/val, etc.
        # ...

    def _get_adapter(self, source_type: str) -> DatasetAdapter:
        """Get adapter for dataset type."""
        adapters = {
            "youtube": YoutubeAdapter(),
            "bridge": BridgeAdapter(),
            # "openx": OpenXAdapter(),
            # "s2s": SomethingSomethingAdapter(),
        }

        if source_type not in adapters:
            raise ValueError(
                f"Unknown dataset type: {source_type}. "
                f"Available: {list(adapters.keys())}"
            )

        return adapters[source_type]
```

---

### Phase 3: Update Config System

**Create new data config** (`config/data/multi_dataset.yaml`):

```yaml
name: multi_dataset
task: laq

# Multi-dataset sources
sources:
  - type: youtube
    root: /mnt/data/datasets/youtube_new
  - type: bridge
    root: /mnt/data/datasets/bridgev2/raw/bridge_data_v2

# Dataset settings
batch_size: 32
num_workers: 4
prefetch_factor: 2
pin_memory: true

# Frame pair settings
image_size: 256
offset: 30

# Subset and split
max_samples: null  # Use all
val_split: 0.1
sampling_strategy: random
sampling_seed: 42

# Dataset mode
pair_level: true
offsets: [30]

# Filtering (optional)
filters: null
return_metadata: false
min_frames: 2
```

**Update existing YouTube config** (`config/data/laq_pairs.yaml`):

```yaml
# Keep for backward compatibility
folder: /mnt/data/datasets/youtube_new/JNBtHDVoNQc_stabilized

# OR use new multi-video approach
sources:
  - type: youtube
    root: /mnt/data/datasets/youtube_new
```

---

### Phase 4: Testing Strategy

**Add tests** (`tests/common/test_adapters.py`):

```python
class TestYoutubeAdapter:
    def test_collect_scenes_single_video(self):
        """Test loading single video."""
        adapter = YoutubeAdapter()
        scenes = adapter.collect_scenes(Path("/mnt/data/datasets/youtube_new"))

        # Should find multiple videos
        assert len(scenes) > 0

        # Check video_id in extras
        video_ids = {s.extras["video_id"] for s in scenes}
        assert "JNBtHDVoNQc_stabilized" in video_ids

    def test_scene_folder_prefixed(self):
        """Test scene folders are prefixed with video name."""
        adapter = YoutubeAdapter()
        scenes = adapter.collect_scenes(Path("/mnt/data/datasets/youtube_new"))

        for scene in scenes:
            # Should be: video_name/scene_folder
            assert "/" in scene.scene_folder
            video_name, scene_name = scene.scene_folder.split("/", 1)
            assert "_stabilized" in video_name


class TestBridgeAdapter:
    def test_collect_scenes(self):
        """Test loading Bridge trajectories."""
        adapter = BridgeAdapter()
        root = Path("/mnt/data/datasets/bridgev2/raw/bridge_data_v2")

        if not root.exists():
            pytest.skip("Bridge dataset not found")

        scenes = adapter.collect_scenes(root)

        # Should find trajectories
        assert len(scenes) > 0

        # Check dataset_type
        for scene in scenes:
            assert scene.extras["dataset_type"] == "bridge"

    def test_get_frame_files(self):
        """Test loading frame files from trajectory."""
        adapter = BridgeAdapter()
        root = Path("/mnt/data/datasets/bridgev2/raw/bridge_data_v2")

        scenes = adapter.collect_scenes(root)
        if len(scenes) == 0:
            pytest.skip("No Bridge scenes found")

        # Get frames from first scene
        frames = adapter.get_frame_files(scenes[0], root)

        assert len(frames) > 0
        assert all(f.suffix == ".jpg" for f in frames)


class TestMultiDatasetLoading:
    def test_load_multiple_sources(self):
        """Test loading from YouTube + Bridge simultaneously."""
        dm = LAQDataModule(
            sources=[
                {"type": "youtube", "root": "/mnt/data/datasets/youtube_new"},
                {"type": "bridge", "root": "/mnt/data/datasets/bridgev2/raw/bridge_data_v2"},
            ],
            image_size=256,
            batch_size=4,
            pair_level=True,
            offsets=[30],
        )

        dm.setup()

        # Should have scenes from both datasets
        # Check dataset_type distribution
        dataset_types = [s.extras["dataset_type"] for s in dm.scenes]
        assert "youtube" in dataset_types
        assert "bridge" in dataset_types
```

---

## Migration Path

### Step 1: Implement adapters (backward compatible)
- Create adapter base class
- Implement YoutubeAdapter (works with existing structure)
- Update LAQDataModule to use adapters internally
- Keep `folder` parameter for backward compatibility

### Step 2: Test with YouTube multi-video
- Change `folder` to point to `/mnt/data/datasets/youtube_new` (parent)
- Verify all videos are loaded

### Step 3: Add BridgeV2 support
- Implement BridgeAdapter
- Test Bridge loading independently
- Create multi-dataset config

### Step 4: Production deployment
- Update experiment configs to use `sources` instead of `folder`
- Deprecate `folder` parameter (keep for compatibility)

---

## Metadata Preservation and Filtering

### SceneMetadata.extras for Each Dataset

**YouTube scenes:**
```python
extras = {
    "dataset_type": "youtube",
    "dataset_name": "youtube_new",
    "video_id": "JNBtHDVoNQc_stabilized",
}
```

**Bridge trajectories:**
```python
extras = {
    "dataset_type": "bridge",
    "dataset_name": "bridge_v2",
    "robot": "datacol1",
    "environment": "toykitchen1",
    "task_category": "many_skills",
    "task_id": "12",
    "timestamp": "2023-04-04_11-47-48",
    "camera_type": "Logitech C920",
    "policy_desc": "human demo",
    "gripper": "default",
    "sequence_length": 50,
    "image_width": 640,
    "image_height": 480,
}
```

### Filtering Examples

**Filter by dataset type:**
```python
# Only YouTube data
dm = LAQDataModule(
    sources=[...],
    filters={"dataset_type": "youtube"},
)

# Only Bridge data
dm = LAQDataModule(
    sources=[...],
    filters={"dataset_type": "bridge"},
)
```

**Filter by environment:**
```python
# Only toykitchen environments from Bridge
dm = LAQDataModule(
    sources=[{"type": "bridge", "root": ...}],
    filters={"environment": lambda env: "toykitchen" in env},
)
```

**Filter by task:**
```python
# Only manipulation tasks
dm = LAQDataModule(
    sources=[{"type": "bridge", "root": ...}],
    filters={"task_category": ["pnp_push_sweep", "stack_blocks"]},
)
```

**Filter by robot:**
```python
# Only datacol1 robots
dm = LAQDataModule(
    sources=[{"type": "bridge", "root": ...}],
    filters={"robot": "datacol1"},
)
```

**Combine datasets with per-source filtering:**
```yaml
sources:
  - type: youtube
    root: /mnt/data/datasets/youtube_new
    # All YouTube data

  - type: bridge
    root: /mnt/data/datasets/bridgev2/raw/bridge_data_v2
    filters:
      task_category: many_skills  # Only many_skills from Bridge
```

---

## Benefits

1. **No file reorganization needed**: Each dataset keeps its native structure
2. **Explicit configuration**: No magic auto-detection, clear source types
3. **Isolated dataset logic**: Each adapter is independent and testable
4. **Unified interface**: All datasets → SceneMetadata → FramePairIndex
5. **Easy to extend**: Adding new dataset = write new adapter
6. **Backward compatible**: Old `folder` parameter still works

---

## Future Extensions

### OpenX Embodiment
```python
class OpenXAdapter(DatasetAdapter):
    """Adapter for OpenX Embodiment dataset (RLDS format)."""
    # Handles tfrecords, episode structure, etc.
```

### Something-Something
```python
class SomethingSomethingAdapter(DatasetAdapter):
    """Adapter for Something-Something v2 dataset."""
    # Handles video-level structure
```

### Per-Dataset Filtering
```yaml
sources:
  - type: youtube
    root: /mnt/data/datasets/youtube_new
    filters:
      has_hands: true  # Only YouTube scenes with hands

  - type: bridge
    root: /mnt/data/datasets/bridgev2/raw
    # No filters for Bridge
```

---

## Next Steps

1. **Implement base adapter + YouTube adapter** (1-2 hours)
2. **Update LAQDataModule to use adapters** (1-2 hours)
3. **Test with YouTube multi-video** (30 min)
4. **Implement Bridge adapter** (2-3 hours, depends on metadata extraction)
5. **Add tests** (1 hour)
6. **Update configs and docs** (30 min)

**Total estimated time**: 1 day
