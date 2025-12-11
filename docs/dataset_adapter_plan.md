# Dataset Adapter Architecture

This document describes the dataset adapter system that unifies multiple datasets (YouTube, BridgeV2) through a common interface.

## Implementation Status

**Completed:**
- ✅ Base adapter interface
- ✅ YouTube adapter (multi-video support)
- ✅ Bridge adapter (50,000+ trajectories)
- ✅ Multi-source LAQDataModule integration
- ✅ Stratified train/val split
- ✅ Per-source and global filtering
- ✅ Metadata-based splits (hold-out environments, robots, etc.)
- ✅ Validation buckets for distribution shift analysis

## Dataset Structures

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
│   └── scenes.csv
└── ...
```

### BridgeV2 Structure  
```
/mnt/data/datasets/bridgev2/raw/
├── bridge_data_v1/
│   └── berkeley/
│       └── toykitchen1/                 ← environment
│           └── close_large4fbox_flaps/  ← task
│               └── 2021-07-30_14-36-57/ ← dated folder
│                   ├── collection_metadata.json
│                   └── raw/traj_group0/
│                       ├── traj0/       ← SCENE (trajectory)
│                       │   ├── images0/
│                       │   │   ├── im_0.jpg
│                       │   │   └── ...
│                       │   └── lang.txt
│                       ├── traj1/
│                       └── ...
├── bridge_data_v2/
│   └── datacol2_toykitchen2/
│       └── many_skills/
│           └── 00/
│               └── 2023-03-08_12-45-22/
│                   └── raw/traj_group0/traj*/
├── rss/
├── icra/
└── flap/
```

**Key insight**: Each `traj*` folder is ONE scene with its own images (50,419 total trajectories).

## Usage

### Multi-Dataset Config (YAML)
```yaml
# config/data/laq_multi_dataset.yaml
sources:
  - type: youtube
    root: /mnt/data/datasets/youtube_new
  - type: bridge
    root: /mnt/data/datasets/bridgev2/raw

batch_size: 32
val_split: 0.2
offsets: [30]
```

### Python API
```python
from common.data import LAQDataModule

dm = LAQDataModule(
    sources=[
        {"type": "youtube", "root": "/mnt/data/datasets/youtube_new"},
        {"type": "bridge", "root": "/mnt/data/datasets/bridgev2/raw"},
    ],
    batch_size=32,
    val_split=0.2,
    offsets=[30],
)
dm.setup()
# Train: 45,378 bridge + 429 youtube
# Val: 5,041 bridge + 107 youtube (stratified split)
```

### Per-Source Filtering
```yaml
sources:
  - type: youtube
    root: /mnt/data/datasets/youtube_new
    filters:
      contains_hand_sam3: true

  - type: bridge
    root: /mnt/data/datasets/bridgev2/raw
    filters:
      environment: toykitchen1
```

### Metadata-Based Validation Split
```yaml
split_mode: metadata
val_scene_filters:
  environment: "toykitchen7"  # Hold out this environment
```

### Validation Buckets
```yaml
val_buckets:
  youtube_only:
    dataset_type: "youtube"
  unseen_robot:
    robot: "minsky"
```

## Metadata Fields

### YouTube
```python
extras = {
    "dataset_type": "youtube",
    "dataset_name": "youtube_new",
    "video_id": "JNBtHDVoNQc_stabilized",
}
```

### Bridge
```python
extras = {
    "dataset_type": "bridge",
    "dataset_name": "bridge_data_v2",  # or bridge_data_v1, rss, icra, flap
    "robot": "widowx",
    "environment": "toykitchen2",
    "task": "many_skills",
    "institution": "datacol2",
    "traj_id": "traj5",
    "language": "put the bowl on the plate",
    "num_frames": 45,
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ LAQDataModule                                           │
│ - Instantiates adapters per source                      │
│ - Collects scenes from all adapters                     │
│ - Applies stratified train/val split                    │
│ - Builds unified FramePairIndex                         │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌─────────────────────┐         ┌─────────────────────┐
│ YoutubeAdapter      │         │ BridgeAdapter       │
│ - Scans video dirs  │         │ - Scans traj dirs   │
│ - Reads scenes.csv  │         │ - Parses JSON meta  │
│ - Prefixes paths    │         │ - Extracts lang.txt │
└─────────────────────┘         └─────────────────────┘
        │                                 │
        └────────────────┬────────────────┘
                         ▼
              ┌──────────────────────────┐
              │ List[SceneMetadata]      │
              │ - scene_folder           │
              │ - num_frames             │
              │ - extras (dataset_type)  │
              └──────────────────────────┘
```

## Dataset Statistics

| Dataset | Scenes | Environments | 
|---------|--------|--------------|
| YouTube | 536 | N/A |
| Bridge (total) | 50,419 | 26 |
| - bridge_data_v1 | 13,192 | |
| - bridge_data_v2 | 24,827 | |
| - rss | 9,161 | |
| - icra | 2,294 | |
| - flap | 945 | |

Top environments in Bridge:
- tabletop: 15,388
- toykitchen2: 13,997
- toysink3: 10,557
- toykitchen_bww: 2,269
- toykitchen1: 2,178
