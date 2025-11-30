# Dataset Metadata Reference

Metadata preserved from each dataset type and how to use it for filtering and tracking.

## Metadata Structure per Dataset

### YouTube Dataset

**Source structure:**
```
youtube_new/
  ├── H6Yts-blLTk_stabilized/
  │   ├── scene_000_part_000/
  │   └── scenes.csv
  └── JNBtHDVoNQc_stabilized/
      └── scenes.csv
```

**SceneMetadata.extras:**
```python
{
    "dataset_type": "youtube",
    "dataset_name": "youtube_new",
    "video_id": "JNBtHDVoNQc_stabilized",  # From folder name
}
```

**Additional from scenes.csv:**
- `has_hands`: Boolean (from hand detection)
- `motion_mean`: Float (from motion tracking)
- Scene-specific extras from CSV columns

---

### BridgeV2 Dataset

**Source structure:**
```
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
```

**SceneMetadata.extras:**
```python
{
    # Dataset identification
    "dataset_type": "bridge",
    "dataset_name": "bridge_v2",

    # From folder hierarchy
    "robot": "datacol1",                    # Robot/datacol ID
    "environment": "toykitchen1",           # Environment name
    "task_category": "many_skills",         # Task category
    "task_id": "12",                        # Task ID (numeric or part of category)
    "timestamp": "2023-04-04_11-47-48",    # Trajectory timestamp
    "trajectory_path": "datacol1_toykitchen1/many_skills/12/2023-04-04_11-47-48",

    # From collection_metadata.json
    "camera_type": "Logitech C920",
    "policy_desc": "human demo",            # "human demo" or policy type
    "gripper": "default",
    "action_space": "x,y,z,roll,pitch,yaw,grasp_continuous",

    # From config.json
    "sequence_length": 50,                  # T parameter (trajectory length)
    "image_width": 640,
    "image_height": 480,
}
```

**Available environments:**
- toykitchen{1-7}
- folding_table
- robot_desk
- laundry_machine
- tabletop_dark_wood
- toysink{1-2}

**Available robots:**
- datacol{1-2}
- deepthought
- minsky

**Available task categories:**
- many_skills (multi-task)
- pnp_push_sweep
- stack_blocks
- fold_cloth
- drawer_pnp
- open/close_microwave
- sweep_granular

---

### OpenX Embodiment (Future)

**Expected structure:**
```
openx/
  └── {dataset_name}/
      └── episodes/
```

**SceneMetadata.extras (planned):**
```python
{
    "dataset_type": "openx",
    "dataset_name": "fractal20220817_data",  # Specific OpenX dataset
    "episode_id": "episode_000123",
    "task": "pick_and_place",
    "robot": "franka",
    # Additional OpenX-specific metadata
}
```

---

### Something-Something v2 (Future)

**Expected structure:**
```
s2s/
  └── {video_id}.webm
```

**SceneMetadata.extras (planned):**
```python
{
    "dataset_type": "something_something",
    "dataset_name": "s2s_v2",
    "video_id": "12345",
    "action_label": "putting something into something",
    # Additional S2S-specific metadata
}
```

---

## Filtering Examples

### Filter by Dataset Type

**Only YouTube data:**
```python
dm = LAQDataModule(
    sources=[
        {"type": "youtube", "root": "/path/to/youtube"},
        {"type": "bridge", "root": "/path/to/bridge"},
    ],
    filters={"dataset_type": "youtube"},  # Only YouTube scenes
)
```

**Only Bridge data:**
```python
dm = LAQDataModule(
    sources=[...],
    filters={"dataset_type": "bridge"},  # Only Bridge trajectories
)
```

### Filter Bridge by Environment

**Only toykitchen environments:**
```python
dm = LAQDataModule(
    sources=[{"type": "bridge", "root": "/path/to/bridge"}],
    filters={"environment": lambda env: "toykitchen" in env},
)
```

**Specific environment:**
```python
dm = LAQDataModule(
    sources=[{"type": "bridge", "root": "/path/to/bridge"}],
    filters={"environment": "toykitchen1"},
)
```

### Filter Bridge by Task

**Only manipulation tasks:**
```python
dm = LAQDataModule(
    sources=[{"type": "bridge", "root": "/path/to/bridge"}],
    filters={"task_category": ["pnp_push_sweep", "stack_blocks"]},
)
```

**Multi-task data only:**
```python
dm = LAQDataModule(
    sources=[{"type": "bridge", "root": "/path/to/bridge"}],
    filters={"task_category": "many_skills"},
)
```

### Filter Bridge by Robot

**Only datacol1:**
```python
dm = LAQDataModule(
    sources=[{"type": "bridge", "root": "/path/to/bridge"}],
    filters={"robot": "datacol1"},
)
```

**Exclude specific robots:**
```python
dm = LAQDataModule(
    sources=[{"type": "bridge", "root": "/path/to/bridge"}],
    filters={"robot": lambda r: r != "minsky"},  # Exclude minsky robot
)
```

### Filter by Policy Type

**Only human demonstrations:**
```python
dm = LAQDataModule(
    sources=[{"type": "bridge", "root": "/path/to/bridge"}],
    filters={"policy_desc": "human demo"},
)
```

### Combine Multiple Filters

**YouTube with hands + Bridge toykitchen:**
```python
dm = LAQDataModule(
    sources=[
        {"type": "youtube", "root": "/path/to/youtube"},
        {"type": "bridge", "root": "/path/to/bridge"},
    ],
    filters={
        # Apply to all datasets
        "dataset_type": ["youtube", "bridge"],
        # YouTube-specific (only applies to YouTube scenes)
        "has_hands": True,
        # Bridge-specific (only applies to Bridge scenes)
        "environment": lambda env: "toykitchen" in env,
    },
)
```

---

## Per-Source Filtering (Recommended)

Instead of global filters, specify filters per source:

```yaml
# config/data/multi_dataset.yaml
sources:
  - type: youtube
    root: /mnt/data/datasets/youtube_new
    filters:
      has_hands: true          # Only YouTube scenes with hands

  - type: bridge
    root: /mnt/data/datasets/bridgev2/raw/bridge_data_v2
    filters:
      environment: toykitchen1  # Only toykitchen1 environment
      task_category: many_skills # Only multi-task data
      policy_desc: human demo    # Only human demos
```

This gives fine-grained control per dataset.

---

## Using Metadata During Training

### Logging Dataset Distribution

```python
# In LAQTask or callback
def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    # Get batch metadata (if return_metadata=True)
    if isinstance(batch, dict) and "metadata" in batch:
        metadata_list = batch["metadata"]

        # Count dataset types in batch
        dataset_counts = {}
        for meta in metadata_list:
            dataset_type = meta["extras"]["dataset_type"]
            dataset_counts[dataset_type] = dataset_counts.get(dataset_type, 0) + 1

        # Log to WandB
        self.log_dict({
            f"batch/dataset_{k}_count": v
            for k, v in dataset_counts.items()
        })
```

### Task-Conditioned Training (Future)

Use metadata to condition the model on task/environment:

```python
# Extract task from metadata
task_category = scene.extras.get("task_category", "unknown")
environment = scene.extras.get("environment", "unknown")

# Create task embedding
task_embedding = task_encoder(task_category, environment)

# Condition model on task
output = model(frames, task_embedding=task_embedding)
```

### Dataset-Specific Evaluation

```python
# Evaluate separately per dataset
for dataset_type in ["youtube", "bridge"]:
    # Filter validation set
    val_subset = [
        s for s in val_dataset
        if s.extras["dataset_type"] == dataset_type
    ]

    # Evaluate
    metrics = evaluate(model, val_subset)

    # Log
    wandb.log({f"val/{dataset_type}/loss": metrics["loss"]})
```

---

## Metadata Access Patterns

### In DataLoader (if return_metadata=True)

```python
dm = LAQDataModule(
    sources=[...],
    return_metadata=True,  # Return dict instead of tensor
)

for batch in dm.train_dataloader():
    frames = batch["frames"]        # Tensor[B, C, 2, H, W]
    metadata = batch["metadata"]    # List[Dict] (length B)

    for meta in metadata:
        dataset_type = meta["extras"]["dataset_type"]
        if dataset_type == "bridge":
            task = meta["extras"]["task_category"]
            robot = meta["extras"]["robot"]
            # Use for conditioning
```

### In Dataset __getitem__

```python
class MetadataAwarePairDataset:
    def __getitem__(self, index):
        pair = self.pairs[index]
        scene = self.scenes[pair.scene_idx]

        # Access metadata
        dataset_type = scene.extras["dataset_type"]

        if dataset_type == "bridge":
            task = scene.extras["task_category"]
            # Load task-specific preprocessing

        # Load and return frames
        frames = self.load_frames(pair)

        if self.return_metadata:
            return {
                "frames": frames,
                "metadata": {
                    "scene_folder": scene.scene_folder,
                    "pair_offset": pair.offset,
                    "extras": scene.extras,
                },
            }
        else:
            return frames
```

---

## Summary

**Preserved metadata enables:**

1. **Rich filtering** - Select specific datasets, environments, tasks, robots
2. **Training visibility** - Track which datasets contribute to each batch
3. **Task conditioning** - Use task/environment metadata for conditioning (future)
4. **Per-dataset evaluation** - Evaluate model separately on each dataset type
5. **Debugging** - Identify issues specific to certain environments/tasks

**Metadata is cheap:**
- Stored in `SceneMetadata.extras` dict (Python dict, minimal overhead)
- Only parsed once during `setup()`
- No runtime cost during training (unless `return_metadata=True`)
