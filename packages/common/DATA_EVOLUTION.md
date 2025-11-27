````markdown
# LAQ Data Pipeline Design Notes

This document captures the **target architecture**, our **current pragmatic direction**, and a **stepwise evolution plan** for the LAQ dataloader / data module.

---

## 1. Goals & Constraints

- **Unit of training**: a *pair* of frames `(frame_t, frame_{t+k})`.
- **Configurable temporal distance**: support multiple offsets `k` (e.g. 10, 30).
- **Scene-level structure**: scenes group frames and carry metadata; training operates on **pairs** derived from scenes.
- **Future modalities**:
  - Long-term motion tracks (numpy, optical flow over many frames).
  - Optical flow **between the two frames** used in the pair.
  - Potential masks / latent actions / tokens (LAPA-style).
- **Multi-source data**:
  - Different datasets, each with own videos/scenes (e.g. `youtube/u4tza9o4wct/scene_001/frame_000.jpg`).
- **Metadata backend**:
  - Currently `scenes.csv` (scene-level).
  - Later: possibly a **global Parquet** with scene/frame/pair-level metadata.
- **Storage backend**:
  - Currently JPEG.
  - Later: possibly `webm` (or similar) for faster sequential decoding.
- **Team reality**: 1–3 researchers, need **fast iteration** and **simple mental model**; avoid heavy overengineering but keep a path to a cleaner “almost optimal” state.

---

## 2. Conceptual “Perfect” Architecture (North Star)

### 2.1 Core abstraction: global *sample index* (pair-level)

- Define a table `sample_index`, where **each row = one training sample** (frame pair).
- Columns (conceptually):

  - Identification:
    - `source` (e.g. `"youtube"`, `"lab_dataset_x"`),
    - `video_id`,
    - `scene_id` (optional),
    - `frame_idx_t`,
    - `offset_k`,
    - `frame_idx_t2 = frame_idx_t + offset_k`.
  - Paths / storage keys:
    - `frame_key_t` (e.g. file path or video+frame index),
    - `frame_key_t2`,
    - `flow_key_t_t2` (optical flow),
    - `motion_track_key` (longer tracks),
    - optional mask keys, etc.
  - Metadata:
    - Scene-level: `stabilized_label`, `max_angle`, `max_trans`, etc.
    - Pair-level: future per-pair metrics (flow magnitude, quality flags, labels).
  - Source-specific tags: dataset name, split hints, etc.

- Physical representation:
  - Initially: Parquet or Arrow table (or pandas DataFrame) loaded into memory.
  - Later: could be sharded per dataset or split (train/val/test).

### 2.2 Storage backends

Keep IO separate from indexing:

```python
class FrameBackend:
    def load_frame(self, row, which: str) -> Image:
        ...

class FlowBackend:
    def load_flow(self, row) -> np.ndarray:
        ...

class MotionBackend:
    def load_motion(self, row) -> np.ndarray:
        ...
````

* Implementations:

  * `FrameBackendImage`: loads JPEG/PNG from disk.
  * `FrameBackendWebM`: decodes frames from a `webm` file (via `av`, `decord`, etc.).
  * `FlowBackendNpy`: loads `.npy` flow arrays.
  * `MotionBackendNpy`: loads motion tracks.

Dataset code doesn’t care about JPEG vs webm, only uses the backend.

### 2.3 Config-driven, modality-aware dataset

* Configuration object:

```python
@dataclass
class DataConfig:
    image_size: int
    use_flow: bool
    use_motion: bool
    return_metadata: bool
    # which offsets to include is encoded in sample_index,
    # or used earlier when building it
```

* Dataset:

```python
class ActionDataset(Dataset):
    def __init__(self, sample_index, frame_backend, flow_backend, motion_backend, config: DataConfig):
        ...

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        row = self.sample_index[idx]

        img1 = self.frame_backend.load_frame(row, which="t")
        img2 = self.frame_backend.load_frame(row, which="t2")
        # transforms → frames: [C, 2, H, W]

        sample = {"frames": frames}

        if self.config.use_flow and "flow_key_t_t2" in row:
            sample["flow"] = flow_tensor

        if self.config.use_motion and "motion_track_key" in row:
            sample["motion"] = motion_tensor

        if self.config.return_metadata:
            sample["metadata"] = metadata_dict

        return sample
```

### 2.4 Splitting & sampling

* Train/val split at **pair-level**:

  * `max_samples` = cap on number of pairs (rows) used.
  * `val_split` = fraction applied on pair count (e.g. 90/10).
* Optionally: a custom Sampler for:

  * Source balancing,
  * Offset balancing,
  * etc.

### 2.5 Scenes and CSV/Parquet

* `scenes.csv` or global Parquet are *inputs* for building `sample_index`.
* Scene-level metadata stays; pair-level index is derived from it.
* Switching from `scenes.csv` to global Parquet is a change in the *builder*, not in dataset logic.

---

## 3. Current Pragmatic Design (Pair-Level Dataset on Top of Scenes)

We keep current `SceneMetadata` and `SceneFilter`, but introduce explicit **pair indexing**.

### 3.1 `SceneMetadata` (existing)

* Represents one scene from `scenes.csv`.
* Contains:

  * Identification: `scene_idx`, `scene_folder`, `start_frame`, `end_frame`,
  * Labels: `label`, `stabilized_label`,
  * Motion metrics: `max_angle`, `max_trans`, etc.,
  * Pointers to masks, motion tracks, etc.,
  * `extras: Dict[str, Any]` for future fields.

### 3.2 Pair index abstraction

```python
@dataclass
class FramePairIndex:
    scene_idx: int
    first_frame_idx: int
    second_frame_idx: int
    offset: int
```

* `scene_idx` is an index into the internal `self.scenes` list.
* `first_frame_idx`, `second_frame_idx`, `offset` define the pair inside a scene.
* This reflects the **real training unit**.

### 3.3 `MetadataAwarePairDataset`

* Builds on top of `SceneMetadata` but enumerates **pairs** explicitly.

Key behavior:

1. Load scenes (from `scenes.csv` or folder structure).
2. Filter scenes using `SceneFilter` and `min_frames`.
3. For each scene and each requested `offset`:

   * Enumerate all valid `(first_idx, second_idx)` pairs with that offset:

     * `second_idx = min(first_idx + offset, last_frame)`.
   * Create `FramePairIndex` entries.
4. Store all pairs in `self.pairs`.

* `__len__()` returns the **number of pairs**.
* `__getitem__(idx)`:

  * Retrieves a `FramePairIndex`,
  * Loads `img1` and `img2` from disk using `scene.scene_folder` and indices,
  * Applies transforms, returns either:

    * `frames: Tensor [C, 2, H, W]`, or
    * Dict with `frames`, `scene_idx`, `first_frame_idx`, `second_frame_idx`, `offset`, and `metadata`.

This solves:

* Deterministic dataset length.
* Reproducibility.
* Ability to:

  * Overfit on a single pair (`max_samples=1`),
  * Precisely control sample count,
  * Reason about train/val splits.

### 3.4 Integration into `LAQDataModule`

* The DataModule keeps its existing init interface but adds:

  * `pair_level: bool` flag,
  * `offsets: List[int]` param (for pair-level mode).

* Behavior:

  * If `use_metadata` and `pair_level=True`:

    * Use `MetadataAwarePairDataset` with `offsets`.
  * If `use_metadata` and `pair_level=False`:

    * Use existing `MetadataAwareDataset` (scene-level, random pair inside).
  * If `use_metadata=False`:

    * Use `ImageVideoDataset` (legacy).

* `max_samples` now limits **number of pairs** when `pair_level=True`.

* `val_split` splits the subset of pairs into train/val indexes.

### 3.5 Overfitting on a single pair (immediate usage)

* Config example:

```yaml
use_metadata: true
pair_level: true
offsets: [30]

max_samples: 1    # exactly one pair
val_split: 0.0    # no val set

batch_size: 1
num_workers: 0
return_metadata: true
```

* Outcome:

  * `len(train_dataset) == 1`,
  * `len(val_dataset) == 0`,
  * Every batch sees the **same** frame pair.

---

## 4. Evolution Plan (Stepwise)

### Stage 0 – Now (done / in progress)

* Add `FramePairIndex` and `MetadataAwarePairDataset`.
* Add `pair_level` and `offsets` to `LAQDataModule`.
* Use this to **overfit on a single pair**.

### Stage 1 – Adopt pair-level as default for metadata-based training

* Prefer `pair_level=True` for normal experiments.
* Interpret `max_samples` and `val_split` as pair-level controls.
* Experiment with different `offsets`:

  * Single offset (`[10]`, `[30]`),
  * Multiple offsets (`[10, 30]`) in one dataset.
* Keep scene-level metadata in `SceneMetadata` and propagate it per pair.

### Stage 2 – Add optional modalities: flow & motion

* Extend `SceneMetadata` or its `extras` with fields pointing to:

  * Flow between `(frame_t, frame_{t+k})`,
  * Motion track data.
* In `MetadataAwarePairDataset.__getitem__`:

  * Guarded by flags like `use_flow`, `use_motion`,
  * Compose file paths from scene + indices,
  * Load numpy arrays, convert to tensors (`sample["flow"]`, `sample["motion"]`).
* Use these as:

  * Additional inputs,
  * Targets for reconstruction loss (VQ-VAE-like experiments).

### Stage 3 – Multi-source and richer metadata

* Evolve `scenes.csv` schema with:

  * `source`,
  * `video_id`,
  * Additional scene-level annotations.
* Keep `scene_folder` as a path-like string for now (e.g. `youtube/u4tza9o4wct/scene_001`).
* Use these fields to derive:

  * Frame paths,
  * Flow/motion paths, etc.
* Start thinking of `FramePairIndex` as the row index of a future `sample_index` table.

### Stage 4 – Global sample index (Parquet) & IO backends (if needed)

Triggered when:

* Number of pairs becomes very large **and/or**
* You need more efficient metadata queries, splits, or joins.

Actions:

1. Extract the logic that builds `self.pairs` into an offline process that creates a **global Parquet sample index**.
2. Replace internal `(scenes + pairs)` representation with:

   * `sample_index` table,
   * `ActionDataset(sample_index, frame_backend, flow_backend, motion_backend, config)`.
3. Factor out path-handling into `FrameBackend`, `FlowBackend`, `MotionBackend` so switching JPEG → webm is mostly backend work.

---

## 5. Key Design Principles (To Keep Us on Track)

* **Pair-level is the ground truth unit**:

  * Internally, always think in terms of frame pairs, not scenes.
* **Scenes are structural and metadata containers**:

  * Use scenes to organize data and store high-level labels;
  * Derive pairs from them.
* **Configurable temporal distance**:

  * Offsets (`k`) are explicit and configured (not hard-coded).
* **Optional modalities, not baked-in assumptions**:

  * Flow, motion, masks are *optional* fields in the sample.
* **Separation of concerns**:

  * Index / metadata (what samples exist?),
  * Backends (how to load raw data?),
  * Dataset (how to turn a row into tensors for training).
* **Iterative refinement**:

  * Implement minimal changes that align with the long-term shape,
  * Only introduce global Parquet, backends, etc. once there’s real need.

---


````markdown
## 6. Reference: Minimal Core Snippets

These are **not** full implementations, just the key shapes to keep things aligned.

### 6.1 `FramePairIndex`

```python
from dataclasses import dataclass

@dataclass
class FramePairIndex:
    scene_idx: int
    first_frame_idx: int
    second_frame_idx: int
    offset: int
````

### 6.2 Pair-level dataset (core idea)

```python
class MetadataAwarePairDataset(Dataset):
    def __init__(
        self,
        folder: str,
        image_size: int = 256,
        offsets: Optional[List[int]] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_frames: int = 2,
        return_metadata: bool = False,
    ):
        self.folder = Path(folder)
        self.image_size = image_size
        self.offsets = offsets or [30]
        self.return_metadata = return_metadata
        self.min_frames = min_frames

        # 1) scenes from csv / folders
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

        # 2) all pairs
        self.pairs = self._build_pairs(self.scenes, self.offsets)

        # 3) transforms
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])

    def _build_pairs(self, scenes, offsets) -> List[FramePairIndex]:
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
```

### 6.3 DataModule integration (essential part)

```python
class LAQDataModule(pl.LightningDataModule):
    def __init__(
        self,
        folder: str,
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
    ):
        ...
        self.pair_level = pair_level
        self.offsets = offsets or [offset]
        ...

    def setup(self, stage: Optional[str] = None):
        if self.use_metadata:
            if self.pair_level:
                full_dataset = MetadataAwarePairDataset(
                    folder=self.folder,
                    image_size=self.image_size,
                    offsets=self.offsets,
                    filters=self.filters,
                    min_frames=self.min_frames,
                    return_metadata=self.return_metadata,
                )
            else:
                full_dataset = MetadataAwareDataset(
                    folder=self.folder,
                    image_size=self.image_size,
                    offset=self.offset,
                    filters=self.filters,
                    return_metadata=self.return_metadata,
                    min_frames=self.min_frames,
                )
        else:
            full_dataset = ImageVideoDataset(
                folder=self.folder,
                image_size=self.image_size,
                offset=self.offset,
            )

        if self.max_samples is not None:
            indices = list(range(min(self.max_samples, len(full_dataset))))
            full_dataset = Subset(full_dataset, indices)

        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_size))

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
```

### 6.4 Debug / overfit config example

```yaml
name: debug_overfit_single_pair
task: laq

folder: /mnt/data/datasets/youtube_new/JNBtHDVoNQc_stabilized

image_size: 256

use_metadata: true
pair_level: true
offsets: [30]          # try [10], [30], etc.

max_samples: 1         # exactly one pair
val_split: 0.0         # no validation set

batch_size: 1
num_workers: 0
prefetch_factor: 2
pin_memory: true

return_metadata: true
filters: null
min_frames: 2
```

---

## 7. Quick Checklist (to stay on track)

* [ ] **Always think pair-level** as the core training unit.
* [ ] Keep `SceneMetadata` for structure + scene-level labels.
* [ ] Build an explicit list of `FramePairIndex` entries per dataset.
* [ ] Let `max_samples` and `val_split` operate at the **pair** level.
* [ ] Add flow/motion as **optional fields** in samples, toggled by config.
* [ ] Only move to global Parquet + backends if/when scale or flexibility requires it.
* [ ] Keep configs simple and explicit for offsets and modalities.

---

## 8. Scalability & Performance Notes

### 8.1 Pair count explosion

- Pairs scale as:  
  `#pairs ≈ Σ_scenes Σ_offsets (num_frames(scene) - offset)`
- For research-scale (<= few million pairs), keeping `self.pairs` in memory as a Python list is usually fine.
- If needed later:
  - Stop pair generation early when reaching `max_pairs`.
  - Or move to an implicit indexing scheme (per-scene prefix sums) instead of a full list.

### 8.2 I/O considerations

- Current: per-pair disk reads of JPEG/PNG.
- Potential future improvements (only if profiling shows I/O bottleneck):
  - Cache frame lists per scene (`_get_frame_files`) at init.
  - Use lighter-weight decode or smaller JPEGs.
  - Switch to video containers (e.g. `webm`) with a `FrameBackendWebM`.
  - Cache frequently used pairs for overfit / eval scenarios.

---

## 9. Future: Transition Toward the “Perfect” State

These steps are **optional** and should be driven by actual pain (scale, complexity).

### 9.1 Introduce a `sample_index` abstraction

- Extract `self.pairs` into a more general internal table:
  - Each entry at least: `(source, video_id, scene_id, frame_idx_t, offset, frame_idx_t2, ...)`.
- Treat this as the canonical “sample index” inside the codebase.
- This is a small step from `List[FramePairIndex]` toward a Parquet-style table.

### 9.2 Move sample_index to Parquet / Arrow (offline build)

- When datasets grow / become more heterogeneous:
  - Build `sample_index` offline from:
    - `scenes.csv`,
    - per-source metadata,
    - precomputed pair-level info (e.g. flow quality).
  - Store it as Parquet or Arrow.
- At training time:
  - Load required split’s Parquet,
  - Feed it into a generalized `ActionDataset`.

### 9.3 Introduce dedicated backends

- Once storage formats diverge (JPEG for some, `webm` for others):
  - Introduce `FrameBackend`, `FlowBackend`, `MotionBackend` with simple interfaces.
  - Map each row in `sample_index` to the correct backend and key.
- This makes changing codec / storage layout a local change (backend), not a dataset rewrite.

---

## 10. Design Invariants to Preserve

When changing or extending the system, keep these invariants:

1. **Index = pair**  
   - Dataset index corresponds to a concrete `(frame_t, frame_{t+k})` pair, not a scene.

2. **Offset is explicit and configurable**  
   - Offsets (temporal distances) are part of configuration and/or stored in the pair index.

3. **Scenes are structural, not the training unit**  
   - Scenes organize frames and hold metadata; training logic works on pairs derived from scenes.

4. **Optional modalities**  
   - Flow, motion, masks, etc. are additional fields that can be switched on/off via config.
   - Core image-pair path should remain simple and stable.

5. **Separation of concerns**  
   - Indexing / metadata definition separate from:
     - How data is stored (JPEG, `webm`),
     - How data is loaded (backends),
     - How samples are batched (DataModule, collate).

6. **Incremental change**  
   - Always prefer adjustments that keep the pair-level abstraction intact,
   - Add complexity (Parquet, backends) only when justified by real constraints.

---