# Action-Frame Filtering (Low Motion / Low Action)

This repo supports configurable per-anchor filtering in the shared LeRobot-v3 pipeline (`packages/common`).

## Where it runs

- Source compile stage in `LeRobotSingleSource.compile`.
- Filtered anchors are materialized once, then reused by samplers in `packages/common/lerobot_v3_sampler.py`.

## Config keys

Configure under `data.filtering` (defaults in `config/data/_lerobot_v3_mix_base.yaml`):

- `enabled`: global on/off.
- `mode`: `none | motion | action | both`.
- `apply_at_sampling`: if true, sampler uses filtered anchors.
- `trim_episode_ends`: apply motion-based endpoint trimming.

Motion section (`data.filtering.motion`):

- `enabled`, `method` (`frame_diff | sparse_flow | two_stage`)
- `frame_gap` (`null` => inferred from requested image deltas)
- `aggregate_all_cameras` (compute motion on all available camera keys for this source)
- `aggregate_reduce` (`mean | max`) to build one aggregate motion score from per-camera scores
- `resize_short_side`, `blur_kernel`, `diff_pixel_threshold`
- `smoothing_window`, `consecutive_active_k`
- `low_threshold`, `high_threshold`, `use_hysteresis`

Action section (`data.filtering.action`):

- `enabled`, `threshold`, `exclude_dims`
- `chunk_size`, `chunk_reduce` (`max | mean`)
- `min_nonzero_ratio`

Semantics:
- `action.chunk_reduce` operates on the action chunk only (time/chunk dimension), not cameras.
- `chunk_reduce=max`: keep if any step in the chunk has high action norm (less aggressive drop).
- `chunk_reduce=mean`: keep only if sustained action across the chunk (more aggressive drop).
- Camera aggregation is controlled separately by `motion.aggregate_all_cameras` and `motion.aggregate_reduce`.

Cache section (`data.filtering.cache`):

- `enabled`
- `reuse_if_config_unchanged`
- `force_recompute`

## Mixed datasets behavior

- Sources with actions: action score is computed and can be used in `action` / `both` mode.
- Sources without actions (`action_key: null`): action criterion is skipped (anchors are not dropped because action is missing).
- Per-source override is supported via `dataset.lerobot.sources[i].filtering`.

## Camera selection behavior

- If `motion.aggregate_all_cameras=false` (default), motion score uses the source's requested primary camera.
- If `motion.aggregate_all_cameras=true`, motion score is computed over camera keys from the source `camera_map` in dataset YAML and aggregated using `motion.aggregate_reduce`.
- If `camera_map` has no keys, it falls back to all camera keys in dataset metadata.
- This camera aggregation affects motion keep/drop and trim logic.
- Action filtering does not use camera keys.

Implication for `lsy_teleop_only`:
- Current config defines four separate sources, each with one `primary` camera key.
- Filtering runs per source, so each source currently scores one camera.
- If you want one aggregate score across multiple perspectives for a single source, define multiple camera roles in the same source `camera_map`.

## Cache location and reuse

- Sidecar cache files are written to:
  - `<dataset_root>/meta/hlrp_action_frame_filter_cache/<split>_<camera_tag>_<fingerprint>.npz`
- Fingerprint includes relevant filtering config + source/delta settings + episode candidate bounds.
- Matching fingerprint => cache hit, scoring is skipped.

Notes:
- `camera_tag` is camera-specific for single-camera mode (for example `fpv`) and `allcamsN` for aggregated multi-camera mode.
- `train_*.npz` and `val_*.npz` are always separate files.
- With multi-source configs (for example one source per camera), you will have multiple files per split.

Quick way to locate cache files:

```bash
find ~/.cache -name "train_*.npz" -path "*hlrp_action_frame_filter_cache*" 2>/dev/null
find / -name "train_*.npz" -path "*hlrp_action_frame_filter_cache*" 2>/dev/null
```

## Debugging and tuning

### Config-first recommendation

Put stable filtering values directly in your dataset config (for example `config/data/lsy_teleop_only.yaml`) and avoid long CLI override chains.

Use temporary CLI overrides only for:
- `training.max_steps` / debug runtime controls
- `data.filtering.cache.force_recompute=true` when intentionally regenerating caches

### Commands

1. Run a short job to create/update cache files:

```bash
python scripts/2_train_stage1_lam.py \
  experiment=stage1_local \
  data=lsy_teleop_only \
  training.max_steps=50 \
  logging.use_wandb=false \
  data.loader.batch_size=16 \
  data.adapter.lerobot_v3.steps_per_epoch=500
```

2. Force cache regeneration after threshold changes:

```bash
python scripts/2_train_stage1_lam.py \
  experiment=stage1_local \
  data=lsy_teleop_only \
  training.max_steps=50 \
  data.filtering.cache.force_recompute=true
```

3. Plot a single cache file:

```bash
python scripts/plot_action_frame_filter_cache.py <path/to/cache_file.npz> --episode-row 0
```

4. Save a plot image while tuning:

```bash
python scripts/plot_action_frame_filter_cache.py <path/to/cache_file.npz> --episode-row 0 --save-path runs/debug/action_frame_filtering_ep0.png
```

5. Compare all cache files for one split in one figure:

```bash
python scripts/plot_action_frame_filter_cache.py \
  --cache-dir <dataset_root>/meta/hlrp_action_frame_filter_cache \
  --split train \
  --all-files \
  --episode-row 0 \
  --fps 15 \
  --save-path runs/debug/action_frame_filtering_all_train_ep0.png
```

The plot shows:
- per-camera `motion_raw` / `motion_smooth` traces (if present)
- aggregated `motion_raw(agg)` / `motion_smooth(<reduce>)`
- trim start/end markers
- motion threshold lines
- `action_score` (if present) + action threshold
- final `keep_mask` on a separate right y-axis (0/1)
- x-axis labels as `anchor_index` and `time_seconds`

Saved plots go exactly to the `--save-path` you pass.

### Recommended tuning order

1. Tune motion first:
- set `mode=motion`
- stabilize `low_threshold`, `high_threshold`, `aggregate_reduce`

2. Tune action second:
- switch to `mode=both`
- tune `threshold`, `chunk_size`, `chunk_reduce`, `min_nonzero_ratio`
- for this teleop dataset, `exclude_dims=[6]` is typically useful because gripper dominates norm scale

3. Lock deployment values in dataset config:
- keep `force_recompute=false`
- run a short confirmation job and validate cache hit/miss behavior in logs

## Deployment checklist

- Filtering values live in dataset config (not ad-hoc CLI).
- `force_recompute=false` for normal runs.
- Cache files exist for both train and val splits.
- Spot-check worst episodes by plotting high-drop rows.
- If moving to another dataset config, copy filtering block and rerun one forced recompute.

## Suggested quick validation commands

```bash
conda run -n hlrp ruff check packages/common/anchor_filtering.py packages/common/action_frame_filtering.py packages/common/lerobot_v3_source.py packages/common/lerobot_v3_sampler.py scripts/plot_action_frame_filter_cache.py
conda run -n hlrp pytest -q tests/test_common.py
```

## Disable entirely

Set:

```yaml
data:
  filtering:
    enabled: false
    mode: none
```
