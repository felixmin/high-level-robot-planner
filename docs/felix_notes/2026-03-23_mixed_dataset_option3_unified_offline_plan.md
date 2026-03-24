# Mixed Dataset Option 3: Unified Offline Dataset / Unified Manifest

This note describes the detailed long-term plan for replacing runtime mixing of source-native datasets with a canonical unified sample space.

This is the cleanest long-term systems design if heterogeneous mixed-data training remains a core path in this repo.

## Position

Treat this as an **offline ingest pipeline**, not a runtime loader tweak.

The right first milestone is:
- unified manifest first
- media rewrite later, only if needed

That means:
- keep source media in place initially
- build one canonical sample index and schema layer
- make runtime training read one normalized sample space
- only materialize rewritten canonical shards if the manifest-first path proves worthwhile

## Goal

Replace runtime mixing of source-native datasets with:
1. a canonical sample schema
2. a unified manifest
3. a small runtime loader
4. a single weighted sampler over normalized samples

Runtime training should see:
- one dataset object
- one manifest-backed sample space
- one DDP sharding story
- one sampler for weights and splits
- no per-source branching in the hot path except lightweight source adapters

## Canonical Unit

Use a **training sample** as the unit, not a raw source row.

A sample should represent:
- source dataset id
- episode id
- anchor time or frame
- resolved temporal window
- normalized observation/action access info
- provenance

This is better than raw row-level unification because training semantics already depend on:
- lookback / lookahead
- `future_seconds`
- fps
- per-source validity windows

## Canonical Schema

Use two layers.

### 1. Global manifest row
One row per effective training anchor.

Suggested fields:
- `sample_id: int64`
- `source_id: int32`
- `source_name: string`
- `episode_id_global: int64`
- `episode_id_source: int64`
- `anchor_index: int32`
- `anchor_timestamp_s: float32`
- `fps: float32`
- `num_frames_episode: int32`
- `split: uint8`
- `task_id: int32 | -1`
- `instruction_id: int32 | -1`
- `weight_group_id: int16`
- `valid_future_dt_s_max: float32`
- `valid_history_dt_s_max: float32`
- `has_rgb_main: bool`
- `has_state: bool`
- `has_language: bool`
- `canonical_action_family: uint8`
- `source_action_family: uint8`
- `payload_ref_id: int64`
- `source_local_index: int64`
- `flags: bitmask`

Purpose:
- enough to sample, shard, validate, and resolve temporal windows
- compact and memory-mappable
- no large Python objects

### 2. Per-source adapter metadata
Stored separately per source:
- canonical image key mapping
- source observation key mapping
- source action adapter config
- source timestamp semantics
- source decode-backend hints
- feature retention / dropped-field metadata
- optional instruction/task lookup tables

This prevents duplicating source-specific metadata on every manifest row.

## Canonical Runtime Sample Contract

At runtime every sample should resolve to:
- `observation.images.image`
- optional extra canonical cameras if needed later
- `observation.state`
- `action`
- `language.instruction`
- `meta.source_name`
- `meta.source_id`
- `meta.fps`
- `meta.timestamp_s`
- `meta.sample_id`

Anything not universally available should be:
- explicit optional fields
- explicit masks
- not hidden by implicit branching

## Runtime Objects And Interfaces

Option 3 only works if the runtime objects are simple and the source-specific complexity is pushed into ingest-time artifacts and small adapters.

### Offline/build-time objects

These exist during ingest/build, not during training:

- `SourceRegistry`
  - one entry per source dataset
  - stores source root, split rules, schema adapter choice, action family, timestamp policy, weighting group
- `SourceScanner`
  - scans one raw source dataset
  - emits per-episode and per-source metadata
- `SourceAdapterSpec`
  - records how one source maps into the canonical schema
  - image key mapping, state mapping, action adapter choice, task/instruction mapping
- `UnifiedManifestBuilder`
  - computes valid anchors
  - resolves temporal semantics
  - writes the unified manifest
- optional later `CanonicalShardWriter`
  - materializes canonical training shards if manifest-first is not enough

### Runtime objects

These exist when training starts:

- `UnifiedDataset`
  - one dataset object over the unified sample space
  - owns a compact manifest view and lightweight source adapters
- `ManifestView`
  - memory-mapped or compact in-memory arrays loaded from the manifest
- `RuntimeSourceAdapter`
  - per-source fetch helper
  - resolves source-local payload fetches into the canonical runtime sample contract
- `UnifiedSampler`
  - applies weights / quotas over the global sample space
- rank sharder
  - standard `DistributedSampler` or custom rank-aware weighted sampler
- `DataLoader`
  - standard PyTorch loader over `UnifiedDataset`

### Runtime dataset interface

The runtime API should be intentionally boring:

```python
class UnifiedDataset(Dataset):
    def __len__(self) -> int: ...
    def __getitem__(self, sample_id: int) -> dict: ...
    def __getitems__(self, sample_ids: list[int]) -> list[dict]: ...
```

The returned canonical sample should be shaped like:

```python
{
    "observation.images.image": ...,
    "observation.state": ...,
    "action": ...,
    "language.instruction": ...,
    "meta.sample_id": ...,
    "meta.source_id": ...,
    "meta.source_name": ...,
    "meta.timestamp_s": ...,
    "meta.fps": ...,
    "meta.flags": ...,
}
```

The trainer should not see source-native branching.

## Per-Source Adapter Metadata Schema

The per-source adapter metadata must be explicit and versioned.

Suggested adapter metadata fields:

- `source_id`
- `source_name`
- `repo_id`
- `root`
- `revision`
- `split_policy`
- `timestamp_mode`
- `fps`
- `rounding_mode`
- `image_key_map`
- `state_key_map`
- `action_adapter_name`
- `action_family`
- `language_field_map`
- `task_field_map`
- `missing_modality_policy`
- `decode_backend_hint`
- `feature_drop_policy`
- `capability_flags`

This metadata should live outside the main manifest rows and be loaded once.

## Sample Identity

Each effective training sample should have one global `sample_id`.

That global `sample_id` is the runtime identity used by:

- the sampler
- rank sharding
- dataset fetch
- logging / debugging

Lookup flow:

- `sample_id`
- `-> manifest row`
- `-> source_id, episode_id, anchor`
- `-> runtime source adapter`
- `-> canonical sample dict`

This is cleaner than runtime source routing through many source-native datasets.

## Sampler And DDP Rank Sharding

This is one of the main benefits of Option 3.

Because there is one global sample space, rank handling becomes straightforward.

### Variant A: standard distributed sampler

Use when:
- weighting is simple
- or an epoch plan is already compiled into the sample space

Pattern:
- one global dataset length
- standard `DistributedSampler`
- `set_epoch()` each epoch

### Variant B: custom rank-aware weighted sampler

Use when:
- source weights / quotas remain dynamic at runtime
- exact weighted behavior matters

Suggested interface:

```python
class UnifiedWeightedSampler(Sampler[int]):
    def __init__(
        self,
        manifest_view,
        source_weights,
        seed: int,
        epoch: int,
        rank: int,
        world_size: int,
        mode: str = "source_weighted_replacement",
    ): ...
```

Recommended rule:

1. generate a deterministic global draw stream
2. each draw chooses:
   - source or weight-group
   - then a valid sample in that group
3. rank `r` consumes positions where `global_index % world_size == r`

Requirements:

- no rank should discover invalidity late
- all validity filtering must be resolved in the manifest or sampler before training
- every rank should get a compatible number of steps for the epoch

## Runtime Fetch Flow

For one batch:

1. sampler emits global `sample_id`s
2. rank sharder assigns ids to the current DDP rank
3. DataLoader worker receives batch indices
4. `UnifiedDataset.__getitems__(sample_ids)` looks up manifest rows
5. rows are grouped by `source_id`
6. corresponding `RuntimeSourceAdapter.fetch_many(...)` calls are made
7. returned canonical samples are restored to the requested order
8. collate runs
9. batch goes to the trainer

### Suggested adapter interface

```python
class RuntimeSourceAdapter:
    def fetch_one(self, episode_id: int, anchor: int) -> dict: ...
    def fetch_many(self, requests: list[tuple[int, int]]) -> list[dict]: ...
```

Inside `fetch_many()` it is fine to:

- bucket by episode
- bucket by video path
- reuse video readers
- apply source-local remapping
- return only canonical sample dicts

## Sequence Diagram

```text
Offline Build Phase
-------------------
SourceRegistry
    -> SourceScanner(source A): scan episodes / frames / timestamps
    -> SourceScanner(source B): scan episodes / frames / timestamps
    -> SourceScanner(source C): scan episodes / frames / timestamps

SourceScanner(*)
    -> SourceAdapterSpec: normalize keys / action family / fps policy
    -> EpisodeTable: write per-episode metadata

UnifiedManifestBuilder
    -> EpisodeTable: read all sources
    -> compute valid anchors and temporal windows
    -> assign global sample_id
    -> write UnifiedManifest
    -> write SourcesTable / AdapterMetadata

Runtime Training Phase
----------------------
Trainer
    -> UnifiedDataset: open manifest + source metadata
    -> UnifiedSampler: create epoch sample stream
    -> RankSharder: shard sample_ids by rank
    -> DataLoader: assign sample_ids to workers

Worker
    -> UnifiedDataset.__getitems__(sample_ids)
    -> ManifestView: lookup rows for sample_ids
    -> group rows by source_id
    -> RuntimeSourceAdapter(source A): fetch_many(...)
    -> RuntimeSourceAdapter(source B): fetch_many(...)
    -> RuntimeSourceAdapter(source C): fetch_many(...)
    -> UnifiedDataset: restore original order
    -> DataLoader collate
    -> Trainer: forward / backward
```

## What Phase 1 Should Omit

The first manifest-first implementation should stay narrow.

Do not include yet:

- full media rewrite
- canonical shard writing
- multi-family action support in one training run
- broad source-specific optional-label explosion
- heavy preprocessing beyond what is needed for canonical manifest correctness
- premature storage optimization

Phase 1 should prove:

- compact parent/runtime state
- simpler rank handling
- simpler weighted sampling
- lower worker RSS or cleaner runtime behavior

## Storage Format

### Phase 1: Manifest-first, media stays in place
Use:
- `manifest.parquet` or Arrow IPC
- per-source metadata tables
- per-source episode tables
- source media still referenced by original paths

Recommended files:
- `manifest.parquet`
- `sources.parquet`
- `episodes.parquet`
- `tasks.parquet`
- `instructions.parquet`
- `source_adapters.yaml`

This is the lowest-risk first implementation.

### Phase 2: Canonical training shards, if needed
Only if Phase 1 still leaves too much decode/IO overhead.

Possible shape:
- `manifest.parquet`
- `shards/shard-000001.idx`
- `shards/shard-000001.bin`

Or Arrow/Parquet + external blob layout if simpler.

Shard contents could include:
- resized / canonicalized images or references
- canonical action vectors
- canonical state vectors
- text ids or text references
- validity masks

Guidelines:
- immutable shards
- 512 MB to 2 GB shard size
- contiguous reads favored over many tiny files

Do not start here unless Phase 1 is clearly worth it.

## Ingest Pipeline Stages

### Stage 0: Source registration
Per source define:
- source id
- root/repo
- split rules
- canonical key mapping
- action adapter
- fps/timestamp policy
- modality presence
- weighting group
- inclusion filters

### Stage 1: Source scan
Extract:
- episodes
- frame counts
- timestamps if present
- fps fallback if timestamps absent
- observation/action availability
- language/task annotations
- per-episode validity ranges

Outputs:
- `episodes.parquet`
- source-local index tables

### Stage 2: Canonicalization
Normalize into canonical sample semantics:
- map observation keys
- map action schema
- resolve timestamps/fps
- compute valid anchor ranges for target temporal settings
- assign global episode ids
- assign source/task ids
- assign split ids

### Stage 3: Manifest build
Build one global manifest row per effective anchor.

At this stage:
- anchors that can never satisfy temporal constraints are excluded
- split assignment is frozen
- source/task weights are attached through compact ids

### Stage 4: Validation
Run strict checks:
- no invalid anchors
- timestamp monotonicity
- fps consistency
- action dimension consistency per action family
- split leakage checks
- modality presence consistency
- sample counts per source / split / task

### Stage 5: Optional materialization
Only if needed:
- rewrite images/clips
- precompute resized tensors or compact image records
- precompute canonical action/state tensors

## Weighting Support

Do not store weights as repeated per-row floats unless necessary.

### A. Source weights
In `sources.parquet`:
- `source_id`
- `base_weight`

### B. Optional group weights
In a `weight_groups` table:
- `weight_group_id`
- `source_id`
- `task_id`
- `split`
- `weight`

### C. Runtime sampling modes
Support:
1. `source_weighted_replacement`
2. `source_quota_epoch`
3. later `task_or_group_weighted`

For first implementation, support:
- source-weighted replacement
- source-quota epoch

## FPS / `future_seconds` Handling

This is one of the strongest reasons to unify.

Rule:
- use timestamps if available
- otherwise use trusted per-source fps

Per source define:
- `timestamp_mode = explicit | inferred_fps`
- `fps`
- rounding rule for temporal offsets

Per manifest row store or make resolvable:
- anchor timestamp
- valid future horizon
- valid history horizon

At runtime, `future_seconds=0.5` becomes a pure lookup or simple source-specific rule.

Do not leave expensive temporal validity logic in the hot path.

## Action / Schema Alignment

This is the hardest part.

Do not assume all sources can be flattened safely into one naive action vector.

Recommended support:
- `source_action_family`
- `canonical_action_family`

Examples:
- cartesian delta pose + gripper
- joint delta
- joint absolute
- velocity-style
- discrete / hybrid

Recommended starting policy:
- strict canonical action family per training run

That means:
- only include sources compatible with the target family initially
- keep observation schema alignment broader and easier

Language/task alignment can store:
- raw string
- optional normalized task id
- optional instruction id

## Incremental Plan

### Phase 1
Build:
- source registry
- episodes table
- manifest table
- source adapters
- runtime loader over source-native media

Do not rewrite media.

Expected win:
- simpler runtime mixer
- compact parent dataset state
- cleaner DDP and weighting

### Phase 2
Benchmark against current runtime mixing.

### Phase 3
Optional canonical media shards if Phase 1 still leaves too much decode/IO overhead.

### Phase 4
Operationalize the build pipeline:
- reproducible build command
- validation reports
- manifest versioning
- partial rebuild support

## Benchmark Plan

Compare:
- current patched runtime mixer
- Option 2 if available
- unified manifest loader

Metrics:
- parent RSS
- worker RSS total / peak
- startup time
- first-batch time
- steady-state samples/s
- p50 / p95 batch time
- source/task exposure histogram
- DDP stability
- implementation complexity / maintenance burden

Minimal matrix:
- same mix
- same batch size
- same future horizon
- same image size
- same backend
- same seed
- test `num_workers=4` and `10`
- then one short training smoke for each

## Success Criteria

A strong win would be:
- `30%+` lower worker RSS
- no mixed-worker failure mode
- equal or better steady-state throughput
- simpler runtime logic
- cleaner deterministic weighting and DDP behavior

## Failure Criteria

This path is not worth continuing if:
- action/schema alignment cost dominates everything
- runtime still depends on a lot of source-specific branching
- throughput does not improve meaningfully over a compact runtime mixer
- ingest cost is too high relative to how often sources change

## Recommendation

Do not jump straight to full canonical-media rewrite.

Do:
1. unified manifest prototype first
2. keep source media in place
3. use source adapters for actual reads
4. benchmark against the current patched runtime path
5. only then decide whether full canonical shards are worth building

Short term:
- keep current patched path as fallback

Next principled experiment:
- unified manifest prototype

Long term:
- if mixed training remains important, this is probably the cleanest end state
