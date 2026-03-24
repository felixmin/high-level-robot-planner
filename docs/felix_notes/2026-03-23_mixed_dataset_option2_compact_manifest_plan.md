# Mixed Dataset Option 2: Compact Manifest + Sample-Level Mixing + `__getitems__`

This note describes the detailed prototype plan for the strongest runtime alternative to block-local scheduling:

- keep sample-level weighted mixing semantics
- make the parent mixed dataset compact and mostly immutable
- use tiny per-source adapters instead of heavy runtime ownership
- implement `__getitems__` so a worker can coalesce a mixed batch by source under the hood

This is the most principled runtime path if preserving training semantics matters.

Related overview notes:

- [2026-03-23_mixed_dataset_design_options.md](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-23_mixed_dataset_design_options.md)
- [2026-03-23_mixed_dataset_option2_prototype_plan.md](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-23_mixed_dataset_option2_prototype_plan.md)

## Position

This option is not the cheapest.

But it is the strongest runtime alternative if the goal is:
- keep true sample-level weighting
- keep genuinely mixed batches
- fix the memory model more directly than the current `mixed_dataset.py`

## Goal

Keep these properties:
- source weights apply at sample level
- batches may still contain samples from multiple datasets
- visible training behavior stays close to the current intent

Change these properties:
- parent mixed dataset becomes compact
- workers stop owning many full per-source `LeRobotDataset` objects
- rank sharding becomes explicit in the sampler
- batch fetch becomes source-coalesced internally through `__getitems__`

## Architecture

Use four pieces.

### 1. `CompactMixedManifest`

One compact global index over the mixed training space.

It stores enough to map a mixed sample id to a source-local fetch.

Suggested runtime fields:
- `source_id: uint16`
- `episode_id: int32`
- `anchor: int32`
- `anchor_valid_end: int32` or equivalent
- `split_id: uint8`
- `fps_or_timebase: float32` or an index into per-source metadata
- optional `task_group_id: uint16`
- optional `weight_group_id: uint16`

Storage:
- NumPy arrays, `np.memmap`, or equivalent compact immutable arrays

Avoid:
- dict-of-dicts
- Pandas in the hot path
- per-entry Python objects

### 2. `SourceAdapter`

A tiny source-specific reader abstraction.

Responsibilities:
- resolve `(episode_id, anchor)` into a source-local fetch
- own compact source metadata only
- perform source-specific key remapping / temporal resolution
- expose batch-friendly fetch

Suggested interface:

```python
class SourceAdapter:
    def fetch_one(self, episode_id: int, anchor: int) -> dict: ...
    def fetch_many(self, requests: list[tuple[int, int]]) -> list[dict]: ...
```

Worker-local behavior:
- tiny LRU of open readers / decoders
- target `1-2` warm sources per worker initially

### 3. `RankShardedWeightedSampler`

A sample-level sampler that is deterministic and rank-aware.

Responsibilities:
- apply source weights
- emit sample-level mixed indices
- be deterministic by `seed`, `epoch`, `rank`, `world_size`
- ensure each rank gets valid sample ids without post-hoc skipping

Recommended rule:
- build or stream a deterministic global draw stream
- global position `i` belongs to rank `i % world_size`
- validity must be resolved before rank consumption

### 4. `MixedDataset.__getitems__(indices)`

This is the core optimization.

Flow:
1. receive a list of mixed sample ids for a batch
2. look up manifest rows
3. build request records `(batch_pos, source_id, episode_id, anchor)`
4. group requests by `source_id`
5. call `adapter.fetch_many(...)` per source
6. scatter fetched samples back into original batch order
7. return the batch items in original order

This preserves the logical mixed batch while improving physical fetch locality.

## Manifest Layout

Split the manifest into two levels.

### Global arrays
Per effective sample entry:
- `sample_source_id`
- `sample_episode_id`
- `sample_anchor`
- optional `sample_split_id`
- optional `sample_group_id`

### Per-source metadata table
Per source:
- `repo_id`
- `root`
- `fps`
- `future_seconds` metadata
- observation/action key mappings
- effective episode lengths
- episode start offsets
- source capability flags

This avoids repeating source metadata per sample.

## Source Adapter Responsibilities

Source adapters should own:
- source root / repo id / revision
- compact episode-level metadata
- camera/key mapping
- temporal lookup rules
- tiny reader / decoder cache

They should not own:
- weighting logic
- rank logic
- batch composition logic
- a large mixed-layer cache

Preferred behavior inside `fetch_many()`:
- bucket requests by episode / file path if useful
- reuse video readers where possible
- resolve remapping there
- return canonical sample dicts ready for collation

## Files / Classes To Change

Likely files:
- [mixed_dataset.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/mixed_dataset.py)
- [sampler.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/sampler.py)
- [lerobot_train.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_train.py)

Prefer one new file at most:
- `compact_mixed_dataset.py`

Possible class split:
- `CompactMixedManifest`
- `CompactSourceAdapter`
- `CompactMixedDataset`

Keep:
- current mix parsing if it is already correct
- current patched path as fallback
- current benchmark harness

Stop relying on:
- `LogicalSource` owning many heavyweight per-source datasets
- repeated runtime ownership of full `LeRobotDataset` instances in the mixed layer
- implicit downstream rank slicing as the main sharding mechanism

## Implementation Plan

### Phase 1: Compact manifest only
- build a compact manifest object from existing mix definitions
- keep current fetch path temporarily
- benchmark parent RSS and worker RSS

Goal:
- prove parent-process memory drops

### Phase 2: Tiny source adapters
- introduce `SourceAdapter`
- stop the mixed path from keeping many full `LeRobotDataset`s alive
- keep `__getitem__` path working first

Goal:
- reduce per-worker replicated state

### Phase 3: `__getitems__`
- implement grouped fetch by source inside one batch
- add `fetch_many()` per source
- benchmark again

Goal:
- recover locality without changing visible batch semantics

### Phase 4: Rank-aware sampler
- move rank logic into the sampler
- make epoch/rank sharding explicit and deterministic

Goal:
- remove post-hoc ambiguity and make mixed DDP behavior debuggable

## Benchmark Plan

### Candidate A
Current patched path:
- `source_block_size`
- `prefetch_factor=1`
- current mixed dataset

### Candidate B
Option 2 path:
- compact manifest
- sample-level weighted sampler
- `__getitems__` source-coalesced fetch

### Matrix
Same:
- mix
- batch size
- image size
- backend
- seed

Run:
- A, `num_workers=4`
- A, `num_workers=10`
- B, `num_workers=4`
- B, `num_workers=10`

Then a short train smoke for A and B.

### Metrics
- parent RSS
- total worker RSS
- peak worker RSS
- steady-state samples/s
- p50/p95 batch time
- source diversity per batch
- source histogram over time
- DDP stability in short multi-process smoke

## Success Criteria

Strong win:
- `30%+` lower worker RSS than A
- same or better steady-state samples/s
- mixed batches remain genuinely mixed
- no new DDP instability

Acceptable partial win:
- materially lower RSS
- small throughput loss that might be recovered later

Failure:
- RSS not materially better
- throughput clearly worse
- source weighting drifts
- new DDP instability or invalid-batch behavior

## Rollback Criteria

Rollback this option if:
- runtime code becomes substantially more complex than the current mixed path without a clear performance win
- `__getitems__` grouped fetch does not materially reduce worker RSS
- throughput is worse than the current patched path after warmup
- source adapters end up recreating most of `LeRobotDataset` complexity anyway

If that happens:
- keep current patched Option 1 as the stable runtime fallback
- revisit Option 3

## Recommendation

This is the right next prototype if the priority is:
- preserve sample-level weighted mixing semantics
- fix the systems behavior more directly
- avoid prematurely committing to source-local training batches

It is not the cheapest path, but it is the strongest runtime alternative to the current batch/block-local workaround.
