# Option 2 Compact Mixed Dataset Selectable Path Plan

Timestamp: 2026-03-23 15:10:00

This is a planning artifact for adding Option 2 to the vendored `lerobot` mixed-dataset path without changing the current default behavior.

## 1. Change summary

Add a second mixed-dataset implementation path that preserves sample-level weighted mixing semantics while reducing per-worker mixed-runtime state and allowing batched source-coalesced fetch through `__getitems__`.

Constraints:

- keep the current mixed path as the default
- keep activation through `dataset.mix_path`
- add one explicit selector for the implementation variant
- keep the change small, loosely coupled, and friendly to a later Option 3 unified-manifest direction
- benchmark the new path locally against the current patched path before cluster use

## 2. Current code and docs fit

Current mixed-dataset seams are:

- [`lerobot/src/lerobot/datasets/factory.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/factory.py)
  - owns mixed dataset construction behind `dataset.mix_path`
- [`lerobot/src/lerobot/datasets/mixed_dataset.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/mixed_dataset.py)
  - owns mix parsing helpers, `LogicalSource`, mixed metadata, `MixedLeRobotDataset`, and the current per-item fetch path
- [`lerobot/src/lerobot/datasets/sampler.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/sampler.py)
  - owns `WeightedSourceSampler`
- [`lerobot/src/lerobot/scripts/lerobot_train.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_train.py)
  - owns mixed-only sampler and loader knobs

Relevant existing tests are:

- [`lerobot/tests/datasets/test_mixed_dataset.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/tests/datasets/test_mixed_dataset.py)
- [`lerobot/tests/training/test_train_sampler.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/tests/training/test_train_sampler.py)

Relevant prior design notes are:

- [`docs/felix_notes/2026-03-23_mixed_dataset_option2_compact_manifest_plan.md`](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-23_mixed_dataset_option2_compact_manifest_plan.md)
- [`docs/felix_notes/2026-03-23_mixed_dataset_decision_framework.md`](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-23_mixed_dataset_decision_framework.md)

The current default path already includes the temporary stability patch:

- `source_block_size=batch_size` for mixed datasets
- `prefetch_factor=1` for mixed datasets

That behavior must remain the default until Option 2 proves itself.

## 3. Design decisions and boundaries

### 3.1 Selector and default

Add one dataset config field:

- `dataset.mix_implementation: str = "legacy"`

Allowed values for this slice:

- `"legacy"`: current `MixedLeRobotDataset` path, unchanged default
- `"compact"`: new Option 2 experimental path

Keep the selection in dataset config instead of in trainer code so the choice stays part of the data-system contract and remains easy to route into Option 3 later.

### 3.2 Minimal architecture

Use one new runtime file:

- [`lerobot/src/lerobot/datasets/compact_mixed_dataset.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/compact_mixed_dataset.py)

Initial Option 2 scope is intentionally narrower than the full long-term note:

- keep existing mix parsing and `LogicalSource` construction in `mixed_dataset.py`
- keep tuple-style sample identity `(source_id, anchor_abs_index)` for the first prototype
- add a compact alternate dataset wrapper that uses `__getitems__` to coalesce a batch by source
- avoid introducing a new global sample-id system yet
- avoid moving DDP sharding into a custom sampler in this first slice

This keeps the prototype close to the current fetch contract while testing the main systems idea:

- preserve sample-level sampling
- preserve mixed batches
- change the fetch path, not the training semantics

### 3.3 Option 2 runtime shape for this slice

Add these classes in `compact_mixed_dataset.py`:

- `CompactMixedBatchRequest`
  - tiny request record for one tuple-style mixed index
- `CompactSourceAdapter`
  - thin wrapper around one `LogicalSource`
  - owns source-local `get_item()` and `get_items()` behavior
- `CompactMixedLeRobotDataset`
  - same external mixed dataset role as `MixedLeRobotDataset`
  - exposes `sources`, `source_weights`, `meta`, `repo_id`, `mix_path`, `build_sampler()`
  - supports `__getitem__` for parity
  - adds `__getitems__` for grouped same-batch fetch

Key boundary:

- the current `LogicalSource` remains the source of per-source metadata and single-sample fetch logic
- Option 2 wraps that logic instead of rewriting it immediately

This is the smallest maintainable branch that can later evolve into:

- a manifest-backed `CompactMixedLeRobotDataset`
- or a unified offline manifest for Option 3

### 3.4 Loader behavior

Do not apply the current block-local sampler tuning to Option 2 by default.

For this slice:

- `legacy` mixed path keeps current behavior:
  - `source_block_size=batch_size`
  - `prefetch_factor=1`
- `compact` mixed path uses:
  - sample-level source switching by default, so `source_block_size=1`
  - `prefetch_factor=1` remains for apples-to-apples comparison unless a benchmark pass justifies changing it

The train script should branch on an explicit dataset capability instead of on `hasattr(dataset, "sources")` alone.

Recommended markers:

- `dataset.mix_implementation`
- `dataset.prefers_source_blocks`

### 3.5 Behavior that must stay frozen

- `dataset.mix_path` remains the activation path for mixed datasets
- `legacy` remains the default mixed implementation
- tuple-style mixed sampler outputs must continue to shard correctly under Accelerate
- mixed batches must still collate the same metadata fields
- the current training path must stay untouched when `dataset.mix_implementation == "legacy"`

## 4. Exact files, classes, and functions to change

### 4.1 Add

- [`lerobot/src/lerobot/datasets/compact_mixed_dataset.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/compact_mixed_dataset.py)
  - add `CompactMixedBatchRequest`
  - add `CompactSourceAdapter`
  - add `CompactMixedLeRobotDataset`

### 4.2 Change

- [`lerobot/src/lerobot/configs/default.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/configs/default.py)
  - add `mix_implementation: str = "legacy"` to `DatasetConfig`
  - validate allowed values in `__post_init__`

- [`lerobot/src/lerobot/datasets/factory.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/factory.py)
  - import `CompactMixedLeRobotDataset`
  - keep existing source construction
  - branch on `cfg.dataset.mix_implementation`
  - return `MixedLeRobotDataset` for `legacy`
  - return `CompactMixedLeRobotDataset` for `compact`

- [`lerobot/src/lerobot/scripts/lerobot_train.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_train.py)
  - replace implicit mixed-dataset loader tuning with explicit dataset capability checks
  - keep current `legacy` source-block behavior
  - avoid forcing source blocking for `compact`

- [`lerobot/tests/datasets/test_mixed_dataset.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/tests/datasets/test_mixed_dataset.py)
  - add config helper coverage for `mix_implementation="compact"`
  - add dataset construction coverage for `CompactMixedLeRobotDataset`
  - add `__getitems__` parity test:
    - grouped batch fetch returns the same items and order as repeated `__getitem__`
  - keep metadata/collation parity tests for `compact`

- [`lerobot/tests/training/test_train_sampler.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/tests/training/test_train_sampler.py)
  - add loader tuning test for:
    - `legacy` mixed path gets `source_block_size=batch_size`
    - `compact` mixed path does not force block-local sampling

- [`tmp_local_runs/bench_mixed_dataloader_memory.py`](/mnt/data/workspace/mare_nostrum_sync/tmp_local_runs/bench_mixed_dataloader_memory.py)
  - add `--mix-implementation legacy|compact`
  - route dataset creation through the new selector
  - preserve the current benchmark outputs so legacy/compact remain directly comparable

### 4.3 Do not change in this slice

- do not remove or rewrite `MixedLeRobotDataset`
- do not rewrite `LogicalSource`
- do not add a custom rank-aware sampler yet
- do not move to a global manifest file format yet
- do not change cluster run scripts in this first implementation slice

## 5. Concrete phased implementation order

### Phase 1: Selector and no-op alternate path scaffold

1. Add `dataset.mix_implementation` to `DatasetConfig` with default `"legacy"`.
2. Add `compact_mixed_dataset.py` with a parity-focused scaffold:
   - construct from existing `LogicalSource` list
   - expose the same basic metadata surface
   - implement `__getitem__`
   - implement `build_sampler()` via the existing `WeightedSourceSampler`
3. Route `factory.make_dataset()` to the alternate class when selected.
4. Add the smallest construction tests.

Success gate:

- all existing legacy tests still pass
- compact selection builds and returns the new class

### Phase 2: Batch-coalesced fetch

1. Implement `CompactSourceAdapter.get_items(anchors)` using one source at a time.
2. Implement `CompactMixedLeRobotDataset.__getitems__(indices)`:
   - normalize tuple indices
   - group by `source_id`
   - fetch source-local chunks
   - restore the original mixed batch order
3. Add parity tests comparing `__getitems__` against repeated `__getitem__`.

Success gate:

- compact batch fetch is behaviorally identical to item-by-item fetch for test fixtures
- batch order is stable

### Phase 3: Train-loader selection logic

1. Add explicit dataset capability markers to both mixed implementations.
2. Update `make_offline_dataloader()` to branch on the capability marker rather than only on `hasattr(dataset, "sources")`.
3. Keep legacy source-blocking intact.
4. Keep compact on sample-level source switching.

Success gate:

- tuple-sampler sharding tests still pass
- legacy behavior is unchanged
- compact path is selectable and does not inherit the legacy block-local tuning by accident

### Phase 4: Local benchmark and iteration

1. Extend the local benchmark script to compare `legacy` vs `compact`.
2. Run at least:
   - `legacy`, `bs=64`, `num_workers=4`
   - `compact`, `bs=64`, `num_workers=4`
   - `legacy`, `bs=64`, `num_workers=10`
   - `compact`, `bs=64`, `num_workers=10`
3. Capture:
   - parent RSS
   - total worker RSS
   - peak worker RSS
   - batches/s
   - samples/s
   - source histogram

Success gate:

- `compact` must not regress sample-level mix exposure
- `compact` should show materially lower worker RSS or better stability at the same worker count

### Phase 5: Local train smoke

Run a short local `lerobot-train` smoke with `dataset.mix_implementation=compact` and compare against `legacy` on the same tiny configuration.

Success gate:

- no regression in basic training startup and first-step execution
- no metadata contract breakage

## 6. Test and validation strategy

### Targeted automated tests

- `pytest lerobot/tests/datasets/test_mixed_dataset.py -q`
- `pytest lerobot/tests/training/test_train_sampler.py -q`

Additions required:

- compact construction test
- compact `__getitems__` parity test
- loader-tuning selection test

### Local benchmarks

Use the temp benchmark script only:

- [`tmp_local_runs/bench_mixed_dataloader_memory.py`](/mnt/data/workspace/mare_nostrum_sync/tmp_local_runs/bench_mixed_dataloader_memory.py)

Required comparison metrics:

- parent RSS
- total and peak worker RSS
- steady-state batches/s
- steady-state samples/s
- per-batch source diversity

### Local smoke

- short local mixed run with `mix_implementation=compact`
- same mix and batch size as the legacy comparison when feasible

### Non-goals for this slice

- no immediate cluster sweep
- no attempt to prove final throughput on MN5 yet

Cluster reruns only make sense if:

- compact passes local parity tests
- compact shows a clear local memory/stability improvement

## 7. Documentation and progress-note impact

Update these notes after implementation:

- [`docs/felix_notes/2026-03-23_mixed_dataset_option2_compact_manifest_plan.md`](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-23_mixed_dataset_option2_compact_manifest_plan.md)
  - mark which part has actually landed
- optionally add one durable execution note under `docs/felix_notes/` if the benchmark results materially change the option ranking

No architecture-doc update is required in this slice because this repo does not currently maintain a living mixed-dataset architecture doc outside the dated notes and plans, and the default runtime architecture remains unchanged.

## 8. Risks, cleanup, and open questions

### Risks

- `__getitems__` may not yield enough benefit if `LogicalSource.get_item()` still dominates memory and decode cost
- keeping tuple-style sample identity means this is only a partial Option 2, not the full manifest-backed form
- if compact requires many special-case branches in train code, the boundary is wrong and should be revised before merging

### Required cleanup

- do not duplicate legacy metadata assembly logic in multiple places if it can be shared cleanly
- avoid a second bespoke sampler if the current `WeightedSourceSampler` already fits this slice
- avoid adding Python-only fallback defaults for the selector outside `DatasetConfig`

### Open questions that do not block implementation

- whether a later second slice should replace tuple indices with explicit compact sample ids
- whether `CompactSourceAdapter` should eventually stop depending on `LogicalSource` entirely
