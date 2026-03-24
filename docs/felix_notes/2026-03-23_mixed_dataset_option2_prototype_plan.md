# Mixed Dataset Option 2 Prototype Plan

This note describes the concrete prototype plan for the current preferred next branch:

- preserve sample-level weighted mixing semantics
- replace heavy mixed runtime state with a compact mixed manifest
- recover source locality during fetch with `__getitems__`
- compare against the current patched batch/block-local path

## Goal

Prototype a runtime mixer that keeps the intended training behavior closer to the current sample-level mix, while reducing worker RAM and improving fetch locality.

This is the "Option 2" path from:

- [2026-03-23_mixed_dataset_design_options.md](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-23_mixed_dataset_design_options.md)

## Design Summary

The core idea is:

- sampling stays sample-level and weighted
- batches may still contain samples from multiple datasets
- but a worker is allowed to fetch a whole batch more intelligently than one sample at a time

So the visible batch may still be:

- `libero, bridge, libero, fractal, bridge, ...`

but inside `__getitems__` we would:

1. group requested samples by source
2. fetch each source-local group together
3. restore the original mixed order
4. hand the mixed batch back to the DataLoader

This preserves logical mixing while recovering source locality under the hood.

## Why This Path

This path tries to address two concerns at once:

- the current mixed runtime representation is too heavy
- pure source-local batching may change optimization behavior too much

So instead of changing the visible sampling semantics first, we change:

- metadata layout
- runtime ownership
- rank sharding
- batch fetch behavior

## Prototype Scope

Keep the prototype small.

Do not try to redesign the whole data stack at once.

The prototype should only aim to prove or disprove the following hypothesis:

- a compact manifest plus `__getitems__` source-coalesced fetch can materially reduce worker RSS while preserving sample-level mixed batches

## Proposed Components

### 1. Compact Mixed Manifest

Create one compact mixed manifest that stores only what the runtime mixer needs.

Per entry or per effective sample space, we want compact arrays for:

- `source_id`
- `episode_id`
- `anchor` or anchor-relative index
- valid temporal range
- timestamps or fps
- split information
- source weight metadata

The important rule:

- avoid large Python object graphs in the parent dataset object
- prefer NumPy arrays, memmap, Arrow-style tables, or similarly compact immutable structures

This does not need to be a final offline-unified dataset.
It only needs to be a compact runtime representation.

### 2. Tiny Source Adapters

Instead of letting the mixed runtime path own many full `LeRobotDataset` objects, introduce a much smaller reader/adapter abstraction.

Each source adapter should hold only what is needed to fetch actual samples:

- source root / repo id / revision
- compact episode metadata
- key remapping metadata
- source-specific temporal metadata
- a very small worker-local open-reader cache

The mixed dataset should not keep many heavyweight source datasets alive in the parent process.

### 3. Rank-Aware Sample-Level Sampler

Keep sampling weighted at the sample level.

But make rank sharding explicit and deterministic in the sampler itself.

The sampler should:

- produce sample-level mixed indices
- respect source weights
- be deterministic by seed and epoch
- shard by rank directly instead of relying on post-hoc rank slicing behavior

That removes a major source of ambiguity when debugging mixed DDP runs.

### 4. `__getitems__(indices)` On The Mixed Dataset

This is the key mechanism.

PyTorch can call `__getitems__` with a list of indices for a batch.
That allows the dataset to optimize batch fetches internally.

The plan:

- accept a list of mixed indices
- map each to `(source_id, local sample identity)`
- group requests by `source_id`
- fetch each group in a source-local way
- reconstruct the requested mixed order
- return the batch items in the original order

This gives locality without forcing source-pure batches.

### 5. Small Worker-Local LRU

Inside workers, keep a tiny LRU of open source readers.

The point is not to keep many sources warm forever.
The point is only to avoid pathological reopen/reinit churn.

A good initial target is:

- `1-2` open source readers per worker

Anything larger should need evidence.

## What Not To Do In The Prototype

- do not redesign every source dataset format
- do not build a full offline unified dataset yet
- do not add many new configuration files
- do not add a large new framework layer
- do not try to solve all throughput problems in one pass

This prototype should stay dense and focused.

## Suggested File Shape

A minimal implementation could look like:

- existing:
  - [lerobot_dataset.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/lerobot_dataset.py)
  - [mixed_dataset.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/mixed_dataset.py)
  - [sampler.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/sampler.py)
  - [lerobot_train.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_train.py)

Possible prototype additions:

- a small new helper inside `mixed_dataset.py`, or
- one new file such as:
  - `compact_mixed_dataset.py`

I would prefer keeping it to one new file at most, or folding the prototype into `mixed_dataset.py` behind a narrow path if that stays readable.

## Benchmark Plan

Benchmark against the current patched path.

Candidate A:

- current patched mixer
- sample-level mixer with `source_block_size`
- mixed batches still formed by the normal DataLoader path

Candidate B:

- compact-manifest mixer
- sample-level weighted sampling preserved
- `__getitems__` source-coalesced fetch

### Compare

1. Parent RSS
2. Total worker RSS
3. Peak worker RSS
4. Steady-state samples/s
5. Steady-state batches/s
6. p50 / p95 batch time
7. Source diversity per batch
8. Effective source histogram over many batches
9. DDP stability in a short train smoke

### Minimal Matrix

Use the same mix, batch size, backend, and seed.

Run:

- A with `num_workers=4`
- A with `num_workers=10`
- B with `num_workers=4`
- B with `num_workers=10`

First with the local dataloader benchmark.
Then with a short real train smoke.

## Success Criteria

The Option 2 prototype is a win if it achieves most of the following:

- materially lower worker RSS than the current patched path
- equal or better steady-state throughput
- preserved source weighting over time
- batches remain truly mixed at sample level
- no new DDP instability in a short train run

A strong result would be something like:

- `30%+` lower worker RSS
- same or better steady-state samples/s
- similar source-diversity profile per batch
- stable short train smoke where the current mixed path is fragile

## Interpretation Rules

If the prototype wins on memory but loses badly on throughput:

- the representation got better, but the fetch path is still too expensive

If it wins on throughput but not memory:

- source-coalesced fetch helps, but heavy parent/worker state is still the bottleneck

If it wins on both:

- this becomes the strongest runtime-mixer direction

If it loses on both:

- we should stop pushing runtime mixing in this direction and either:
  - lean harder into block-local scheduling, or
  - begin the unified-manifest/offline path

## Current Recommendation

Build this prototype before committing further to source-local batching as the main answer.

Reason:

- it is the cleanest way to test whether we can preserve the intended sample-level semantics while fixing the systems behavior more directly
- it gives us a more principled A/B than simply tuning `source_block_size`
