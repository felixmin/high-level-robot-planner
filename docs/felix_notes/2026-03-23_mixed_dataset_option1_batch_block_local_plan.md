# Mixed Dataset Option 1: Batch- / Block-Local Runtime Mixing

This note describes the detailed plan for the current pragmatic runtime-mixer path:

- keep the current mixed runtime dataset
- keep standard per-source `LeRobotDataset`
- reduce source churn by making source choice block-local instead of fully sample-local
- treat this as a systems optimization with explicit training-risk checks

Related overview notes:

- [2026-03-23_mixed_dataset_design_options.md](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-23_mixed_dataset_design_options.md)
- [2026-03-23_mixed_dataset_option2_prototype_plan.md](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-23_mixed_dataset_option2_prototype_plan.md)

## Position

This is the fastest path to a more stable mixed loader.

It is **not** the cleanest semantics-preserving path.
It changes training order and can change optimization behavior.
So it should be treated as a controlled training change, not just a loader tweak.

## Goal

Bound worker churn and host RAM by making source selection coarser in time.

Keep:
- current mix YAML format
- current `MixedLeRobotDataset`
- current source-specific remapping logic
- current runtime path through `lerobot-train`

Change:
- source selection happens in short source-local blocks
- mixed-dataset loader settings become explicit and mixed-specific
- rank behavior becomes more explicit in the sampler

## Architecture

### Main idea

A sampler epoch is still a stream of mixed sample indices.

But the stream is generated in short source blocks:

1. choose a source by configured weight
2. emit `source_block_size` valid samples from that source
3. choose the next source by weight
4. repeat

This means most batches become source-pure or source-dominant, but only for short intervals.

### Weighting

Weights stay source-level exactly as today.

The weighting rule should be:
- source weights govern **emitted samples over the epoch**
- not just the number of blocks

The simplest safe form is:
- fixed `source_block_size`
- weighted source draw per block
- uniform valid-anchor draw within the chosen source

### Rank sharding

Preferred design:
- generate a deterministic global mixed stream from `(seed, epoch)`
- shard by rank in the sampler itself
- rank `r` consumes every `world_size`-th global position

This is better than relying on downstream batch slicing wrappers.

## Code Shape

### Keep

- [mixed_dataset.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/mixed_dataset.py)
- [lerobot_dataset.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/lerobot_dataset.py)
- current mix YAML and logical-source parsing

### Main files to change

- [sampler.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/sampler.py)
- [lerobot_train.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_train.py)
- possibly minor cleanup in [mixed_dataset.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/mixed_dataset.py)

## Detailed Plan

### 1. Sampler hardening

In [sampler.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/sampler.py):

Add or formalize:
- `source_block_size`
- `rank`
- `world_size`
- optional diagnostics
- optional `source_block_jitter=false` initially

Requirements:
- only valid per-source anchors are emitted
- source histogram is inspectable
- source-switch count is inspectable
- unique sources touched per `N` samples is inspectable

### 2. Mixed-loader config cleanup

In [lerobot_train.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_train.py):

Make mixed-only loader knobs explicit:
- `prefetch_factor`
- `num_workers`
- optionally `pin_memory`

Recommended initial defaults for heavy mixed runs:
- `prefetch_factor=1`
- `num_workers=4`
- `source_block_size=batch_size`

Do not hardcode these invisibly; they should be inspectable in config or logs.

### 3. Mixed dataset cleanup

In [mixed_dataset.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/mixed_dataset.py):

Keep current fetch behavior.

Do not expand responsibilities.

Only keep:
- bounded cache pieces that are still justified
- minimal debug hooks that remain useful for benchmarking

Remove or gate ad hoc investigation-only code once the path is stable.

## Optimization Risk Mitigation

This option can alter training behavior.

Mitigations:
- keep blocks short
- start with `source_block_size=batch_size`
- only test `2 * batch_size` after the first comparison passes
- preserve exact epoch-level source weighting
- watch for:
  - per-source loss drift
  - codebook collapse by source
  - rare-source starvation
- if quality regresses, try:
  - few-source blocks
  - or gradient accumulation across different-source microbatches

## Benchmark Plan

Compare three settings locally first:

1. `source_block_size=1`
2. `source_block_size=batch_size`
3. `source_block_size=2*batch_size`

For each, run:
- `num_workers=4`
- `num_workers=10`

Collect:
- parent RSS
- total worker RSS
- peak worker RSS
- steady-state samples/s
- p50/p95 batch time
- source switch count
- unique sources touched after 20/40 steps
- source diversity per batch

Then run short local train smokes for:
- `source_block_size=1`
- `source_block_size=batch_size`

Collect:
- first 50-100 steps
- loss trajectory
- step time
- codebook stats
- any instability

Then cluster validation:
- `4 GPU`
- `bs=64`
- `num_workers=4`
- compare `source_block_size=1` vs `source_block_size=batch_size`
- test `2*batch_size` only if the first comparison is positive

Collect:
- completion / OOM / worker-kill
- time to step 100 / 200 / 400
- steady-state `data_s`, `updt_s`
- host memory if available

## Success Criteria

Minimum win:
- materially lower worker RSS than `source_block_size=1`
- no worker-kill or host-RAM OOM at the tested worker count
- similar steady-state throughput

Strong win:
- `25-30%+` lower worker RSS
- clear reduction in source-switch count
- equal or better steady-state samples/s
- successful mixed 4-GPU validation where the baseline is unstable

Training acceptability:
- no obvious early loss degradation
- no rare-source starvation
- no source-specific collapse signals
- no new DDP instability

## Failure Criteria

Reject or demote this option if:
- throughput does not improve materially after warmup
- memory improves only slightly while training semantics clearly change
- gains disappear once backend or worker count is controlled
- the main benefit is only first-batch smoothing
- optimization risk is real and the systems win is weak

## Rollback Criteria

Rollback to the current stable fallback if:
- there is no stable mixed 4-GPU improvement over current patched settings
- training quality regresses meaningfully
- block-local scheduling becomes a growing pile of special cases
- evidence shows backend/decode or heavy dataset state dominates much more than source order

## Recommendation

Pursue this option only as a disciplined systems optimization with hard benchmark gates.

If it wins clearly, it is the fastest path to an operational mixed loader.
If it does not, do not keep tuning it indefinitely; move to Option 2 or Option 3.
