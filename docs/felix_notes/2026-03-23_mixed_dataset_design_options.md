# Mixed Dataset Design Options

This note summarizes the current design options for heterogeneous LeRobot dataset mixing after the recent mixed `rlfv_lam` failures and local/cluster investigations.

## Current Situation

What we now know:

- The current mixed runtime path can be made stable enough for MN5 with the recent `source_block_size` and lower-prefetch patch.
- That patch improves worker memory behavior, but it is still not obviously the clean long-term answer.
- The mixed-path failure mode looked primarily like host-RAM pressure and worker/process replication, not CUDA OOM.
- The root cause is likely not just "too much mixing", but the interaction of:
  - sample-level source switching
  - heavy per-source dataset/runtime state
  - video decode cost
  - DataLoader worker replication
  - DDP rank fan-out

So the real question is not just "how do we stabilize the current mixer?", but:

- what dataset/loader abstraction should we actually want?

## Option 1: Keep Runtime Mixing And Lean Into Batch-Level Scheduling

This is the current patched direction.

Idea:

- keep the runtime mixed dataset
- keep standard source datasets underneath
- make source choice coarser, e.g. batch- or block-local
- reduce source churn per worker/rank
- preserve custom weights over many batches rather than within each batch

### Why It Looks Attractive

- smallest change from the current implementation
- already produced a stable cluster point for the `octo24 + libero` mixed run
- directly reduces source churn, which was strongly correlated with worker RSS growth
- easy to explain operationally

### Risks

- this may optimize around the symptom rather than the real bottleneck
- it changes training semantics, not just systems behavior
- source-local batches can increase gradient autocorrelation and source-specific optimizer drift
- uneven weights can become burstier in time even if epoch-level proportions stay correct

### When It Is Reasonable

- if the systems gain is clearly large
- if short blocks are enough
- if training quality does not regress
- if we need a practical near-term solution more than a clean redesign

### What To Validate

- steady-state samples/s, not just first-batch time
- worker RSS vs source-block size
- optimization behavior vs the original sample-level mix
- sensitivity for rare datasets and heterogeneous sources

## Option 2: Preserve Sample-Level Mixing Semantics, But Replace The Runtime Representation

This is the strongest alternative to batch-level scheduling.

Idea:

- keep sample-level weighted mixing logically
- do not let the runtime mixed dataset own many full `LeRobotDataset` objects
- replace heavy mixed runtime state with a compact global manifest
- use small source adapters and rank-aware sample-level sampling
- add `__getitems__(indices)` so a worker can group a mixed batch by source internally, fetch efficiently, then restore the original mixed order

This keeps the visible training behavior much closer to the current intended semantics.

### Target Shape

- one compact mixed manifest:
  - `source_id`
  - `episode_id`
  - `anchor`
  - valid temporal range
  - timestamps or fps
  - split info
  - source weight metadata
- tiny `SourceAdapter` objects instead of full runtime `LeRobotDataset` ownership inside the mixed path
- rank-aware sample-level sampler
- `MixedLeRobotDataset.__getitems__` groups by source internally

### Why It Is Appealing

- preserves sample-level weighted mixing semantics
- allows mixed batches to stay truly mixed
- attacks the memory model more directly than source-blocking
- can recover source locality during fetch without making the batch source-local
- probably a cleaner long-term runtime mixer than the current `mixed_dataset.py`

### Risks

- not a small patch anymore
- harder than the current batch/block-local stabilization
- still fights the physical reality of per-sample video decode more than a fully unified offline dataset would
- if implemented badly, it can become a hidden batch-local design with extra complexity

### When It Is Reasonable

- if preserving current training semantics matters
- if we want a cleaner runtime mixer instead of a training-behavior change
- if we are willing to invest in a new compact mixed representation

### What To Validate

- worker RSS vs the current patched mixer
- steady-state samples/s
- source diversity per batch
- effective exposure to rare datasets
- DDP stability with true mixed batches

## Option 3: Build A Proper Unified Offline Dataset

This is the cleanest long-term systems design, but also the largest investment.

Idea:

- stop mixing source-native datasets directly at runtime
- build a canonical unified dataset artifact offline
- train against one normalized sample space
- let runtime mixing only mean sampling policy over already-normalized samples

This is not just "store everything in one folder".
It would require a real offline canonicalization pipeline.

### What Would Need To Exist

- one global manifest or canonical sample index
- one normalized observation schema
- one normalized action schema, or at least a clean action-family policy
- canonical temporal metadata so `future_seconds` is well-defined
- deterministic split assignment
- source/task weighting metadata
- likely partial media normalization or canonical training-ready references

### Why It Is Appealing

- simplest runtime training loop
- easiest DDP sharding story
- one loader, one schema, one place to optimize throughput
- best chance at high aggregate throughput once built properly
- avoids carrying source-native quirks in the hot path

### Risks

- highest offline engineering cost
- schema alignment is hard, especially actions
- storage duplication risk
- rebuild cost when sources change
- less flexible for quick inclusion/exclusion experiments unless the manifest layer is designed carefully

### When It Is Reasonable

- if this mixed-data path is going to be used repeatedly
- if we care about long-term throughput, simplicity, and reproducibility
- if we are willing to build and maintain a real ingest pipeline

### Best Incremental Form

The most realistic first step is probably not a full media rewrite.
A better first move is:

- unified manifest first
- source media left in place initially
- canonical indexing, timestamps/fps, key mapping, split metadata, and weighting
- then decide later whether a fully materialized canonical dataset is worth it

## Recommendation Right Now

I would not commit fully to Option 1 yet, even though it produced the first stable mixed cluster point.

Why:

- it is a systems patch with real optimization tradeoffs
- we do not yet know whether it is solving the real bottleneck or only masking it

The most defensible next step is:

1. keep the current patched path as the short-term stable fallback
2. prototype Option 2 locally
3. compare Option 2 against the current patched Option 1 on:
   - worker RSS
   - steady-state throughput
   - source diversity per batch
   - DDP stability
4. only if Option 2 is still too expensive, decide whether to:
   - lean further into Option 1, or
   - start a true unified-manifest pipeline toward Option 3

## Practical Reading Of The Tradeoff

- Option 1:
  - easiest near-term stabilization
  - most likely to change optimization behavior
- Option 2:
  - best runtime design if we want to preserve sample-level semantics
  - medium engineering cost
- Option 3:
  - cleanest long-term data system
  - highest upfront cost

If we care most about preserving training semantics, Option 2 currently looks like the best next branch.
If we care most about immediate stability with limited engineering, Option 1 remains the practical fallback.
If this will become a central long-term training path across many datasets, Option 3 is probably the real end state.
