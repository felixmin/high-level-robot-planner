# Mixed Dataset Decision Framework

This note is the explicit decision framework for the current mixed-dataset design branches.

It uses these option names:

- **Option 1**: batch- / block-local runtime mixing
- **Option 2**: compact-manifest runtime mixing with sample-level semantics and `__getitems__` source-coalesced fetch
- **Option 3**: unified offline dataset / unified manifest path

Companion notes:

- [2026-03-23_mixed_dataset_option1_batch_block_local_plan.md](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-23_mixed_dataset_option1_batch_block_local_plan.md)
- [2026-03-23_mixed_dataset_option2_compact_manifest_plan.md](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-23_mixed_dataset_option2_compact_manifest_plan.md)
- [2026-03-23_mixed_dataset_option3_unified_offline_plan.md](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-23_mixed_dataset_option3_unified_offline_plan.md)

## What We Are Actually Choosing Between

### Option 1
Changes the **mixing schedule** to improve systems behavior.

- easiest near-term stabilization
- most likely to change optimization behavior
- smallest implementation cost

### Option 2
Keeps **sample-level mixing semantics** but changes the **runtime representation and fetch path**.

- strongest runtime alternative if semantics matter
- medium implementation cost
- best direct test of whether we can preserve the intended training behavior without the current memory blowup pattern

### Option 3
Changes the **data system itself**.

- cleanest long-term design
- highest upfront cost
- strongest long-term throughput / simplicity story if this path stays central

## Primary Decision Criteria

Use these in order.

### 1. Do we need to preserve sample-level training behavior?

If **yes**, prefer:
- Option 2 first
- Option 3 if runtime mixing remains too expensive

If **no**, Option 1 becomes much more attractive.

### 2. Do we need a practical stabilization soon?

If **yes**, prefer:
- Option 1 first
- keep hard rollback gates

If **no**, Option 2 is usually the better next experiment.

### 3. Is mixed training becoming a central long-term path?

If **yes**, Option 3 becomes more attractive.

If **no**, avoid paying the offline-ingest cost too early.

### 4. Is the current bottleneck really source order?

If **yes**, Option 1 should win clearly in benchmarks.

If **no**, Option 2 or Option 3 are the better investments.

## Unknowns That Still Matter

These are the key unknowns, and they are the reason not to choose on taste alone.

### Unknown 1: Does source-blocking actually improve steady-state throughput?

If it only improves first-batch behavior or source-switch counts, it is not the main fix.

### Unknown 2: Does Option 1 measurably change training quality?

We expect some optimization risk.
We do not yet know how large it is for this stack.

### Unknown 3: Can Option 2 actually cut worker RSS enough to matter?

If `__getitems__` coalescing plus compact manifest does not materially reduce RSS, Option 2 loses much of its reason to exist.

### Unknown 4: Is Option 3 worth the operational cost?

If source schemas or actions are too expensive to canonicalize relative to the benefit, the long-term story changes.

## Benchmark Matrix To Decide

Use one common benchmark matrix across options.

### Common setup
Same:
- mix
- batch size
- future horizon
- image size
- backend
- seed

Measure:
- parent RSS
- total worker RSS
- peak worker RSS
- steady-state samples/s
- p50 / p95 batch time
- first-batch time
- source diversity per batch
- source histogram over time
- DDP stability
- short-train early loss trajectory

### Option 1 benchmark set
- `source_block_size=1`
- `source_block_size=batch_size`
- `source_block_size=2*batch_size`
- each with `num_workers=4` and `10`

### Option 2 benchmark set
- current patched baseline
- compact-manifest + `__getitems__` candidate
- each with `num_workers=4` and `10`

### Option 3 benchmark set
- unified manifest loader vs current patched path
- initially only manifest-first, media left in place
- then short train smoke

## Go / No-Go Gates

### Choose Option 1 if all are true
- we need a practical stabilization quickly
- it gives a clear systems win after warmup
- it does not introduce obvious early training regression
- the gain is large enough to justify the semantic change

Recommended threshold:
- at least `25%+` lower worker RSS
- and at least similar steady-state throughput
- and no meaningful early loss deterioration

Reject Option 1 if:
- gain is mostly startup smoothing
- throughput stays flat after warmup
- backend / worker count dominates much more than source-block size
- training behavior changes noticeably without a clear systems win

### Choose Option 2 if all are true
- preserving sample-level weighted mixing semantics matters
- compact manifest reduces parent / worker memory materially
- `__getitems__` grouped fetch recovers enough locality to keep throughput competitive
- DDP behavior becomes clearer, not harder

Recommended threshold:
- `30%+` lower worker RSS than current patched path
- same or better steady-state throughput
- mixed batches remain genuinely mixed
- no new DDP instability

Reject Option 2 if:
- it becomes much more complex without a clear win
- source adapters recreate most of `LeRobotDataset` complexity
- `__getitems__` grouping does not materially help memory or throughput

### Choose Option 3 if all are true
- mixed training will remain a core long-term path
- we are willing to build and maintain an ingest pipeline
- manifest-first prototype already shows clear runtime simplification and/or memory wins
- action/schema alignment cost is acceptable

Recommended threshold:
- materially simpler runtime path
- clearly lower worker RSS
- equal or better throughput
- cleaner deterministic weighting and DDP story

Reject Option 3 if:
- schema alignment cost dominates everything
- runtime still needs lots of source-specific branching
- the operational burden is too high for the expected reuse

## Recommended Decision Order

Use this order to avoid overcommitting too early.

### Step 1
Keep current patched path as the operational fallback.

### Step 2
Prototype Option 2.

Reason:
- it is the strongest test of whether we can preserve intended semantics while fixing the memory model more directly

### Step 3
Keep Option 1 as the practical systems comparison, not as an assumed winner.

Reason:
- it may still win on cost/benefit even if it is less principled

### Step 4
Only move toward Option 3 if:
- mixed training is clearly staying important
- and the runtime paths still feel too compromised or too expensive

## Current Recommendation

Right now, the most defensible sequencing is:

1. operational fallback = current patched path
2. next prototype = Option 2
3. systems comparison branch = Option 1
4. long-term investment branch = Option 3 if the path proves important enough

So the current call is:
- do **not** assume Option 1 is the answer
- do **not** jump straight to Option 3
- prototype Option 2 next and use the explicit benchmark gates above to decide what survives
