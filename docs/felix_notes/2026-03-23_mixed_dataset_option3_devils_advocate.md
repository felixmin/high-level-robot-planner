# Mixed Dataset Option 3: Devil's Advocate

This note records the strongest arguments against jumping too quickly to Option 3:

- unified manifest / unified offline dataset
- canonical schema
- one runtime sample space

The point is not that Option 3 is wrong.
The point is that it is expensive and can fail in subtle ways if we overestimate what it solves.

## Main Critique

Option 3 is attractive because it promises:

- one dataset
- one schema
- one sampler
- one DDP story

But that can hide three real risks:

1. it may move complexity out of the training loop without reducing the true hot-path cost enough
2. it may require much more schema/action canonicalization work than the current project can justify
3. it may overfit the data system to today’s sources and become expensive to evolve

## What Option 3 Does Not Automatically Fix

### 1. Decode / backend cost

If the real bottleneck is still:

- video seek cost
- decode cost
- filesystem latency
- backend behavior

then a unified manifest alone may not buy enough.

It will likely improve:

- parent-process memory
- worker replication of heavy metadata
- sharding clarity

But it will not magically make video decode cheap.

That is why the manifest-first version should be benchmarked before any full media rewrite.

### 2. Schema complexity

Option 3 only looks simple at runtime because the complexity is paid up front.

That up-front cost can be large:

- action family alignment
- timestamp/fps semantics
- missing modalities
- per-dataset labels
- source-specific quirks

If those differences are broader than expected, the canonical schema can become:

- too weak to be useful
- or too complicated to stay maintainable

### 3. Rebuild and maintenance burden

Every new source needs:

- source registration
- adapter logic
- validation
- manifest rebuild or partial rebuild

If datasets change often, this can become a workflow tax.

Option 2 is weaker architecturally, but more flexible operationally.

## Hard Questions Option 3 Must Answer

Before committing, Option 3 should answer:

1. Which action families are actually in scope for one unified run?
2. Which per-dataset labels are canonical and which stay source-specific?
3. How often do we expect new sources or schema changes?
4. Does manifest-first already give enough of the systems gain?
5. Is a canonical media rewrite actually necessary later?

If the answers are weak or unstable, Option 3 may be premature.

## Operational Risk

Option 3 has the highest project-management cost:

- new build pipeline
- new validation surface
- new storage/versioning story
- new failure modes outside training

That is acceptable only if mixed training is a real long-term product surface in this repo, not just an experiment lane.

## When Option 3 Is Probably Premature

Option 3 is probably too early if:

- we have not yet proven that Option 2 cannot deliver enough
- source schemas are still moving quickly
- action alignment is still unresolved
- the real bottleneck may still be decode/backend rather than manifest/state
- the team needs iteration speed more than architectural cleanliness

## When Option 3 Is Worth It

Option 3 becomes worth it if:

- mixed training is clearly staying central
- we keep revisiting the same runtime-mixing problems
- manifest-first shows measurable systems gains
- action/schema boundaries can be made explicit and stable

## Recommended Guardrail

Do not start with a full canonical-media rewrite.

Start with:

- unified manifest
- source adapters
- source media left in place
- explicit benchmark against current runtime paths

Only after that should we decide whether Option 3 deserves deeper investment.
