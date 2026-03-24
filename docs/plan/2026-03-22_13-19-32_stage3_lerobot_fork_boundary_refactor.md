# Stage 3 LeRobot Fork Boundary Refactor Plan

## 1. Scope and decisions

This plan covers the current Stage 3 fork surface around:

- mixed dataset support
- gradient accumulation config support
- exact latent/action accumulation semantics
- processor override wiring
- dataset sampler / dataloader creation

Explicit out-of-scope note:

- the legacy `latent_smol` cleanup has already landed and is no longer part of the remaining fork-boundary problem
- this plan should not talk about `latent_smol` as if it still contributes implicit legacy surface

Planned ownership decisions:

| Area | Decision | Why |
| --- | --- | --- |
| A. `mixed_dataset` | Keep in vendored `lerobot/`, but remove implicit policy-coupled resizing from dataset factory | This is a reusable data/runtime feature, but `cfg.policy.image_size -> dataset resize` is the wrong ownership seam |
| B. `grad_accum_steps` | Keep in vendored `lerobot/` as-is | It is already a generic trainer feature in config + `Accelerator` setup |
| C. exact latent/action accumulation | Keep one generic train loop, but move the HLRP-only accumulation path behind a thin explicit branch into HLRP-owned helper code | This removes the largest policy-specific block from generic train internals without introducing a second trainer too early |
| D. processor override logic in train script | Move out of entrypoints into a shared runtime-helper seam adjacent to processor construction | Runtime override assembly needs runtime context from train/eval/record/inference, so it should not be re-centralized as ad hoc logic in entrypoints or overloaded into `policies/factory.py` |
| E. dataset sampler creation in train script | Keep in fork, but move out of top-level train script into a small generic training/dataloader helper | The behavior is useful and generic enough for the fork, but it does not belong in the entrypoint |

Change-size expectation:

- A: medium
- B: none or tiny
- C: medium to large
- D: medium
- E: small to medium

## 2. Real structural problem

The main problem is not simply "the LeRobot fork is too large". The sharper problem is that `lerobot/src/lerobot/scripts/lerobot_train.py` currently owns three different layers at once:

1. generic offline training orchestration
2. generic-but-fork-local dataset and sampler behavior
3. HLRP-specific Stage 3 policy semantics

That mixed ownership shows up concretely in the current code:

- exact accumulation in [`lerobot_train.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_train.py) depends on HLRP-only output keys such as `_action_loss_tensor`, `_latent_loss_tensor`, and branch-specific denominators
- processor override assembly in the same script knows too much about saved processors, runtime device wiring, rename maps, and normalization overrides
- processor/runtime override assembly is duplicated across multiple entrypoints, not just training, including [`lerobot_eval.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_eval.py), [`lerobot_record.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_record.py), and inference-side setup such as [`policy_server.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/async_inference/policy_server.py)
- dataset factory derives `MixedLeRobotDataset` visual resizing from `cfg.policy.image_size`, which couples generic dataset creation to one policy config
- sampler creation is generic enough to keep, but it still clutters the train entrypoint

The refactor should therefore reduce the broader "entrypoints own runtime adaptation logic" pattern, with `lerobot_train.py` as the worst instance, not try to erase every repo-local LeRobot extension.

Just as important, the remaining fork surface is broader than `lerobot_train.py`. The largest durable fork surface that still appears justified today is still dataset/runtime code in vendored LeRobot, especially:

- [`lerobot/src/lerobot/datasets/mixed_dataset.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/mixed_dataset.py)
- [`lerobot/src/lerobot/datasets/sampler.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/sampler.py)
- [`lerobot/src/lerobot/datasets/factory.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/factory.py)

So this initiative is primarily an ownership cleanup and boundary clarification, not a promise of major line-count reduction across the whole fork.

## 3. Frozen behavior surface

The following behavior must stay stable unless the implementation plan explicitly says otherwise:

### Stage 3 data / config behavior

- `dataset.mix_path` remains the Stage 3 activation path for mixed datasets.
- Mixed datasets continue to expose:
  - `hlrp_action_supervised`
  - `hlrp_latent_supervised`
  - source metadata fields already used by tests and reporting
- Stage 3 mixed-data complementary metadata remains batch/transition round-trippable for the fields already covered in processor tests:
  - `hlrp_action_supervised`
  - `hlrp_latent_supervised`
  - `hlrp_source_name`
  - `dataset_source_name`
- `grad_accum_steps` stays a valid TrainPipelineConfig field and still controls effective batch size through `Accelerator(gradient_accumulation_steps=...)`.
- Stage 3 Hydra defaults in [`config/experiment/_stage3_base.yaml`](/mnt/data/workspace/code/high-level-robot-planner/config/experiment/_stage3_base.yaml) remain the source of truth for defaults, not Python-side fallback logic.

### Training semantics

- HLRP exact accumulation must still weight action and latent losses by their true supervised denominator mass across the full accumulation window, including multi-process execution.
- HLRP exact accumulation must preserve current behavior for:
  - zero-mass windows
  - alternating mode
  - one-rank-zero-mass / other-ranks-nonzero windows
- `stage3_training_mode` semantics stay unchanged:
  - `action`
  - `latent`
  - `multitask`
  - `alternating`
- The saved/logged effective batch size continues to reflect `batch_size * grad_accum_steps * num_processes`.
- Denominator-driving mask inputs remain stable:
  - `action_is_pad`
  - image padding masks
  - `valid_pair` construction for latent supervision
- The exact accumulation refactor must preserve the actual tensors seen by both branches:
  - action branch sees the same image/state/action tensors that currently reach the policy path after dataset/preprocessor handling
  - latent branch sees the same tensors that currently feed Stage-1 teacher target construction before the policy-side Stage-1 teacher resize

### Dataset / sampler semantics

- `dataset.build_sampler()` remains the preferred hook when a dataset owns sample order.
- Mixed dataset weighted sampling behavior must stay deterministic under the existing sampler contract and existing tests.
- Resume / epoch semantics for dataset-owned samplers must stay correct.
- Current DataLoader settings stay frozen unless the implementation intentionally revalidates them:
  - `pin_memory`
  - `prefetch_factor`
  - `shuffle`
  - `drop_last`
  - sampler-vs-shuffle selection
- The current preparation timing stays frozen:
  - build the `DataLoader`
  - then pass that loader through `accelerate.prepare(...)`
- Accelerate sharding must continue to split dataset-owned samplers without cross-rank overlap, including tuple-style `(source_id, anchor)` sampler outputs used by mixed datasets.
- Mixed dataset caching behavior should stay stable enough that multiple logical sources over the same physical dataset still share one cached `LeRobotDataset` instance unless the refactor deliberately changes that contract with updated tests.
- The exact-accumulation microbatch buffering envelope stays bounded to the current update-window shape:
  - memory/object lifetime remains O(`grad_accum_steps`) for prefetched batches and denominator metadata
  - the refactor must not accidentally retain extra per-window tensors or extend their lifetime across steps

### Checkpoint / processor behavior

- Saved normalization stats sidecar remains the primary source of truth.
- Mixed-dataset normalization fallback rebuild continues to use authoritative mix metadata, not `dataset.repo_id` alone.
- Pretrained/resume processor behavior must stay functionally equivalent for active Stage 3 configs.
- Cross-path normalization behavior must remain equivalent after the processor/runtime refactor for:
  - train
  - resume
  - eval
  - inference
- The command-builder contract stays unchanged for active Stage 3 configs:
  - repo-local relative `dataset.mix_path` still resolves to an absolute path before launch
  - `grad_accum_steps` still forwards into the emitted LeRobot command
  - Hydra Stage 3 dataset presets still resolve to the same `dataset.id`, `dataset.repo_id`, and `dataset.mix_path` values currently asserted in config tests

## 4. Candidate structures considered

### Option 1: keep everything in `lerobot_train.py`

Rejected.

Why:

- simplest short-term edit, but it preserves the central ownership problem
- exact accumulation remains encoded as generic-trainer behavior even though it is effectively one policy's loss contract
- train-script complexity continues to grow

### Option 2: add a generic trainer hook / accumulation strategy API

Rejected for the first refactor pass.

Why:

- this risks adding abstraction for a single known policy
- the current exact accumulation path is very HLRP-shaped: two named branches, branch-specific denominators, special metrics merging, and policy methods that are not generic trainer concepts
- a generic hook surface would likely be larger and harder to understand than a thin explicit HLRP path

### Option 3: one explicit HLRP split, without a generic hook framework

Considered, but not the preferred first cut.

Shape:

- keep generic CLI/config loading and generic reusable data behavior in vendored LeRobot
- keep generic dataset and dataloader features in vendored LeRobot
- move HLRP exact-accumulation semantics behind one explicit handoff seam
- move processor override assembly and dataloader construction into focused helper/factory modules

Two viable handoff seams are available:

- `scripts/6_train_lerobot.py` selects an HLRP-owned Stage 3 train entrypoint directly
- `lerobot_train.py` performs one explicit `policy.type == "hlrp_smolvla_shared"` dispatch

Why not first:

- cleaner ownership than the status quo, but likely larger than necessary
- the launcher-owned entrypoint duplicates more setup surface unless the shared helpers are extracted first
- the train-script dispatch version is smaller, but still broader than a surgical extraction of only the HLRP-specific accumulation seam

### Option 4: keep one generic training loop, but extract the HLRP-only accumulation path behind a narrow explicit branch

Chosen.

Shape:

- keep one generic `lerobot-train` entrypoint and one main offline-training loop
- keep generic config validation, accelerator setup, checkpointing, and eval in vendored LeRobot
- move exact accumulation math, HLRP-only output conventions, and HLRP-specific metric merging into an HLRP-owned helper module
- use one explicit `hlrp_smolvla_shared` branch near the update-step path, rather than a generic strategy framework or a second full trainer
- move processor override assembly and dataloader construction into focused helper/factory modules

Why this is the smallest understandable change:

- smaller than a full HLRP-owned trainer or launcher-owned entrypoint split
- still removes the HLRP-shaped exact-accumulation logic from generic code bodies
- avoids creating a generic strategy framework for one policy
- preserves a single training entrypoint and loop shape
- keeps reusable data features in the fork where they already fit naturally

## 5. Preferred target structure

### A. `mixed_dataset` stays in vendored `lerobot/`

Keep:

- [`lerobot/src/lerobot/datasets/mixed_dataset.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/mixed_dataset.py)
- [`lerobot/src/lerobot/datasets/sampler.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/sampler.py)
- [`lerobot/src/lerobot/datasets/factory.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/factory.py)
- existing mixed-dataset tests in [`tests/lerobot/`](/mnt/data/workspace/code/high-level-robot-planner/tests/lerobot)

Refactor:

- remove implicit `visual_target_size = getattr(cfg.policy, "image_size", None)` from dataset factory
- stop treating policy image size as a hidden dataset-construction default

Preferred replacement seam:

- make visual resizing explicit and dataset-owned
- first choice: an explicit dataset-side config field for mixed datasets
- acceptable fallback: an explicit processor-side resize step if that path is already simpler for Stage 3

Resize-removal gate:

- before removing the hidden coupling, capture one deterministic batch per active Stage 3 dataset preset and record:
  - per-key tensor shapes and dtypes after dataset loading / dataloader collation
  - the tensors actually consumed by the action branch
  - the tensors actually consumed by the latent-target path before policy-side Stage-1 teacher resize
- explicitly account for the two image-size paths in play today:
  - mixed-dataset resize toward policy image size
  - policy-side latent-target resize toward Stage-1 teacher image size
- only remove or relocate ownership after a before/after check shows that both action and latent branches still see intentionally validated tensors
- if ownership moves, gate rollout on before/after throughput and memory measurements, not only on shape assertions

Non-goal:

- do not move mixed-dataset support out of vendored LeRobot in this initiative

Hydra/config implication:

- first verify whether active Stage 3 mixes actually rely on the hidden resize
- if they do not, delete the hidden behavior without replacement
- if they do, add one explicit config value in Hydra and thread it directly
- do not infer it in Python from policy config

### B. `grad_accum_steps` stays where it is

Keep the existing generic ownership in:

- [`lerobot/src/lerobot/configs/train.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/configs/train.py)
- [`lerobot/src/lerobot/scripts/lerobot_train.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_train.py)
- [`scripts/6_train_lerobot.py`](/mnt/data/workspace/code/high-level-robot-planner/scripts/6_train_lerobot.py)

Planned work here is limited to:

- keeping tests that prove the Stage 3 launcher forwards it correctly
- ensuring the HLRP-specific accumulation move does not reintroduce a second competing accumulation control path

### C. Move exact accumulation out of generic `lerobot_train.py`

Preferred boundary:

- keep generic LeRobot bootstrap and the main training loop generic
- keep the seam inside [`lerobot_train.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_train.py), but make it a thin explicit `hlrp_smolvla_shared` branch at the update-step boundary
- move HLRP-specific exact accumulation into an HLRP-owned helper module under [`lerobot_policy_hlrp/src/lerobot_policy_hlrp/`](/mnt/data/workspace/code/high-level-robot-planner/lerobot_policy_hlrp/src/lerobot_policy_hlrp)

Recommended shape:

- generic config validation, accelerator setup, logging/bootstrap, checkpoint/eval orchestration, and normal non-HLRP train behavior remain generic
- generic policies stay on the generic update path
- the HLRP helper owns:
  - exact denominator prefetching
  - exact scaled-loss construction
  - branch-aware metric merging and HLRP-only output-dict conventions
  - any HLRP-only step lifecycle around `begin_training_step()` / `end_training_step()`

Generic/HLRP seam contract:

- generic `lerobot_train.py` should know only whether a policy uses:
  - the generic update path
  - or an HLRP-owned update helper
- generic code must not traffic in HLRP-private output keys such as:
  - `_action_loss_tensor`
  - `_latent_loss_tensor`
  - `_action_loss_denominator_exact`
  - `_latent_loss_denominator_exact`
- `_build_exact_scaled_loss`, `_merge_microbatch_output_dicts`, and `_format_supervision_batch_log` move with the HLRP helper unless a helper is proven generic after the extraction
- after the seam is in place, generic code may only receive:
  - generic train-step state (`loss`, `grad_norm`, `lr`, timings)
  - already-merged public scalar metrics suitable for generic logging
  - an optional already-formatted supplemental HLRP log line, if generic logging still prints it
- do not leave a thin dispatch branch while most HLRP-shaped merge/logging machinery remains in `lerobot_train.py`

Important simplicity rule:

- do not introduce a generic plugin registry or generic loss-strategy framework in the first pass
- use an explicit policy-type dispatch
- only generalize later if a second real policy needs the same seam

Code-size rule:

- prefer extracting small reusable generic helpers only where duplication is obvious and local
- accept a modest amount of explicit HLRP-owned code if that is clearer than new abstraction
- do not keep HLRP-specific helper functions in `lerobot_train.py` just because current tests import them from there; move the tests with the helpers in the same change unless a helper is truly generic

### D. Move processor override logic out of the train script

Current bad seam:

- `lerobot_train.py` assembles `preprocessor_overrides` and `postprocessor_overrides` itself
- similar runtime override assembly also exists in other entrypoints such as eval and record
- this makes top-level scripts responsible for processor semantics

Preferred new seam:

- move runtime override assembly into one small shared runtime-helper module adjacent to processor construction, with [`lerobot/src/lerobot/policies/factory.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/policies/factory.py) remaining the processor-construction owner
- do not solve this by stuffing more runtime-context logic directly into `policies/factory.py`

Practical first-cut design:

- use one shared runtime-helper seam, but do not force all callers through one parameter-sprawl helper
- model the caller contexts explicitly:
  - train / resume
  - eval
  - record
  - inference
- each caller should provide only the context it actually has
- inference-side paths must not be forced to resolve dataset stats or dataset metadata eagerly when they do not need them
- the preferred shape is:
  - one adjacent runtime-helper module near policy/factory ownership
  - small per-caller wrappers or context adapters
  - one shared core that builds only the overrides relevant to the provided context
- `make_pre_post_processors(...)` remains the processor-construction owner, but context resolution should not become an unstructured kwargs bag

Why this is preferable:

- the factory already contains policy-specific processor handling, including special cases such as Groot
- it keeps training orchestration simpler without inventing a new subsystem
- it addresses the broader duplication pattern instead of only moving logic out of one script
- it matches the real problem better than making `policies/factory.py` absorb train/eval/record/inference runtime concerns

### E. Move dataset sampler creation out of the train script, but keep it generic

Keep the behavior in vendored LeRobot, but move the implementation out of the entrypoint into a focused helper module.

Preferred destination:

- a small helper under vendored LeRobot training/dataloader code

Ownership after refactor:

- datasets still own `build_sampler()` when present
- generic dataloader helper decides between:
  - dataset-owned sampler
  - default `EpisodeAwareSampler`
  - shuffle behavior
- tests currently import `make_offline_dataloader` from [`lerobot_train.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_train.py), so this move should be treated as a real public-ish seam change inside the fork, not just a private cleanup
- update those tests to the new helper import in the same change instead of preserving a long-lived compatibility wrapper in `lerobot_train.py`

Non-goal:

- do not move this into HLRP-only code
- this is generic enough to stay in the forked trainer infra

## 6. Implementation phases

### Phase 1: make boundaries explicit without changing semantics

- move sampler construction into a dedicated helper module
- move processor-override assembly into factory/helper code
- update internal tests/imports to the new helper homes in the same patch
- keep behavior stable and preserve existing tests

Exit condition:

- `lerobot_train.py` is smaller and no longer manually assembles processor overrides or sampler creation

### Phase 2: surgically extract HLRP exact accumulation from the generic update path

- introduce one explicit `hlrp_smolvla_shared` branch at the update-step boundary
- move HLRP exact-accumulation update logic into HLRP-owned helper code
- keep generic non-HLRP policies on the generic LeRobot training path

Exit condition:

- generic `lerobot_train.py` no longer implements HLRP-only `_action_loss_tensor` / `_latent_loss_tensor` conventions internally
- the remaining HLRP-specific logic in the train script is limited to a thin dispatch branch
- HLRP-specific helper tests move with the extracted helpers rather than pinning those helpers to the old train-script module

### Phase 3: remove implicit policy-driven mixed-dataset resizing

- delete the hidden `cfg.policy.image_size -> visual_target_size` coupling
- replace it with one explicit ownership seam
- update active Stage 3 Hydra/configs only if needed to preserve current runnable setups

Exit condition:

- dataset behavior is driven by explicit dataset/processor config, not by hidden policy coupling in Python

### Phase 4: docs and cleanup

- update Stage 3 docs and any nearby notes that still describe the old ownership
- remove obsolete helper branches immediately
- do not keep compatibility-only internal code paths unless an active config/checkpoint requires a narrow deserialization shim

## 7. Concrete file targets

Primary implementation targets:

- [`lerobot/src/lerobot/scripts/lerobot_train.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/lerobot_train.py)
- [`lerobot/src/lerobot/policies/factory.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/policies/factory.py)
- [`lerobot/src/lerobot/async_inference/policy_server.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/async_inference/policy_server.py)
- [`lerobot/src/lerobot/datasets/sampler.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/sampler.py)
- [`lerobot/src/lerobot/datasets/factory.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/factory.py)
- [`lerobot/src/lerobot/datasets/mixed_dataset.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/mixed_dataset.py)
- [`lerobot_policy_hlrp/src/lerobot_policy_hlrp/`](/mnt/data/workspace/code/high-level-robot-planner/lerobot_policy_hlrp/src/lerobot_policy_hlrp)
- [`scripts/6_train_lerobot.py`](/mnt/data/workspace/code/high-level-robot-planner/scripts/6_train_lerobot.py) only where command/config handoff needs to reflect the new boundary

Hydra/config surfaces to check while implementing:

- [`config/config.yaml`](/mnt/data/workspace/code/high-level-robot-planner/config/config.yaml)
- [`config/experiment/_stage3_base.yaml`](/mnt/data/workspace/code/high-level-robot-planner/config/experiment/_stage3_base.yaml)
- [`config/experiment/stage3_local.yaml`](/mnt/data/workspace/code/high-level-robot-planner/config/experiment/stage3_local.yaml)
- [`config/experiment/stage3_cluster.yaml`](/mnt/data/workspace/code/high-level-robot-planner/config/experiment/stage3_cluster.yaml)
- Stage 3 dataset presets under [`config/stage3_dataset/`](/mnt/data/workspace/code/high-level-robot-planner/config/stage3_dataset)

## 8. Test and validation plan

### Must-preserve tests

- keep/update Stage 3 launcher coverage in [`tests/scripts/test_train_lerobot_command_builder.py`](/mnt/data/workspace/code/high-level-robot-planner/tests/scripts/test_train_lerobot_command_builder.py)
- keep mixed-dataset coverage in [`tests/lerobot/test_mixed_dataset_feature_key_mapping.py`](/mnt/data/workspace/code/high-level-robot-planner/tests/lerobot/test_mixed_dataset_feature_key_mapping.py)
- keep checkpoint stats fallback coverage in [`tests/lerobot/test_hlrp_checkpoint_stats.py`](/mnt/data/workspace/code/high-level-robot-planner/tests/lerobot/test_hlrp_checkpoint_stats.py)
- keep Hydra coverage in [`tests/config/test_hydra_configs.py`](/mnt/data/workspace/code/high-level-robot-planner/tests/config/test_hydra_configs.py)
- keep sampler sharding and dataset-owned sampler coverage in [`lerobot/tests/training/test_train_sampler.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/tests/training/test_train_sampler.py)
- keep exact-loss merge/scaling coverage in [`lerobot/tests/training/test_train_logging.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/tests/training/test_train_logging.py)
- keep Stage 3 complementary-metadata round-trip coverage in [`lerobot/tests/processor/test_converters.py`](/mnt/data/workspace/code/high-level-robot-planner/lerobot/tests/processor/test_converters.py)

### New tests to add

- HLRP exact accumulation regression test:
  - same synthetic supervision mix with `grad_accum_steps=1` and `grad_accum_steps>1`
  - verify equivalent exact loss weighting behavior
  - verify logged merged `batch_*_supervised_*` quantities remain identical
  - cover zero-mass windows, alternating mode, and one-rank-zero-mass cases
  - freeze denominator-driving masks (`action_is_pad`, image padding masks, `valid_pair`) in the fixture
- train-path dispatch test:
  - HLRP policy goes through HLRP-owned path
  - a normal policy stays on the generic path
- processor factory test:
  - processor runtime overrides come from factory/helper code, not the train/eval/record paths
  - include at least one inference-side caller if that path reuses the same seam
  - confirm inference paths do not require dataset stats/meta when they are not needed
- dataloader helper test:
  - dataset-owned sampler is still preferred
  - fallback `EpisodeAwareSampler` behavior stays unchanged
  - freeze current `pin_memory`, `prefetch_factor`, `shuffle`, `drop_last`, and `accelerate.prepare(...)` loader timing behavior
- mixed-dataset resize ownership test:
  - no hidden resize derived from policy config
  - explicit resize configuration, if retained, is honored
  - add this before removing the current implicit coupling, since there is no strong existing test locking it down today
  - freeze the tensors seen by both the action branch and the latent-target path across the ownership move
- normalization-equivalence test:
  - train / resume / eval / inference paths stay equivalent where they should share the same processor overrides and stats handling

### Runtime validation

- local Stage 3 smoke run on the active HLRP policy after the split with:
  - fixed seed
  - fixed sampler epoch behavior
  - same supervision-mix realization where possible
- confirm:
  - logs still show correct effective batch size
  - exact accumulation metrics remain sane
  - exact accumulation parity against a pre-refactor control run matches the frozen acceptance criteria, not a vague "close enough"
  - mixed-dataset batches still carry supervision/source metadata
  - checkpoint normalization sidecar behavior is unchanged
  - if resize ownership moves to processor-side code, dataloading/update time and memory do not regress materially on the smoke run

Experiment continuity acceptance criteria:

- same seed
- same sampler epoch behavior
- same realized supervision-mix counts where determinism permits
- same denominator totals and merged batch supervision fractions on the fixed smoke window
- any tolerated numerical drift must be predeclared narrowly and justified by the ownership move, not accepted informally

## 9. Documentation and progress-note impact

Concrete docs to update when this refactor lands:

- [`docs/felix_notes/2026-03-10_stage3_weighted_supervision_mixing.md`](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/2026-03-10_stage3_weighted_supervision_mixing.md)
  - update the ownership story around mixed-dataset resizing and exact accumulation location
- [`docs/felix_notes/foundation_model/stage2_stage3_interface_map.md`](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/foundation_model/stage2_stage3_interface_map.md)
  - update the Stage 2 -> Stage 3 interface notes if action/latent branch tensor or processor ownership descriptions move
- [`docs/felix_notes/hlrp_action_expert.md`](/mnt/data/workspace/code/high-level-robot-planner/docs/felix_notes/hlrp_action_expert.md)
  - update processor ownership notes if runtime override assembly moves out of entrypoints
- add one concise progress note under `docs/progress/` only if the implementation reveals a meaningful migration or validation result worth preserving beyond the plan
## 10. Rejected follow-ups for this change

- do not attempt to upstream the whole refactor during this initiative
- do not create a generic train-hook abstraction unless a second real user appears
- do not preserve old and new accumulation paths side by side once the HLRP split lands
- do not add Python-side hidden defaults for new dataset/trainer behavior that should live in Hydra

## 11. Residual risks

- the thin HLRP dispatch branch may still leave a small amount of policy awareness in `lerobot_train.py`
- moving processor ownership can expose previously hidden assumptions around resume/pretrained loading
- if processor override logic is only moved out of train and not actually shared, the duplication problem will persist in eval/record/inference
- removing implicit mixed-dataset resize may reveal that some active Stage 3 configs were relying on it silently
- moving helper functions out of `lerobot_train.py` can create avoidable churn if test-covered generic helpers lose a stable import home
- if the HLRP helper path grows beyond a narrow dispatch, a second refactor to a fuller trainer split may still be needed later
- moving resize later in the pipeline can shift cost from dataset workers to training/inference processors, so the smoke validation must check for an unwanted compute or memory regression

## 12. Approval summary

This plan intentionally keeps reusable fork-local capabilities in vendored LeRobot:

- mixed dataset support
- `grad_accum_steps`
- generic dataloader / sampler behavior

It moves only the clearly HLRP-shaped semantics out of the generic train internals:

- exact latent/action accumulation
- train-script-owned processor semantics

That is the smallest refactor that materially improves the fork boundary without turning the training stack into either a framework of callbacks or a prematurely duplicated second trainer.
