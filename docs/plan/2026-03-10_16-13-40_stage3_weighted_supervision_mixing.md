# Stage 3 Weighted Supervision Mixing Plan

## 1. Change summary

Replace the Stage 3 policy-internal ratio mask with repo-owned dataset-side logical sources in the vendored `lerobot/` runtime in this repo.

Canonical behavior:

- `stage3_dataset` selects a committed repo-local mix YAML.
- The mix YAML defines named logical sources over whole episodes.
- Each source sets:
  - episode coverage,
  - supervision mode,
  - sampling weight.
- The dataset stamps fixed reserved booleans into each item:
  - `hlrp_action_supervised`
  - `hlrp_latent_supervised`
- The HLRP policy reads those booleans and masks losses accordingly.

The config separates supervision coverage and sampling weight as knobs, while acknowledging that oversampling also changes optimization pressure and therefore experiment meaning.

Weighted sampling stays with replacement in v1. We target expected source draw proportions, not strict exhaustion guarantees.

## 2. Ownership and activation

### Runtime owner

The only runtime to change is the vendored `lerobot/` tree in this repo.

- `../lerobot_dataset_mixer` is prior art only.
- No Stage 3 runtime dependency on sibling checkouts.

### Config owner

Supervision ownership lives entirely in `stage3_dataset`.

- Mix YAMLs live under `config/stage3_dataset_mix/`.
- `config/stage3_dataset/*.yaml` points `lerobot.dataset.mix_path` to one of those files.
- `stage3_profile` remains model/training-only and must not encode supervision split semantics.

### `mix_path` resolution owner

Because `lerobot-train` runs from the run directory, repo-local mix paths must be resolved to absolute paths before launch.

Planned owner:

- `scripts/6_train_lerobot.py` resolves `lerobot.dataset.mix_path` against `workspace_root` and emits an absolute `--dataset.mix_path=...` CLI value.

This must be covered by command-builder tests.

## 3. Final design

### Mix schema

Implement a local mix loader in vendored `lerobot/`. Each source supports:

- `name: str`
- `repo_id: str`
- `root: str | null`
- `revision: str | null`
- `weight: float`
- `episodes: list[int] | null`
- `exclude_episodes: list[int] | null`
- `supervision: latent_only | multitask`
- `video_backend: str | null`
- `tolerance_s: float | null`

Rules:

- `episodes` is explicit inclusion.
- `exclude_episodes` excludes from the full dataset.
- specifying both is invalid.
- source overlap is invalid in v1.
- empty effective sources are invalid.
- all sources in one mix must match on physical dataset compatibility:
  - camera keys and image/video layout,
  - action/state/task feature schema,
  - normalization/stat feature availability and shapes,
  - delta-timestamp compatibility for the active policy.

### Dataset identity contract

Use two different identities deliberately:

- `dataset.repo_id` may be a logical Stage 3 dataset id for mixed runs
- the mix definition is authoritative for the actual physical source datasets via per-source `repo_id` / `root` / `revision` / related fields

Implications:

- mixed datasets spanning multiple physical datasets are allowed in principle
- downstream Stage 3 code must stop interpreting `dataset.repo_id` as the physical dataset identity
- downstream fallback/rebuild paths must read resolved mix metadata or resolved mixed stats instead

Concrete migration targets include:

- `scripts/6_train_lerobot.py`
- `lerobot_policy_hlrp/.../checkpoint_stats.py`
- any Stage 3 stats/checkpoint/reporting code that currently assumes one physical dataset from `dataset.repo_id`

### Supervision contract

Keep the batch contract fixed and narrow:

- `latent_only` source stamps:
  - `hlrp_action_supervised=false`
  - `hlrp_latent_supervised=true`
- `multitask` source stamps:
  - `hlrp_action_supervised=true`
  - `hlrp_latent_supervised=true`

No configurable field names. No arbitrary fixed-field bag.

### Policy contract and compatibility decision

Update the HLRP Stage 3 policy to require the reserved supervision booleans in training and remove the runtime ratio-split path in this same initiative.

Compatibility decision:

- active Stage 3 configs/tests/docs are migrated now
- old ratio-based Stage 3 runtime configs are allowed to break
- if checkpoint/config deserialization needs a narrow compatibility shim, keep it deserialization-only and do not preserve old runtime behavior

### Normalization-stats / checkpoint fallback

Checkpoint sidecar normalization stats remain the primary source of truth.

If sidecar rebuild is required:

- rebuild from the resolved mixed dataset stats derived from the authoritative mix metadata
- do not attempt to rebuild from `dataset.repo_id` alone

This makes `checkpoint_stats.py` a concrete migration target in this initiative.

### `stage3_training_mode` compatibility

Add fail-fast validation before training:

- `action` requires at least one action-supervised source
- `latent` requires at least one latent-supervised source
- `multitask` and `alternating` require at least one source with each needed supervision branch over the run
- invalid or ambiguous combinations are rejected at preflight / launch time

## 4. Concrete implementation steps

### Phase 1: Repo-owned mix support in local `lerobot/`

Implement the minimal weighted logical-source dataset path directly in vendored `lerobot/`.

Expected file areas:

- `lerobot/src/lerobot/configs/default.py`
  - add `mix_path: str | None = None`
- `lerobot/src/lerobot/datasets/`
  - add local mix config loader
  - add local logical-source dataset implementation
  - add or adapt weighted sampler
- `lerobot/src/lerobot/datasets/factory.py`
  - route `dataset.mix_path` through the local mix loader
- `lerobot/src/lerobot/scripts/lerobot_train.py`
  - use `dataset.build_sampler()` when present
  - verify interaction with resume/epoch logic rather than assuming the hook is isolated

### Phase 2: Stage 3 launcher and policy updates

Update:

- `scripts/6_train_lerobot.py`
  - resolve repo-local `mix_path` to an absolute path before launch
  - persist stable mix information in the saved train config so fallback rebuild can recover the authoritative source metadata
- `config/lerobot_policy/smolvla_shared_base.yaml`
- `lerobot_policy_hlrp/.../configuration_hlrp_smolvla_shared.py`
- `lerobot_policy_hlrp/.../modeling_hlrp_smolvla_shared.py`

Changes:

- remove `action_subset_ratio`, `action_subset_key`, and `latent_scope` from active Stage 3 config wiring
- require the reserved supervision booleans from the dataset path
- retain `stage3_training_mode`
- log three different quantities during runtime:
  - sampled source proportions,
  - action-supervised sample mass,
  - latent-supervised valid-pair mass after `valid_pair` masking

### Phase 3: Single-source-of-truth preflight

Add one preflight path that reuses the same mix loader / source-selection code as runtime, not a second independent parser.

Preflight reports per source:

- selected episode count
- effective anchor count after `drop_n_last_frames`
- supervision mode
- sampling weight
- normalized expected source draw probability
- physical dataset identity
 - logical dataset id plus authoritative physical source metadata

Preflight also validates:

- disjoint episode coverage
- valid `stage3_training_mode` vs source supervision combination
- non-empty effective iteration space

### Phase 4: Hydra/config migration

Replace the legacy Hydra wiring explicitly:

- `config/experiment/_stage3_base.yaml`
  - delete forwarding of ratio/latent-scope fields
  - keep `stage3_dataset` as the only supervision selector
- `config/stage3_profile/action_scratch.yaml`
  - delete old ratio/latent-scope entries
- `config/stage3_profile/multitask_scratch.yaml`
  - delete old ratio/latent-scope entries
- `config/experiment/stage3__latent_vs_multitask_sweep.yaml`
  - replace ratio-based variants with `stage3_dataset` mix variants

Add committed mix YAMLs under `config/stage3_dataset_mix/` and matching `config/stage3_dataset/*.yaml` presets.

Low-label protocol:

- use committed episode manifests generated by a repo-owned utility
- require either stratified subset construction or multiple subset seeds for comparison experiments
- store the manifest or seed list in repo-tracked config/docs

## 5. Migration inventory

Update or remove these legacy surfaces:

- `config/experiment/_stage3_base.yaml`
- `config/stage3_profile/action_scratch.yaml`
- `config/stage3_profile/multitask_scratch.yaml`
- `config/experiment/stage3__latent_vs_multitask_sweep.yaml`
- `tests/config/test_hydra_configs.py`
- `tests/config/test_hlrp_smolvla_shared_config.py`
- `tests/scripts/test_train_lerobot_command_builder.py`
- `tests/stage2/test_stage2_stage3_transform_parity.py`
- `tests/lerobot/test_hlrp_smolvla_shared_normalization_stats.py`
- `scripts/6_train_lerobot.py`
- `docs/felix_notes/plan/stage3_split_latent_cotrain.md`
- `docs/felix_notes/data/lerobotv3_integration.md`
- `docs/felix_notes/foundation_model/stage2_stage3_interface_map.md`

## 6. Test and validation strategy

### Config and launch tests

- `tests/scripts/test_train_lerobot_command_builder.py`
  - repo-local relative `mix_path` becomes an absolute repo path
  - emitted command includes absolute `--dataset.mix_path=...`
- `tests/config/test_hydra_configs.py`
  - `stage3_dataset=<mix preset>` is sufficient to activate the new path
  - invalid `stage3_profile` / `stage3_dataset` supervision combinations fail
- `tests/config/test_hlrp_smolvla_shared_config.py`
  - active policy config no longer expects ratio fields
  - any deserialization-only compatibility shim is covered separately

### Actual LeRobot path tests

Add vendored-LeRobot tests that exercise the real training/collation path, not only policy units:

- reserved supervision booleans survive collation into batch tensors
- mixed dataset iteration yields non-empty real batches
- logical dataset id and authoritative physical source metadata are both preserved through the resolved dataset object / saved config
- normalization-stat fallback rebuild uses mix metadata / resolved mixed stats rather than `dataset.repo_id`

### Sampler / DDP rollout gate

Before cluster rollout, define and prove the sampler contract:

- `__len__` is the per-rank epoch length used by the trainer
- per-rank sample counts are correct
- reseeding across epochs is explicit via `set_epoch()` or equivalent
- rank sharding does not accidentally duplicate the full unsharded sample stream
- resume/epoch logic in the trainer remains correct with `dataset.build_sampler()`

Required tests:

- multi-rank sampler test for disjoint per-rank sampling behavior
- deterministic epoch-to-epoch reseeding test
- sampled source proportions under actual workers/ranks are close to expected proportions over a meaningful window

### Correctness and scientific validation

1. Metadata preflight for each committed mix:
   - exact episode coverage
   - exact effective anchor counts
   - expected source draw proportions
2. Small real-iteration smoke check:
   - materialize a few batches through the real dataloader path
   - confirm non-empty runtime iteration and supervision fields
3. Local Stage 3 smoke training:
   - compare realized source proportions to expected proportions
   - measure realized latent supervision mass after `valid_pair` masking
4. Deterministic held-out evaluation protocol for comparing mix regimes:
   - fixed eval tasks/checkpoints/seeded subset definitions
   - compare balanced vs action-heavy mixes on the same held-out setup

### Performance validation

Measure separately:

- sampler overhead
- dataset/video decode overhead
- worker memory / duplicate reader cost from duplicated logical sources

If duplicate source objects/readers are materially expensive, record the minimal mitigation needed before broader rollout.

## 7. Risks and rollout notes

Main risks:

- duplicated logical sources may duplicate dataset objects, readers, and worker memory
- `dataset.build_sampler()` may interact with trainer epoch/resume logic
- weighted-with-replacement sampling only gives expected, not exact, source proportions
- realized latent supervision mass can differ from source coverage because of `valid_pair` masking

Rollout:

- start with local single-GPU smoke runs
- land the sampler/DDP contract proof or minimal fix before cluster use
- migrate active Stage 3 configs/docs in the same change series
