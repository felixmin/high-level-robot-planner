# Option 2 Selectable Compact Mixed Dataset Plan

Timestamp: 2026-03-23 11:54:08

This plan supersedes the earlier draft for the same initiative by grounding the work in the repo's current code state: the vendored LeRobot tree already contains a partial compact-path prototype and a prototype selector. The implementation goal is to make that path a clean, selectable Option 2 branch while preserving today's default runtime behavior when the selector is unset.

## 1. Change summary

Land Option 2 as an alternate mixed-dataset implementation inside vendored LeRobot by promoting the existing `compact_mixed_dataset.py` path into a supported, test-covered branch behind one explicit selector.

Required end state:

- `dataset.mix_path` still activates mixed-dataset mode
- default mixed behavior stays on the current legacy path
- Option 2 is only used when explicitly selected
- dataloader behavior branches through an explicit dataset contract, not `hasattr(dataset, "sources")`
- the change stays narrow enough that a later Option 3 can add a third implementation without rewriting this slice

## 2. Current code and docs fit

Relevant current code:

- `lerobot/src/lerobot/configs/default.py`
  - already has a prototype selector: `DatasetConfig.mix_impl` with values `"current"` and `"compact_manifest"`
- `lerobot/src/lerobot/datasets/factory.py`
  - already branches mixed-dataset construction on that prototype selector
  - constructs `LogicalSource` for the legacy path and `CompactSourceAdapter` for the compact path
- `lerobot/src/lerobot/datasets/mixed_dataset.py`
  - owns the canonical mix config parser and the current default `MixedLeRobotDataset`
  - exposes `loader_hints()` with `sampler_mode="source_block"`
- `lerobot/src/lerobot/datasets/compact_mixed_dataset.py`
  - already contains the Option 2 prototype: `CompactManifest`, `CompactSourceAdapter`, `CompactMixedDataset`, `__getitems__`, and `WeightedSampleIndexSampler`
  - already exposes `loader_hints()` with `sampler_mode="sample_level"`
- `lerobot/src/lerobot/scripts/lerobot_train.py`
  - already uses `loader_hints()` instead of checking `dataset.sources` to decide `source_block_size`
  - the mixed-dataset distinction is therefore already close to the desired seam
- `scripts/6_train_lerobot.py`
  - forwards extra dataset keys, so a selector can be passed from HLRP Hydra into vendored LeRobot without changing the basic launcher structure

Relevant tests already present:

- `lerobot/tests/datasets/test_mixed_dataset.py`
- `lerobot/tests/training/test_train_sampler.py`
- `lerobot/tests/configs/test_default.py`
- `tests/lerobot/test_mixed_dataset_feature_key_mapping.py`
- `tests/scripts/test_train_lerobot_command_builder.py`
- `tests/config/test_hydra_configs.py`

Relevant notes:

- `docs/felix_notes/2026-03-23_mixed_dataset_design_options.md`
- `docs/felix_notes/2026-03-23_mixed_dataset_option2_compact_manifest_plan.md`
- `docs/felix_notes/2026-03-23_mixed_dataset_decision_framework.md`

## 3. Design decisions and boundaries

### 3.1 Canonical selector

Use one explicit public selector on the vendored LeRobot dataset config:

- `dataset.mix_implementation: "legacy" | "compact"`

Decision:

- replace the prototype-only `mix_impl: "current" | "compact_manifest"` surface rather than adding a second selector
- default to `"legacy"`
- do not add compatibility aliases in Python

Reasoning:

- the repo already has one prototype selector; keeping both would create unnecessary surface area
- `legacy` and `compact` are clearer names and leave room for a future Option 3 value
- fail-fast renaming is cleaner than carrying both names

### 3.2 Keep `dataset.mix_path` as the only mixed activation switch

Do not change mixed activation semantics:

- `mix_path is None` -> normal single-dataset path
- `mix_path is not None` and selector unset -> legacy mixed path
- `mix_path is not None` and selector is `compact` -> Option 2 compact path

No implicit activation by dataset type, sampler type, or path contents.

### 3.3 Reuse the existing compact prototype

Treat `lerobot/src/lerobot/datasets/compact_mixed_dataset.py` as the Option 2 base, not as a throwaway prototype and not as justification for a second independent implementation.

This slice should keep:

- `CompactManifest`
- `CompactSourceAdapter`
- the compact dataset class and `__getitems__`
- the compact sample-level sampler that emits flat sample ids

This slice should not yet add:

- a new offline manifest format
- a new global sample-id schema shared with Option 3
- a rank-aware custom distributed sampler
- a second compact dataset file

### 3.4 Dataloader contract

The training path should distinguish legacy mixed vs compact mixed through the existing `loader_hints()` seam, not through `hasattr(dataset, "sources")`.

Operational contract:

- legacy mixed dataset returns `sampler_mode="source_block"`
- compact mixed dataset returns `sampler_mode="sample_level"`
- `make_offline_dataloader()` only passes `source_block_size=batch_size` for `source_block`
- compact keeps `source_block_size=1` semantics

Optional observability cleanup:

- rename or remove the prototype `mixed_impl` loader-hint field so debug output matches the canonical selector name

### 3.5 What stays out of scope

Do not change yet:

- `WeightedSourceSampler` behavior for the legacy path
- mix config file format
- Stage-3 dataset YAML defaults across `config/stage3_dataset/*.yaml`
- Option 3 manifest design
- distributed sharding strategy beyond the current Accelerate path

That keeps this landing narrow and preserves a clean future branch for Option 3.

## 4. Exact files, classes, and functions to change

### Vendored LeRobot config surface

- `lerobot/src/lerobot/configs/default.py`
  - change `DatasetConfig`
  - replace `mix_impl` with `mix_implementation`
  - allowed values: `"legacy"`, `"compact"`
  - default: `"legacy"`

### Mixed dataset construction

- `lerobot/src/lerobot/datasets/factory.py`
  - change `make_dataset()`
  - branch on `cfg.dataset.mix_implementation`
  - keep the existing legacy branch with `LogicalSource` + `MixedLeRobotDataset`
  - keep the compact branch based on `CompactSourceAdapter` + compact dataset class
  - remove prototype names (`current`, `compact_manifest`) from this file

### Legacy mixed runtime contract

- `lerobot/src/lerobot/datasets/mixed_dataset.py`
  - keep `MixedLeRobotDataset` as the default path
  - update `MixedLeRobotDataset.loader_hints()` to report canonical naming for the implementation if that field remains exposed
  - no behavior change to `build_sampler()`

### Compact Option 2 runtime

- `lerobot/src/lerobot/datasets/compact_mixed_dataset.py`
  - keep the current compact implementation as the Option 2 code path
  - make the exported class naming intentional and stable
  - keep `__getitems__` as the compact-path differentiator
  - keep the sample-level sampler local to this module
  - if both `CompactMixedDataset` and `CompactMixedLeRobotDataset` remain, choose one as the canonical name and remove the other alias in the same slice

### Dataloader behavior

- `lerobot/src/lerobot/scripts/lerobot_train.py`
  - change `make_offline_dataloader()`
  - keep `loader_hints()` as the only implementation distinction
  - remove dead locals tied to mixed detection if they are not used
  - preserve `prefetch_factor=1` for both mixed paths unless benchmarking justifies a later change

### HLRP stage-3 forwarding

- `scripts/6_train_lerobot.py`
  - verify `_write_lerobot_train_config()` forwards `lerobot.dataset.mix_implementation` unchanged
  - only change this file if the selector rename requires explicit key handling or test protection

### Tests

- `lerobot/tests/configs/test_default.py`
  - add selector-validity coverage for `mix_implementation`
- `lerobot/tests/datasets/test_mixed_dataset.py`
  - parameterize core mixed-dataset parity tests across `legacy` and `compact`
  - add compact-only `__getitems__` parity coverage
- `lerobot/tests/training/test_train_sampler.py`
  - add explicit loader-hint / `source_block_size` behavior tests for legacy vs compact
- `tests/lerobot/test_mixed_dataset_feature_key_mapping.py`
  - add one compact-path remapping/parity test if the current test coverage only exercises `LogicalSource`
- `tests/scripts/test_train_lerobot_command_builder.py`
  - add a guard that `lerobot.dataset.mix_implementation=compact` is forwarded into the generated CLI/config
- `tests/config/test_hydra_configs.py`
  - add one composition-level check that stage-3 config can carry the selector override without changing defaults

## 5. Concrete phased implementation order

### Phase 1: Canonicalize the selector without changing default behavior

1. Replace `mix_impl` with `mix_implementation` in vendored LeRobot config.
2. Update `factory.make_dataset()` to use `legacy` vs `compact`.
3. Keep the default on `legacy`.
4. Add config tests proving unset selector still resolves to the legacy path.

Success gate:

- existing mixed-dataset behavior is unchanged when no selector is provided
- selecting `compact` constructs the compact dataset class

### Phase 2: Harden the compact path as a supported alternate runtime

1. Make class naming and exports in `compact_mixed_dataset.py` intentional.
2. Confirm `loader_hints()` exposes the compact contract cleanly.
3. Add compact-path parity tests for item fetch, batch fetch, and mixed metadata.

Success gate:

- compact path returns the same sample content and ordering as repeated item fetch
- compact path remains isolated from the legacy sampler contract

### Phase 3: Lock in dataloader branching

1. Keep `make_offline_dataloader()` driven by `loader_hints()`.
2. Add tests that legacy mixed datasets still receive `source_block_size=batch_size`.
3. Add tests that compact mixed datasets never receive forced source blocks.

Success gate:

- no change in legacy runtime behavior
- compact path stays on sample-level mixing

### Phase 4: HLRP stage-3 selection path

1. Verify stage-3 command/config building forwards the new selector.
2. Add one regression test for that forwarding path.
3. Keep experiment YAMLs unchanged in this slice; selection happens by explicit override only.

Success gate:

- `lerobot.dataset.mix_implementation=compact` reaches vendored LeRobot unchanged
- default HLRP stage-3 configs remain behaviorally identical

## 6. Test and benchmark strategy

### Unit and integration tests

Required passing test set:

- `lerobot/tests/configs/test_default.py`
- `lerobot/tests/datasets/test_mixed_dataset.py`
- `lerobot/tests/training/test_train_sampler.py`
- `tests/lerobot/test_mixed_dataset_feature_key_mapping.py`
- `tests/scripts/test_train_lerobot_command_builder.py`
- `tests/config/test_hydra_configs.py`

Key assertions:

- selector default is legacy
- `mix_path` activation still works unchanged
- legacy and compact both expose the same mixed supervision/source metadata contract
- compact `__getitems__` preserves content and batch order
- legacy keeps source-block sampler behavior
- compact does not inherit source-block sampler behavior

### Benchmark plan

Do not commit a benchmark harness in the initial landing unless the existing test suite is insufficient.

Benchmark as a separate validation pass using the same mix file and hardware across both implementations:

- implementation: `legacy` vs `compact`
- batch size: `64`
- workers: `4` and `10`
- same `mix_path`, image size, policy, and seed

Measure:

- steady-state batches/s or samples/s
- p50 and p95 batch latency over a fixed warm run
- parent RSS
- summed worker RSS
- source histogram per batch for sanity

Go/no-go target for adopting the alternate path in experiments:

- default legacy path unchanged
- compact path shows a clear memory win or a clear locality/throughput win without semantic regressions

## 7. Documentation and progress-note impact

This slice should only add or update the canonical plan artifact during planning.

For later implementation:

- no user-facing docs change is required to land the code
- if the compact path benchmarks well enough to be used in experiments, add a short `docs/progress/` note or equivalent run note describing:
  - selector name
  - default behavior
  - benchmark outcome
  - when to use `compact`

## 8. Risks, cleanup, open questions, and Hydra/default ownership

### Primary risks

- the repo already contains partial Option 2 integration, so the real risk is leaving it half-renamed and half-tested
- class naming drift (`CompactMixedDataset` vs `CompactMixedLeRobotDataset`) can create needless confusion if not resolved in the same slice
- if tests only cover construction, loader behavior can silently regress back toward legacy source blocking

### Required cleanup

Remove prototype-only adjacent code in the same change:

- `mix_impl`
- `"current"`
- `"compact_manifest"`
- any dead loader-hint keys or locals that only exist for the prototype naming

Do not keep both the old and new selector names.

### Hydra/default ownership

Source of truth for this slice:

- vendored LeRobot `DatasetConfig` owns the default selector value because that is the actual runtime config schema

HLRP Hydra stance for this slice:

- do not add `mix_implementation` to every stage-3 dataset YAML yet
- keep HLRP defaults unchanged and rely on the vendored LeRobot default for the unset case
- use explicit Hydra override when selecting the compact path

This is the minimal clean split between the vendored runtime config and HLRP experiment config. If compact becomes the normal experimental path later, then add the field explicitly to the stage-3 dataset configs at that time.

### Open questions that matter

1. Should the compact dataset keep both class names for compatibility, or should the implementation standardize on a single exported class name immediately?
2. Is there any out-of-repo consumer already using `mix_impl=current|compact_manifest`? If yes, that migration needs to be called out before implementation; if no, remove the prototype names outright.
