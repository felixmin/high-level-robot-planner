# Terminology and Naming Guidelines

This document defines canonical naming across Stage 1/2/3 code, configs, logs, and docs.

## Goals

- Use one term per concept.
- Separate pipeline-stage names from model-family names.
- Keep Stage 2 naming architecture-agnostic at the abstraction layer.
- Remove legacy `LAQ/laq` and ambiguous `foundation` usage over time.

## Canonical Terms

- `stage1`: Pipeline step for latent action model pretraining.
- `lam`: Model family used in Stage 1 (replaces `laq`).
- `stage2`: Pipeline step for policy pretraining.
- `policy`: Generic model abstraction used in Stage 2 and Stage 3.
- `vla`: A specific policy family (not a generic synonym for Stage 2).
- `stage3`: Pipeline step for downstream fine-tuning/evaluation.

## Naming Rules

### 1) Stage names vs model names

- Use `stageN` only for orchestration, pipeline references, run naming, and stage dependencies.
- Use model-family names (`lam`, `vla`, `xvla`, etc.) for model-specific fields, methods, and modules.

Examples:

- Good: `model.stage1.checkpoint` (stage dependency).
- Good: `lam_checkpoint_path` (LAM-specific teacher/model argument).
- Avoid: using `stage1_*` as prefix for LAM-specific hyperparameters.

### 2) Stage 2 abstraction naming

Stage 2 abstractions must not be VLA-specific.

- Preferred module name: `PolicyLightningModule`.
- Preferred backend interface name: `PolicyBackend`.

For the batch type, use stage-scoped naming instead of policy-family naming:

- Preferred batch name: `Stage2Batch`.

Rationale: Stage 1 already uses `Stage1Batch`; stage-scoped batch names avoid ambiguity and keep boundaries clear.

### 3) VLA naming scope

- Keep `vla` only in concrete VLA-family implementations.
- Do not use `vla` in generic Stage 2 interfaces.

### 4) Legacy `foundation` naming

- Treat `foundation` as legacy namespace.
- Do not introduce new public names with `foundation`.
- Migrate to `stage2` or `policy` names depending on whether the concept is stage-level or abstraction-level.

## Config Conventions

- Stage dependency links:
  - `model.stage1.checkpoint`
  - `model.stage2.artifact`

- LAM-specific settings:
  - `lam_checkpoint_path`
  - `lam_future_frames`
  - `lam_camera_keyss`
  - `lam_resize_hw`
  - `lam_code_seq_len`
  - `lam_codebook_size`

- Optional future generic teacher schema:
  - `teacher.type=lam`
  - `teacher.checkpoint_path=...`
  - `teacher.future_frames=...`
  - `teacher.camera_keys=[...]`

## Scripts and Artifact Names

- Scripts:
  - `2_train_stage1_lam.py`
  - `4_train_stage2_policy.py`
  - `6_train_stage3_policy.py`

- Checkpoints/artifacts:
  - `stage1-lam-step{step:06d}.ckpt`
  - `stage2-policy-...`
  - `stage2_policy_artifact.pt`

## Logger and Metric Names

- Logger namespaces:
  - `stage1.training`
  - `stage2.training`
  - `stage3.training`

- Metric prefixes:
  - `stage1/...`
  - `stage2/...`
  - `stage3/...`

- Model-specific metric groups:
  - `lam/...`
  - `policy/...`
  - `vla/...` (only for VLA-specific internals)

## Legacy-to-Canonical Mapping

- `LAQ` -> `LAM`
- `laq_*` -> `lam_*` (model-specific)
- `foundation` (generic Stage 2 naming) -> `stage2` or `policy`
- `VLABackend` (generic interface) -> `PolicyBackend`
- `VLATokenBackendLightningModule` -> `PolicyLightningModule`
- `FoundationBatch` -> `Stage2Batch`

## Migration Plan

1. Freeze terminology in docs and new PRs.
2. Rename config keys and CLI overrides (`laq_*` -> `lam_*`) with temporary aliases.
3. Rename Stage 2 generic interfaces (`VLA*` -> `Policy*`, `FoundationBatch` -> `Stage2Batch`) with aliases.
4. Rename scripts and public docs/examples to canonical names.
5. Remove aliases after one cleanup cycle.

## Repository Notes (current state)

- Stage 1 currently uses `Stage1Batch`.
- Stage 2/3 transfer paths currently use `FoundationBatch`.
- This guideline sets the target terminology regardless of current transitional naming.
