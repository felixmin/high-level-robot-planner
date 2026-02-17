# Stage 2 -> Stage 3 Interface Map (SmolVLA Shared)

## Summary

Current handoff from Stage 2 to Stage 3 is **artifact-only**.

- Stage 2 (`scripts/4_train_foundation.py`) writes:
  - `<run_dir>/artifacts/smolvla_shared_stage2_artifact.pt`
- Stage 3 (`scripts/6_train_lerobot.py`) forwards:
  - `--policy.stage2_artifact=...`
- LeRobot policy (`HLRPSmolVLASharedPolicy`) loads the artifact and applies:
  - manifest compatibility checks (model/dtype/image/latent/flow params)
  - controlled state load where only `action_head.*` may be missing

There is no legacy Stage-2 checkpoint fallback in this path.

## Artifact Contract

Schema version:
- `smolvla_shared.v1`

Payload keys:
- `schema_version`
- `manifest`
- `core_state_dict`

`manifest` fields:
- `schema_version`
- `model_name`
- `torch_dtype`
- `image_size`
- `action_dim`
- `latent_vector_dim`
- `flow_hidden_dim`
- `flow_steps`
- `min_period`
- `max_period`
- `time_beta_alpha`
- `time_beta_beta`
- `source_backend`
- `source_training_mode`
- `source_run_dir`
- `source_global_step`

## Failure Behavior

Fail-fast behavior:
- Missing artifact path -> `FileNotFoundError`
- Bad schema/version -> `ValueError`
- Malformed payload/state dict -> `TypeError`/`KeyError`
- Stage-3 config/manifest mismatch -> `ValueError`
- Unexpected model key mismatch -> `RuntimeError`
- Missing keys other than `action_head.*` -> `RuntimeError`

## Minimal Flow

1. Run Stage 2 with `model.backend=smolvla_shared` and a valid `model.laq.checkpoint`.
2. Wait for artifact at `artifacts/smolvla_shared_stage2_artifact.pt`.
3. Run Stage 3 with `lerobot.stage2_artifact=<artifact_path>`.
4. Train/eval in LeRobot with shared core warm-started from artifact.
