# LAM Shared Pixel Decoder Multi-Objective Plan

## 1. Change summary

Expand Stage-1 pixel reconstruction from one loss path into three training objectives that all reuse the existing shared pixel decoder and pixel head:

- real latent action tokens -> reconstruct the future frame
- invalid token sequence `[-1] * code_seq_len` -> reconstruct the current frame
- random token sequence -> reconstruct the future frame with a small configurable loss weight

The implementation should keep one `pixel_decoder` module and one `pixel_to_pixels` head, run them multiple times per forward pass, and move the new behavior into Hydra-owned `pixel_decoder` config instead of Python-side defaults.

## 2. Current code and docs fit

- [packages/lam/models/decoder_losses.py](/mnt/data/workspace/code/high-level-robot-planner/packages/lam/models/decoder_losses.py) currently has a single `compute_pixel_loss()` branch that decodes future pixels from `pixel_context + action_tokens` and logs `pixel_loss`.
- [packages/lam/models/latent_action_model.py](/mnt/data/workspace/code/high-level-robot-planner/packages/lam/models/latent_action_model.py) already has the right shared inputs for this change:
  - one pixel-context projection from the first frame
  - one `pixel_decoder`
  - one `pixel_to_pixels`
  - one `decode()` path used by inference/visualization fallback
- [packages/lam/task.py](/mnt/data/workspace/code/high-level-robot-planner/packages/lam/task.py) and [packages/lam/checkpoints.py](/mnt/data/workspace/code/high-level-robot-planner/packages/lam/checkpoints.py) both still parse `use_pixel_decoder` as a flat bool.
- [config/model/lam.yaml](/mnt/data/workspace/code/high-level-robot-planner/config/model/lam.yaml) has no nested pixel-decoder block yet.
- Existing LAM tests cover model construction, forward loss, inference, and Hydra parsing, but there is no checkpoint-focused LAM test file today.
- `docs/progress/` does not exist in this repo, so there is no existing progress-note target to update.

## 3. Design decisions and boundaries

### Final config shape

Replace `use_pixel_decoder` as the source of truth with a narrow nested Hydra block in [config/model/lam.yaml](/mnt/data/workspace/code/high-level-robot-planner/config/model/lam.yaml):

```yaml
pixel_decoder:
  enabled: true
  real_future:
    enabled: true
    loss_weight: 1.0
  invalid_current:
    enabled: true
    loss_weight: 1.0
    invalid_token_id: -1
  random_future:
    enabled: true
    loss_weight: 0.05
```

This stays specific to the three requested objectives instead of introducing a generic branch framework.

### Shared-weight rule

- Keep `pixel_decoder.*` and `pixel_to_pixels.*` as the only trainable decoder/head weights for pixel reconstruction.
- Do not instantiate extra decoder or head modules for the new objectives.
- `decode()` and `inference()` continue to use the real-token future reconstruction path only.

### Invalid-token representation

- Treat the invalid sequence as synthetic token ids of shape `[B, code_seq_len]` filled with `-1`.
- Never pass `-1` through NSVQ indexing.
- Add a dedicated pixel-decoder-only invalid conditioning embedding/tensor that is expanded into the decoder action grid when `invalid_current.enabled=true`.
- Real code ids remain `0..codebook_size-1`, so the invalid branch cannot collide with real VQ ids or reuse a real codebook entry by accident.

### Random-token branch behavior

- Sample valid random code ids uniformly from `0..codebook_size-1`.
- Convert them to action embeddings through the existing VQ codebook lookup, but detach the sampled embeddings before the pixel loss branch uses them.
- This keeps the random branch as a decoder/context regularizer instead of injecting random codebook-training pressure.
- Default `random_future.loss_weight` should stay small (`0.05` by default, and not above `0.1` in baseline configs).

### Backward-compatibility boundary

- Old configs and checkpoint hyperparameters that only contain `use_pixel_decoder` must still load.
- Upgrade missing `pixel_decoder` blocks inside config-parsing code by synthesizing a legacy-equivalent block:
  - `enabled = use_pixel_decoder`
  - `real_future.enabled = use_pixel_decoder`
  - `real_future.loss_weight = 1.0`
  - `invalid_current.enabled = false`
  - `random_future.enabled = false`
- Keep existing state-dict names `pixel_decoder.*` and `pixel_to_pixels.*` unchanged so old checkpoints still load strictly when the synthesized legacy config is used.
- New invalid-branch parameters must only exist when that branch is enabled; otherwise old checkpoints would gain unexpected missing keys under strict load.

### Out of scope

- No change to DINO, flow, or aux-decoder behavior beyond plumbing around the shared pixel loss path.
- No new inference mode for invalid/random branches.
- No generic token-language semantics for `-1` outside this Stage-1 pixel-decoder training feature.

## 4. Concrete implementation steps

### Phase 1: Hydra ownership and config normalization

Update:

- [config/model/lam.yaml](/mnt/data/workspace/code/high-level-robot-planner/config/model/lam.yaml)
- [packages/lam/task.py](/mnt/data/workspace/code/high-level-robot-planner/packages/lam/task.py)
- [packages/lam/checkpoints.py](/mnt/data/workspace/code/high-level-robot-planner/packages/lam/checkpoints.py)

Changes:

- Add the nested `pixel_decoder` block to Hydra defaults.
- Add one shared normalization path so both task construction and checkpoint reconstruction translate legacy `use_pixel_decoder` configs into the new block instead of duplicating ad hoc parsing.
- Treat the nested block as the only model-constructor input after normalization.

### Phase 2: Model/config dataclasses and decoder wiring

Update:

- [packages/lam/models/latent_action_model.py](/mnt/data/workspace/code/high-level-robot-planner/packages/lam/models/latent_action_model.py)

Changes:

- Add minimal pixel-decoder config dataclasses or equivalent validated containers next to `DinoConfig`.
- Replace the flat `use_pixel_decoder` constructor flag with the normalized pixel-decoder config object.
- Keep `_get_enabled_training_decoders()` keyed off `pixel_decoder.enabled`.
- Keep `_initialize_decoder_modules()` building only one `pixel_decoder` and one `pixel_to_pixels`.
- Add the small amount of new model state needed for invalid conditioning, guarded behind `invalid_current.enabled`.
- Add helpers for:
  - building invalid branch action tokens from the dedicated invalid embedding
  - sampling random valid ids and mapping them to detached embeddings

### Phase 3: Multi-branch pixel loss orchestration

Update:

- [packages/lam/models/decoder_losses.py](/mnt/data/workspace/code/high-level-robot-planner/packages/lam/models/decoder_losses.py)
- [packages/lam/models/latent_action_model.py](/mnt/data/workspace/code/high-level-robot-planner/packages/lam/models/latent_action_model.py)

Changes:

- Replace `compute_pixel_loss()` with one shared multi-branch pixel-loss routine.
- Reuse the same decoder/head three times with different action-token sources and targets:
  - real tokens + future frame
  - invalid tokens + current frame
  - random tokens + future frame
- Return one summed weighted pixel loss term for total-loss composition.
- Log branch-specific metrics such as:
  - `pixel_real_future_loss`
  - `pixel_invalid_current_loss`
  - `pixel_random_future_loss`
  - `pixel_random_future_weight`
  - `pixel_loss` as the aggregate weighted pixel objective for continuity
- Keep `return_recons_only` and `decode()` pinned to the real-token future reconstruction path so visualization behavior does not become ambiguous.

### Phase 4: Compatibility cleanup and narrow follow-through

Update:

- [packages/lam/inference.py](/mnt/data/workspace/code/high-level-robot-planner/packages/lam/inference.py) only if the new invalid-conditioning state should also be pruned from inference wrappers
- Any direct LAM config fixtures that should migrate off `use_pixel_decoder`

Changes:

- Preserve legacy config loading, but remove new runtime dependence on `use_pixel_decoder` once the normalized block exists.
- Keep visualization/inference fallback checks keyed to `pixel_decoder` module existence, not to individual branch toggles.

## 5. Test and validation strategy

Update:

- [tests/lam/models/test_lam_model.py](/mnt/data/workspace/code/high-level-robot-planner/tests/lam/models/test_lam_model.py)
- [tests/lam/test_task.py](/mnt/data/workspace/code/high-level-robot-planner/tests/lam/test_task.py)
- Add [tests/lam/test_checkpoints.py](/mnt/data/workspace/code/high-level-robot-planner/tests/lam/test_checkpoints.py)
- Touch stage-2 tests only where they construct minimal LAM configs and need the new normalized shape

Targeted coverage:

- model init with the nested `pixel_decoder` block
- legacy bool-only config upgrade still builds the old single-branch behavior
- forward pass with all three pixel branches enabled logs separate metrics and returns a finite total loss
- invalid branch uses the dedicated invalid conditioning path rather than NSVQ indexing
- random branch weight comes from config and stays small by default
- inference and `return_recons_only` still return the real-token future reconstruction
- checkpoint reconstruction from old hyperparameters missing `pixel_decoder` still succeeds
- strict old-checkpoint weight loading remains valid when branch upgrade synthesizes the legacy-equivalent config

Recommended targeted commands during implementation:

- `conda run -n hlrp pytest -q tests/lam/models/test_lam_model.py tests/lam/test_task.py tests/lam/test_checkpoints.py`
- Add the smallest necessary stage-2 regression test if a stage-2 fixture breaks because it assumed `use_pixel_decoder` was the source of truth.

## 6. Documentation and progress-note impact

- Update inline comments in [config/model/lam.yaml](/mnt/data/workspace/code/high-level-robot-planner/config/model/lam.yaml) so the three pixel objectives and their default weights are clear.
- No `docs/progress/` update is required for this initiative because that directory does not exist in the current repo.
- No broader docs change is required unless implementation surfaces a user-facing checkpoint-loading caveat worth documenting.

## 7. Risks, cleanup, and open questions that matter

- The invalid-current branch may dominate early training if it is weighted equally with the main future branch. Keep the config explicit so weights can be reduced quickly without code edits.
- Running the shared decoder three times increases Stage-1 step cost. The plan keeps scope small by avoiding a larger architectural refactor, but implementation should treat this as a measurable runtime regression risk.
- Strict checkpoint compatibility depends on branch-gated parameter creation. If invalid-branch state is always created, old strict loads will break.
- Open question: whether `invalid_current.loss_weight` should default to `1.0` or a lower value in the baseline Hydra config. The requested behavior suggests equal weighting, but this should be validated in the first short run.

## 8. Legacy-code removal and Hydra-default ownership

- After the normalized `pixel_decoder` block is in place, `use_pixel_decoder` should remain only as a legacy input shim in config-loading code.
- New runtime logic should not branch on the legacy bool once the normalized config object exists.
- Avoid a generic branch registry or plugin system. The repo only needs three explicit pixel objectives.
- Hydra owns all branch enable flags, loss weights, and the invalid token id default. Python should only validate and consume the normalized config.
