# Stage-3 Split Cotraining Plan (Latent on Y, Action+Latent on X)

## Objective
Add a strict Stage-3 training mode behavior for one run:
- `X` subset: optimize **action flow + latent flow**
- `Y` subset: optimize **latent flow only**

Target use case: e.g. first 10% supervised (`X`) and remaining 90% latent-only (`Y`) on Libero.

## What Exists Today

### Current Stage-3 policy path
- `HLRPSmolVLASharedPolicy.forward(...)` already computes:
  - action flow loss in `action|multitask`
  - latent flow loss in `latent|multitask`
- In `multitask`, action loss is computed on the **full batch**.
- Latent path already has per-sample filtering (`valid_pair`) from online LAQ pair validity.

### Batch IDs available from LeRobot
LeRobot includes stable identifiers in training batches:
- `index` (global frame index)
- `episode_index`

So we can derive a deterministic per-sample supervision mask in policy code without changing dataset format.

## Target Workflow

1. Build `FoundationBatch` once for the current step (already done).
2. Compute action-supervision mask from configured key (`index` or `episode_index`).
3. If action loss is active:
   - slice `FoundationBatch` by mask
   - slice action targets by same mask
   - compute action flow only on selected samples
4. If latent loss is active:
   - keep current LAQ valid-pair filtering behavior
5. Combine losses according to existing mode logic.

## Design Decisions

- Keep existing `stage3_training_mode` semantics unchanged.
- Add required policy config fields:
  - `action_subset_ratio` in `(0, 1]`
  - `action_subset_key` in `{index, episode_index}`
- Add required latent-branch scope field:
  - `latent_scope` in `{all, action_subset}`
- Implement **prefix split** semantics:
  - supervised if `batch[key] < floor(ratio * total_count)`
  - total_count = `dataset_meta.total_frames` for `index`
  - total_count = `dataset_meta.total_episodes` for `episode_index`
- No dataset/loader rewrite, no fallback routing.

## Dataflow Diagram

```mermaid
flowchart TD
    A[LeRobot batch]\n(index, episode_index, images, state, action, task) --> B[HLRP Policy Forward]
    B --> C[Build FoundationBatch]
    B --> D[Build action targets]
    B --> E[Compute action supervision mask\nkey + ratio + dataset_meta]

    E -->|mask True| F[Slice FoundationBatch + targets for action branch]
    F --> G[action_flow_loss]

    B --> H[LAQ online target + valid_pair]
    H --> I[Slice FoundationBatch for latent-valid samples]
    I --> J[latent_flow_loss]

    G --> K[weighted sum by mode]
    J --> K
    K --> L[total loss + metrics]
```

## Config Surface

Policy fields (Hydra -> `lerobot.policy.*`):
- `action_subset_ratio`: float
- `action_subset_key`: `index` or `episode_index`
- `latent_scope`: `all` or `action_subset`

Recommended values:
- standard full cotrain: `ratio=1.0`
- 10% supervised split by frame index: `ratio=0.1`, `key=index`
- 10% supervised split by episodes: `ratio=0.1`, `key=episode_index`
- apples-to-apples latent scope: `latent_scope=action_subset`
- semi-supervised latent scope: `latent_scope=all`

## File-Level Change Plan

1. `lerobot_policy_hlrp/.../configuration_hlrp_smolvla_shared.py`
- add config fields
- validate ratio/key in `__post_init__`

2. `lerobot_policy_hlrp/.../modeling_hlrp_smolvla_shared.py`
- add helper to compute action supervision threshold
- add helper to compute per-batch action-supervision mask
- apply mask before `core.action_flow_loss(...)`
- log metrics: selected action samples and fraction

3. `config/experiment/stage3_hlrp_libero_action_scratch.yaml`
4. `config/experiment/stage3_hlrp_libero_multitask_scratch.yaml`
- set explicit policy fields (`ratio`, `key`)

5. tests
- extend config tests for new required fields
- add lightweight policy mask behavior tests (CPU-only)

## Validation Plan (No full GPU run)

1. Hydra compose tests (stage-3 experiments load).
2. Unit tests for config validation.
3. Unit tests for policy action-supervision mask behavior:
   - ratio `1.0` selects all
   - ratio `0.1` selects expected prefix subset
4. existing stage2/stage3 transform parity tests to ensure no regression.

## Notes
- This change is sample-level selection. It is orthogonal to `alternating` (step-level mode scheduling).
- Alternating can still be used together with this mask.
