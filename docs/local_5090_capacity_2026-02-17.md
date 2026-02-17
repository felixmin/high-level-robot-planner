# Local 5090 Capacity Check (2026-02-17)

## Goal
Validate what is practical on the local workstation (RTX 5090 32GB VRAM) for quick iteration, while cluster jobs are queued.

## Environment
- Stage 2: `conda run -n hlrp`
- Stage 3: `conda run -n lerobot`
- Policy install path: `lerobot_policy_hlrp` via `scripts/6_train_lerobot.py`

## Stage 2 Sweep (Synthetic OXE)
Config pattern:
- `experiment=vla_smol_flow_shared`
- `model.laq.checkpoint=/mnt/data/workspace/code/high-level-robot-planner/laq-stepstep052500.ckpt`
- `data=laq_oxe_all_smoke`
- `data.adapter.tf.debug.use_synthetic_data=true`
- `training.max_steps=15`
- `logging.use_wandb=false`

Results:

| batch_size | status | elapsed_sec | peak_vram_mib | peak_gpu_util_pct |
|---|---|---:|---:|---:|
| 16 | ok | 47.57 | 14444 | 43 |
| 32 | ok | 47.02 | 18966 | 94 |
| 48 | ok | 49.90 | 23268 | 92 |
| 64 | ok | 50.31 | 27506 | 98 |

Notes:
- `batch_size=64` still fits on local 5090 for Stage 2 synthetic tests.
- These elapsed values include startup overhead (LAQ + TF stack init), so they are conservative for pure train-step timing.

## Stage 2 Sweep (Real OXE Smoke Data)
Config pattern:
- `experiment=vla_smol_flow_shared`
- `model.laq.checkpoint=/mnt/data/workspace/code/high-level-robot-planner/laq-stepstep052500.ckpt`
- `data=laq_oxe_all_smoke`
- `training.max_steps=10`
- `training.validation.check_interval=10`
- `training.validation.limit_batches=1`
- `logging.use_wandb=false`

Results:

| batch_size | status | elapsed_sec | peak_vram_mib | peak_gpu_util_pct |
|---|---|---:|---:|---:|
| 32 | ok | 106.67 | 20028 | 89 |
| 48 | ok | 105.11 | 25221 | 98 |
| 64 | ok | 106.51 | 30931 | 95 |
| 80 | fail (OOM) | 70.59 | 32039 | 99 |

Notes:
- On real data, local Stage 2 is stable up to `batch_size=64` and OOMs at `80`.
- Compared to synthetic-data sweeps, real-data memory is higher and close to device ceiling at `64`.
- For this smoke dataset shape, one train epoch is `62` batches at `batch_size=64`.
- Lightning enforces `training.validation.check_interval <= epoch_length` in this setup.
  - `check_interval=100` failed (`100 > 62`).
  - `check_interval=50` worked.

### Important Stage 2 control detail (`max_steps` vs `max_epochs`)
- A run with:
  - `training.max_steps=100`
  - `training.max_epochs=1`
  stopped at epoch end (`62/62`) before reaching 100 steps.
- Confirmed fix:
  - setting `training.max_epochs=-1` allows step-based stopping to dominate.
  - Example verified run: `batch_size=64`, `max_steps=70`, `max_epochs=-1` reached `[Step 70]` and exported artifact successfully.

Practical implication:
- For local/cluster step-targeted Stage 2 runs where you want exact step counts across epoch boundaries, set:
  - `training.max_epochs=-1`
  - plus desired `training.max_steps=<N>`.

### Latest Stage 2 “real” local run (W&B enabled)
Run:
- `experiment=vla_smol_flow_shared`
- `batch_size=64`
- `training.max_steps=500`
- `training.max_epochs=-1`
- `training.validation.check_interval=50`
- `logging.use_wandb=true`

Result:
- success
- elapsed: `560.07s`
- peak VRAM: `31063 MiB`
- final `val/loss`: `1.20995`
- artifact:
  - `/mnt/data/workspace/runs/hlrp/2026-02-17_19-44-07_local_s2_real_bs64_steps500_fixval_194403/artifacts/smolvla_shared_stage2_artifact.pt`
- W&B:
  - `https://wandb.ai/felixmin/hlrp/runs/676kzhjy`

## Stage 3 Sweep (LeRobot + Stage2 Artifact, CUDA)
Artifact used:
- `/mnt/data/workspace/runs/hlrp/2026-02-17_18-49-23_vla_smol_flow_shared/artifacts/smolvla_shared_stage2_artifact.pt`

Config pattern:
- `experiment=lerobot_hlrp_smolvla_shared_smoke`
- `lerobot.policy_device=cuda`
- `lerobot.stage2_artifact=<artifact>`
- `logging.use_wandb=false`

### 10-step sweep

| batch_size | status | elapsed_sec | peak_vram_mib | peak_gpu_util_pct |
|---|---|---:|---:|---:|
| 1 | ok | 15.43 | 8652 | 10 |
| 2 | ok | 15.03 | 8660 | 45 |
| 4 | ok | 15.16 | 8714 | 44 |
| 8 | ok | 16.46 | 9554 | 41 |
| 16 | ok | 17.87 | 11480 | 28 |
| 32 | ok | 19.88 | 16265 | 30 |
| 64 | ok | 26.16 | 25420 | 93 |
| 80 (5 steps) | ok | 21.44 | 29264 | 95 |
| 96 (5 steps) | fail | 20.49 | 8840 | 81 |

Notes:
- Practical local Stage 3 range is up to about `batch_size=80` on this setup.
- `batch_size=96` failed with non-zero exit (likely memory/runtime boundary).

### Longer “significant” local run
- Run: `batch_size=64`, `steps=100`, CUDA, artifact warm-start
- Result: success
- Wall time: `122.48s` (includes launcher overhead)
- Peak VRAM: `25440 MiB`
- Per-log train timing around steady state:
  - `updt_s ~ 0.23s`
  - `data_s ~ 0.5-1.1s`

Interpretation:
- Local Stage 3 can run meaningful short-to-medium experiments reliably (e.g., 100-1000 steps) at high batch sizes.

### Stage 3 E2E check from latest real-data Stage 2 artifact
Artifact used:
- `/mnt/data/workspace/runs/hlrp/2026-02-17_19-14-39_local_s2_real_probe_bs64_191435/artifacts/smolvla_shared_stage2_artifact.pt`

Run:
- `experiment=lerobot_hlrp_smolvla_shared_smoke`
- `lerobot.stage2_artifact=<artifact>`
- `lerobot.steps=100`
- `lerobot.batch_size=32`
- `logging.use_wandb=false`

Result:
- success
- wall time: `37.41s`
- peak VRAM: `16183 MiB`
- artifact loading reported compatible manifest and expected optional missing keys only (`action_head.weight`, `action_head.bias`).

### Latest Stage 3 “real” local run from Stage 2 artifact (W&B enabled)
Artifact used:
- `/mnt/data/workspace/runs/hlrp/2026-02-17_19-44-07_local_s2_real_bs64_steps500_fixval_194403/artifacts/smolvla_shared_stage2_artifact.pt`

Run:
- `experiment=lerobot_hlrp_smolvla_shared_smoke`
- `lerobot.steps=500`
- `lerobot.batch_size=32`
- `lerobot.wandb_enable=true`
- `logging.use_wandb=true`

Result:
- success
- elapsed: `162.44s`
- peak VRAM: `16028 MiB`
- reached and checkpointed at step 500
- W&B:
  - `https://wandb.ai/felixmin/lerobot/runs/16nokzcu`

## Integration sanity checks completed
- Stage 2 exports artifact successfully.
- Stage 3 consumes artifact successfully on CUDA with strict manifest compatibility checks.
- Stage 3 launcher import path fixed via `PYTHONPATH=<repo>/packages` in scripts 6/7.

## Recommendations
1. Local fast iteration:
   - Stage 2 smoke/capacity: `batch_size 48-64`.
   - Stage 3 smoke/debug: `batch_size 32-64`.
2. For heavier local Stage 3 tests, use `batch_size=64` first; push to `80` only if needed.
3. For Stage 2 runs targeting fixed step counts, set `training.max_epochs=-1` so `training.max_steps` is not cut by epoch-end.
4. Keep cluster for long/full runs; use local for rapid contract/interface iteration.
