# SmolVLA Shared Backend (Stage 2 + Stage 3)

## Overview

This backend reimplements the SmolVLA-style flow setup with a shared trunk and split heads:

- Shared trunk: SmolVLM image+language encoder (`packages/foundation/backends/smolvla_shared/model.py`)
- Latent head: flow-matching on flattened LAQ vectors `[B, S*D]`
- Real-action head: initialized in shared core, trained in Stage 3 (LeRobot)

Backend code is split by responsibility:

- `packages/foundation/backends/smolvla_shared/config.py`
- `packages/foundation/backends/smolvla_shared/preprocess.py`
- `packages/foundation/backends/smolvla_shared/flow.py`
- `packages/foundation/backends/smolvla_shared/losses.py`
- `packages/foundation/backends/smolvla_shared/model.py`
- `packages/foundation/backends/smolvla_shared_backend.py`

## Stage 2 Training

Use:

- `config/model/foundation_smol_flow_shared.yaml`
- `config/experiment/vla_smol_flow_shared.yaml`

Stage 2 mode is `model.training_mode=latent_flow`.
Only latent-flow loss is optimized in this stage.

Example:

```bash
python scripts/submit_job.py \
  experiment=vla_smol_flow_shared \
  cluster=lrz_x100 \
  model.laq.checkpoint=/dss/.../laq.ckpt \
  experiment.name=vla_smol_flow_shared_smoke
```

## Stage 3 LeRobot Training

New policy type:

- `hlrp_smolvla_shared`

Config:

- `config/experiment/lerobot_hlrp_smolvla_shared_smoke.yaml`

Optional Stage-2 checkpoint handoff:

- `lerobot.stage2_checkpoint=/path/to/stage2.ckpt`

Example:

```bash
python scripts/submit_job.py \
  experiment=lerobot_hlrp_smolvla_shared_smoke \
  cluster=lrz_x100 \
  lerobot.stage2_checkpoint=/dss/.../stage2.ckpt \
  experiment.name=lerobot_hlrp_smolvla_shared_smoke
```

## Stage 3 Rollout

Rollout/eval entrypoint:

- `scripts/7_rollout_lerobot.py`

Experiment config:

- `config/experiment/lerobot_hlrp_smolvla_shared_rollout.yaml`

Required field:

- `lerobot_eval.policy_path` (path to `pretrained_model` dir)

Example:

```bash
python scripts/submit_job.py \
  experiment=lerobot_hlrp_smolvla_shared_rollout \
  cluster=lrz_x100 \
  lerobot_eval.policy_path=/dss/.../checkpoints/000050/pretrained_model \
  experiment.name=lerobot_hlrp_smolvla_shared_rollout
```
