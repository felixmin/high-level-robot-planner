# Run Note: 2026-03-15_23-59-15_stage3_local_libero_5pct_mt_bs32_latent0p1

## Meta

- Date: 2026-03-15
- Status: running
- Mode: local
- Host: tueilsy-st-022
- Code Commit: 04970c0557a41f7b3d76d15cabf04df7ed8f7883 (dirty: config/data/octo24*.yaml, containers/Dockerfile.unified, docs/runs/, scripts/submit_job.py)
- Worktree State: dirty
- Logical Cluster Target:
- Stage: stage3
- Script: `scripts/6_train_lerobot.py`
- Base Experiment: `stage3_local`
- Config Path: `config/experiment/runs/2026-03-15_23-59-15_stage3_local_libero_5pct_mt_bs32_latent0p1.yaml`
- Experiment Name: `stage3_local` (launched before documented config was created; run dir uses default naming)
- Intended Run Dir: `/mnt/data/workspace/runs_root/runs/2026-03-15_23-59-15_stage3_local`
- Final Run Dir: `/mnt/data/workspace/runs_root/runs/2026-03-15_23-59-15_stage3_local`
- LRZ Job ID:
- MCML Job ID:

## Purpose

- Ablation baseline for `2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1`.
- Train Stage 3 multitask on only the 5% action-labeled Libero subset (84 episodes), discarding the 95% latent-only data.
- Same LAM checkpoint, same latent loss weight 0.1, but batch size 32 (profile default) instead of 64.
- Measures whether the extra 95% latent-only training data actually helps or hurts compared to multitask on the labeled subset alone.

## Config Delta Vs Default

- Override `stage3_dataset` from `libero` (full dataset) to `libero_5pct` (84 multitask-only episodes).
- Set `artifacts.lam_checkpoint_path` to `/mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt`.
- Override `lerobot.policy.latent_loss_weight` from `1.0` to `0.1`.
- Override `lerobot.num_workers` from `8` to `4`.
- `lerobot.batch_size` stays at profile default `32` (vs 64 in the comparison run).
- All other settings (model, scheduler, optimizer, eval, steps=100k) remain at `_stage3_base` / `multitask_scratch` defaults.

## Upstream Artifacts / Checkpoints

- Type: `stage1 checkpoint`
  Source Run: `2026-03-14_19-27-26_stage1_local`
  Path: `/mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt`
  Notes: Libero-only LAM checkpoint used for latent supervision, same as in the comparison run.

## Launch Command

```bash
conda run -n lerobot python scripts/6_train_lerobot.py \
  experiment=stage3_local \
  stage3_dataset=libero_5pct \
  artifacts.lam_checkpoint_path=/mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt \
  lerobot.batch_size=32 \
  lerobot.num_workers=4 \
  lerobot.policy.latent_loss_weight=0.1 \
  logging.use_wandb=true \
  lerobot.wandb.enable=true
```

tmux session: `stage3_5pct`

## Results / Findings

- Pending.
