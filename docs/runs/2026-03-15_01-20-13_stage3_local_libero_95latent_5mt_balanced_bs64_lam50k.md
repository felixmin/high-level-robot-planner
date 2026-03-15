# Run Note: 2026-03-15_01-20-13_stage3_local_libero_95latent_5mt_balanced_bs64_lam50k

## Meta

- Date: 2026-03-15
- Status: canceled
- Mode: local
- Host: tueilsy-st-022
- Code Commit: 7038b673a6b9752420383941de425c3f1adb0848
- Worktree State: dirty (`containers/Dockerfile.unified`, `containers/README.md`, `containers/requirements.unified.txt`, `packages/common/lerobot_v3_data.py`, `packages/common/lerobot_v3_source.py`, `scripts/submit_job.py`, `EXPERIMENT.txt`)
- Logical Cluster Target:
- Stage: stage3
- Script: scripts/6_train_lerobot.py
- Base Experiment: stage3_local
- Config Path: config/experiment/runs/2026-03-15_01-20-13_stage3_local_libero_95latent_5mt_balanced_bs64_lam50k.yaml
- Experiment Name: 2026-03-15_01-20-13_stage3_local_libero_95latent_5mt_balanced_bs64_lam50k
- Intended Run Dir: /mnt/data/workspace/runs_root/runs/2026-03-15_01-20-13_stage3_local_libero_95latent_5mt_balanced_bs64_lam50k
- Final Run Dir: /mnt/data/workspace/runs_root/runs/2026-03-15_01-20-13_stage3_local_libero_95latent_5mt_balanced_bs64_lam50k
- LRZ Job ID:
- MCML Job ID:

## Purpose

- Start a new local Stage 3 Libero cotraining run that matches the prior balanced 95% latent / 5% multitask setup at batch size 64, but uses the newer local Stage 1 checkpoint written on 2026-03-15 after roughly 50k steps.

## Config Delta Vs Default

- Override `stage3_profile` from `action_scratch` to `multitask_scratch`.
- Override the Stage 3 dataset from `libero` to `libero_5pct_latent_rest_balanced`.
- Override `stage3_profile.batch_size` from `32` to `64`.
- Override `artifacts.lam_checkpoint_path` to `/mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt`.
- Override `logging.runs_dir` to `${logging.root_dir}/runs/${experiment.name}` to avoid duplicate timestamps in the run directory.

## Upstream Artifacts / Checkpoints

- Type: stage1 checkpoint
  Source Run: unknown
  Path: /mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt
  Notes: Undocumented local Stage 1 run directory. `last.ckpt` was written on 2026-03-15 01:11 CET after the explicit `stage1-lam-stepstep=050000.ckpt`.

## Launch Command

```bash
tmux new-window -d -t train -n s3_95lat_bs64 "cd /mnt/data/workspace/code/high-level-robot-planner && conda run -n lerobot python scripts/6_train_lerobot.py experiment=runs/2026-03-15_01-20-13_stage3_local_libero_95latent_5mt_balanced_bs64_lam50k; exec bash"
```

## Results / Findings

- Canceled by user request before meaningful training progress to restart with `lerobot.policy.latent_loss_weight=0.1`.
