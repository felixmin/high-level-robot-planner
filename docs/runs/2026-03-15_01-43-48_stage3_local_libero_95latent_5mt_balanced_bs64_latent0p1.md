# Run Note: 2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1

## Meta

- Date: 2026-03-15
- Status: running
- Mode: local
- Host: tueilsy-st-022
- Code Commit: 7038b673a6b9752420383941de425c3f1adb0848
- Worktree State: dirty (`containers/Dockerfile.unified`, `containers/README.md`, `containers/requirements.unified.txt`, `packages/common/lerobot_v3_data.py`, `packages/common/lerobot_v3_source.py`, `scripts/submit_job.py`, `EXPERIMENT.txt`)
- Logical Cluster Target:
- Stage: stage3
- Script: scripts/6_train_lerobot.py
- Base Experiment: stage3_local
- Config Path: config/experiment/runs/2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1.yaml
- Experiment Name: 2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1
- Intended Run Dir: /mnt/data/workspace/runs_root/runs/2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1
- Final Run Dir: /mnt/data/workspace/runs_root/runs/2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1
- LRZ Job ID:
- MCML Job ID:

## Purpose

- Restart the local Stage 3 Libero cotraining run with the same balanced 95% latent / 5% multitask mix, batch size 64, and latest local Stage 1 checkpoint, but underweight the latent loss by 10x relative to the action loss.

## Config Delta Vs Default

- Override `stage3_profile` from `action_scratch` to `multitask_scratch`.
- Override the Stage 3 dataset from `libero` to `libero_5pct_latent_rest_balanced`.
- Override `stage3_profile.batch_size` from `32` to `64`.
- Override `artifacts.lam_checkpoint_path` to `/mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt`.
- Keep `lerobot.policy.init_mode=scratch` and `lerobot.policy.stage2_artifact=null` so this run does not resume from any Stage 3 checkpoint.
- Override `lerobot.policy.latent_loss_weight` from `1.0` to `0.1`.
- Override `logging.runs_dir` to `${logging.root_dir}/runs/${experiment.name}` to avoid duplicate timestamps in the run directory.

## Upstream Artifacts / Checkpoints

- Type: stage1 checkpoint
  Source Run: unknown
  Path: /mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt
  Notes: Undocumented local Stage 1 run directory. `last.ckpt` was written on 2026-03-15 01:11 CET after the explicit `stage1-lam-stepstep=050000.ckpt`.

- Type: stage3 checkpoint
  Source Run: 2026-03-15_01-20-13_stage3_local_libero_95latent_5mt_balanced_bs64_lam50k
  Path: /mnt/data/workspace/runs_root/runs/2026-03-15_01-20-13_stage3_local_libero_95latent_5mt_balanced_bs64_lam50k
  Notes: Prior launch with the same settings except equal action/latent loss weighting. Kept on disk after cancel for possible later inspection or resume.

## Launch Command

```bash
tmux new-window -d -t train -n s3_95lat_lw01 "cd /mnt/data/workspace/code/high-level-robot-planner && conda run -n lerobot python scripts/6_train_lerobot.py experiment=runs/2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1; exec bash"
```

## Results / Findings

- Pending.
