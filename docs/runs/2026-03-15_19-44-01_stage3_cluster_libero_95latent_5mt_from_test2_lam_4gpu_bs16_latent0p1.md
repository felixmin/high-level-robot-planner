# Run Note: 2026-03-15_19-44-01_stage3_cluster_libero_95latent_5mt_from_test2_lam_4gpu_bs16_latent0p1

## Meta

- Date: 2026-03-15
- Status: running
- Mode: cluster
- Host: tueilsy-st-022
- Code Commit: 11d0897ce1d0b46f52faeb3fa4e42ef884de1192
- Worktree State: dirty; local worktree had unrelated run-doc changes during launch.
- Logical Cluster Target: lrz_x100
- Stage: stage3
- Script: scripts/6_train_lerobot.py
- Base Experiment: stage3_cluster
- Config Path: config/experiment/runs/2026-03-15_19-44-01_stage3_cluster_libero_95latent_5mt_from_test2_lam_4gpu_bs16_latent0p1.yaml
- Experiment Name: 2026-03-15_19-44-01_stage3_cluster_libero_95latent_5mt_from_test2_lam_4gpu_bs16_latent0p1
- Intended Run Dir: /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_19-44-01_stage3_cluster_libero_95latent_5mt_from_test2_lam_4gpu_bs16_latent0p1
- Final Run Dir: /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_19-44-01_stage3_cluster_libero_95latent_5mt_from_test2_lam_4gpu_bs16_latent0p1
- LRZ Job ID: 5519360
- MCML Job ID:

## Purpose

- Start the corrected Stage 3 cluster multitask Libero run on the `libero_5pct_latent_rest_balanced` dataset using the tiny `test2` Stage-1 checkpoint as latent supervision, with 4 GPUs, effective batch size 64, and latent loss weight 0.1.

## Config Delta Vs Default

- `cluster=lrz_x100`
- `stage3_profile=multitask_scratch`
- `stage3_dataset=libero_5pct_latent_rest_balanced`
- `cluster.compute.gpus_per_node=4`
- `cluster.compute.time_limit=08:00:00`
- `artifacts.lam_checkpoint_path` points to the successful `test2` tiny Stage-1 checkpoint
- `lerobot.batch_size=16` so 4-GPU effective batch size remains `64`
- `lerobot.policy.latent_loss_weight=0.1`
- `experiment.name` and `logging.runs_dir` use the dated documented-run stem

## Upstream Artifacts / Checkpoints

- Type: stage1 checkpoint
  Source Run: 2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask
  Path: /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask_mcml/checkpoints/last.ckpt
  Notes: Tiny but valid Stage-1 LAM checkpoint used to provide latent supervision for multitask Stage 3.

## Launch Command

```bash
python scripts/submit_job.py \
  experiment=stage3_cluster \
  stage3_profile=multitask_scratch \
  stage3_dataset=libero_5pct_latent_rest_balanced \
  cluster=lrz_x100 \
  cluster.compute.gpus_per_node=4 \
  cluster.compute.time_limit=08:00:00 \
  experiment.name=2026-03-15_19-44-01_stage3_cluster_libero_95latent_5mt_from_test2_lam_4gpu_bs16_latent0p1 \
  artifacts.lam_checkpoint_path=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask_mcml/checkpoints/last.ckpt \
  lerobot.batch_size=16 \
  lerobot.policy.latent_loss_weight=0.1 \
  logging.runs_dir=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_19-44-01_stage3_cluster_libero_95latent_5mt_from_test2_lam_4gpu_bs16_latent0p1
```

## Results / Findings

- Submitted to LRZ as job `5519360`.
- Hydra recorded the intended overrides correctly:
  - full `artifacts.lam_checkpoint_path`
  - `lerobot.batch_size=16`
  - `lerobot.policy.latent_loss_weight=0.1`
  - `stage3_dataset=libero_5pct_latent_rest_balanced`
- The generated `lerobot_train_config.json` confirms the downstream LeRobot config also resolved to:
  - `batch_size=16`
  - `policy.latent_loss_weight=0.1`
  - `policy.action_loss_weight=1.0`
  - the full source Stage-1 checkpoint path
