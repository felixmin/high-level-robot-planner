# Run Note: 2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask

## Meta

- Date: 2026-03-15
- Status: completed
- Mode: cluster
- Host: tueilsy-st-022
- Code Commit: 8e6c5b0a46eb7197b9b0752c0294b8dba6092212
- Worktree State: dirty; local docs updates are uncommitted during launch.
- Logical Cluster Target: lrz cluster
- Stage: stage1
- Script: scripts/2_train_stage1_lam.py
- Base Experiment: stage1_cluster
- Config Path: config/experiment/runs/2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask.yaml
- Experiment Name: 2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask
- Intended Run Dir: /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask_{lrz|mcml}
- Final Run Dir: /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask_mcml
- LRZ Job ID: 5519305
- MCML Job ID: 5519306

## Purpose

- Produce a minimally trained but valid Stage 1 LAM checkpoint for later Stage 3 multitask Libero runs, while avoiding the corrupted shared-cache video path hit by the previous Octo24-based attempt.

## Config Delta Vs Default

- `experiment.name` set to the documented dated run stem.
- `experiment.description` narrowed to a checkpoint-bootstrap purpose.
- `logging.runs_dir` pinned to `${logging.root_dir}/runs/${experiment.name}` for documented-run naming.
- `logging.tags` narrowed to a test2 checkpoint-bootstrap run.
- `data=test2` to avoid the corrupted cache entry encountered in the previous run.
- `training.max_steps=50` to keep the run short.
- `training.num_sanity_val_steps=0` to avoid startup-only sanity validation overhead.

## Upstream Artifacts / Checkpoints

- Type: stage1 checkpoint
  Source Run: none
  Path: none
  Notes: This run is the source checkpoint producer for a later Stage 3 multitask training run.

## Launch Command

```bash
# LRZ queue
python scripts/submit_job.py \
  experiment=stage1_cluster \
  experiment.name=2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask \
  experiment.description="Short cluster Stage 1 run on test2 to produce a valid LAM checkpoint for later Stage 3 multitask training." \
  data=test2 \
  training.max_steps=50 \
  training.num_sanity_val_steps=0 \
  logging.tags=[stage1,cluster,test2,tiny-ckpt,lam,stage3-bootstrap] \
  cluster=lrz_x100 \
  cluster.compute.time_limit=00:30:00 \
  logging.runs_dir=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask_lrz

# MCML queue
python scripts/submit_job.py \
  experiment=stage1_cluster \
  experiment.name=2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask \
  experiment.description="Short cluster Stage 1 run on test2 to produce a valid LAM checkpoint for later Stage 3 multitask training." \
  data=test2 \
  training.max_steps=50 \
  training.num_sanity_val_steps=0 \
  logging.tags=[stage1,cluster,test2,tiny-ckpt,lam,stage3-bootstrap] \
  cluster=mcml_x100 \
  cluster.compute.time_limit=00:30:00 \
  logging.runs_dir=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask_mcml
```

## Results / Findings

- Submitted to LRZ and MCML with a 30-minute limit.
- Both duplicate jobs started on A100 nodes; the LRZ duplicate `5519305` was canceled after start to avoid burning both allocations.
- Active run was MCML job `5519306`, which completed successfully in about 1.5 minutes of cluster runtime.
- Training reached the requested `50` steps on `test2` without hitting the earlier corrupted shared-cache video path from the Octo24 attempt.
- Produced checkpoint:
  `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask_mcml/checkpoints/last.ckpt`
