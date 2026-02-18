# Local Stage-2 Validation OOM Investigation (2026-02-18)

## Context
- Host: `tueilsy-st-022` (workstation, 64 GB RAM, RTX 5090 32 GB VRAM)
- Repo: `/mnt/data/workspace/code/high-level-robot-planner`
- Failed run:
  - `/mnt/data/workspace/runs/hlrp/2026-02-18_17-13-55_local_s2_bs48_spe0_reasonable_v6_long`

## Failure Evidence
- Kernel OOM at:
  - `2026-02-18 17:34:37`
- OOM log:
  - `Out of memory: Killed process 2256658 (python) ... anon-rss:39798588kB`
- Run reached train step 2000 and generated:
  - `visualizations/train_samples_step002000.*`
  - `visualizations/val_samples_step002000.*`
- Immediately after that, logs show a second full TFDS initialization:
  - `python-prefetch mixing: starting 29 workers`
  - followed by 29 dataset `Load dataset info` + TFDS split construction (`train[90%:]`)

## Working Hypothesis
- Train pipeline remains resident (persistent iterator + long-lived TF pipeline state).
- Validation initialization brings up a second 29-dataset pipeline.
- Combined host memory exceeds local limit and kernel kills Python during/after val dataloader initialization.

## Config Used In Failed Run (key overrides)
- `data=oxe_local_spe0`
- `data.loader.batch_size=48`
- `data.adapter.tf.sampling.samples_per_episode=0`
- `data.adapter.tf.mixing.strategy=python`
- `data.adapter.tf.mixing.mix_block_length=4`
- `data.adapter.tf.mixing.python_prefetch_queue_size=1`
- `data.adapter.tf.mixing.python_prefetch_min_ready_datasets=1`
- `data.adapter.tf.tfds_read.skip_steps_decoding=true`
- `data.adapter.tf.pipeline.emit_encoded_pairs=true`
- `data.adapter.tf.tfds_read.decode_parallelism=4`
- `data.adapter.tf.pipeline.transform_parallelism=4`
- `training.validation.check_interval=2000`
- `training.validation.limit_batches=20`
- `training.validation.visualization.enabled=true`

## Isolation Ladder (local)
1. Baseline tiny-val sanity:
   - `data=oxe_local_spe0_tiny_val`, `batch_size=48`, val every 10, limit val batches 4.
2. Full-val low pressure:
   - `data=oxe_local_spe0`, `batch_size=8`, val every 10, limit val batches 2.
3. Full-val same batch with non-persistent iterator:
   - `data=oxe_local_spe0`, `batch_size=48`, `data.adapter.tf.iterator.persistent=false`, val every 10.
4. Scale-up:
   - Increase batch and/or val limits until failure boundary is found.

## Goal
- Identify the smallest config change that keeps validation stable locally.
- Keep full 29-dataset training mix (`samples_per_episode=0`) and then find max stable throughput before OOM.

## Probe Results (2026-02-18 evening)

### Local probes
- `local_s2_valoom_probe2_tinyval_py` (`bs=48`, `tiny_val`, val every 10):
  - Reproduced OOM at `18:28:40`.
  - Kernel:
    - `Out of memory: Killed process 2286800 (python) ... anon-rss:38894720kB`
  - Log pattern before OOM:
    - repeated `python-prefetch mixing: starting 29 workers` + `initial ready=1/29`
    - repeated re-init cycles without meaningful step progress.

- `local_s2_valoom_probe3_singleval_bs48` (`bs=48`, `tiny_val`, one val target):
  - Still showed repeated 29-worker restarts and RAM growth.
  - Stopped manually before hard OOM.

- `local_s2_valoom_probe4_tinyval_bs8` (`bs=8`, `tiny_val`, val every 10):
  - Same repeated restart pattern.
  - Lower batch did not remove the issue.
  - Stopped manually.

- `local_s2_valoom_probe5_fullval_bs8_once` (`bs=8`, full val split):
  - Repeated restarts observed as well.
  - Stopped manually to avoid OOM while investigating.

### Cluster check
- Job `5486298` (`cluster_s2_bs48_spe0_highram_v2_lrz`):
  - Did not crash from OOM.
  - Reached validation at step `2000` and `4000`.
  - Then ended due walltime (`TIMEOUT`), not functional failure.

## Code Fix Applied

- File: `packages/common/adapters/oxe.py`
- Change:
  - In python-prefetch worker loop, changed blocking queue put:
    - from `queues[i].put(item)` (can block indefinitely)
    - to timed put with stop-check:
      - `queues[i].put(item, timeout=0.1)` with `except Full: continue`
- Rationale:
  - Prevent background worker threads from being stuck forever on full queues during iterator shutdown.
  - Reduce accumulation of stale daemon workers/pipelines across iterator restarts.

## Post-fix signal
- `local_s2_valoom_probe6_fix_bs8` (same stress recipe as probe4):
  - Immediate runaway memory growth was reduced compared to pre-fix behavior.
  - Repeated restarts still occur and can still drive RAM up over time.
  - Indicates the fix mitigates one leak path, but restart frequency remains a primary pressure source.

## Current Interpretation
- The OOM is not just raw batch-size pressure.
- Primary trigger appears to be repeated dataset iterator restart cycles in python-mixer mode with 29 datasets, which repeatedly spin up prefetch workers and TF pipelines.
- Validation can amplify this, but restart behavior itself is already enough to push host RAM over time on workstation.

## Next Recommended Experiments
1. Disable python-mixer locally (`data.adapter.tf.mixing.strategy=sample`) and retest short runs with validation.
2. Keep python-mixer only on cluster/high-RAM runs.
3. For local python-mixer debug, reduce dataset count (small subset preset) before scaling back up.
4. Add explicit local run guardrail:
   - larger `validation.check_interval`
   - `validation.limit_batches` very small
   - monitor RSS every N seconds in `unified.log`.
