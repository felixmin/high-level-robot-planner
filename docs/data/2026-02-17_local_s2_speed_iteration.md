# Local Stage-2 Speed Iteration (2026-02-17)

## Goal
Improve local Stage-2 training throughput for `data=oxe_local_spe0` (29 datasets).

## Baseline (current config)
Run:
- `/mnt/data/workspace/runs/hlrp/2026-02-17_22-41-17_local_s2_overnight_bs48_steps25k_oxe_local_spe0`
- Output log: `/mnt/data/workspace/runs/hlrp/2026-02-17_22-41-17_local_s2_overnight_bs48_steps25k_oxe_local_spe0/wandb/wandb/run-20260217_224142-940e0wj4/files/output.log`

Config:
- `experiment=vla_smol_flow_shared`
- `cluster=local_dev`
- `data=oxe_local_spe0`
- `data.loader.batch_size=48`
- `training.max_steps=25000`
- `training.max_epochs=-1`
- `training.validation.check_interval=2000`
- `training.validation.visualization.enabled=true`
- `training.train_visualization.enabled=true`

Observed throughput line:
- `Epoch 0: ... 98/25000 ... 0.09it/s ... [Step 100]`

Derived throughput:
- `it/s`: `0.09`
- `samples/s`: `0.09 * 48 = 4.32`

## Next config to test
Same base run, with throughput overrides from prior OXE docs:
- `data.adapter.tf.tfds_read.decode_parallelism=4`
- `data.adapter.tf.tfds_read.interleave_parallelism=4`
- `data.adapter.tf.pipeline.transform_parallelism=4`
- `data.adapter.tf.pipeline.interleave_parallelism=4`
- `data.adapter.tf.prefetch.final_stream_buffer=8`
- `data.adapter.tf.prefetch.per_dataset_stream_buffer=0`
- `data.adapter.tf.mixing.strategy=python`
- `data.adapter.tf.mixing.parallelism_mode=sqrt`
- `data.adapter.tf.mixing.mix_block_length=4`
- `data.adapter.tf.mixing.python_prefetch_queue_size=4`
- `data.adapter.tf.mixing.python_prefetch_min_ready_datasets=8`

## Results
| label | run | it/s | samples/s | notes |
|---|---|---:|---:|---|
| baseline | `2026-02-17_22-41-17_local_s2_overnight_bs48_steps25k_oxe_local_spe0` | 0.09 | 4.32 | from output.log at step 100 |
| next (fast overrides) | `2026-02-17_23-13-07_local_s2_speed_iter_fast_oxe29_bs48` | 1.00 (step 100), 1.54 (step 200) | 48.00 (step 100), 73.92 (step 200) | strong speedup; process later exited with `SIGKILL` (likely OOM) around step 243 |
| intermediate (reduced RAM) | `2026-02-17_23-24-43_local_s2_speed_iter_mid_oxe29_bs48` | 1.12 (step 100), 1.66 (step 200), 1.97 (step 300) | 53.76 (step 100), 79.68 (step 200), 94.56 (step 300) | also exited with `SIGKILL` later |
| stable attempt (lower RAM, profiler off) | `2026-02-17_23-31-15_local_s2_speed_iter_stable_oxe29_bs48` | 1.12 (step 100), 1.68 (step 200), 2.00 (step 300) | 53.76 (step 100), 80.64 (step 200), 96.00 (step 300) | `pin_memory=false`, `final_stream_buffer=2`, `queue_size=1`, `min_ready=2`, `profiler=false`; still exited with `SIGKILL` |
| choose mixer, bs48 | `2026-02-17_23-36-44_local_s2_speed_iter_choose32_oxe29_bs48` | n/a | n/a | reached sanity/epoch start, then exited with `SIGKILL` before step metrics |
| choose mixer, bs32 (retry2) | `2026-02-17_23-59-00_local_s2_speed_iter_choose32_oxe29_bs32_retry2` | n/a | n/a | long startup, then stalled at epoch start (`0/500`) and was manually stopped |
| python lowmem, bs32 | `2026-02-18_00-06-44_local_s2_speed_iter_py_lowmem_oxe29_bs32` | 1.30 (100), 2.06 (200), 2.53 (300), 2.86 (400), 3.09 (500) | 41.60, 65.92, 80.96, 91.52, 98.88 | completed step 500, then parent process still died with `SIGKILL` |
| python lowmem, bs24 | `2026-02-18_00-11-29_local_s2_speed_iter_py_lowmem_oxe29_bs24` | 1.34 (100), 2.20 (200), 2.78 (300), 3.19 (400), 3.50 (500) | 32.16, 52.80, 66.72, 76.56, 84.00 | completed step 500, then parent process died with `SIGKILL` |
| python lowmem, bs24, no checkpoint, s300 | `2026-02-18_00-15-44_local_s2_speed_iter_py_lowmem_bs24_nockpt_s300` | 1.39 (100), 2.27 (200), 2.85 (300) | 33.36, 54.48, 68.40 | disabling checkpoint did not remove post-run `SIGKILL` |
| python lowmem, bs24, no checkpoint, nw=0, s300 | `2026-02-18_00-19-23_local_s2_speed_iter_py_lowmem_bs24_nw0_s300` | 1.34 (100), 2.19 (200), 2.77 (300) | 32.16, 52.56, 66.48 | still ended with post-run `SIGKILL`; setting `num_workers=0` did not fix termination |
| sample+auto, p2 (invalid wiring) | `2026-02-18_00-53-04_local_s2_speed_iter_sample_auto_p2_bs48_s200` | n/a | n/a | failed fast: `num_parallel_calls=2` with `cycle_length=1` (tf/pipeline interleave constraint) |
| sample+auto, p2c2e2 | `2026-02-18_00-54-19_local_s2_speed_iter_sample_auto_p2c2e2_bs48_s120` | n/a | n/a | valid config, but startup stayed at `Epoch 0: 0/120` while constructing 29 TFDS pipelines from GCS (`source=auto`); manually stopped |
| sample+auto, p2c2e2 (retry-long) | `2026-02-18_01-00-48_local_s2_speed_iter_sample_auto_p2c2e2_bs48_s120_retrylong` | n/a | n/a | waited longer; still no first step (`Epoch 0: 0/120`), then kernel OOM kill at `01:05:59` (`anon-rss ~36.3 GB`) |

## Direct delta
- Baseline vs fast at step 100:
  - `0.09 -> 1.00 it/s` (`~11.1x`)
  - `4.32 -> 48.00 samples/s` (`~11.1x`)
- Baseline vs intermediate at step 100:
  - `0.09 -> 1.12 it/s` (`~12.4x`)
  - `4.32 -> 53.76 samples/s` (`~12.4x`)

## Current interpretation
- Best speed observed so far is in the python-mixing low-memory family (`queue_size=1`, `min_ready=2`, `final_stream_buffer=2`) with `bs32`/`bs24`.
- Failure pattern changed:
  - old configs (`bs48`) died mid-epoch.
  - new low-memory configs usually finish target steps, then still die with parent `SIGKILL` after training loop completion.
- Disabling checkpoint did not eliminate the post-run `SIGKILL`, so checkpoint save is not the only trigger.

## Root cause check (system logs)
`journalctl -k` confirms host-level OOM kills for these runs (not Python exceptions):
- `00:10:46` killed `python` pid `1864402` (`local_s2_speed_iter_py_lowmem_oxe29_bs32`)
- `00:15:12` killed `python` pid `1870896` (`local_s2_speed_iter_py_lowmem_oxe29_bs24`)
- `00:18:55` killed `python` pid `1877108` (`local_s2_speed_iter_py_lowmem_bs24_nockpt_s300`)
- `00:22:37` killed `python` pid `1883168` (`local_s2_speed_iter_py_lowmem_bs24_nw0_s300`)

Observed from kernel log:
- `Out of memory: Killed process <pid> (python) ... anon-rss ~36 GB`
- global OOM context (`global_oom`, `task_memcg=/user.slice/...`)

Conclusion:
- The failures are caused by system RAM pressure leading to kernel OOM kill.
- This affects both mid-run and post-run/finalization phases depending on config.

## 2026-02-18 Param-by-param ladder (10-step, bs48, source=auto)
Method:
- Start from a known working baseline.
- Change exactly one parameter per run.
- Keep all other knobs fixed.

Base command family:
- `experiment=vla_smol_flow_shared cluster=local_dev data=oxe_local_spe0 data.loader.batch_size=48 training.max_steps=10 training.max_epochs=-1 training.validation.check_interval=1000 training.validation.num_sanity_val_steps=0 training.validation.visualization.enabled=false training.train_visualization.enabled=false training.profiler.enabled=false logging.use_wandb=true`

| label | run | changed override(s) | status | steps/s | samples/s | notes |
|---|---|---|---|---:|---:|---|
| baseline | `2026-02-18_01-09-55_paramscan01_baseline_bs48_s10` | none | pass | 0.03861 | 1.85305 | artifact exported |
| decode p2 | `2026-02-18_01-15-25_paramscan02_decode_p2_bs48_s10` | `data.adapter.tf.tfds_read.decode_parallelism=2` | pass | 0.03702 | 1.77691 | slower than baseline |
| transform p2 | `2026-02-18_01-21-14_paramscan03_transform_p2_bs48_s10` | `data.adapter.tf.pipeline.transform_parallelism=2` | pass | 0.03815 | 1.83098 | close to baseline |
| episode conc2 | `2026-02-18_01-26-40_paramscan04_episode_conc2_bs48_s10` | `data.adapter.tf.pipeline.episode_concurrency=2` | pass | 0.03434 | 1.64849 | slowest; highest RAM peak seen in this ladder |
| pipe interleave2 | `2026-02-18_01-32-31_paramscan05_pipe_interleave2_bs48_s10` | `data.adapter.tf.pipeline.interleave_parallelism=2` | fail | n/a | n/a | invalid combo (`num_parallel_calls=2` with `cycle_length=1`) |
| tfds interleave2 | `2026-02-18_01-33-00_paramscan06_tfds_interleave2_bs48_s10` | `data.adapter.tf.tfds_read.interleave_parallelism=2` | pass | 0.03897 | 1.87050 | best speed in this ladder |
| tfds interleave2 + decode2 | `2026-02-18_01-38-20_paramscan07_tfds_inter2_decode2_bs48_s10` | `data.adapter.tf.tfds_read.interleave_parallelism=2`, `data.adapter.tf.tfds_read.decode_parallelism=2` | pass | 0.03726 | 1.78871 | slower than `tfds interleave2` alone |

Decision from ladder:
- Use `data.adapter.tf.tfds_read.interleave_parallelism=2` as the only performance override for the overnight run.
- Do not use `data.adapter.tf.pipeline.episode_concurrency=2` for overnight; it hurts throughput and increases RAM pressure.
- Keep `data.adapter.tf.pipeline.interleave_parallelism=1` unless `cycle_length` is changed compatibly.

## 2026-02-18 Validation + RAM-floor probes (local, 29 datasets)
Goal:
- Find a local Stage-2 config that
  1) runs validation,
  2) keeps at least 5 GB free RAM (`MemAvailable >= 5120 MB`),
  3) stays on `data=oxe_local_spe0` composition.

Method:
- Short runs with explicit memory sampling every 5s from `/proc/meminfo`.
- Probe command family based on `experiment=vla_smol_flow_shared` and
  `model.laq.checkpoint=/mnt/data/workspace/code/high-level-robot-planner/laq-stepstep052500.ckpt`.

| label | run | key overrides vs baseline | min_avail_mb | outcome |
|---|---|---|---:|---|
| A | `local_s2_valram_probe_a_bs8` | `data=oxe_local_spe0`, `bs=8`, `check_interval=1`, `limit_batches=1`, `num_sanity_val_steps=0`, `iterator.persistent=false` | 5501 | stayed above RAM floor but did not complete (manual stop; prolonged dataloader setup/rebuild path) |
| B | `local_s2_valram_probe_b_tinyval_bs8` | `data=oxe_local_spe0_tiny_val` with val split `train[:1%]` | 15874 | failed in validation init: `ValueError: Instruction [] corresponds to no data!` |
| C | `local_s2_valram_probe_c_tinyval_abs1_bs8` | val split changed to `train[:1]` | 17116 (before stop) | no OOM; validation dataset construction progressed, then stalled waiting for mixed stream (manual stop) |
| D | `local_s2_valram_probe_d_tinyval_abs1_bs8_persist` | same as C but persistent iterator | 33083 (before stop) | no OOM; still stalled in short probe window (manual stop) |
| E | `local_s2_valram_probe_e_sanityval_abs1_bs8` | `num_sanity_val_steps=1`, val split `train[:1]` | 32972 (before stop) | no OOM; still stalled after val pipeline construction (manual stop) |
| F | `local_s2_valram_probe_f_sanityval_abs10_bs8` | val split `train[:10]` | 32925 (before stop) | no OOM; still stalled after val pipeline construction (manual stop) |
| G | `local_s2_valram_probe_g_sanityval_fullval_bs8` | reverted to full val split (`train[90%:]`), sanity val | 24458 (before stop) | no OOM; still long-running/stalled in short probe window (manual stop) |
| H | `local_s2_valram_probe_h_fullval_bs8_pythonmix` | full val + `mixing.strategy=python`, queue/min-ready=1 | 29151 (before stop) | no OOM; python prefetch showed `ready=0/29` and stalled |
| I | `local_s2_valram_probe_i_fullval_bs8_spe1` | full val + `samples_per_episode=1` | 24458 (before stop) | no OOM; still long-running after val pipeline build in short probe window |
| J | `local_s2_valram_probe_j_stepval_warmup_bs8` | full 29 datasets + `check_interval=10`, `limit_batches=1`, `samples_per_episode=1` | 25680 (before stop) | no OOM; on workstation `source=auto` resolved to `gs://` datasets, so run was stopped and excluded from speed decisions |
| K | `local_s2_valram_probe_k_sample_auto_bs8` | `mixing.strategy=sample`, `bs=8`, `check_interval=5`, `limit_batches=1`, `num_sanity_val_steps=1` | 31521 (before stop) | also resolved to GCS (`source=auto`) and spent probe window constructing datasets/splits; stopped and excluded from speed conclusions |

Interpretation from this sweep:
- RAM floor target is achievable with wide margin under low-memory settings (`bs=8`, no viz/profiler, small prefetch).
- Main blocker in short local probes is not OOM but validation batch materialization/stall for multi-dataset mixed val streams.
- Tiny val splits can fail when split resolves to empty (`train[:1%]`) or produce mixed-stream starvation.
- Full-val splits avoid empty-split errors but still take long enough that short local probes can appear stalled after pipeline setup.

## 2026-02-18 Cluster failure follow-up

Failed run:
- `5485027` (`cluster_s2_bs48_steps25k_oxe_cluster_spe0_lrz`) failed before training startup.

Root cause:
- Container runtime/NVML startup issue on node via Pyxis/Enroot, not a training config error:
  - `nvidia-container-cli: detection error: nvml error: unknown error`
  - `pyxis: container start failed`

Action taken:
- Resubmitted LRZ retry with same training config:
  - `5486040` (`cluster_s2_bs48_steps25k_oxe_cluster_spe0_lrz_retry_nvml`)
- Canceled queued MCML twin (`5485032`) after LRZ retry started, following cancel-on-first-start strategy.

Outcome of retry:
- `5486040` ended by walltime (`TIMEOUT`, 15 min smoke window), not by container/runtime crash.
- W&B run output shows training progressed to at least:
  - `[Step 100]`
  - `[Step 200]`
  - `[Step 300]`
  - `[Step 400]`
  - `[Step 500]`
  - `[Step 600]`

Next short override test (startup tuning):
- Submitted with only override changes:
  - `data.adapter.tf.mixing.python_prefetch_min_ready_datasets=1`
  - `data.adapter.tf.mixing.python_prefetch_queue_size=1`
- Jobs:
  - LRZ: `5486075` (`cluster_s2_bs48_steps25k_15m_prefetch1_lrz`) -> running
  - MCML: `5486076` (`cluster_s2_bs48_steps25k_15m_prefetch1_mcml`) -> canceled after LRZ started

Observed outcome for `5486075`:
- Ended by 15-minute walltime (`TIMEOUT`), no container/runtime crash.
- Reached at least `[Step 600]` within the 15-minute smoke window (similar to prior baseline smoke).

## 2026-02-18 New concurrent runs (cluster + local)

Goal:
- Start one high-RAM cluster candidate and one local bs48/spe0 run in parallel.
- Keep `samples_per_episode=0`.

### Cluster run (high-RAM candidate, vetted overrides only)

Submitted jobs:
- LRZ: `5486289` (`cluster_s2_bs48_spe0_highram_v1_lrz`)
- MCML: `5486290` (`cluster_s2_bs48_spe0_highram_v1_mcml`) -> canceled after LRZ started.

Base:
- `experiment=vla_smol_flow_shared`
- `data=oxe_cluster_spe0`
- `data.loader.batch_size=48`
- `data.adapter.tf.sampling.samples_per_episode=0`
- `training.max_steps=25000`
- `training.max_epochs=-1`
- `cluster.compute.time_limit=00:45:00`
- `training.validation.check_interval=2000`
- `training.validation.visualization.enabled=true`
- `training.train_visualization.enabled=true`
- `logging.use_wandb=true`

Adapter overrides for this candidate:
- `data.adapter.tf.mixing.mix_block_length=8`
- `data.adapter.tf.mixing.python_prefetch_queue_size=8`
- `data.adapter.tf.mixing.python_prefetch_min_ready_datasets=8`
- `data.adapter.tf.prefetch.final_stream_buffer=8`
- `data.adapter.tf.prefetch.per_dataset_stream_buffer=1`
- `data.adapter.tf.prefetch.episode_queue_buffer=2`
- `data.adapter.tf.tfds_read.decode_parallelism=6`
- `data.adapter.tf.tfds_read.interleave_parallelism=6`
- `data.adapter.tf.pipeline.episode_concurrency=6`
- `data.adapter.tf.pipeline.transform_parallelism=12`
- `data.adapter.tf.pipeline.interleave_parallelism=6`
- `data.adapter.tf.mixing.per_dataset_private_threadpool_size=8`

Early status:
- run started on LRZ (`lrz-dgx-a100-004`)
- val sanity artifact exists: `visualizations/val_samples_step000000.{png,json}`
- startup progressed to `initial ready=8/29` in both sanity and train phases.

### Local run (workstation, GCS-backed path, bs48/spe0)

Run:
- `/mnt/data/workspace/runs/hlrp/2026-02-18_16-53-45_local_s2_bs48_spe0_reasonable_v1`

Base:
- `experiment=vla_smol_flow_shared`
- `cluster=local_dev`
- `data=oxe_local_spe0`
- `data.loader.batch_size=48`
- `data.adapter.tf.sampling.samples_per_episode=0`
- `training.max_steps=25000`
- `training.max_epochs=-1`
- `training.validation.check_interval=2000`
- `training.validation.visualization.enabled=true`
- `training.train_visualization.enabled=true`
- `logging.use_wandb=true`

Adapter overrides:
- `data.adapter.tf.mixing.strategy=python`
- `data.adapter.tf.mixing.mix_block_length=4`
- `data.adapter.tf.mixing.parallelism_mode=sqrt`
- `data.adapter.tf.mixing.python_prefetch_queue_size=1`
- `data.adapter.tf.mixing.python_prefetch_min_ready_datasets=1`
- `data.adapter.tf.tfds_read.skip_steps_decoding=true`
- `data.adapter.tf.tfds_read.cycle_length=4`
- `data.adapter.tf.tfds_read.decode_parallelism=2`
- `data.adapter.tf.tfds_read.interleave_parallelism=2`
- `data.adapter.tf.pipeline.emit_encoded_pairs=true`
- `data.adapter.tf.pipeline.episode_concurrency=2`
- `data.adapter.tf.pipeline.transform_parallelism=4`
- `data.adapter.tf.pipeline.interleave_parallelism=2`
- `data.adapter.tf.prefetch.final_stream_buffer=2`
- `data.adapter.tf.prefetch.per_dataset_stream_buffer=0`
- `data.adapter.tf.prefetch.episode_queue_buffer=0`

Early status:
- run alive and in sanity-check startup (`Sanity Checking: 0/?`)
- startup reached `python-prefetch mixing: initial ready=1/29 (elapsed~25.8s)`
- no train-step lines yet at latest check.

## 2026-02-18 Cluster high-RAM v2 + local bs48 iterations

### Cluster: high-RAM v2 (running)

Submitted:
- LRZ: `5486298` (`cluster_s2_bs48_spe0_highram_v2_lrz`)
- MCML: `5486299` (`cluster_s2_bs48_spe0_highram_v2_mcml`) -> canceled after LRZ started.

Config (vs `data=oxe_cluster_spe0`):
- `data.loader.batch_size=48`
- `data.adapter.tf.sampling.samples_per_episode=0`
- `data.adapter.tf.mixing.mix_block_length=8`
- `data.adapter.tf.mixing.python_prefetch_queue_size=2`
- `data.adapter.tf.mixing.python_prefetch_min_ready_datasets=4`
- `data.adapter.tf.prefetch.final_stream_buffer=4`
- `data.adapter.tf.prefetch.per_dataset_stream_buffer=1`
- `data.adapter.tf.prefetch.episode_queue_buffer=1`
- `data.adapter.tf.mixing.per_dataset_private_threadpool_size=4`
- `+cluster.compute.cpus_per_task=32`
- `+cluster.compute.mem_gb=220`
- `cluster.compute.time_limit=01:30:00`
- `training.max_steps=25000`
- `training.max_epochs=-1`
- `training.profiler.enabled=false`
- `training.validation.num_sanity_val_steps=0`
- `training.validation.check_interval=2000`
- `training.validation.visualization.enabled=true`
- `training.train_visualization.enabled=true`
- `logging.use_wandb=true`

Status snapshot:
- running on `lrz-dgx-a100-004`
- reached at least `[Step 800]` without OOM/container errors.

### Local: bs48/spe0 iterations (GCS, source=auto)

Runs and outcome:
- `2026-02-18_17-04-03_local_s2_bs48_spe0_reasonable_v3`
  - conservative python mixer (`skip_steps_decoding=true`, decode/interleave low)
  - stable but too slow (`~0.03 it/s` early)
- `2026-02-18_17-07-15_local_s2_bs48_spe0_fast_v4`
  - faster decode/interleave settings
  - failed with `RuntimeError: Background prefetch failed: InvalidArgumentError()`
- `2026-02-18_17-09-57_local_s2_bs48_spe0_stable_repro_v5`
  - repro of earlier fast settings
  - failed with same prefetch `InvalidArgumentError()` (during sanity-val path)
- `2026-02-18_17-12-05_local_s2_bs48_probe_v6`
  - safe fast probe settings:
    - `strategy=python`, `mix_block_length=4`, `queue=1`, `min_ready=1`
    - `skip_steps_decoding=true`, `emit_encoded_pairs=true`
    - `decode_parallelism=4`, `interleave_parallelism=1`
    - `transform_parallelism=4`, `pipeline.interleave_parallelism=1`
    - `prefetch.final_stream_buffer=2`, `per_dataset_stream_buffer=0`, `episode_queue_buffer=0`
  - completed `30/30` in `49s` (~`0.60 it/s`) with no prefetch error.

Current local long run:
- `2026-02-18_17-13-55_local_s2_bs48_spe0_reasonable_v6_long`
- same adapter settings as `v6` probe, with:
  - `training.max_steps=25000`
  - `training.validation.num_sanity_val_steps=0`
  - `training.validation.check_interval=2000`
  - `training.validation.limit_batches=20`
  - `training.validation.visualization.enabled=true`
  - `training.train_visualization.enabled=true`
  - `logging.use_wandb=true`
- status snapshot: reached `24/25000` at about `0.78 it/s`; no error at snapshot.
