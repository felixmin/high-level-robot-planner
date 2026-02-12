# Dataset Additions Log

## Current State (2026-02-10)
- Expanded OXE TF streaming with additional datasets that exist in the LRZ cluster mirror and on `gs://gresearch/robotics/`:
  `aloha_mobile`, `droid`, `berkeley_autolab_ur5`, `jaco_play`, `kuka`, `taco_play`, `roboturk`,
  plus a larger set used for multi-dataset scaling experiments:
  `bc_z`, `berkeley_cable_routing`, `columbia_cairlab_pusht_real`, `mimic_play`,
  `berkeley_fanuc_manipulation`, `dobbe`, `uiuc_d3field`,
  `ucsd_kitchen_dataset_converted_externally_to_rlds`, `ucsd_pick_and_place_dataset_converted_externally_to_rlds`,
  `furniture_bench_dataset_converted_externally_to_rlds`, `maniskill_dataset_converted_externally_to_rlds`,
  `robo_set`, `stanford_hydra_dataset_converted_externally_to_rlds`, `stanford_robocook_converted_externally_to_rlds`,
  `spoc`, `tidybot`, `toto`, `viola`, `vima_converted_externally_to_rlds`, `utaustin_mutex`, `fmb`.
- Added a matching multi-dataset preset: `config/data/laq_oxe_cluster_mirror_extended.yaml` (smoke test: `config/data/laq_oxe_cluster_mirror_extended_smoke.yaml`).
- Added a training experiment config using the full extended dataset set: `config/experiment/laq_oxe_cluster_mirror_extended_val_3.yaml`.
- Roboturk metadata streaming is stable again (no segfault) via a scan-free per-episode pairing path for state-less datasets.
- Added a 30+ dataset preset for scaling: `config/data/laq_oxe_cluster_mirror_large_gcs_choose32.yaml`.
- Throughput investigation for local GCS streaming points to cross-dataset switching + cold-start stalls as the primary slowdown mechanism; see log entries below and `scripts/bench_oxe_dataloader_detailed.py`.
- For local GCS streaming runs, use `data=laq_oxe_cluster_mirror_extended_gcs_fast` (adapter preset: `config/data/adapter/oxe_tf_gcs_fast.yaml`).
  - Current recommended defaults in that adapter: `parallelism_mode=sqrt`, `mix_block_length=128`, `final_stream_buffer=16` (batch_size=128).
- For 30+ datasets, prefer `data=laq_oxe_cluster_mirror_large_gcs_choose32` (adapter preset: `config/data/adapter/oxe_tf_gcs_large_choose32.yaml`).
  - Key knobs: `prefetch.per_dataset_stream_buffer=0` and `mixing.selector_run_length>=32` to avoid long-tail stalls.

## Log
- 2026-02-02: Removed an earlier attempt to add more `language_table_*_oracle_sim` datasets (not desired for this expansion).
- 2026-02-02: Added support + configs for 7 additional non-language-table OXE datasets (cluster mirror aligned).
- 2026-02-02: Isolated a Roboturk crash: `roboturk` + `return_metadata=true` segfaulted when `samples_per_episode=0` (full scan path). The fast path (`samples_per_episode=1`) and the non-metadata path (`return_metadata=false`) worked. Root cause was inside the TF `Dataset.scan()` pipeline (reproducible even without PyTorch).
- 2026-02-02: Fixed Roboturk metadata crash by bypassing `Dataset.scan()` for `state_dim<=0` datasets and using `Dataset.zip(frames, frames.skip(offset))` + windowed action reductions instead (Roboturk now works with `return_metadata=true` and `samples_per_episode=0`).
- 2026-02-03: Fixed `return_metadata=true` for the python “keep-hot” mixer:
  - Python mixing now propagates full metadata (episode_id/language/action/initial_state/dataset_name/robot) instead of only `dataset_name`.
  - Updated the “image-only” placeholders in `OXE_DATASETS` for the 21 added cluster-mirror datasets with correct `instruction_key`, `instruction_in_step`, and action settings so metadata extraction doesn’t fail.
  - Added a training-ready all-pairs + metadata preset: `config/data/laq_oxe_cluster_mirror_large_gcs_python_hot_allpairs_meta.yaml`.
- 2026-02-10: Fixed remaining metadata path breakages found by 32-dataset priming:
  - `kuka`: state stream in TFDS skip-decoding is string-backed in this setup; switched to `state_dim=0` (`state_key=None`) for metadata compatibility.
  - `mimic_play`: `state` is a nested dict; switched to `state_key=state/ee_pose` with `state_dim=7`.
  - Added a cluster-local one-sample preset: `config/data/laq_oxe_cluster_mirror_large_local_python_hot_samples1.yaml`.
  - Added cluster trial runbook for samples1 mode: `docs/cluster_samples1_local_trial.md`.
- 2026-02-10: Re-ran full 32-dataset metadata priming on CPU with the all-pairs metadata preset and confirmed clean setup across all datasets:
  - Command: `conda run -n hlrp python scripts/bench_oxe_dataloader_detailed.py ... data=laq_oxe_cluster_mirror_large_gcs_python_hot_allpairs_meta benchmark.prime_each_dataset_samples=1 benchmark.prime_materialize=true benchmark.warmup_steps=0 benchmark.steps=0`.
  - Run: `runs/2026-02-10_15-06-03_laq_oxe_local`.
  - Result: all datasets primed (`kuka`, `mimic_play`, `fmb` included) and exited with `Benchmark warmup_steps+steps == 0; exiting after setup/priming.` (no `priming failed` lines).
- 2026-02-02: CPU-only smoke throughput checks (GCS, metadata enabled, batch_size=2, `oxe_tf_minimal`, `tfds_read.source=gcs`). These numbers are noisy and include remote I/O variance, but adding datasets clearly reduces throughput:
  - 4-dataset smoke (`data=laq_oxe_all_smoke`, warmup=2, steps=10): mean=0.545s, p50=0.0004s, p90=2.650s, ~3.7 samples/s.
  - 7-dataset smoke (`data=laq_oxe_cluster_mirror_extended_smoke`, warmup=2, steps=10): mean=4.110s, p50=0.0014s, p90=12.486s, ~0.5 samples/s.
- 2026-02-02: CPU dataloader smoke benchmarks (batch_size=16, warmup=20, steps=50, `oxe_tf_minimal`):
  - 4-dataset baseline (`data=laq_oxe_all_smoke`): p50=0.0034s, p90=0.0084s, ~222 samples/s.
  - extended (`data=laq_oxe_cluster_mirror_extended_smoke`): p50=0.0037s, p90=0.0109s, ~141 samples/s (mean includes startup/multi-builder overhead).
- 2026-02-03: Added finer-grained per-batch benchmarking in `scripts/bench_oxe_dataloader_detailed.py`:
  - Records warmup batch latencies too (so cold-start is visible).
  - In TF-only mode, can optionally materialize tensors (`benchmark.tf_materialize`) to measure real I/O/decoding rather than iterator dequeue time.
- 2026-02-03: Confirmed the “more datasets = incredibly slow” observation is mostly *switching* + *first-seen dataset cold-start*, not a steady linear degradation:
  - Baseline (11 datasets, GCS, metadata enabled, batch_size=2, `mix_block_length=1`): mean=2.5945s, p90=6.5064s, ~0.77 samples/s.
    - Largest observed stalls (single batches): `kuka` ~24s, `aloha_mobile` ~27s, plus multi-second stalls for `droid` / `rt1` / `berkeley_autolab_ur5`.
    - “Switch batches” dominate latency (stay batches are near-zero).
  - Tuned settings (block mixing + small per-dataset buffering + more TFDS/pipeline parallelism) materially reduce switching frequency and improve sustained throughput (in longer runs ~5 samples/s at batch_size=2), but cold-start spikes still exist for certain datasets.
- 2026-02-03: Added per-dataset “priming” to `scripts/bench_oxe_dataloader_detailed.py` (`benchmark.prime_each_dataset_samples`) to separate one-time TFDS/GCS startup costs from steady-state throughput.
- 2026-02-03: Multi-dataset (11 datasets) large-batch GCS streaming is *near target* in TF-only benchmarking with priming:
  - `batch_size=128`, `parallelism_mode=sqrt`, `mix_block_length=128`, `final_stream_buffer=8`, `return_metadata=false`, TF mode (no materialize): mean batch time ~0.502s ⇒ ~1.99 batches/s ⇒ ~255 samples/s.
  - Remaining issue: rare long-tail stalls (single batches up to ~12s) still reduce mean; next steps focus on reducing these tail events (or moving them fully into an explicit startup warmup).
- 2026-02-03: Increasing post-mix buffering improved mean throughput and crossed the 2 batches/s target in TF-only mode:
  - `final_stream_buffer=16` (same setup otherwise): mean batch time ~0.488s ⇒ ~2.05 batches/s ⇒ ~263 samples/s (tail stalls still exist).
- 2026-02-03: Added a more detailed write-up of throughput experiments + config knobs: `docs/oxe_throughput_experiments.md`.
- 2026-02-03: Tried adding a “startup warm blocks per dataset” mixing knob; it triggered a tf.data shape error (`expected [128,...] but got []`) and was removed.
- 2026-02-03: Attempted tf.data per-edge latency stats via `tf.data.experimental.StatsAggregator`; the TensorFlow build in `hlrp` did not expose this API, so the feature was removed (we rely on per-batch timing + metadata attribution instead).
- 2026-02-03: Fixed TF GPU initialization warnings in OXE TFDS builder init by routing TF import through `_import_tensorflow_cpu_only()` (so TF doesn’t see GPUs even if TFDS imports TF early).
- 2026-02-03: Added nested key-path support for observation/action/instruction extraction in the OXE adapter (`packages/common/adapters/oxe.py`) so datasets like `mimic_play` can use image keys like `image/front_image_1`.
- 2026-02-03: Added a helper for quick schema inspection: `scripts/inspect_oxe_dataset_sample.py` (prints candidate image keypaths from a TFDS builder dir).
- 2026-02-03: Added 21 more cluster-mirror datasets to `OXE_DATASETS` (image keys verified by sampling one episode on GCS) and a 32-dataset list config: `config/data/dataset/oxe_cluster_mirror_large.yaml`.
- 2026-02-03: Found a major scaling pitfall: buffering *blocks per dataset* does not scale to 30+ datasets (it causes huge long-tail stalls).
  - 32 datasets, `mix_block_length=128`, `per_dataset_stream_buffer=1`, `selector_run_length=4`: `runs/2026-02-03_03-35-22_laq_hf_local` → mean 4.11s ⇒ ~0.24 batches/s (~31 samples/s), p99 ~37.6s, max ~49.3s.
- 2026-02-03: Found a practical fix: disable per-dataset block prefetch and amortize switches with longer `selector_run_length`.
  - 32 datasets, `per_dataset_stream_buffer=0`, `selector_run_length=32`: `runs/2026-02-03_03-45-58_laq_hf_local` → mean 0.286s ⇒ ~3.49 batches/s (~447 samples/s), p99 ~6.6s.
  - 32 datasets, `per_dataset_stream_buffer=0`, `selector_run_length=64`: `runs/2026-02-03_03-41-12_laq_hf_local` → mean 0.185s ⇒ ~5.41 batches/s (~693 samples/s), p99 ~3.5s.
- 2026-02-03: Confirmed that shorter run-lengths can re-introduce long-tail stalls even with per-dataset block prefetch disabled:
  - 32 datasets, `per_dataset_stream_buffer=0`, `selector_run_length=16`: `runs/2026-02-03_03-43-06_laq_hf_local` → mean 1.28s ⇒ ~0.78 batches/s (~100 samples/s), p99 ~25.1s.
