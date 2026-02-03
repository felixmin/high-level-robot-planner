# Dataset Additions Log

## Current State (2026-02-02)
- Expanded OXE TF streaming with additional datasets that exist in the LRZ cluster mirror and on `gs://gresearch/robotics/`:
  `aloha_mobile`, `droid`, `berkeley_autolab_ur5`, `jaco_play`, `kuka`, `taco_play`, `roboturk`.
- Added a matching multi-dataset preset: `config/data/laq_oxe_cluster_mirror_extended.yaml` (smoke test: `config/data/laq_oxe_cluster_mirror_extended_smoke.yaml`).
- Added a training experiment config using the full extended dataset set: `config/experiment/laq_oxe_cluster_mirror_extended_val_3.yaml`.
- Roboturk metadata streaming is stable again (no segfault) via a scan-free per-episode pairing path for state-less datasets.
- OXE TF streaming remains CPU-only (TensorFlow GPUs are disabled in the adapter); prefer running benchmarks/training on CPU right now.
- Throughput investigation for local GCS streaming points to cross-dataset switching + cold-start stalls as the primary slowdown mechanism; see log entries below and `scripts/bench_oxe_dataloader_detailed.py`.
- For local GCS streaming runs, use `data=laq_oxe_cluster_mirror_extended_gcs_fast` (adapter preset: `config/data/adapter/oxe_tf_gcs_fast.yaml`).
  - Current recommended defaults in that adapter: `parallelism_mode=sqrt`, `mix_block_length=128`, `final_stream_buffer=8` (batch_size=128).

## Log
- 2026-02-02: Removed an earlier attempt to add more `language_table_*_oracle_sim` datasets (not desired for this expansion).
- 2026-02-02: Added support + configs for 7 additional non-language-table OXE datasets (cluster mirror aligned).
- 2026-02-02: Isolated a Roboturk crash: `roboturk` + `return_metadata=true` segfaulted when `samples_per_episode=0` (full scan path). The fast path (`samples_per_episode=1`) and the non-metadata path (`return_metadata=false`) worked. Root cause was inside the TF `Dataset.scan()` pipeline (reproducible even without PyTorch).
- 2026-02-02: Fixed Roboturk metadata crash by bypassing `Dataset.scan()` for `state_dim<=0` datasets and using `Dataset.zip(frames, frames.skip(offset))` + windowed action reductions instead (Roboturk now works with `return_metadata=true` and `samples_per_episode=0`).
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
