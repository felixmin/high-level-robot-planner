# OXE (GCS) Throughput Experiments

Goal: local training directly from `gs://gresearch/robotics/...` with **≥2 batches/s at `batch_size=128`**
(≈256 samples/s), while scaling to **30+ datasets** without throughput collapse.

This doc focuses on *multi-dataset* throughput and specifically the **long-tail batch stalls**
that appear when mixing many datasets over GCS.

## Setup / measurement notes
- Benchmark script: `scripts/bench_oxe_dataloader_detailed.py`
- Data preset (11 datasets): `data=laq_oxe_cluster_mirror_extended_gcs_fast`
- Default adapter: `config/data/adapter/oxe_tf_gcs_fast.yaml`
- Unless stated otherwise:
  - `data.preprocess.return_metadata=false` (LAQ training doesn’t need metadata in the hot path)
  - `benchmark.prime_each_dataset_samples=1` (pays per-dataset TFDS/GCS “first element” costs up front)
  - TF-only benchmark mode: `+benchmark.detailed_mode=tf` and `+benchmark.tf_materialize=false`
    - Important: `tf_materialize=false` measures iterator stalls (buffer starvation / I/O waits) rather than
      forcing full tensor materialization. This is still the correct signal for “dataloader starvation”.

## What causes long-tail stalls (working theory)
In multi-dataset mode, the mixer (`choose_from_datasets()` / `sample_from_datasets()`) only advances the
currently-selected dataset pipeline. If the selected dataset has no ready buffered element (common on a
switch), the *next batch* blocks on:
- opening TFRecord shards over GCS,
- TFDS file interleave / read scheduling,
- decode + resize,
- episode→pair sampling logic,
until enough elements exist to form the next output batch.

This matches the observed pattern:
- many batches are “instant” (p50 ~0.0003s in TF mode),
- a few batches take multiple seconds (8–12s+), dominating the mean.

## Key knobs (and when to use them)
- `data.adapter.tf.prefetch.final_stream_buffer`
  - Post-mix buffer of *batches*. Increasing this often improves mean throughput by smoothing rare stalls.
  - RAM cost ~ O(buffer * batch_bytes).
- `data.adapter.tf.prefetch.per_dataset_stream_buffer`
  - With `mix_block_length>1`, this buffers *blocks* per dataset (expensive; multiplies by dataset count).
- `data.adapter.tf.mixing.mix_block_length`
  - Setting to the training batch size (128) avoids per-sample switching overhead.
- `data.adapter.tf.mixing.parallelism_mode`
  - `sqrt` has been a good “safe default” to avoid per-dataset thread explosion while still giving each
    dataset some parallelism.
- `data.adapter.tf.mixing.selector_run_length` (only affects `strategy=choose`)
  - Repeat each dataset choice for N consecutive selections to reduce switching frequency. May help or hurt;
    needs benchmarking because it changes access patterns.

## Experiments (11 datasets, GCS)

### E0: `final_stream_buffer=8` (near target)
- Run dir: `runs/2026-02-03_01-32-16_laq_hf_local`
- Result: mean ~0.502s ⇒ **~1.99 batches/s** ⇒ **~255 samples/s**
- Tail: max ~11.8s (rare, but still present)

### E1: increase post-mix buffering (`final_stream_buffer=16`)
- Run dir: `runs/2026-02-03_01-38-55_laq_hf_local`
- Result: mean ~0.488s ⇒ **~2.05 batches/s** ⇒ **~263 samples/s**
- Observation: this crosses the 2 batches/s target in TF-only mode by smoothing the long tail.

### E1b: per-dataset block prefetch helps (but has diminishing returns)
Short torch-mode runs (100 measured batches) suggest:
- `per_dataset_stream_buffer=0` (disabled): `runs/2026-02-03_02-22-01_laq_hf_local` ⇒ ~3.00 batches/s (~384 samples/s)
- `per_dataset_stream_buffer=1` (default): `runs/2026-02-03_02-31-58_laq_hf_local` ⇒ ~3.01 batches/s (~385 samples/s)
- `per_dataset_stream_buffer=2`: `runs/2026-02-03_02-27-13_laq_hf_local` ⇒ ~3.02 batches/s (~386 samples/s)

Variance dominates at this horizon; keep `per_dataset_stream_buffer=1` as the “sane default” for scaling.

### E3: end-to-end (PyTorch) throughput is comfortably above target
- Run dir: `runs/2026-02-03_02-04-29_laq_hf_local`
- Settings: `detailed_mode=torch`, `batch_size=128`, `final_stream_buffer=16`, priming enabled
- Result: mean ~0.300s ⇒ **~3.34 batches/s** ⇒ **~427 samples/s**
- Tail: rare multi-second stalls still occur, but do not prevent achieving ≥2 batches/s in steady state.

### E4: “interleave-like” Python mixer (background prefetch threads)
This is an experimental alternative to tf.data mixing, implemented in `scripts/bench_oxe_python_prefetch_mixer.py`.

- Queue size 1:
  - Run dir: `runs/2026-02-03_02-43-11_laq_hf_local`
  - Result: **~2.19 batches/s (~280 samples/s)**, p90 ~0.74s, mean ~0.46s
- Queue size 2:
  - Run dir: `runs/2026-02-03_02-46-15_laq_hf_local`
  - Result: **~2.52 batches/s (~322 samples/s)**, p90 ~0.26s, mean ~0.40s

Takeaway: this approach can meet the target with enough buffering, but it is slower than the current
tf.data `choose` + block mixing path in end-to-end throughput (and it adds many threads + memory).

### E2: `parallelism_mode=sqrt` (important baseline)
- `parallelism_mode=divide` tended to underutilize per-dataset parallelism as dataset count grows.
- `parallelism_mode=sqrt` keeps throughput high without exploding threads for 30+ datasets.

## Current recommended settings (local GCS, batch_size=128)
Use `data=laq_oxe_cluster_mirror_extended_gcs_fast` and start from:
- `mix_block_length=128`
- `parallelism_mode=sqrt`
- `final_stream_buffer=16` (if RAM allows; 8 is a lower-RAM fallback)

## Experiments (32 datasets, GCS)

Dataset preset (32): `data=laq_oxe_cluster_mirror_large_gcs_fast_large` (dataset list: `config/data/dataset/oxe_cluster_mirror_large.yaml`).

### E5: naive “keep blocks hot per dataset” does not scale
Settings:
- `mix_block_length=128` (batch mixing)
- `prefetch.per_dataset_stream_buffer=1` (buffers *blocks* per dataset)
- `mixing.strategy=choose`, `mixing.selector_run_length=4`

Run dir: `runs/2026-02-03_03-35-22_laq_hf_local`
- Result: mean ~4.11s ⇒ **~0.24 batches/s** ⇒ **~31 samples/s**
- Tail: p90 ~13.45s, p99 ~37.63s, max ~49.25s

Interpretation: with 30+ datasets, “1 prefetched block per dataset” means TF is trying to keep ~N_datasets blocks
ready concurrently. This is expensive and leads to severe long-tail stalls (buffer starvation) during switches.

### E6: fix by *not* prefetching blocks per dataset + amortize switching
Key change:
- set `prefetch.per_dataset_stream_buffer=0`
- increase `mixing.selector_run_length` to reduce switching frequency (while keeping `mix_block_length=128`)

Run dir: `runs/2026-02-03_03-45-58_laq_hf_local` (`selector_run_length=32`)
- Result: mean ~0.286s ⇒ **~3.49 batches/s** ⇒ **~447 samples/s**
- Tail: p99 ~6.60s, max ~10.79s

Run dir: `runs/2026-02-03_03-41-12_laq_hf_local` (`selector_run_length=64`)
- Result: mean ~0.185s ⇒ **~5.41 batches/s** ⇒ **~693 samples/s**
- Tail: p99 ~3.51s, max ~4.58s

Takeaway: for 30+ datasets, it is better to keep *one* dataset streaming smoothly for longer stretches than to try to
keep all datasets “hot” at once.

### E7: too-short selector runs reintroduce long-tail stalls
Run dir: `runs/2026-02-03_03-43-06_laq_hf_local` (`selector_run_length=16`, `per_dataset_stream_buffer=0`)
- Result: mean ~1.28s ⇒ **~0.78 batches/s** ⇒ **~100 samples/s**
- Tail: p99 ~25.10s, max ~33.00s

Conclusion: `selector_run_length` is the main knob that trades off “how frequently we switch datasets” vs throughput.

## Current recommended settings (local GCS, 30+ datasets, batch_size=128)
Use:
- `data=laq_oxe_cluster_mirror_large_gcs_choose32` (adapter: `config/data/adapter/oxe_tf_gcs_large_choose32.yaml`)

Start from:
- `mix_block_length=128`
- `mixing.strategy=choose`
- `mixing.selector_run_length=32` (increase to 64 if you want more throughput and can tolerate longer per-dataset streaks)
- `prefetch.per_dataset_stream_buffer=0`

## Next experiments (planned)
1) Validate end-to-end (PyTorch) throughput in `detailed_mode=torch` (TF→Torch conversion included).
2) Reduce the remaining long-tail stalls without relying only on post-mix buffering:
   - sweep `mixing.selector_run_length` for the chosen dataset count (it dominates switch frequency),
   - only consider `per_dataset_stream_buffer>0` for *small* dataset counts; it scales poorly to 30+ datasets,
   - if needed, add targeted timing around TFDS read/decode stages (the TF build here did not expose
     `tf.data.experimental.StatsAggregator`, so “per-edge” stats are not currently available).

## Notes on discarded ideas
- A “startup warm blocks per dataset” approach was attempted (to force each dataset to produce a few blocks
  before normal mixing), but it triggered a tf.data shape error at iterator creation for batched output
  (`expected [128,...] but got []`). It was removed to keep the pipeline stable.
