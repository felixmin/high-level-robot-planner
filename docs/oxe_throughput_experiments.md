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

### Alternative (preferred for composite batches): python “keep-hot” mixing
Use:
- `data=laq_oxe_cluster_mirror_large_gcs_python_hot` (adapter: `config/data/adapter/oxe_tf_gcs_python_hot.yaml`)

Start from:
- `data.adapter.tf.mixing.strategy=python`
- `data.adapter.tf.mixing.mix_block_length=8` (works for `batch_size=64` and `batch_size=128`; for bs=128, `mix_block_length=16` was slightly better in our short runs)
- `data.adapter.tf.mixing.python_prefetch_min_ready_datasets=8`

This produces **composite batches** (multiple datasets within a batch) while staying fast in steady-state.

## Why “just shuffle more” is hard (decoded images are too big)
Attempting to use a large post-mix shuffle buffer on the *decoded* (uint8) stream is prohibitively expensive
and can OOM on local training-from-GCS:

- Run dir: `runs/2026-02-03_11-18-34_laq_hf_local`
- Override: `data.adapter.tf.train.global_stream_shuffle_buffer=2048`
- Observation: TF spent minutes filling the shuffle buffer and the process was killed while still far below 2048.

This motivates an Octo-like ordering: shuffle/mix while samples are still lightweight (encoded bytes), and only
decode/resize after mixing.

## Octo-like experiments: mix before decode/resize (SkipDecoding on RLDS `steps`)
Implementation (HLRP):
- TFDS read option: `data.adapter.tf.tfds_read.skip_steps_decoding=true`
- Post-mix decode: `data.adapter.tf.pipeline.post_mix_decode_resize=true`
- Adapter preset: `config/data/adapter/oxe_tf_gcs_octo_like.yaml`
- Data presets:
  - Smoke: `data=laq_oxe_cluster_mirror_extended_smoke_gcs_octo_like`
  - Full:  `data=laq_oxe_cluster_mirror_large_gcs_octo_like`

Key idea: TFDS returns images as `tf.string` (encoded bytes) when skipping `steps` decoding. We then:
1) form frame-pairs using the encoded strings (cheap),
2) mix/shuffle the encoded pairs (much cheaper than shuffling uint8 images),
3) decode+resize after mixing.

### E8 (smoke): 7 datasets, encoded shuffle buffer 1024, decode/resize after mix
- Run dir: `runs/2026-02-03_11-31-55_laq_hf_local`
- Data: `data=laq_oxe_cluster_mirror_extended_smoke_gcs_octo_like`
- Note: first warmup batch took ~65s due to initial TFDS init + filling `global_stream_shuffle_buffer=1024`.
  After the shuffle buffer filled, `next()` was near-instant for the short measured window (prefetch stayed full).

TODO: run longer horizons + torch-mode to confirm steady-state end-to-end throughput (and whether p99 stalls
return once prefetch drains).

### E9 (full): 32 datasets, diagnosing “first batch” latency (per-dataset cold starts)
Observation: for the 32-dataset mixture, the *first* element from many datasets is **multi-second** even when
we disable global shuffle and keep parallelism low. This explains why per-sample mixing (`mix_block_length=1`)
can take minutes to produce the first batch: a batch of 64 samples will touch ~28 distinct datasets in expectation.

Method: prime each dataset once (fetch 1 element from each child pipeline; encoded pairs, no post-mix decode)
- Run dir: `runs/2026-02-03_13-56-40_laq_hf_local`
- Command shape:
  - `benchmark.warmup_steps=0 benchmark.steps=0 benchmark.prime_each_dataset_samples=1`
  - `data=laq_oxe_cluster_mirror_large_gcs_octo_like`
  - `data.adapter.tf.train.global_stream_shuffle_buffer=0`
  - `data.adapter.tf.tfds_read.cycle_length=1` (reduce per-dataset parallel reads)
- Results (sample):
  - `language_table`: ~9s
  - `stanford_robocook_converted_externally_to_rlds`: ~20s
  - `spoc`: ~12s
  - worst observed in this run: `toto` ~36s, `fmb` ~22s

Conclusion: we need either (a) **parallel priming / per-dataset background prefetch** to overlap these cold
starts, and/or (b) a mixing scheme that does not require touching ~O(#datasets) distinct datasets per batch
to make progress.

### E10 (full): Python “keep-hot” mixer v0 (blocking on empty queues) is catastrophically slow
This was the first integrated Python mixer implementation (`mixing.strategy=python`) which kept per-dataset
background threads and a small per-dataset queue of *blocks* (`mix_block_length`).

Run dir: `runs/2026-02-03_14-37-20_laq_hf_local`
- Settings (shape):
  - `data=laq_oxe_cluster_mirror_large_gcs_octo_like`
  - `mixing.strategy=python`, `mix_block_length=8`, `batch_size=64`
  - `pipeline.emit_encoded_pairs=true`, `pipeline.post_mix_decode_resize=false` (decode happens in python mixer)
  - `python_prefetch_min_ready_datasets=8`, `python_prefetch_queue_size=2`
- Result (measured only; warmup excluded): **mean 11.545s**, p50 0.236s, p99 55.016s ⇒ **0.087 batches/s** (5.5 samples/s)

Diagnosis: the consumer sampled datasets by weight and did a blocking `Queue.get()`. When it selected a dataset
whose queue was temporarily empty, it stalled for tens of seconds waiting for that dataset to produce its next block.

### E11 (full): Python “keep-hot” mixer v1 (non-blocking selection) eliminates long-tail stalls
Fix: when selecting the next dataset block, **do not block on an empty queue**. Instead, resample until a non-empty
queue is found (fallback to a ready-subset selection if needed). This keeps mixing “as weighted as possible” while
remaining non-blocking in the common case.

#### E11a: `batch_size=64`, `mix_block_length=8` (composite batches, fast)
Run dir: `runs/2026-02-03_14-52-26_laq_hf_local`
- Result (measured): **mean 0.0916s**, p99 0.1156s ⇒ **10.914 batches/s** ⇒ **698.5 samples/s**
- Mixing check: batches contain multiple datasets (example counts in a single batch):
  - `{'mimic_play': 24, 'robo_set': 8, 'robonet': 8, 'bc_z': 8, ...}`
- Note: warmup batch time is dominated by initial queue-fill to `min_ready=8` (~50s). After that, throughput is stable.

#### E11b: `batch_size=128`, `mix_block_length=16` (composite batches, meets target)
Run dir: `runs/2026-02-03_14-54-15_laq_hf_local`
- Result (measured): **mean 0.3443s**, p99 0.4786s ⇒ **2.904 batches/s** ⇒ **371.7 samples/s**
- This comfortably exceeds the original target (≥2 batches/s at bs=128) while still producing composite batches.

### E12: `samples_per_episode=1` is still very slow (episode-level I/O dominates)
We historically saw this mode as slow; the python “keep-hot” mixer does **not** fix it because the bottleneck is
within each dataset pipeline: each output sample requires advancing to a new episode (and thus reading/parsing a new
episode record), so per-episode overhead and wasted episode payload dominates.

Settings (both runs): `data=laq_oxe_cluster_mirror_large_gcs_python_hot`, `batch_size=64`, `mix_block_length=8`
- Baseline (`samples_per_episode=0`, yield all pairs per episode):
  - Run dir: `runs/2026-02-03_15-29-10_laq_hf_local`
  - Result: mean **0.1221s**, p99 **0.2070s** ⇒ **8.192 batches/s** ⇒ **524.3 samples/s**
- One-sample-per-episode (`samples_per_episode=1`):
  - Run dir: `runs/2026-02-03_15-26-11_laq_hf_local`
  - Result: mean **3.4303s**, p99 **7.1142s** ⇒ **0.292 batches/s** ⇒ **18.7 samples/s**

### E13: `samples_per_episode=1` baseline (longer horizon + metadata enabled)
Settings: `data=laq_oxe_cluster_mirror_large_gcs_python_hot`, `batch_size=64`, `mix_block_length=8`,
`samples_per_episode=1`, `return_metadata=true`, `warmup_steps=2`, `measured_steps=30`
- Run dir: `runs/2026-02-03_17-29-33_laq_hf_local`
- Result (measured): **mean 3.2938s**, p99 **10.7063s** ⇒ **0.304 batches/s** ⇒ **19.4 samples/s**
- Note: tail stalls still present (max ~12s) even after the long-tail “switch stall” fix; these stalls now appear
  to be driven by per-episode overhead (not by mixer blocking).

### E14: improve `samples_per_episode=1` by reducing python block size
Hypothesis: when `samples_per_episode=1`, each dataset worker must read **one episode per sample**. Smaller
`mix_block_length` should reduce the “time-to-first-block” per dataset (fewer episodes per block), improving
steady-state throughput.

Settings: same as E13, but:
- `data.adapter.tf.mixing.mix_block_length=4`
- `data.adapter.tf.mixing.python_prefetch_queue_size=4`

- Run dir: `runs/2026-02-03_17-34-53_laq_hf_local`
- Result (measured): **mean 2.9819s**, p99 **8.1507s** ⇒ **0.335 batches/s** ⇒ **21.5 samples/s**

### E15: episode prefetch does not help (and can hurt)
Settings: E14 + `data.adapter.tf.prefetch.episode_queue_buffer=64` (≈2 prefetched episodes per dataset)
- Run dir: `runs/2026-02-03_17-37-42_laq_hf_local`
- Result (measured): **mean 3.3728s**, p99 **9.0295s** ⇒ **0.296 batches/s** ⇒ **19.0 samples/s**

### E16: higher per-dataset read/episode concurrency hurts (startup + throughput)
Settings: E14 + `data.adapter.tf.pipeline.episode_concurrency=16` and `data.adapter.tf.tfds_read.cycle_length=16`
- Run dir: `runs/2026-02-03_17-40-43_laq_hf_local`
- Observation: much slower “initial ready” (~111s vs ~59s)
- Result (measured): **mean 3.4541s**, p99 **10.5383s** ⇒ **0.290 batches/s** ⇒ **18.5 samples/s**

### E17: smaller blocks (2) are not better than 4
Settings: E13 but `mix_block_length=2`, `python_prefetch_queue_size=8`
- Run dir: `runs/2026-02-03_17-44-50_laq_hf_local`
- Result (measured): **mean 3.2109s**, p99 **9.3983s** ⇒ **0.311 batches/s** ⇒ **19.9 samples/s**

### E18: larger per-dataset queue (blocks) does not help
Settings: E14 but `python_prefetch_queue_size=8`
- Run dir: `runs/2026-02-03_17-47-49_laq_hf_local`
- Result (measured): **mean 3.2158s**, p99 **6.7859s** ⇒ **0.311 batches/s** ⇒ **19.9 samples/s**

## Current best config for `samples_per_episode=1` (so far)
Best among the tried variants is E14:
- `mix_block_length=4`, `python_prefetch_queue_size=4`
- Achieves **~21.5 samples/s** at `batch_size=64` on the 32-dataset GCS mixture.

This is still **far** below the “all pairs per episode” mode and far below the overall throughput targets. This
supports the working theory that `samples_per_episode=1` is fundamentally limited by *per-episode* RLDS/TFDS read
and parse overhead (which is amortized in the `samples_per_episode=0` / “all pairs” mode).

## Next experiments (planned)
We hit the throughput target with composite batches; remaining work is mostly ergonomics + scaling:
1) Turn the “python keep-hot” approach into a clean training preset (Hydra config), so LAQ training can use it directly.
2) Reduce startup cost (queue-fill / cold-start overlap) without hurting steady-state:
   - tune `python_prefetch_min_ready_datasets` (mixing quality vs. startup latency),
   - tune `mix_block_length` (more datasets per batch vs. prefetch/block assembly cost).
3) Stress-test with >32 datasets (expected future: 30+ → 50+), and monitor memory (one TF pipeline per dataset worker).

## Notes on discarded ideas
- A “startup warm blocks per dataset” approach was attempted (to force each dataset to produce a few blocks
  before normal mixing), but it triggered a tf.data shape error at iterator creation for batched output
  (`expected [128,...] but got []`). It was removed to keep the pipeline stable.
