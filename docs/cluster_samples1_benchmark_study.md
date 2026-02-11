# Cluster `samples_per_episode=1` Benchmark Study

Last updated: 2026-02-10

## Goal
- Quantify metadata overhead (`return_metadata=true/false`) under matched settings.
- Measure throughput scaling with different CPU allocations.
- Inspect how batch mixture changes over time (potential shuffle/prefetch effects).

## Fixed Setup
- Experiment: `laq_oxe_samples1_bench`
- Dataset preset: `laq_oxe_cluster_mirror_large_local_python_hot_samples1_no_aloha_mimic`
- Container: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/lam.sqsh`
- Logging root: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay`
- Batch size: `64`
- Samples per episode: `1`

## Run Log
| Date | Job ID | Label | CPUs | Metadata | Warmup | Steps | Decode/Interleave | Mix block / queue | batches/s | samples/s | p50 (s) | p90 (s) | Notes |
|---|---:|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|---|
| 2026-02-10 | 5477707 | prior baseline | 16 | false | 4 | 40 | 4/4 | 4/4 | 2.353 | 150.583 | 0.3925 | 0.5501 | earlier short run |
| 2026-02-10 | 5477725 | prior sweep | 32 | false | 4 | 40 | 8/8 | 4/4 | 2.636 | 168.710 | 0.3715 | 0.4414 | best short run so far |
| 2026-02-10 | 5477726 | prior sweep | 32 | false | 4 | 40 | 8/8 | 4/4 | 2.494 | 159.586 | 0.3943 | 0.5110 | same run-dir as 5477725, higher tail |
| 2026-02-10 | 5477727 | prior sweep | 32 | false | 4 | 40 | 8/8 | 2/8 | 2.561 | 163.873 | 0.3885 | 0.4648 |  |
| 2026-02-10 | 5477728 | prior sweep | 32 | false | 4 | 40 | 8/8 | 8/8 | 2.409 | 154.150 | 0.4025 | 0.5776 |  |
| 2026-02-10 | 5477729 | prior metadata diag | 32 | true | 2 | 20 | 4/4 | 4/4 | 1.874 | 119.915 | 0.4409 | 0.6723 | short run, sensitive to first outlier |

## New Campaign (This Turn)
Planned:
1. Matched metadata A/B at fixed CPU count.
2. CPU sweep (lower/higher CPUs) on non-metadata hot path.
3. Long metadata run for mixture-over-time analysis and plots.

### Dataloader Change Applied
- Added `data.adapter.tf.metadata.mode` with:
  - `full` (default): existing full metadata payload.
  - `dataset_only`: keep `return_metadata=true` API path but emit only `dataset_name`/`dataset_type` in the python mixer fast path.
- Goal: preserve dataset-mixture introspection while removing expensive per-batch metadata conversions.
- Files changed:
  - `config/data/adapter/oxe_tf_low_ram.yaml`
  - `packages/common/data.py`
  - `packages/common/adapters/oxe.py`

### Completed This Turn
| Date | Job ID | Label | CPUs | Metadata | Metadata mode | Warmup | Steps | batches/s | samples/s | p50 (s) | p90 (s) | p99 (s) | Notes |
|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-02-10 | 5477758 | metadata A/B | 32 | true | dataset_only | 4 | 80 | 2.592 | 165.862 | 0.3634 | 0.4885 | 0.6162 | new optimized path |
| 2026-02-10 | 5477759 | metadata A/B | 32 | true | full | 4 | 80 | 2.378 | 152.203 | 0.4075 | 0.4963 | 0.6454 | matched control |
| 2026-02-10 | 5477760 | CPU sweep | 8 | false | full | 4 | 80 | 1.070 | 68.505 | 0.8869 | 1.2272 | 2.2229 | clearly under-provisioned |
| 2026-02-10 | 5477761 | CPU sweep | 16 | false | full | 4 | 80 | 2.247 | 143.782 | 0.4321 | 0.5897 | 0.6841 | overlapped with other jobs |
| 2026-02-10 | 5477762 | CPU sweep | 32 | false | full | 4 | 80 | 2.193 | 140.354 | 0.4427 | 0.5851 | 0.6721 | overlapped with other jobs |
| 2026-02-10 | 5477775 | CPU sweep (isolated) | 16 | false | full | 4 | 80 | 2.191 | 140.200 | 0.4492 | 0.6054 | 0.6908 | clean rerun |
| 2026-02-10 | 5477778 | CPU sweep (isolated) | 32 | false | full | 4 | 80 | 2.363 | 151.227 | 0.4259 | 0.5181 | 0.5940 | clean rerun |
| 2026-02-10 | 5477763 | long mixture run | 32 | true | dataset_only | 20 | 1200 | 1.922 | 122.978 | 0.5110 | 0.6472 | 0.8073 | ran to metrics+artifacts; then cancelled due teardown hang |
| 2026-02-10 | 5477782 | TF ablation (full meta) | 32 | true | full | 8 | 160 | 2.590 | 165.776 | 0.3805 | 0.4744 | 0.5167 | `samples_per_episode=1`, baseline (4/4, mix4/q4) |
| 2026-02-10 | 5477783 | TF ablation (full meta) | 32 | true | full | 8 | 160 | 1.971 | 126.158 | 0.5024 | 0.6599 | 0.7614 | `samples_per_episode=1`, read 8/8 |
| 2026-02-10 | 5477784 | TF ablation (full meta) | 32 | true | full | 8 | 160 | 2.098 | 134.282 | 0.4794 | 0.5689 | 0.6962 | `samples_per_episode=1`, read 2/2 |
| 2026-02-10 | 5477785 | TF ablation (full meta) | 32 | true | full | 8 | 160 | 1.912 | 122.377 | 0.5186 | 0.6406 | 0.7281 | `samples_per_episode=1`, mix2/q4 |
| 2026-02-10 | 5477786 | TF ablation (full meta) | 32 | true | full | 8 | 160 | 2.555 | 163.518 | 0.3839 | 0.5155 | 0.6091 | `samples_per_episode=1`, mix8/q4 |
| 2026-02-10 | 5477787 | TF ablation (full meta) | 32 | true | full | 8 | 160 | 2.046 | 130.955 | 0.4667 | 0.5996 | 0.8635 | `samples_per_episode=1`, mix4/q8 |
| 2026-02-10 | 5477781 | long full-metadata run | 32 | true | full | 20 | 1200 | 2.310 | 147.840 | 0.4232 | 0.5424 | 0.6671 | `samples_per_episode=1`, sustained run |
| 2026-02-10 | 5477788 | samples sweep | 32 | true | full | 8 | 160 | 1.862 | 119.176 | 0.5217 | 0.6912 | 0.9327 | `samples_per_episode=2` |
| 2026-02-10 | 5477789 | samples sweep | 32 | true | full | 8 | 160 | 1.806 | 115.598 | 0.5408 | 0.6976 | 0.9215 | `samples_per_episode=5` |
| 2026-02-10 | 5477790 | samples sweep | 32 | true | full | 8 | 160 | 2.276 | 145.694 | 0.4262 | 0.5435 | 0.6368 | `samples_per_episode=10` |
| 2026-02-10 | 5477791 | samples sweep | 32 | true | full | 8 | 160 | 2.422 | 154.993 | 0.4045 | 0.5217 | 0.5869 | `samples_per_episode=0` (all pairs) |

Interpretation (current):
- `dataset_only` vs `full` at same settings improved throughput by ~`+8.99%` samples/s (`165.862` vs `152.203`).
- This closes much of the previously observed metadata penalty while keeping per-batch dataset-mixture visibility.
- CPU scaling is non-linear; `8` CPUs is clearly too low.
- Isolated reruns show `32` CPUs > `16` CPUs on no-metadata hot path (`151.227` vs `140.200` samples/s, ~`+7.9%`).
- Long full-metadata reference (`5477781`) reached `147.840 samples/s` over `1200` measured batches (`samples_per_episode=1`), i.e. close to the 150 target in sustained mode.
- TF settings under full metadata:
  - Best in this batch: baseline-like `4/4` read with larger block mixing (`mix8/q4`, `5477786`: `163.518 samples/s`).
  - Read parallelism extremes (`8/8`, `2/2`) underperformed baseline in these runs.
  - Increasing queue depth alone (`mix4/q8`) hurt throughput.
- `samples_per_episode` sweep with full metadata (same TF baseline):
  - `2`: `119.176 samples/s`
  - `5`: `115.598 samples/s`
  - `10`: `145.694 samples/s`
  - `all (0)`: `154.993 samples/s`
  - Behavior is non-monotonic; the strongest points in this sweep were `10` and `all`.

### Long-Run Mixture Artifacts
- Run: `5477763` (1200 measured batches)
- Artifact directory:
  - `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-02-10_18-58-44_laq_oxe_samples1_bench/mixture_analysis`
- Files:
  - `mixture_over_time.svg`
  - `mixture_windows_every_stride.svg` (5 sequential batches every 100 batches)
  - `mixture_summary.txt`
- Summary highlights (`mixture_summary.txt`):
  - measured_batches: `1200`
  - unique_dominant_datasets: `29`
  - dominant_switch_rate: `0.9658`

### Long-Run Mixture Artifacts (Full Metadata)
- Run: `5477781` (1200 measured batches, `metadata.mode=full`)
- Artifact directory:
  - `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-02-10_19-23-24_laq_oxe_samples1_bench/mixture_analysis`
- Files:
  - `mixture_over_time.svg`
  - `mixture_windows_every_stride.svg`
  - `mixture_summary.txt`

## Training Takeaways (Actionable)
For actual training runs on this cluster setup:

1. If full metadata is required and `samples_per_episode=1`:
- Use `CPUs=32`.
- Use `tfds_read.decode_parallelism=4`, `tfds_read.interleave_parallelism=4`, `pipeline.interleave_parallelism=4`.
- Use python mixing with `mix_block_length=4` or `8`, `python_prefetch_queue_size=4`.
- Expected sustained throughput range from this study: ~`148-164 samples/s` (depending on short vs long horizon and exact mixing block).

2. If pure throughput is the priority (and training allows it):
- `samples_per_episode=0` (all pairs) was fastest in the full-metadata sweep (`~155 samples/s` in short run).
- `samples_per_episode=10` was close (`~146 samples/s`), and much faster than `2` or `5`.

3. Avoid in this environment:
- `decode/interleave=8/8` (consistently worse in our ablation).
- `python_prefetch_queue_size=8` with `mix_block=4` (degraded throughput).
- `CPUs=8` (severe throughput drop).

4. Interpretation caveat:
- Short runs can over/under-estimate due to startup and first-seen stalls.
- Prefer warmup>=`8` and measured steps>=`160`; for decisions, use long runs (~`1200` measured).

## Further Investigation
High-priority follow-ups:

1. Confirm best mixing block in long full-metadata runs:
- Run `mix_block=4` vs `mix_block=8` with `steps=1200` and identical seed/settings.
- Decide final production default from sustained result, not short benchmark.

2. Repeat `samples_per_episode` sweep with longer runs:
- Validate the observed non-monotonic pattern (`2/5` slower, `10/all` faster).
- Check whether this holds across multiple seeds and at least two nodes.

3. Add end-to-end training validation (not only dataloader benchmark):
- Measure actual train-step throughput and utilization for top 2-3 dataloader configs.
- Confirm no regressions in validation logic that consumes full metadata.

4. Investigate startup and teardown overhead:
- Startup: evaluate `prime_each_dataset_samples` for lowering early instability.
- Teardown: occasional “metrics written but job still RUNNING” behavior should be traced in cleanup path.

5. Dataset-level bottleneck profiling:
- Use `batch_times.jsonl` to rank recurrent slow dominant datasets and large-gap returns.
- Check whether specific datasets need custom per-dataset handling (weights/parallelism/skip settings).

## Plot Outputs
Generated from `batch_times.jsonl`:
- `mixture_windows_every_stride.svg`: 5 sequential batches sampled every stride (100 in long runs).
- `mixture_over_time.svg`: stacked mixture share over batch index.
- `mixture_summary.txt`: compact numeric summary (switch-rate, dominant coverage, top datasets).

## Final Findings
- Metadata overhead is mostly conversion-path overhead (`tf -> numpy -> python`) rather than metadata byte volume.
- Full metadata with `samples_per_episode=1` can sustain near-target performance (`147.84 samples/s` in long run).
- Best-performing settings in this study favored moderate read parallelism (`4/4`) and python mixing with controlled block/queue (`4/4` or `8/4`).
