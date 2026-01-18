# OXE Local Experiment Log

## Baseline setup
- Trigger command: `conda run -n hlrp python scripts/2_train_laq.py experiment=laq_oxe_local …`
- Profiler: Lightning `SimpleProfiler`; output at `runs/<timestamp>_laq_oxe_local/profiles/fit-fit-profile.txt`.
- Config path: `config/experiment/laq_oxe_local.yaml` (sourced defaults plus per-run overrides via Hydra CLI).
- Goal: maximize throughput within ~64 GB RAM while iterating on GPU utilization and tf.data shuffle/prefetch knobs.

## Key knobs
- `data.batch_size`: larger batches push more work to GPU but increase per-batch load time.
- `data.episode_shuffle_buffer` / `data.pair_shuffle_buffer`: fill time increases with buffer size (large buffers delay startup), but once filled the pipeline mixes samples better and keeps the dataloader busy. We explored values from 0 up to 5 000.
- `data.prefetch_buffer`: controls tf.data prefetch; 32/64 gave consistent gains, -1 enables AUTOTUNE.
- `data.num_parallel_episodes` & `data.num_parallel_calls`: higher values add TF-side concurrency; we now split them per dataset to avoid oversubscription.
- `data.num_workers`: stayed at 0 (tf.data already parallelized). Future work can enable Torch workers once iterable length issues are resolved.
- `training.max_steps` / `data.samples_per_episode`: longer runs (600 steps) fill shuffle buffers and expose steady-state behaviors; short runs (≤300) focus on short iteration time.

## Experiment results
| Run label | Overrides (diff from `laq_oxe_local`) | Dataloader % (SimpleProfiler) | Peak RSS (MB) | Wall time | Notes |
|-----------|----------------------------------------|------------------------------|--------------|-----------|-------|
| base | `batch_size=24`, `prefetch_buffer=64`, no shuffle/pair buffers, 32 episodes/calls | ~77% | ~9 000 | ~1:32 | Quick startup, shuffle disabled—dataloader dominates. |
| batch32 + shuffle200/1000 | added `batch_size=32`, `episode_shuffle_buffer=200`, `pair_shuffle_buffer=1000`, prefetch=32, 16 episodes/calls | ~77% | ~13 600 | ~1:39 | Shuffle buffers now fill; dataloader still majority of time. |
| batch32 + shuffle200/5000 + long run | same as above but shuffle pair=5 000, `prefetch_buffer=64`, `max_steps=600`, `num_parallel_*`=32 | ~83% | ~15 800 | ~3:53 | Shuffle takes longer to fill but keeps pipeline busy; GPU still underutilized. |
| batch32 + shuffle200/0 + prefetch32 | removed pair shuffle (reset to 0), prefetch=32, `num_parallel_*`=16, short run | ~78% | ~9 000 | ~2:54 | Lower memory, dataloader still dominant, shows shuffle buffer critical for mixing. |
| batch32 + shuffle200/1000 + concurrent runs | `num_parallel_*`=32, prefetch=64, pair shuffle=1 000, `max_steps=600` | ~82% | ~15–15.8k | ~3:53 | Final baseline for ongoing tuning: shuffle buffers on, dataloader still consuming ~80% of time; GPU remains the bottleneck target. |

## Findings
- The dataloader (tf.data interleave + shuffle/prefetch) absorbs 75–83% of execution time even after aggressive parallelism—GPU utilization stays low.
- Large shuffle buffers (1 000–5 000) are feasible locally; they fill in a few minutes for this small split and then sustain throughput, but they raise RSS and delay startup.
- Increasing `prefetch_buffer` and `batch_size` improves overlap but not enough to shift the bottleneck.
- Next steps: enable PyTorch `num_workers` once iterable length issues are solved and keep tuning shuffle/prefetch concurrency to raise GPU utilization while monitoring RAM.

## Run archive
Each row below points to the corresponding `runs/<timestamp>_laq_oxe_local` folder and matches one of the experiments in the table.

| Run dir | Main overrides | Notes |
|---------|----------------|-------|
| `2026-01-18_19-28-30_laq_oxe_local` | `batch_size=24`, `prefetch_buffer=64`, no shuffle | Baseline, quick turnaround with minimal buffers. |
| `2026-01-18_19-35-18_laq_oxe_local` | `batch_size=16` → `32`, `prefetch=32`, shuffle=200/5000 | Added pair shuffle and TF parallelism (used for the “batch32 + shuffle…” rows). |
| `2026-01-18_20-27-48_laq_oxe_local` | `batch_size=32`, `prefetch=32`, `num_parallel_*`=16, long run | Used for the long/steady-state column with shuffle buffers reducing to 0 and prefetch 32. |
| `2026-01-18_20-33-49_laq_oxe_local` | `batch_size=32`, `prefetch=32`, shuffle=200/1000, `num_parallel_*`=16, `max_steps=600` | Candidate for the final shuffle-heavy baseline with `pair_shuffle_buffer=1000`. |
| `2026-01-18_20-44-10_laq_oxe_local` | `batch_size=32`, `prefetch=64`, shuffle=200/5000, `num_parallel_*`=32 | Largest shuffle buffer run; detailed profiler captured at `profiles/fit-fit-profile.txt`. |

---

## Post-Refactoring Experiments (2026-01-18 evening)

After the data loader refactoring (commits `1d8701e` and `e1169c3`), we noticed performance degradation. The refactoring added multi-dataset support via `MultiOXEFramePairDataset` which divides parallelism settings by the number of datasets.

### Key Insight: Parallelism Division Problem

In `MultiOXEFramePairDataset._init_datasets()` (oxe.py lines 947-959):
```python
num_parallel_episodes_per_ds = max(1, int(self.num_parallel_episodes) // n_datasets)
num_parallel_calls_per_ds = max(1, int(self.num_parallel_calls) // n_datasets)
```

With 4 datasets and `num_parallel_*=32`, each dataset only gets 8 parallelism, reducing overall throughput.

### Experiment Results (post-refactoring)

| Run | Config | it/s | samples/s | Dataloader % | Notes |
|-----|--------|------|-----------|--------------|-------|
| baseline_postrefac | `batch_size=24`, `prefetch=64`, `num_parallel_*=32` | 3.1 | 74.4 | 82.5% | Baseline with minimal buffers |
| batch128 | `batch_size=128` | 1.16 | 148.5 | 74.1% | **2x samples/s** - bigger batches help |
| high_parallel | `batch_size=128`, `num_parallel_*=128/64`, `prefetch=128`, `episode_prefetch=8` | 1.0 | 128 | - | **Slower** - likely thread contention |
| autotune | `batch_size=256`, `num_parallel_calls=-1`, `prefetch=-1` (AUTOTUNE) | 0.55 | 140.8 | 76% | AUTOTUNE overhead + larger batch doesn't help |
| workers2 (aborted) | `batch_size=128`, `num_workers=2`, `num_parallel_*=16` | - | - | - | **OOM** - each worker creates separate TF pipeline (~6GB each) |

### Key Findings

1. **Batch size is crucial**: `batch_size=128` gives ~2x samples/s over `batch_size=24` (148 vs 74 samples/s)
2. **GPU is severely underutilized**: Only using ~2GB of 32GB VRAM at batch_size=128
3. **More parallelism ≠ faster**: Higher `num_parallel_*` settings cause thread contention
4. **AUTOTUNE is slower**: `-1` values add overhead vs manual tuning
5. **num_workers > 0 causes OOM**: Each PyTorch worker creates its own TF pipeline, consuming ~6GB RAM each

### Run Archive (post-refactoring)

| Run dir | Overrides | Result |
|---------|-----------|--------|
| `2026-01-18_21-27-40_laq_oxe_local` | baseline `batch_size=24` | 3.1 it/s, 82.5% dataloader |
| `2026-01-18_21-30-20_laq_oxe_local` | `batch_size=128` | **1.16 it/s**, 74.1% dataloader |
| `2026-01-18_21-35-09_laq_oxe_local` | `batch_size=128`, high parallelism | 1.0 it/s (worse) |
| `2026-01-18_21-41-10_laq_oxe_local` | `batch_size=256`, AUTOTUNE | 0.55 it/s (worse) |

### Bandwidth Analysis (Critical Finding)

**GCS Network is NOT the bottleneck!**

| Metric | Value |
|--------|-------|
| GCS raw download speed | ~70 MB/s |
| Required bandwidth @ 148 pairs/s | ~55 MB/s |
| **Headroom** | **+27%** |

**tf.data Pipeline Benchmarks:**

| Stage | Throughput | Notes |
|-------|-----------|-------|
| Image decode only | 218 pairs/s | Raw TFDS iteration |
| + Image resize (256x256) | 194 pairs/s | 11% overhead |
| + Interleave (simulated) | 191 pairs/s | Minimal overhead |
| **Actual training** | **148 pairs/s** | 23% gap unexplained |

The ~43 pairs/s gap (191 vs 148) must come from:
1. Frame pair construction (ring buffer/TensorArray operations)
2. Metadata extraction (action, state, language tensors)
3. Multi-dataset mixing overhead (sample_from_datasets)
4. TF-to-PyTorch conversion (dlpack/numpy copy)

### Additional Experiments

| Run | Config | it/s | samples/s | Notes |
|-----|--------|------|-----------|-------|
| shuffle_buffers | `batch_size=128`, `episode_prefetch=4`, `episode_shuffle=100`, `pair_shuffle=500` | 1.10 | 140.8 | **Worse** - shuffle fill time adds overhead |
| batch256 | `batch_size=256` | 0.66 | 169 | Slightly better samples/s but slower convergence |
| image128 | `batch_size=128`, `image_size=128` | 1.48 | 189 | 27% faster (not viable for production) |

### Run Archive (continued)

| Run dir | Overrides | Result |
|---------|-----------|--------|
| `2026-01-18_21-57-32_laq_oxe_local` | `batch_size=128`, shuffle buffers | 1.10 it/s |
| `2026-01-18_22-02-22_laq_oxe_local` | `batch_size=128`, `image_size=128` | 1.48 it/s |
| `2026-01-18_22-06-02_laq_oxe_local` | `batch_size=256` | 0.66 it/s, 169 samples/s |

### **Critical Discovery: Multi-Dataset Overhead**

**Single-dataset vs Multi-dataset Performance:**

| Config | it/s | samples/s | Improvement |
|--------|------|-----------|-------------|
| Multi-dataset (4 datasets), batch=128 | 1.16 | 148 | baseline |
| Single-dataset (language_table), batch=128 | **1.60** | **205** | **+38%** |
| Single-dataset (language_table), batch=256 | **1.04** | **266** | **+80%** |

**Root cause of multi-dataset slowdown:**
1. `MultiOXEFramePairDataset` divides parallelism by n_datasets (oxe.py lines 947-959)
2. `sample_from_datasets()` mixing has coordination overhead
3. Each dataset's TF pipeline competes for resources

**Recommendation:** For maximum throughput, use single large dataset or fix the parallelism division logic in `MultiOXEFramePairDataset`.

### Run Archive (single-dataset experiments)

| Run dir | Config | Result |
|---------|--------|--------|
| `2026-01-18_22-26-15_laq_oxe_local` | single lang_table 1k eps | 1.30 it/s (early exit) |
| `2026-01-18_22-27-14_laq_oxe_local` | single lang_table 10k eps, batch=128 | **1.60 it/s = 205 samples/s** |
| `2026-01-18_22-30-41_laq_oxe_local` | single lang_table 10k eps, batch=256 | **1.04 it/s = 266 samples/s** |

### GPU Utilization Results

With single-dataset (language_table 50k episodes) and batch_size=256:
- **GPU Utilization: 100%**
- **GPU Memory: 26GB / 32GB (81%)**
- **Throughput: 0.91 it/s = 233 samples/s**

We have successfully shifted from **I/O-bound to GPU-bound**!

### Next Steps

1. ~~Profile GCS latency~~ - **Done**: Network NOT bottleneck
2. ~~Test single-dataset mode~~ - **Done**: 38% faster without multi-dataset overhead
3. ~~Profile GPU utilization~~ - **Done**: 100% GPU util achieved!
4. **Fix MultiOXEFramePairDataset**: Either:
   - Remove parallelism division by n_datasets
   - Use sequential interleave instead of `sample_from_datasets()`
5. **Scale up**: Test if higher batch sizes (384, 512) can push even further
6. **Mixed precision**: Try `precision=bf16` for potentially 2x throughput
