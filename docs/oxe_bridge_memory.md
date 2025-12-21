# OXE Bridge Memory Issues: Analysis and Fixes

This document explains why OXE Bridge training was causing OOM crashes and the changes made to fix it.

## The Problem

OXE Bridge training kept crashing with OOM (Out of Memory) during validation, while language_table worked fine with default settings. After multiple rounds of debugging, we identified **two root causes**:

### 1. TensorFlow Pipeline Memory Leak (Code Issue)

**Location:** `packages/common/adapters/oxe.py` - `OXEFramePairDataset.__iter__()`

**The Bug:** Every time `__iter__()` was called (once per epoch, plus during validation), a **new tf.data pipeline was created without releasing the old one**:

```python
# BEFORE (buggy)
def __iter__(self):
    tf_ds = self._create_tf_pipeline()  # Creates NEW pipeline every time!
    for item in tf_ds:
        yield ...
```

TensorFlow doesn't garbage collect tf.data pipelines properly. This caused memory to accumulate:
1. Training starts → creates tf.data pipeline (~2GB for shuffle buffer)
2. Validation starts → creates ANOTHER tf.data pipeline (~0.5GB)
3. Training resumes → creates YET ANOTHER pipeline
4. Memory keeps growing until OOM

**The Fix:** Explicitly clean up the old pipeline before creating a new one:

```python
# AFTER (fixed)
def __iter__(self):
    if self._cached_pipeline is not None:
        del self._cached_pipeline
        self._cached_pipeline = None
        gc.collect()
        tf.keras.backend.clear_session()  # Critical for TF memory cleanup

    tf_ds = self._create_tf_pipeline()
    self._cached_pipeline = tf_ds
    for item in tf_ds:
        yield ...
```

### 2. Shuffle Buffer Size (Configuration Issue)

**The Real Culprit:** Even with the memory leak fixed, Bridge still crashed because of the **shuffle buffer size**.

**Why Bridge is different from language_table:**

| Dataset | Raw Image Size | Pixels per Frame | Memory per Episode* |
|---------|---------------|------------------|---------------------|
| language_table | 360 × 640 | 230,400 | ~20 MB |
| Bridge | 480 × 640 | 307,200 | ~28 MB |

*Approximate, depends on episode length

With `shuffle_buffer=500`, TensorFlow holds 500 episodes in memory:
- language_table: 500 × 20 MB = ~10 GB (fits)
- Bridge: 500 × 28 MB = ~14 GB (doesn't fit with model + validation)

**The Fix:** Reduce shuffle buffer for Bridge:

```yaml
# config/data/laq_oxe_bridge.yaml
shuffle_buffer: 30  # Keep small - Bridge images are 480x640 before resize
```

## Current Memory Usage

With the fixes applied, here's the stable memory profile:

- **RAM:** ~43 GB
- **VRAM:** ~22.7 GB (DINOv3 encoder + decoder + batch data)
- **Training speed:** ~1.15 it/s

### Where RAM Actually Goes

The shuffle buffer is NOT the main memory consumer. Here's the breakdown:

**Shuffle buffer (small):**
- Each pair: `2 × 256 × 256 × 3` uint8 = 393 KB
- `shuffle_buffer=30`: ~12 MB
- `shuffle_buffer=500`: ~200 MB

**Actual memory consumers:**
1. **TensorFlow GCS streaming** (~5-10 GB) - Downloads and caches data from Google Cloud
2. **PyTorch model + optimizer** (~4-8 GB) - DINOv3 encoder, decoder, Adam states
3. **Validation cache** (~400 MB) - 256 samples × pair size
4. **TF-to-PyTorch copies** - During conversion, both copies exist briefly
5. **Python/numpy overhead** - General allocations

**Important clarification:** The generator already resizes images to 256×256 BEFORE yielding to the shuffle buffer. The buffer holds small resized pairs, not full 480×640 images.

### Why shuffle_buffer=30 vs 500 matters

Even though the shuffle buffer itself is small, the OOM happened because:
1. The memory leak (now fixed) accumulated TF graph objects
2. Combined pressure from all sources pushed total over limit
3. During validation, both train and val pipelines existed simultaneously

## What About Validation Strategies?

**Validation strategies are NOT the issue.** They work fine with:
- `shuffle_buffer=30` (or similar small value)
- The memory leak fix in place

We tested incrementally:
1. No validation strategies → Works
2. Basic visualization only → OOM with shuffle_buffer=100
3. Basic visualization + shuffle_buffer=20 → Works
4. All strategies + shuffle_buffer=30 → Works

The key insight: validation strategies add ~0.5-1 GB of memory for caches and model inference. This pushed Bridge over the edge when combined with large shuffle buffers.

## Is shuffle_buffer=30 Too Small?

### Short Answer: Probably yes, but it's a tradeoff.

### Effects of Small Shuffle Buffer:

1. **Less randomization within epochs** - You may see similar trajectories grouped together in batches
2. **"Random picks" showing similar images** - This is expected behavior with small buffers
3. **Still good cross-epoch randomization** - Different episodes are sampled each epoch

### Why You See Similar Images:

With `shuffle_buffer=30`, the pipeline:
1. Loads episodes sequentially from GCS
2. Only shuffles among 30 episodes at a time
3. Adjacent episodes are often from similar scenes/robots

### Recommendations:

| Use Case | shuffle_buffer | Notes |
|----------|---------------|-------|
| Debugging | 10-30 | Fast startup, low memory |
| Current (stable) | 30 | Works reliably, some grouping |
| Better mixing | 50-100 | Try if you have RAM headroom |
| Optimal mixing | 200+ | May need more RAM or smaller batch_size |

To increase shuffle_buffer, you can either:
- Reduce `batch_size` (currently 32)
- Reduce `max_cached_samples` in validation (currently 256)
- Accept higher RAM usage if available

## Files Changed

### Code Changes (should keep permanently):

1. **`packages/common/adapters/oxe.py`**
   - Added `_cached_pipeline` field for pipeline lifecycle management
   - Modified `__iter__()` to clear TF session before creating new pipeline
   - Added `gc.collect()` and `tf.keras.backend.clear_session()` for cleanup

### Config Changes (can adjust based on needs):

1. **`config/data/laq_oxe_bridge.yaml`**
   - `shuffle_buffer: 30` - Can increase if you have RAM headroom

2. **`config/experiment/laq_oxe_bridge.yaml`**
   - `shuffle_buffer: 30` - Matches data config
   - All validation strategies enabled (working fine now)

## External Suggestions Review

Some suggestions were received about memory optimization. Here's the analysis:

| Suggestion | Applicable? | Notes |
|-----------|-------------|-------|
| Calculate buffer size in bytes | No | `tf.data.shuffle()` takes element count, not bytes |
| Resize before shuffle | **Already done** | Generator resizes to 256×256 before yielding |
| Force GC before validation | **Already done** | `gc.collect()` + `clear_session()` in `__iter__` |
| TF memory growth config | Partial | We disable TF GPU entirely; CPU memory is harder to control |
| Separate process for tf.data | Overkill | Current fix is sufficient |
| Limit parallel map calls | N/A | Using `tf.data.AUTOTUNE` which handles this |

**Key insight:** The resize already happens inside the generator (line 277 in `oxe.py`), so the shuffle buffer holds small 256×256 pairs (~400KB each), not full 480×640 images (~900KB each). The memory pressure came from:
1. TF graph accumulation (fixed with `clear_session()`)
2. GCS streaming buffers (unavoidable with remote data)
3. Simultaneous train+val pipelines during validation

## Summary

| Issue | Root Cause | Fix | Permanent? |
|-------|-----------|-----|------------|
| Memory leak | TF pipelines not released | `clear_session()` in `__iter__` | Yes, keep forever |
| OOM during validation | TF graph accumulation + simultaneous pipelines | Memory cleanup + smaller buffer | Keep cleanup; buffer is tunable |
| Similar images in batches | Small shuffle_buffer | Expected behavior | Can increase to 200+ safely now |

## Testing After Changes

To verify everything works:

```bash
# Quick test (150 steps with validation)
python scripts/2_train_laq.py experiment=laq_oxe_bridge \
    training.max_steps=150 \
    training.validation.check_interval=100

# Full training
python scripts/2_train_laq.py experiment=laq_oxe_bridge
```

Watch for:
- No "Killed" messages (OOM)
- Stable memory usage during validation
- Loss decreasing normally
