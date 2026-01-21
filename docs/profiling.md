# Profiling LAQ Training

Guide for profiling LAQ training performance to identify bottlenecks.

## Quick Summary

**For most debugging:** Use SimpleProfiler (low overhead, easy)
**For deep GPU analysis:** Use PyTorchProfiler (high overhead, detailed)

## Profiler Types

### 1. SimpleProfiler (Recommended for general use)
- **Overhead:** ~5% (negligible)
- **Output:** Text file with time per function
- **Use case:** Quick check of where time is spent
- **Safe to leave on:** Yes

### 2. AdvancedProfiler
- **Overhead:** ~10-15%
- **Output:** Detailed text file with call counts
- **Use case:** More detailed function-level profiling
- **Safe to leave on:** For debugging runs only

### 3. PyTorchProfiler
- **Overhead:** ~20-50% (significant!)
- **Output:** Chrome trace (JSON) + text summary
- **Use case:** GPU bottleneck analysis, CUDA kernel debugging
- **Safe to leave on:** NO - only for targeted debugging

---

## Usage Examples

### Enable SimpleProfiler (Quick Check)

```bash
# Enable via CLI
python scripts/2_train_laq.py experiment=laq_debug \
    training.profiler.enabled=true \
    training.profiler.type=simple

# Output: ./profiles/profile.txt
```

**Example output:**
```
Action                              | Total Time (s)
-----------------------------------------------------------------
[LightningModule]LatentActionQuantization.forward | 2.456
[Callback]on_validation_epoch_end   | 1.234
[Optimizer]AdamW.step                | 0.543
```

### Enable PyTorchProfiler (Deep Dive)

```bash
# For GPU bottleneck analysis
python scripts/2_train_laq.py experiment=laq_debug \
    training.profiler.enabled=true \
    training.profiler.type=pytorch \
    training.epochs=2  # Keep short - high overhead!

# Output: ./profiles/profile.json
```

**View results:**
1. Open Chrome browser
2. Go to `chrome://tracing`
3. Load `./profiles/profile.json`
4. See timeline of CPU/GPU operations

### Customize Output Location

```bash
python scripts/2_train_laq.py experiment=laq_debug \
    training.profiler.enabled=true \
    training.profiler.dirpath=./debug_profiles \
    training.profiler.filename=laq_bottleneck
```

---

## Diagnosing Slow Training

### Symptom: Slow between epochs

**Likely causes:**
1. Visualization callbacks generating reconstructions
2. Codebook replacement on GPU→CPU sync
3. Checkpoint saving

**How to check:**
```bash
# Enable SimpleProfiler
python scripts/2_train_laq.py experiment=laq_debug \
    training.profiler.enabled=true \
    training.epochs=2

# Look for:
# - [Callback]on_validation_epoch_end  ← Visualization
# - [Callback]on_train_epoch_end       ← Visualization
```

**Quick fixes:**
```bash
# Disable visualization
python scripts/2_train_laq.py experiment=laq_debug \
    training.validation.visualize_train=false \
    training.validation.visualize_val=false

# Or reduce frequency
python scripts/2_train_laq.py experiment=laq_debug \
    training.validation.interval_epochs=10
```

### Symptom: Slow forward pass

**Likely causes:**
1. Model too large for batch size
2. Inefficient attention operations
3. CPU-GPU data transfer

**How to check:**
```bash
# Use PyTorchProfiler for GPU details
python scripts/2_train_laq.py experiment=laq_debug \
    training.profiler.enabled=true \
    training.profiler.type=pytorch \
    training.epochs=1 \
    data.loader.batch_size=4  # Reduce to test
```

**Look for in trace:**
- GPU idle time (underutilization)
- `aten::to` calls (data movement)
- Long-running kernels

### Symptom: Slow data loading

**Likely causes:**
1. Too few workers
2. Slow disk I/O
3. Complex transforms

**How to check:**
```bash
# Enable SimpleProfiler and vary workers
python scripts/2_train_laq.py experiment=laq_debug \
    training.profiler.enabled=true \
    data.loader.num_workers=0  # CPU-bound

python scripts/2_train_laq.py experiment=laq_debug \
    training.profiler.enabled=true \
    data.loader.num_workers=4  # Should be faster
```

---

## Configuration Reference

In `config/training/laq_optimizer.yaml`:

```yaml
profiler:
  enabled: false           # Enable profiling
  type: simple             # 'simple', 'advanced', 'pytorch'
  dirpath: ./profiles      # Output directory
  filename: profile        # Base filename
```

**CLI overrides:**
```bash
# Enable
training.profiler.enabled=true

# Change type
training.profiler.type=pytorch

# Change output
training.profiler.dirpath=./my_profiles
training.profiler.filename=my_run
```

---

## Interpreting Results

### SimpleProfiler Output

```
Action                              | Total Time (s)
-----------------------------------------------------------------
[LightningModule]forward            | 10.5     ← Model forward pass
[Callback]on_validation_epoch_end   | 5.2      ← Validation (reconstructions)
[Optimizer]step                     | 2.1      ← Optimizer update
```

**What to look for:**
- Callbacks taking >1s → Likely visualization
- Forward pass >50% of epoch time → Normal
- Optimizer step >20% → Check batch size

### PyTorchProfiler Output

Chrome trace shows:
- **Timeline:** When CPU/GPU are active
- **Gaps:** Idle time (underutilization)
- **Synchronization:** GPU→CPU transfers (blocking)

**Common patterns:**
```
GPU: [███forward███]__[sync]__[███forward███]
CPU: ____wait____[.cpu()][process]____wait____
      ↑ GPU-CPU sync blocks here!
```

---

## Permanent Profiling (Optional)

If you want SimpleProfiler always available:

Edit `config/training/laq_optimizer.yaml`:
```yaml
profiler:
  enabled: true      # Always on
  type: simple       # Low overhead
```

Then disable when not needed:
```bash
python scripts/2_train_laq.py training.profiler.enabled=false
```

---

## Best Practices

1. **Start with SimpleProfiler** - Quick, low overhead
2. **Disable for production** - No profiling in final runs
3. **Short epochs for PyTorchProfiler** - High overhead!
4. **Profile early** - Catch issues before long training runs
5. **Compare runs** - Before/after optimization changes

---

## Common Bottlenecks & Fixes

| Bottleneck | Symptom | Fix |
|------------|---------|-----|
| Visualization | Slow between epochs | Reduce frequency or disable |
| GPU→CPU sync | `.cpu()` in trace | Use `.cpu(non_blocking=True)` or batch transfers |
| Small batch size | GPU underutilized | Increase batch size |
| Few data workers | CPU-bound loading | Increase `num_workers` |
| Large model | OOM / slow forward | Reduce model size or batch size |

---

## Getting Help

If profiling reveals an unexpected bottleneck:

1. Save the profile output
2. Note your configuration (`--cfg job`)
3. Document the symptom vs expectation
4. Share profile + config for debugging
