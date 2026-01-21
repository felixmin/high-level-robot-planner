# Validation System Refactor Plan

This document outlines issues found in the validation system and proposed solutions.

## Issues Identified

### 1. Fixed Count Split Targets Scenes Not Samples (CRITICAL)

**Problem:** `val_counts_per_dataset` is interpreted as scene count, but users expect sample (frame pair) count.

Example with `val_counts_per_dataset: {'youtube': 1000, 'bridge': 1000}`:
- YouTube: 2 scenes selected → 16,345 samples (YouTube scenes are very long)
- Bridge: 29 scenes selected → 1,024 samples

**Solution:** Change to target sample counts instead of scene counts. Accumulate scenes until reaching the target sample count.

### 2. Imbalanced Validation Caching

**Problem:** With `limit_val_batches=0.1`, we only see 10% of mixed val loader. If that 10% is 95% bridge, YouTube gets underrepresented in cache even with stratified sampling.

**Current flow:**
```
Mixed Val Loader → limit_val_batches → Cache (stratified but still imbalanced)
```

**Solution:** Use validation buckets for routing/caching:
```yaml
# Config
training:
  validation:
    buckets:
      youtube: {filters: {dataset_name: "youtube"}}
      bridge: {filters: {dataset_name: "bridge"}}
```
```
Main Val Loader → loss computation (full or limited)
Bucket Caches → targeted visualization (guaranteed samples from each bucket)
```

### 3. LatentTransferStrategy API Mismatch (CRITICAL)

**Problem:** The strategy calls:
```python
source_latents = pl_module.model.encode(source_frames)  # Returns tuple!
```

But `encode()` returns `(first_tokens, last_tokens)` tuple, not a latent tensor.

**Solution:** Use the proper encoding path:
```python
# Option A: Use full forward with return_only_codebook_ids
indices = pl_module.model(frames, return_only_codebook_ids=True)
latents = pl_module.model.vq.codebooks[indices]

# Option B: Add a get_latents() helper method to the model
```

### 4. Validation Strategies Don't Run in Debug Mode

**Problem:** With 3 validations and `every_n_validations=10/20`, heavy strategies never run.

**Solution:**
- In debug mode, run heavy strategies at least once (on last validation)
- Or add `run_on_final: true` option to strategies

### 5. Train Visualization May Miss Buckets

**Problem:** Currently samples one batch from train dataloader which may not contain all bucket types.

**Solution:** Use per-bucket train dataloaders for visualization, similar to validation.

## Proposed Architecture

### Dataloaders
```
Training:
  - Main Mixed Loader → actual training
  - Per-Bucket Loaders → visualization only (small batches)

Validation:
  - Main Mixed Loader → loss computation
  - Per-Bucket Loaders → visualization & targeted metrics
```

### Config
```yaml
data:
  # Per-bucket sample counts for balanced validation
  val_samples_per_dataset:
    youtube: 1000
    bridge: 1000

  # Create per-bucket dataloaders
  create_bucket_loaders: true

validation:
  strategies:
    basic:
      enabled: true
      use_bucket_loaders: true  # Use per-bucket loaders for viz
      samples_per_bucket: 8
```

### Implementation Steps

1. **Fix `_split_scenes_by_fixed_count`** to target sample counts
   - Accumulate scenes until reaching sample target
   - Rename parameter to `val_samples_per_dataset` for clarity

2. **Add per-bucket dataloaders** to LAQDataModule
   - `train_bucket_dataloader(bucket_name)`
   - `val_bucket_dataloader(bucket_name)`
   - These are lightweight iterators for viz only

3. **Fix LatentTransferStrategy**
   - Use `model(frames, return_only_codebook_ids=True)` then lookup codes
   - Or add proper `get_latents()` method to model

4. **Update BasicVisualizationStrategy**
   - Use bucket loaders instead of filtering from mixed cache
   - Guarantees samples from each bucket

5. **Add run_on_final option** to heavy strategies
   - Ensures clustering/transfer analysis runs at least once even in debug

## Migration

Old config:
```yaml
split_mode: fixed_count
val_counts_per_dataset:  # scenes!
  youtube: 50
  bridge: 200
```

New config:
```yaml
split_mode: fixed_samples
val_samples_per_dataset:  # actual samples!
  youtube: 1000
  bridge: 1000
```
