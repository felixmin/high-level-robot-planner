# Normal LAQ Training Guide

Quick reference for setting up and running normal (production) LAQ training.

## Quick Start

For normal training on your full dataset:

```bash
# Scene-level (on-the-fly sampling, memory efficient - RECOMMENDED)
python scripts/2_train_laq.py experiment=laq_normal data=laq_scenes

# OR pair-level (pre-computed pairs, deterministic)
python scripts/2_train_laq.py experiment=laq_normal data=laq_pairs
```

## Understanding the Three Data Modes

### 1. **Scene-Level** (Recommended for Full Dataset)
- **What it is**: Samples frame pairs **on-the-fly** each epoch
- **Dataset size**: Number of scenes (e.g., 100 scenes)
- **Memory usage**: Low (loads 2 frames at a time)
- **Repeatability**: Different pairs each epoch (more diverse)
- **Use case**: Full dataset training, memory-constrained environments

```bash
python scripts/2_train_laq.py experiment=laq_normal data=laq_scenes
```

### 2. **Pair-Level** (Deterministic, Pre-computed)
- **What it is**: Pre-computes ALL valid frame pairs during setup
- **Dataset size**: Number of pairs (e.g., 10,000 pairs from 100 scenes)
- **Memory usage**: Medium (depends on dataset size)
- **Repeatability**: Same pairs every run (deterministic)
- **Use case**: Standard training, research reproducibility

```bash
python scripts/2_train_laq.py experiment=laq_normal data=laq_pairs
```

**With multi-scale training:**
```bash
python scripts/2_train_laq.py experiment=laq_normal data=laq_pairs data.offsets=[15,30,60]
```

### 3. **Debug** (Small Subset)
- **What it is**: Uses small random subset for quick iteration
- **Dataset size**: 10 samples (configurable)
- **Memory usage**: Very low
- **Use case**: Quick testing, code debugging

```bash
python scripts/2_train_laq.py experiment=laq_debug
```

---

## Configuration Files to Know

### Experiment Configs
Located in `config/experiment/`:

| File | Use | Data Mode | Epochs | Batch Size |
|------|-----|-----------|--------|-----------|
| `laq_debug.yaml` | Quick iteration | Scene-level | 5 | 8 |
| `laq_normal.yaml` | Production training | Pair-level | 50 | 32 |
| `laq_full.yaml` | Full training (LRZ) | WebDataset | 100 | 256 |

### Data Configs
Located in `config/data/`:

| File | pair_level | offsets | Use |
|------|-----------|---------|-----|
| `debug_frames.yaml` | false | - | Debug mode |
| `laq_scenes.yaml` | false | - | Scene-level training |
| `laq_pairs.yaml` | true | [30] | Pair-level training |

---

## Answering Your Questions

### "Why is `offsets` a list?"

Multiple offsets allow training on different time scales:
- `offset=15`: Fast transitions (short gaps)
- `offset=30`: Normal transitions (medium gaps)
- `offset=60`: Slow transitions (long gaps)

**In pair-level mode**, each offset creates pairs:
```
Scene with 100 frames + 3 offsets = ~198 pairs total
  Offset 15: ~86 pairs
  Offset 30: ~71 pairs
  Offset 60: ~41 pairs
```

**For single offset training** (most common):
```yaml
offsets: [30]  # Just one offset
```

### "For normal training: pair mode or not?"

**Scene-level (recommended for full data):**
- ✓ Memory efficient
- ✓ More sample diversity (new pairs each epoch)
- ✓ Works with any dataset size
- ✗ Slightly slower (on-the-fly sampling)

**Pair-level (good for deterministic training):**
- ✓ Deterministic, reproducible
- ✓ Faster training (pre-computed pairs)
- ✓ Good for multi-scale training
- ✗ Uses more memory
- ✗ Fixed pairs every run

**Recommendation**: Use `laq_scenes` for full dataset, `laq_pairs` if you want determinism.

### "What about shuffling? Shouldn't DataLoader handle it?"

Good catch! DataLoader shuffle handles two different things:

1. **DataLoader shuffle** (`shuffle=True`):
   - Randomizes order of samples within each epoch
   - Train dataloader: shuffled
   - Val dataloader: fixed order

2. **Our random sampling** (new):
   - Controls WHICH samples are selected from the full dataset
   - When you use `max_samples=10`, ensures diverse samples (not just first 10)
   - Happens once during setup

Both work together:
- Random sampling selects diverse samples
- DataLoader shuffle randomizes their presentation order

---

## Common Commands

### Full dataset with scene-level sampling
```bash
python scripts/2_train_laq.py experiment=laq_normal data=laq_scenes
```

### Full dataset with pair-level sampling
```bash
python scripts/2_train_laq.py experiment=laq_normal data=laq_pairs
```

### Multi-scale training (3 frame gaps)
```bash
python scripts/2_train_laq.py experiment=laq_normal data=laq_pairs data.offsets=[15,30,60]
```

### Higher learning rate
```bash
python scripts/2_train_laq.py experiment=laq_normal training.optimizer.lr=5e-4
```

### Different batch size
```bash
python scripts/2_train_laq.py experiment=laq_normal data.batch_size=64
```

### More epochs
```bash
python scripts/2_train_laq.py experiment=laq_normal training.epochs=100
```

### With WandB logging
```bash
python scripts/2_train_laq.py experiment=laq_normal logging.use_wandb=true
```

### Enable profiling (to debug slowness)
```bash
python scripts/2_train_laq.py experiment=laq_normal training.profiler.enabled=true training.profiler.type=simple
```

---

## Setup Checklist

Before running normal training:

- [ ] Update dataset folder in `config/data/laq_scenes.yaml` or `config/data/laq_pairs.yaml`
- [ ] Verify you have enough disk space for your dataset
- [ ] Choose scene-level or pair-level based on your needs
- [ ] Verify batch size fits in GPU memory:
  - RTX 5090: batch_size=32-64 with bf16 precision
  - RTX A100: batch_size=128+ with bf16 precision
- [ ] Set `num_workers` to your CPU core count (not more)
- [ ] Test with `experiment=laq_debug` first to verify setup
- [ ] Run full training with `experiment=laq_normal`

---

## Troubleshooting

### "Out of memory" error
1. Reduce batch size: `data.batch_size=16`
2. Switch to scene-level: `data=laq_scenes` (lower memory)
3. Enable gradient checkpointing (if supported)

### "Training is slow"
1. Check CPU bottleneck: Increase `num_workers` to match CPU cores
2. Profile with: `training.profiler.enabled=true`
3. Check if GPU is underutilized: Maybe batch_size is too small

### "Results differ between runs"
1. Set fixed seed: `seed=42`
2. Use pair-level mode for deterministic pairs: `data=laq_pairs`
3. Use same `sampling_seed`: `data.sampling_seed=42`

### "Want to resume from checkpoint"
```bash
python scripts/2_train_laq.py experiment=laq_normal \
    trainer.resume_from_checkpoint=./checkpoints/laq-epoch10.ckpt
```

---

## See Also

- `docs/data_modes.md` - Detailed explanation of data loading modes
- `docs/profiling.md` - How to profile and debug performance
- `config/experiment/laq_normal.yaml` - Complete experiment config
- `config/data/laq_scenes.yaml` - Scene-level data config
- `config/data/laq_pairs.yaml` - Pair-level data config
