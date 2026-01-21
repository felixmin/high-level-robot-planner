# Normal LAQ Training Guide

Quick reference for setting up and running normal (production) LAQ training.

## Quick Start

```bash
# Debug mode - small subset for quick iteration
python scripts/2_train_laq.py experiment=laq_debug

# Normal training - full dataset
python scripts/2_train_laq.py experiment=laq_normal

# With custom parameters
python scripts/2_train_laq.py experiment=laq_normal data.loader.batch_size=64 training.epochs=100
```

---

## Multi-Dataset Training

LAQ training uses a multi-dataset adapter system. Configure datasets via the `sources` list:

```yaml
# config/data/laq_multi_dataset.yaml
sources:
  - type: youtube
    root: /mnt/data/datasets/youtube_new
    filters:
      contains_hand_sam3: true  # Only YouTube scenes with hands

  - type: bridge
    root: /mnt/data/datasets/bridgev2/raw/bridge_data_v2
    filters:
      environment: toykitchen1  # Only toykitchen1 environment
```

**Supported dataset types:**
- `youtube`: YouTube videos with scenes.csv metadata
- `bridge`: BridgeV2 robot trajectories

---

## Holding Out Data for Validation (Distribution Shift)

### Ratio-Based Split (Default)
```yaml
split_mode: ratio
val_split: 0.1  # 10% for validation
```

### Metadata-Based Split
Hold out specific videos, datasets, or environments:

```yaml
split_mode: metadata
val_scene_filters:
  video_id: "JNBtHDVoNQc_stabilized"  # Hold out one video
```

**Examples:**
```yaml
# Hold out entire dataset type
val_scene_filters:
  dataset_type: "bridge"

# Leave-one-environment-out
val_scene_filters:
  environment: "toykitchen7"

# Hold out specific robot
val_scene_filters:
  robot: "minsky"
```

---

## Validation Buckets (Distribution Shift Analysis)

Create named subsets of validation data for separate evaluation:

```yaml
training:
  validation:
    buckets:
      youtube_only:
        filters:
          dataset_name: "youtube"
      bridge_only:
        filters:
          dataset_name: "bridge"
      unseen_environment:
        filters:
          environment: "folding_table"
```

Buckets are routed and cached by `ValidationStrategyCallback` (no bucket dataloaders
on the DataModule).

---

## Configuration Files to Know

### Experiment Configs
Located in `config/experiment/`:

| File | Use | Epochs | Batch Size |
|------|-----|--------|-----------|
| `laq_debug.yaml` | Quick iteration | 3 | 4 |
| `laq_normal.yaml` | Production training | 5000 | 32 |
| `laq_full.yaml` | Full training (LRZ) | 100 | 256 |

### Data Config
Located in `config/data/`:

| File | Description |
|------|-------------|
| `laq_multi_dataset.yaml` | Multi-source dataset with YouTube + Bridge |

---

## Key Parameters

### Frame Offsets
Multiple offsets allow training on different time scales:
- `offset=15`: Fast transitions (short gaps)
- `offset=30`: Normal transitions (medium gaps)
- `offset=60`: Slow transitions (long gaps)

```yaml
offsets: [30]  # Single offset (most common)
offsets: [15, 30, 60]  # Multi-scale training
```

### Max Samples
Limit dataset size for debugging:
```yaml
max_samples: 20  # Use only 20 frame pairs
max_samples: null  # Use all pairs (default)
```

### Sampling Strategy
Control how subsets are selected:
```yaml
sampling_strategy: "random"  # Diverse samples (default)
sampling_strategy: "sequential"  # First N samples
sampling_seed: 42  # Reproducible sampling
```

---

## Common Commands

### Debug mode (quick test)
```bash
python scripts/2_train_laq.py experiment=laq_debug
```

### Full dataset training
```bash
python scripts/2_train_laq.py experiment=laq_normal
```

### Multi-scale training (3 frame gaps)
```bash
python scripts/2_train_laq.py experiment=laq_normal data.dataset.local_files.pair_offsets_frames=[15,30,60]
```

### Higher learning rate
```bash
python scripts/2_train_laq.py experiment=laq_normal training.optimizer.lr=5e-4
```

### Different batch size
```bash
python scripts/2_train_laq.py experiment=laq_normal data.loader.batch_size=64
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

- [ ] Update dataset sources in `config/data/laq_multi_dataset.yaml`
- [ ] Verify you have enough disk space for your dataset
- [ ] Verify batch size fits in GPU memory:
  - RTX 5090: batch_size=32-64 with bf16 precision
  - RTX A100: batch_size=128+ with bf16 precision
- [ ] Set `data.loader.num_workers` to your CPU core count (not more)
- [ ] Test with `experiment=laq_debug` first to verify setup
- [ ] Run full training with `experiment=laq_normal`

---

## Troubleshooting

### "Out of memory" error
1. Reduce batch size: `data.loader.batch_size=16`
2. Enable gradient checkpointing (if supported)
3. Reduce `data.loader.num_workers`

### "Training is slow"
1. Check CPU bottleneck: Increase `data.loader.num_workers` to match CPU cores
2. Profile with: `training.profiler.enabled=true`
3. Check if GPU is underutilized: Maybe batch_size is too small

### "Results differ between runs"
1. Set fixed seed: `seed=42`
2. Local-files split: `data.split.seed=42` (and optionally `data.subset.seed=42`)
3. OXE streaming (TF): `data.adapter.tf.sampling.seed=42`

### "Want to resume from checkpoint"
```bash
python scripts/2_train_laq.py experiment=laq_normal \
    trainer.resume_from_checkpoint=./checkpoints/laq-epoch10.ckpt
```

---

## See Also

- `docs/profiling.md` - How to profile and debug performance
- `config/experiment/laq_normal.yaml` - Complete experiment config
- `config/data/laq_multi_dataset.yaml` - Multi-dataset data config
