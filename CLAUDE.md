# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Our project is a three-stage robot learning system that learns policies from videos without action labels.

**Three Training Stages:**
1. **Stage 1 (LAQ)**: VQ-VAE compressing frame-to-frame transitions into discrete latent codes
2. **Stage 2 (Foundation)**: Vision-Language model predicting latent actions from images + text
3. **Stage 3 (Finetuning)**: Adapting the foundation model to output continuous robot commands

**Infrastructure:**
- **Local Development:** RTX 5090 (24GB VRAM) for single-GPU training and debugging
- **Production:** LRZ AI cluster (H100 GPUs, GPFS storage, Slurm scheduler) for multi-node training

## Development Commands

### Quick Start
```bash
# Create conda environment
conda env create -f environment.yml

# Activate and install
conda activate hlrp
# install pytorch 2.9.1

# Verify setup
python scripts/0_setup_environment.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Specific test file
pytest tests/test_hydra_configs.py -v

# With coverage
pytest --cov=packages --cov-report=html tests/
```

### Code Quality
```bash
# Format code
black packages/ scripts/ tests/

# Lint
ruff check packages/ scripts/ tests/
```

## Project Architecture

### Repository Structure
- **packages/**: Installable Python packages with shared code
  - `common/` - Shared utilities, logging, data interfaces
  - `laq/` - Stage 1: Latent action quantization (VQ-VAE)
  - `foundation/` - Stage 2: Vision-Language-Action model
  - `low_level/` - Stage 3: Action decoding
- **config/**: Hydra YAML configurations (modular, composable)
  - `experiment/` - Complete experiment setups (laq_debug, laq_full, vla_7b)
  - `model/`, `data/`, `training/`, `cluster/` - Config components
- **scripts/**: Training entry points (numbered 0-5 for each stage)
- **slurm/**: LRZ job submission templates
- **containers/**: Enroot/Docker definitions for LRZ

### Key Architectural Decisions

**Modular Monorepo:** Single repository with installable packages for tight coupling between stages (LAQ vocabulary changes cascade to Foundation and Low-Level). Enables atomic commits and simplified dependency management.

**Hybrid Training Framework:**
- **Stage 1 & 3 (LAQ, Finetuning):** PyTorch Lightning for standard supervised learning with automatic distributed training (DDP) and checkpointing
- **Stage 2 (Foundation):** Lightning Fabric for multi-node FSDP training with fine control over training loops (raw loop control while keeping ecosystem consistent)

**Data Pipeline:** WebDataset with TAR shards for offline preprocessing. Critical for LRZ's GPFS filesystem which is optimized for large sequential reads, not millions of small files. Pattern: Raw Videos → Preprocessing → Sharded TARs → WebDataset Loader → Training.

### Configuration System

Uses **Hydra** (1.3+) for composable, modular configuration. Experiments compose components with explicit package paths:

```yaml
# config/experiment/laq_debug.yaml
defaults:
  - /model@model: laq
  - /data@data: debug_frames
  - /training@training: laq_optimizer
  - /cluster@cluster: local_dev
```

Package paths (`@model:`, `@data:`, etc.) explicitly place configs under their keys. Component configs are clean without package directives. Override from CLI:
```bash
python scripts/2_train_laq.py experiment=laq_full data.batch_size=512 training.optimizer.lr=5e-5
```

## Infrastructure

**LRZ Cluster (H100 multi-node training):**
- See `docs/lrz_workflow.md` for setup, job submission, and monitoring

**Local RTX 5090 (single-GPU development):**
- 24GB VRAM: Use batch size 8-16, enable `mixed_precision=bf16` via Hydra config
- Stage 1 (LAQ) and Stage 3 (Finetuning) fully trainable on RTX 5090
- Stage 2 (Foundation) requires gradient checkpointing for larger models

## Testing

Add unit and integration tests as features are implemented. Current tests focus on configuration validation. Run tests with:
```bash
pytest tests/
pytest tests/test_hydra_configs.py -v
pytest --cov=packages --cov-report=html tests/
```

## Training Script Pattern

Each training script follows this template:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # TODO: Implement training
```

Scripts use Hydra decorators to automatically load and compose configurations. CLI arguments become Hydra overrides (e.g., `experiment=laq_debug` loads `config/experiment/laq_debug.yaml`).

## Dependencies

Python 3.12 with PyTorch 2.9.1. Install via:
```bash
conda env create -f environment.yml
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

Key dependencies: pytorch-lightning, transformers, webdataset, hydra-core, wandb, accelerate, timm. See `environment.yml` for complete list.

## Common Workflows

### Running a Single Training Stage
```bash
# Stage 1: LAQ on local machine
python scripts/2_train_laq.py experiment=laq_debug

# Stage 1: LAQ on LRZ (full)
sbatch slurm/train.sbatch scripts/2_train_laq.py experiment=laq_full

# Stage 2: Foundation (multi-node on LRZ)
sbatch slurm/train.sbatch scripts/4_train_foundation.py experiment=vla_7b
```

### Modifying Configuration
Edit YAML files in `config/` hierarchy or override via CLI:
```bash
python scripts/2_train_laq.py experiment=laq_debug data.batch_size=16 training.epochs=10
```

### Debugging Training Scripts
Before running full training:
1. Test config loads: `pytest tests/test_hydra_configs.py -v`
2. Verify setup: `python scripts/0_setup_environment.py`
3. Check configuration: `python scripts/2_train_laq.py experiment=laq_debug --help`

### Data Loading Modes (LAQ Training)

**Scene-Level** (`data=laq_scenes`) - Recommended for full dataset:
- Samples frame pairs on-the-fly each epoch
- Memory efficient, diverse sampling
```bash
python scripts/2_train_laq.py experiment=laq_normal data=laq_scenes
```

**Pair-Level** (`data=laq_pairs`) - Deterministic training:
- Pre-computes all valid frame pairs upfront
- Supports multiple offsets: `offsets=[15, 30, 60]`
```bash
python scripts/2_train_laq.py experiment=laq_normal data=laq_pairs
```

**Debug Mode** (`data=debug_frames`) - Quick iteration:
- Small random subset (`max_samples=10`)
- Fast for testing code changes
```bash
python scripts/2_train_laq.py experiment=laq_debug
```

**See:** `docs/normal_training_guide.md` for detailed examples

### Data Filtering

Filter scenes by metadata (YAML-compatible):
```yaml
filters:
  # Comparison operators (use lists for YAML)
  max_trans: [">", 10.0]
  label: ["!=", "static"]

  # Multiple allowed values
  task_category: ["pnp_push_sweep", "stack_blocks"]

  # Boolean and equality
  has_hands: true
  environment: toykitchen1
```

### Performance Profiling

Enable profiling to debug slow training:
```bash
# Low overhead (~5%), always safe to use
python scripts/2_train_laq.py experiment=laq_debug \
    training.profiler.enabled=true \
    training.profiler.type=simple

# High overhead (~20-50%), detailed GPU analysis
python scripts/2_train_laq.py experiment=laq_debug \
    training.profiler.enabled=true \
    training.profiler.type=pytorch \
    training.epochs=2
```

**See:** `docs/profiling.md` for detailed profiling guide

## Implementation Notes

- **LAQ Training** (Stage 1): Implement in `scripts/2_train_laq.py`. Use PyTorch Lightning for standard supervised learning with DDP.
- **Foundation Training** (Stage 2): Implement in `scripts/4_train_foundation.py`. Use Lightning Fabric for FSDP multi-node training with fine training loop control.
- **Data Loading**:
  - Stage 1 (LAQ): Supports both scene-level and pair-level modes via `LAQDataModule`. Use pair-level for overfitting and debugging.
  - Stage 2/3: Implement WebDataset-based loaders in `packages/common/` for streaming TAR shards.
- **Logging**: Use `packages/common/logging.py` for consistent logging across stages.
- **Checkpointing**: Lightning handles checkpoint saving; configure paths via Hydra config.

## References

- **Hydra Documentation**: Configuration system for composable configs
- **PyTorch Lightning**: Framework for Stages 1 and 3
- **Lightning Fabric**: Framework for Stage 2 multi-node training
- **WebDataset**: Streaming loader for TAR-based datasets
- **LRZ Cluster**: See `docs/lrz_workflow.md` for cluster-specific guidance
