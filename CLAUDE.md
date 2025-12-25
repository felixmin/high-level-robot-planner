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
  - /data@data: laq_multi_dataset
  - /training@training: laq_optimizer
  - /cluster@cluster: local_dev
```

Package paths (`@model:`, `@data:`, etc.) explicitly place configs under their keys. Component configs are clean without package directives. Override from CLI:
```bash
python scripts/2_train_laq.py experiment=laq_full data.batch_size=512 training.optimizer.lr=5e-5
```

## Infrastructure

**LRZ Cluster (H100 multi-node training):**
- See `docs/job_submission.md` for job submission and sweeps
- See `docs/lrz_workflow.md` for cluster setup and monitoring

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

# Stage 1: LAQ on LRZ cluster
python scripts/submit_job.py experiment=laq_full

# Stage 2: Foundation on LRZ (multi-node)
python scripts/submit_job.py --script 4_train_foundation experiment=vla_7b
```

### Submitting Sweeps (Multiple Jobs)
```bash
# Define sweep in experiment config with sweep.params
# See config/experiment/laq_lr_sweep.yaml for example

# Submit sweep (one job per parameter combination)
python scripts/submit_job.py experiment=laq_lr_sweep

# Dry run to preview jobs
python scripts/submit_job.py --dry-run experiment=laq_lr_sweep
```

See `docs/job_submission.md` for full documentation.

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

### Data Loading (LAQ Training)

LAQ training supports two data loading modes:

#### 1. Local Multi-Dataset Loading (Bridge, YouTube)

Uses multi-dataset loading with adapters. Configure datasets via the `sources` list:

```bash
# Debug mode - small subset for quick iteration
python scripts/2_train_laq.py experiment=laq_debug

# Normal training - full dataset
python scripts/2_train_laq.py experiment=laq_normal
```

Key parameters:
- `max_pairs`: Limit frame pairs for debugging
- `offsets`: Frame offsets for pair generation (e.g., `offsets=[15, 30, 60]`)
- `val_split`: Train/val split ratio (default 0.1)

Data loading pre-computes all valid frame pairs across datasets for deterministic training.

#### 2. OXE Streaming (Open X-Embodiment)

Stream data from Google Cloud Storage using TensorFlow Datasets:

```bash
# Language Table from OXE
python scripts/2_train_laq.py experiment=laq_oxe

# Bridge from OXE
python scripts/2_train_laq.py data=laq_oxe_bridge training.epochs=100
```

**Available OXE Datasets**:
- `language_table` - 442k episodes, 2D tabletop manipulation
- `language_table_blocktorelative_oracle_sim` - 200k episodes, oracle agent
- `bridge` - 25,460 train episodes, WidowX kitchen manipulation

**OXE Config Example** (`config/data/laq_oxe_bridge.yaml`):
```yaml
dataset_name: bridge
train_split: "train[:90%]"
val_split: "train[90%:]"
offset: 5  # Frame offset (steps)
image_size: 256
batch_size: 32
return_metadata: true  # Required for validation strategies
```

**Key Differences**:
- **Streaming**: OXE data streams from GCS, no local storage needed
- **TFDS splits**: Use TensorFlow Datasets split syntax (e.g., `train[:1000]`, `train[90%:]`)
- **Metadata**: OXE provides `action`, `initial_state`, and `instruction` automatically

### Multi-Dataset Training

Train on multiple datasets using adapters:
```yaml
sources:
  - type: youtube
    root: /mnt/data/datasets/youtube_new
    filters:
      contains_hand_sam3: true

  - type: bridge
    root: /mnt/data/datasets/bridgev2/raw/bridge_data_v2
    filters:
      environment: toykitchen1
```

### Metadata-Based Train/Val Splits

Hold out specific data for validation (distribution shift analysis):
```yaml
split_mode: metadata
val_scene_filters:
  video_id: "holdout_video"  # Hold out specific video
  # OR: dataset_type: "bridge"  # Hold out entire dataset
  # OR: environment: "toykitchen7"  # Leave-one-out
```

### Validation Buckets

Create named validation subsets for analysis:
```yaml
val_buckets:
  youtube_only:
    dataset_type: "youtube"
  unseen_robot:
    robot: "minsky"
```

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
  contains_hand_sam3: true
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

### Validation Configuration

The validation system uses **bucket-strategy binding** for flexible, multi-dataset validation:

```yaml
validation:
  # Validate 100 times per epoch (important for large datasets)
  check_interval: 0.01

  # Fixed samples: diverse across datasets, tracked every validation
  num_fixed_samples: 8
  # Random samples: different each time for diversity
  num_random_samples: 8
  max_cached_samples: 1024  # Per-bucket cache limit

  # Define validation buckets (data subsets with filters)
  buckets:
    youtube_iid:
      filters: {dataset_type: "youtube"}
      max_samples: 100
    bridge_iid:
      filters: {dataset_type: "bridge", environment: ["!=", "toykitchen7"]}
      max_samples: 100
    bridge_holdout:
      filters: {dataset_type: "bridge", environment: "toykitchen7"}
      max_samples: 100
      is_holdout: true  # Distribution shift data
    language_table:
      filters: {dataset_type: "oxe", dataset_name: "language_table"}
      max_samples: 200

  # Bind strategies to buckets
  strategy_bucket_bindings:
    basic_visualization:
      buckets: all  # Run on all buckets
    latent_transfer:
      buckets: [bridge_iid, bridge_holdout]
      compare_buckets: true  # Run separately per bucket with metric suffix
    action_token_scatter:
      buckets: [language_table]  # Only bucket with action metadata
    clustering:
      buckets: all

  strategies:
    basic_visualization:
      enabled: true
      visualize_train: true
      visualize_val: true

    # Latent transfer: test action transfer between scenes
    latent_transfer:
      enabled: true
      every_n_validations: 10
      num_pairs: 256

    # Action visualization (requires action metadata)
    action_token_scatter:
      enabled: true
      every_n_validations: 3
      num_samples: 1000

    # Clustering: analyze latent action distribution
    clustering:
      enabled: true
      every_n_validations: 20
      num_clusters: 16
```

**Key Features**:
- **Bucket-aware routing**: Samples routed to matching buckets based on filters
- **Strategy-bucket binding**: Strategies declare which buckets they operate on
- **Compare mode**: Run separately on each bucket for distribution shift analysis (e.g., `val/latent_transfer_mse_bridge_iid` vs `val/latent_transfer_mse_bridge_holdout`)
- **Automatic applicability checks**: Strategies skip execution if insufficient data
- **Metadata requirements**: Strategies declare required metadata (e.g., `action`, `initial_state`)

**Validation Strategy Compatibility**:

| Strategy | Requires | Compatible Datasets |
|----------|----------|---------------------|
| `basic_visualization` | frames | All |
| `latent_transfer` | frames | All |
| `clustering` | codes | All |
| `codebook_histogram` | codes | All |
| `action_token_scatter` | codes + `action` metadata | language_table, bridge |
| `state_sequence_scatter` | codes + `initial_state` metadata | language_table, bridge |

## Implementation Notes

- **LAQ Training** (Stage 1): Implemented in `scripts/2_train_laq.py`. Uses PyTorch Lightning for standard supervised learning with DDP.
- **Foundation Training** (Stage 2): Implement in `scripts/4_train_foundation.py`. Use Lightning Fabric for FSDP multi-node training with fine training loop control.
- **Data Loading**:
  - **Local datasets** (Bridge, YouTube): Uses `LAQDataModule` with multi-source adapters (`MultiSourcePairDataset`). Pre-computes all valid frame pairs for deterministic training.
  - **OXE datasets** (language_table, bridge): Uses `OXEDataModule` with `OXEFramePairDataset`. Streams from GCS using TensorFlow Datasets with `tf.data` pipelines.
  - **Auto-detection**: Training script automatically selects `OXEDataModule` if `dataset_name` field present in config.
  - Stage 2/3: Implement WebDataset-based loaders in `packages/common/` for streaming TAR shards.
- **Validation**: Uses `ValidationStrategyCallback` with bucket-strategy binding:
  - **Architecture**: Bucket-aware routing where samples are filtered to named buckets based on metadata
  - **Strategies**: Each strategy declares metadata requirements (e.g., `action`, `initial_state`) and bucket bindings
  - **Applicability checks**: Strategies automatically skip execution if insufficient data via `can_run()` method
  - **Compare mode**: Strategies can run separately on each bucket for distribution shift analysis
  - **Implementations**: Basic visualization, latent transfer, clustering, action/state scatter plots
- **OXE Adapters**: Located in `packages/common/adapters/oxe.py`:
  - Handles dict-based actions (Bridge) vs flat arrays (language_table)
  - Handles string instructions (Bridge) vs encoded tensors (language_table)
  - Configurable via `OXEDatasetConfig` registry
- **Logging**: Use `packages/common/logging.py` for consistent logging across stages.
- **Checkpointing**: Lightning handles checkpoint saving; configure paths via Hydra config.

## References

- **Hydra Documentation**: Configuration system for composable configs
- **PyTorch Lightning**: Framework for Stages 1 and 3
- **Lightning Fabric**: Framework for Stage 2 multi-node training
- **WebDataset**: Streaming loader for TAR-based datasets
- **LRZ Cluster**: See `docs/lrz_workflow.md` for cluster-specific guidance
