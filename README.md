# LAPA: Latent Action Pretraining from Videos

A three-stage robot learning system that learns policies from videos without action labels.

## Overview

LAPA consists of three training stages:

1. **Stage 1 (LAQ)**: VQ-VAE that compresses frame-to-frame transitions into discrete latent codes
2. **Stage 2 (Foundation)**: 7B Vision-Language model that predicts latent actions from image + text
3. **Stage 3 (Finetuning)**: Adapts the foundation model to output continuous robot commands

## Repository Structure

```
lapa-project/
├── packages/              # Installable Python packages
│   ├── common/           # Shared utilities
│   ├── laq/              # Stage 1: Latent action quantization
│   ├── foundation/       # Stage 2: Vision-Language-Action model
│   └── low_level/        # Stage 3: Action decoding
├── config/               # Hydra configurations
│   ├── experiment/       # Complete experiment configs
│   ├── model/           # Model architectures
│   ├── data/            # Dataset configurations
│   ├── training/        # Training hyperparameters
│   └── cluster/         # Cluster-specific settings
├── scripts/             # Training entry points
├── slurm/               # LRZ job submission templates
├── containers/          # Docker/Enroot definitions
└── tests/               # Unit and integration tests
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd lapa-project

# Create conda environment
# For macOS (Apple Silicon):
conda env create -f environment.yml

# For Linux with CUDA (LRZ cluster):
conda env create -f environment_cuda.yml

# Activate environment
conda activate lapa

# Install LAPA packages in editable mode
pip install -e .

# Verify setup
python scripts/0_setup_environment.py
```

### 2. Run LAQ Training (Debug Mode)

```bash
# Local training on small dataset
python scripts/2_train_laq.py experiment=laq_debug
```

### 3. Run on LRZ Cluster

```bash
# Submit LAQ training job
sbatch slurm/train.sbatch scripts/2_train_laq.py experiment=laq_full

# Submit foundation training (multi-node)
sbatch slurm/train.sbatch scripts/4_train_foundation.py experiment=vla_7b
```

## Configuration Management

LAPA uses [Hydra](https://hydra.cc) for configuration management. You can:

### Compose Configurations

```yaml
# config/experiment/my_experiment.yaml
defaults:
  - override /model: laq
  - override /data: openx_webdataset
  - override /training: laq_optimizer
  - override /cluster: lrz_h100
```

### Override from Command Line

```bash
python scripts/2_train_laq.py \
  experiment=laq_full \
  data.batch_size=512 \
  training.optimizer.lr=5e-5 \
  cluster.compute.num_nodes=2
```

## Training Stages

### Stage 1: LAQ Training

```bash
# 1. Preprocess videos to WebDataset
python scripts/1_videos_to_webdataset.py

# 2. Train LAQ
python scripts/2_train_laq.py experiment=laq_full

# 3. Generate latent labels
python scripts/3_generate_latent_labels.py \
  laq_checkpoint=checkpoints/laq_final.ckpt
```

### Stage 2: Foundation Training

```bash
# Train 7B VLA model with FSDP (multi-node)
sbatch slurm/train.sbatch \
  scripts/4_train_foundation.py \
  experiment=vla_7b
```

### Stage 3: Action Finetuning

```bash
# Finetune on robot demonstrations
python scripts/5_finetune_actions.py \
  foundation_checkpoint=checkpoints/vla_foundation.ckpt
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_hydra_configs.py -v

# With coverage
pytest --cov=packages --cov-report=html
```

### Code Formatting

```bash
# Format code
black packages/ scripts/ tests/

# Lint
ruff check packages/ scripts/ tests/
```

## LRZ Cluster Setup

### Container Build

```bash
cd containers
./build_container.sh
```

### Directory Structure on LRZ

```bash
/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/
├── datasets/
│   ├── openx_frames/              # Stage 1 input
│   └── openx_latent_labeled/      # Stage 2 input
├── checkpoints/
│   ├── laq_final.ckpt
│   └── vla_foundation.ckpt
├── logs/
└── containers/
    └── lapa.sqsh
```

## Dependencies

- Python >= 3.9, < 3.12
- PyTorch >= 2.1.0
- PyTorch Lightning >= 2.1.0
- Hydra >= 1.3.0
- Transformers >= 4.35.0
- WebDataset >= 0.2.86
- Weights & Biases >= 0.16.0

See `pyproject.toml` for complete dependency list.

## Documentation

- [Hydra Documentation](docs/hydra.md) - Configuration system reference
- [LRZ Cluster Workflow](docs/lrz_workflow.md) - LRZ AI Systems deployment guide

## Project Status

- [x] Phase 0: Infrastructure setup
- [ ] Phase 1: LAQ implementation (Weeks 2-4)
- [ ] Phase 2: Foundation policy (Weeks 5-8)
- [ ] Phase 3: Action finetuning (Weeks 9-10)
- [ ] Phase 4: Inference & evaluation (Week 11)
- [ ] Phase 5: Optimization & documentation (Week 12)

## Citation

```bibtex
@misc{lapa2024,
  title={LAPA: Latent Action Pretraining from Videos},
  author={Your Team},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details

