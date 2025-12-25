# Job Submission Guide

**Date:** 2025-12-24
**Status:** Working

This guide explains how to submit training jobs to the LRZ cluster using `scripts/submit_job.py`.

---

## Quick Start

```bash
# Submit a single job
python scripts/submit_job.py experiment=laq_oxe_debug

# Submit with custom time limit
python scripts/submit_job.py --time 04:00:00 experiment=laq_full

# Dry run (preview without submitting)
python scripts/submit_job.py --dry-run experiment=laq_oxe_debug

# Submit a sweep (multiple jobs)
python scripts/submit_job.py experiment=laq_lr_sweep
```

---

## How It Works

The `submit_job.py` script:

1. **Runs on the login node** (no GPU/torch required)
2. **Loads Hydra config** to get experiment info and sweep parameters
3. **Generates sbatch scripts** with container directives
4. **Submits via `sbatch`** command

This approach bypasses Hydra's submitit launcher (which requires torch imports) by generating sbatch scripts directly.

---

## Single Job Submission

### Basic Usage

```bash
python scripts/submit_job.py experiment=laq_oxe_debug
```

### With Overrides

```bash
python scripts/submit_job.py experiment=laq_full \
    training.epochs=50 \
    data.batch_size=64
```

### Custom Resources

```bash
python scripts/submit_job.py \
    --partition mcml-hgx-h100-94x4 \
    --gpus 4 \
    --time 08:00:00 \
    --mem 128G \
    --cpus 16 \
    experiment=laq_full
```

### Different Training Script

```bash
# Run foundation training instead of LAQ
python scripts/submit_job.py --script 4_train_foundation experiment=vla_7b
```

---

## Sweep Submission

Sweeps submit multiple jobs with different parameter combinations.

### Define Sweep in Experiment Config

```yaml
# config/experiment/laq_lr_sweep.yaml
# @package _global_

defaults:
  - /model@model: laq
  - /data@data: laq_oxe
  - /training@training: laq_optimizer
  - /cluster@cluster: local_dev

# Sweep parameters - comma-separated values
sweep:
  params:
    training.optimizer.lr: 1e-4, 5e-5, 1e-5
    seed: 42, 123

experiment:
  name: laq_lr_sweep
  description: "Learning rate sweep for LAQ"

# ... rest of config
```

### Submit Sweep

```bash
python scripts/submit_job.py experiment=laq_lr_sweep
```

**Output:**
```
🔄 SWEEP MODE: 6 jobs
  Sweep parameters:
    training.optimizer.lr: ['1e-4', '5e-5', '1e-5']
    seed: ['42', '123']

Submitting job 1/6: ['training.optimizer.lr=1e-4', 'seed=42']
  Submitted batch job 5423108
Submitting job 2/6: ['training.optimizer.lr=1e-4', 'seed=123']
  Submitted batch job 5423109
...

Job IDs:
  5423108: ['training.optimizer.lr=1e-4', 'seed=42']
  5423109: ['training.optimizer.lr=1e-4', 'seed=123']
  ...
```

### How Sweeps Work

1. **Parse `sweep.params`** from experiment config
2. **Split comma-separated values** into lists
3. **Generate Cartesian product** of all combinations
4. **Submit separate job** for each combination
5. **Unique job names** include parameter values (e.g., `hlrp_laq_lr_sweep_lr1e4_seed42`)

### Sweep Parameter Syntax

```yaml
sweep:
  params:
    # Multiple values (comma-separated)
    training.optimizer.lr: 1e-4, 5e-5, 1e-5

    # Seeds for reproducibility
    seed: 42, 123, 456

    # Any config path works
    model.codebook_size: 8, 16, 32
    data.batch_size: 32, 64
```

**Note:** Uses `sweep.params` (not `hydra.sweeper.params`) because Hydra's internal config isn't accessible via `compose()`.

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--script`, `-s` | `2_train_laq` | Training script (without .py) |
| `--partition`, `-p` | `mcml-hgx-h100-94x4` | Slurm partition |
| `--gpus`, `-g` | `1` | Number of GPUs |
| `--time`, `-t` | `24:00:00` | Time limit (HH:MM:SS) |
| `--mem` | `64G` | Memory per node |
| `--cpus` | `8` | CPUs per task |
| `--container` | `lam.sqsh` | Container image path |
| `--dry-run` | `false` | Print script without submitting |

---

## Container Configuration

The default container is `lam.sqsh` which has all dependencies pre-installed:
- PyTorch with CUDA
- PyTorch Lightning
- Hydra, OmegaConf
- Transformers, timm
- TensorFlow, tensorflow-datasets
- WandB, einops, webdataset

### Override Container

```bash
# Via CLI
python scripts/submit_job.py --container /path/to/custom.sqsh experiment=laq_debug

# Via environment variable
export HLRP_CONTAINER_IMAGE=/path/to/custom.sqsh
python scripts/submit_job.py experiment=laq_debug
```

---

## Monitoring Jobs

### Check Queue Status

```bash
ssh ai 'squeue --me'
```

**Output:**
```
JOBID PARTITION     NAME     USER ST  TIME  NODES NODELIST(REASON)
5423108 mcml-hgx- hlrp_laq go98qik2  R  5:23      1 mcml-hgx-h100-006
5423109 mcml-hgx- hlrp_laq go98qik2 PD  0:00      1 (Priority)
```

Status codes:
- `R` = Running
- `PD` = Pending (waiting for resources)

### View Job Output

```bash
# Live output
ssh ai 'tail -f /dss/.../outputs/logs/5423108.out'

# Error log
ssh ai 'cat /dss/.../outputs/logs/5423108.err'
```

### Cancel Jobs

```bash
# Cancel single job
ssh ai 'scancel 5423108'

# Cancel all your jobs
ssh ai 'scancel --me'
```

---

## Generated sbatch Script

Each job generates a script like:

```bash
#!/bin/bash
#SBATCH --job-name=hlrp_laq_lr_sweep_lr1e4_seed42
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dss/.../outputs/logs/%j.out
#SBATCH --error=/dss/.../outputs/logs/%j.err
#SBATCH --container-image=/dss/.../containers/lam.sqsh
#SBATCH --container-mounts=/dss/...:/dss/...
#SBATCH --container-workdir=/dss/.../high-level-robot-planner

# Environment setup
export PYTHONPATH=/dss/.../packages:$PYTHONPATH
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

nvidia-smi

# Run training with overrides
python scripts/2_train_laq.py experiment=laq_lr_sweep training.optimizer.lr=1e-4 seed=42
```

---

## Example Sweep Configs

### Learning Rate Sweep

```yaml
# config/experiment/laq_lr_sweep.yaml
sweep:
  params:
    training.optimizer.lr: 1e-4, 5e-5, 1e-5
    seed: 42, 123
```
Submits 6 jobs (3 LRs × 2 seeds).

### Model Architecture Sweep

```yaml
# config/experiment/laq_arch_sweep.yaml
sweep:
  params:
    model.codebook_size: 8, 16, 32, 64
    model.code_dim: 32, 64
```
Submits 8 jobs (4 codebook sizes × 2 code dims).

### Dataset Comparison

```yaml
# config/experiment/laq_dataset_sweep.yaml
sweep:
  params:
    data.dataset_name: language_table, bridge
    seed: 42, 123, 456
```
Submits 6 jobs (2 datasets × 3 seeds).

---

## Troubleshooting

### "Container image not found"

The default container path is for the LRZ cluster. Set your own:
```bash
export HLRP_CONTAINER_IMAGE=/path/to/your/container.sqsh
```

### Job stuck in "Priority" state

Normal on shared clusters. Jobs wait for resources. Check estimated start:
```bash
ssh ai 'squeue --me --start'
```

### "No module named 'torch'" on login node

This is expected! The submit script doesn't need torch - it only parses Hydra configs. Training runs inside the container on compute nodes.

### Sweep parameters not detected

Make sure you use `sweep.params` (not `hydra.sweeper.params`):
```yaml
# Correct
sweep:
  params:
    training.optimizer.lr: 1e-4, 5e-5

# Won't work (Hydra internal config)
hydra:
  sweeper:
    params:
      training.optimizer.lr: 1e-4, 5e-5
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/submit_job.py` | Main submission script |
| `config/experiment/*.yaml` | Experiment configs (can include sweep.params) |
| `outputs/logs/*.out` | Job stdout logs |
| `outputs/logs/*.err` | Job stderr logs |

---

## Design Decisions

### Why Not Hydra's Submitit Launcher?

We use a custom `submit_job.py` script instead of Hydra's built-in submitit launcher for these reasons:

**1. Training scripts import torch at module level**
```python
# scripts/2_train_laq.py
import torch  # ← Fails on login node (no torch installed)

@hydra.main(...)
def main(cfg):
    ...
```

Hydra's submitit launcher (`-m` flag) executes the script on the login node to parse the config, then pickles the function. This fails because torch isn't installed on the login node.

**Solution comparison:**
- **Hydra submitit:** Requires refactoring all scripts to lazy-import torch inside `main()`
- **Our approach:** Parse configs separately, generate sbatch scripts directly

**2. Container integration is simpler**

- **Hydra submitit:** Requires wrapper scripts or pickle compatibility inside container
- **Our approach:** Uses SBATCH container directives natively (`#SBATCH --container-image=...`)

**3. No pickle mechanism needed**

- **Hydra submitit:** Pickles Python functions, requires submitit installed in container
- **Our approach:** Generates shell scripts that call `python scripts/X.py` with overrides

### Comparison

| Feature | Hydra Submitit | submit_job.py |
|---------|---------------|---------------|
| Requires torch on login node | Yes | No |
| Sweep support | Built-in (`-m` flag) | Via `sweep.params` |
| Container support | Needs wrapper | Native SBATCH directives |
| Pickle compatibility | Required | Not needed |
| Config syntax | `hydra.sweeper.params` | `sweep.params` |
| Refactoring required | Move imports to `main()` | None |

### Alternative Approaches Considered

**1. Pip install at runtime:** Install missing deps when job starts
- **Issue:** PyTorch version conflicts between base container and pytorch-lightning
- **When it works:** If using container with compatible PyTorch version

**2. Submitit with SlurmExecutor:** Use submitit directly without Hydra
- **Issue:** Requires submitit installed in container (base containers don't have it)
- **When it works:** If building custom container with submitit included

**3. Hydra multirun with custom launcher:** Write custom Hydra launcher plugin
- **Issue:** Complex, reinvents what sbatch already does
- **When it works:** If you need Hydra's advanced sweep algorithms (Bayesian optimization, etc.)

Our approach (direct sbatch generation) is the simplest solution that works with minimal changes to existing code and containers.
