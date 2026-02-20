# LRZ AI Systems Workflow Guide

Complete guide for using this project on LRZ AI Systems.

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Building Containers](#building-containers)
3. [Submitting Jobs](#submitting-jobs)
4. [Multi-Node Training](#multi-node-training)
5. [Common Issues](#common-issues)

## Initial Setup

### 1. Login to LRZ

```bash
# From TUM network or VPN
ssh -A <TUM-ID>@login.ai.lrz.de

# Check your storage access
dssusrinfo all
```

You should see:
```
pn57pi-dss-0001 at /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001
```

### 2. Run Setup Script

```bash
# Download and run setup
cd /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001
git clone <repo-url> project-repo
cd project-repo
# Follow installation steps in README.md
```

This creates:
- Environment variables in `~/.bashrc`
- Directory structure on DSS storage
- Helpful aliases for job management

### 3. Configure WandB

```bash
echo "export WANDB_API_KEY=<YOUR_KEY>" >> ~/.bashrc
source ~/.bashrc
```

## Building Containers

### Building the Container

```bash
# Build on LRZ in project containers/ directory
cd /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/project-repo/containers
./build_container.sh  # or enroot build commands if needed

# Test the container
enroot create --name project-test /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/containers/project.sqsh
enroot start --root --rw --mount /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001:/workspace project-test

# Inside container: verify setup
python scripts/0_setup_environment.py
exit
```

## Submitting Jobs

### Check Resource Availability

```bash
# View available nodes
sinfo

# Check MCML nodes specifically
sinfo -p mcml-hgx-h100-94x4
```

### Single-Node Training

```bash
# Submit training job (see README.md for available experiments)
sbatch slurm/train.sbatch scripts/2_train_laq.py experiment=laq_oxe_cluster
sbatch slurm/train.sbatch scripts/4_train_foundation.py experiment=vla_smol_flow_shared
```

### Monitor Jobs

```bash
# View your jobs
squeue --user=$USER -M mcml-hgx-h100

# Check job details
scontrol show job <JOB_ID>

# View job output
tail -f logs/slurm-<JOB_ID>.out

# Cancel job
scancel <JOB_ID> -M mcml-hgx-h100
```

## Multi-Node Training

### Foundation Model Training

```bash
# Multi-node training configured in experiment config
sbatch slurm/train.sbatch scripts/4_train_foundation.py experiment=vla_smol_flow_shared
```

Configure nodes in `config/experiment/vla_smol_flow_shared.yaml` via:
```yaml
cluster:
  compute:
    num_nodes: 4
    gpus_per_node: 4
```

### NCCL Configuration

`scripts/submit_job.py` exports reasonable defaults:
- `NCCL_SOCKET_IFNAME=ib0`
- `NCCL_DEBUG=WARN`

Slurm automatically sets:
- `MASTER_ADDR`: First node in allocation
- `MASTER_PORT`: 29500

## Interactive Sessions

For debugging or testing:

```bash
# Request 1 GPU for 1 hour
salloc -p mcml-hgx-h100-94x4 -q mcml --gres=gpu:1

# Start interactive shell on compute node
srun --pty bash

# Check GPU
nvidia-smi

# Load environment
source ~/.bashrc

# Run commands
python scripts/0_setup_environment.py

# Exit
exit  # From compute node
exit  # From allocation
```

## Common Issues

### 1. Container Not Found

**Error**: `No such file or directory: /raid/enroot/data/user-XXX/project`

**Solution**: Create container first:
```bash
enroot create --name project-test /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/containers/project.sqsh
```

### 2. NCCL Timeout

**Error**: `NCCL timeout` or `Connection refused`

**Solution**: Check InfiniBand settings:
```bash
# In your sbatch or interactive session
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
```

### 3. Out of Storage

**Error**: `No space left on device`

**Check usage**:
```bash
dssusrinfo all
```

**Clean up**:
```bash
# Remove old checkpoints
rm -rf /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/checkpoints/old_experiment_*

# Clean enroot cache
rm -rf /raid/enroot/data/user-$(id -u)/*
```

### 4. Job Pending for Long Time

**Check queue**:
```bash
squeue -p mcml-hgx-h100-94x4
```

**Estimate start time**:
```bash
squeue --start --job <JOB_ID>
```

**Try different partition** if urgent:
```bash
# Use lrz partition (non-MCML, might have different limits)
sbatch --partition=lrz-hgx-h100-94x4 ...
```

## Storage Best Practices

### Directory Structure

```
/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/
├── project-repo/            # Code repository
├── datasets/                # Training data
│   └── (WebDataset shards)
├── checkpoints/             # Model checkpoints (auto-saved)
├── logs/                    # Slurm logs
└── containers/              # Enroot containers
    └── project.sqsh
```

### Data Organization

- Store raw videos and datasets under `datasets/`
- Checkpoints auto-save to `checkpoints/<experiment_name>/`
- Logs auto-save to `logs/` directory by Slurm

## Performance Tips

1. **Use InfiniBand**: Always set `NCCL_SOCKET_IFNAME=ib0`
2. **Batch Size**: H100 has 80GB, use larger batches
3. **Data Loading**: Set `num_workers=8` or higher
4. **Gradient Accumulation**: For larger effective batch sizes
5. **Mixed Precision**: Always use `bf16` on H100

## Useful Commands

```bash
# Check node specs
scontrol show node <node_name>

# View your recent jobs
sacct --user=$USER --starttime=$(date -d '7 days ago' +%Y-%m-%d)

# Job efficiency report
seff <JOB_ID>

# Real-time GPU monitoring during job
ssh <compute_node> nvidia-smi -l 1
```

## Additional Resources

- LRZ AI Systems Docs: https://doku.lrz.de/display/PUBLIC/LRZ+AI+Systems
- MCML Guidelines: https://doku.lrz.de/display/PUBLIC/MCML+Members
- Slurm Commands: https://slurm.schedmd.com/pdfs/summary.pdf
- NCCL Tuning: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
