#!/bin/bash
#SBATCH --job-name=lapa_train
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Universal Slurm training script for LAPA
# Usage: sbatch slurm/train.sbatch scripts/2_train_laq.py experiment=laq_full

set -e  # Exit on error

# Print job information
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "Working Directory: $(pwd)"
echo "================================================================"

# Environment setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# NCCL settings for InfiniBand
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

# PyTorch settings
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Suppress tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "================================================================"

# Run training script
# All arguments after the script name are passed to Python
srun python "$@"

echo "================================================================"
echo "Training completed successfully!"
echo "================================================================"

