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
#SBATCH --container-image=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/containers/lapa.sqsh
#SBATCH --container-mounts=/dss/dssmcmlfs01:/data
#SBATCH --container-workdir=/workspace

# Training script with Enroot container
# Usage: sbatch slurm/train_container.sbatch scripts/2_train_laq.py experiment=laq_full

set -e

echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Container: lapa.sqsh"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "================================================================"

# Environment setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# Run in container
srun python "$@"

echo "Training completed!"

