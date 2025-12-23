#!/bin/bash
#SBATCH -J laq_language_table           # Job name
#SBATCH -p lrz-hgx-h100-94x4            # H100 partition
#SBATCH --gres=gpu:1                    # 1 GPU (single GPU training for LAQ)
#SBATCH --cpus-per-task=8               # CPU cores for data loading
#SBATCH --mem=64G                       # Memory
#SBATCH --time=24:00:00                 # 24 hours (adjust as needed)
#SBATCH -o /dss/dsshome1/00/go98qik2/workspace/runs/logs/laq_lt_%j.out
#SBATCH -e /dss/dsshome1/00/go98qik2/workspace/runs/logs/laq_lt_%j.err
#SBATCH --container-image=/dss/dsshome1/00/go98qik2/workspace/containers/lam.sqsh
#SBATCH --container-mounts=/dss/dsshome1/00/go98qik2/workspace:/workspace
#SBATCH --container-workdir=/workspace

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

nvidia-smi

# Run training with language table experiment config
python scripts/2_train_laq.py experiment=laq_oxe_language_table

echo "=== Job finished at $(date) ==="
