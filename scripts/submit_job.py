#!/usr/bin/env python3
"""
Submit training jobs to Slurm by generating sbatch scripts.

This script runs on the login node (no torch required) and generates
sbatch scripts that run inside Enroot containers on compute nodes.

Usage:
    # Submit single job
    python scripts/submit_job.py experiment=laq_oxe_debug

    # Override parameters
    python scripts/submit_job.py experiment=laq_oxe_debug training.epochs=10

    # Dry run (print script, don't submit)
    python scripts/submit_job.py --dry-run experiment=laq_oxe_debug

    # Custom resources
    python scripts/submit_job.py --time 01:00:00 --gpus 2 experiment=laq_full
"""

import argparse
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


# Project root (resolved at import time on login node)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def generate_sbatch_script(
    script: str,
    overrides: list,
    partition: str,
    gpus: int,
    time: str,
    mem: str,
    cpus: int,
    container_image: str,
    job_name: str,
    pip_install: bool = False,
) -> str:
    """Generate sbatch script content."""

    # Build the python command with overrides
    override_str = " ".join(overrides) if overrides else ""
    python_cmd = f"python scripts/{script}.py {override_str}".strip()

    # Output directory
    logs_dir = PROJECT_ROOT / "outputs" / "logs"

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --qos=mcml
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={logs_dir}/%j.out
#SBATCH --error={logs_dir}/%j.err
#SBATCH --container-image={container_image}
#SBATCH --container-mounts={PROJECT_ROOT}:{PROJECT_ROOT}
#SBATCH --container-workdir={PROJECT_ROOT}

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "========================================"

# Environment setup
export PYTHONPATH={PROJECT_ROOT}/packages:$PYTHONPATH
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
{f'''
# Install missing dependencies (--pip-install flag)
echo "Installing dependencies..."
pip install --quiet pytorch-lightning hydra-core omegaconf transformers timm wandb einops webdataset accelerate tensorflow tensorflow-datasets
''' if pip_install else ''}
# Show GPU info
nvidia-smi

# Run training
{python_cmd}

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"
"""
    return script_content


def main():
    parser = argparse.ArgumentParser(
        description="Submit training jobs to Slurm",
        epilog="Additional arguments are passed as Hydra overrides"
    )
    parser.add_argument(
        "--script", "-s",
        default="2_train_laq",
        help="Training script to run (without .py)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch script, don't submit"
    )
    parser.add_argument(
        "--partition", "-p",
        default="mcml-hgx-h100-94x4",
        help="Slurm partition"
    )
    parser.add_argument(
        "--gpus", "-g",
        type=int,
        default=1,
        help="Number of GPUs"
    )
    parser.add_argument(
        "--time", "-t",
        default="24:00:00",
        help="Time limit (HH:MM:SS)"
    )
    parser.add_argument(
        "--mem",
        default="64G",
        help="Memory per node"
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="CPUs per task"
    )
    parser.add_argument(
        "--container",
        default=None,
        help="Container image path (default: from HLRP_CONTAINER_IMAGE or built-in)"
    )
    parser.add_argument(
        "--pip-install",
        action="store_true",
        help="Install missing dependencies via pip at job start (slower but ensures deps)"
    )

    args, overrides = parser.parse_known_args()

    # Load config to show experiment info and get job name
    config_dir = str(PROJECT_ROOT / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # Container image
    container_image = args.container or os.environ.get(
        "HLRP_CONTAINER_IMAGE",
        "/dss/dsshome1/00/go98qik2/workspace/containers/nvidia+pytorch+25.06-py3.sqsh"
    )

    # Job name from experiment
    job_name = f"hlrp_{cfg.experiment.name}"

    print("=" * 60)
    print("HLRP Job Submission")
    print("=" * 60)
    print(f"\nScript: {args.script}.py")
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Description: {cfg.experiment.description}")
    print(f"\nSlurm settings:")
    print(f"  Partition: {args.partition}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Time: {args.time}")
    print(f"  Memory: {args.mem}")
    print(f"  CPUs: {args.cpus}")
    print(f"  Container: {container_image}")

    # Generate sbatch script
    sbatch_content = generate_sbatch_script(
        script=args.script,
        overrides=overrides,
        partition=args.partition,
        gpus=args.gpus,
        time=args.time,
        mem=args.mem,
        cpus=args.cpus,
        container_image=container_image,
        job_name=job_name,
        pip_install=args.pip_install,
    )

    if args.dry_run:
        print("\n[DRY RUN] Generated sbatch script:")
        print("-" * 60)
        print(sbatch_content)
        print("-" * 60)
        return

    # Create logs directory
    logs_dir = PROJECT_ROOT / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Write and submit sbatch script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sbatch', delete=False) as f:
        f.write(sbatch_content)
        sbatch_file = f.name

    try:
        print("\nSubmitting job...")
        result = subprocess.run(
            ["sbatch", sbatch_file],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse job ID from output like "Submitted batch job 12345"
        output = result.stdout.strip()
        job_id = output.split()[-1] if output else "unknown"

        print(f"\n{output}")
        print(f"\nMonitor with:")
        print(f"  squeue --me")
        print(f"  tail -f {logs_dir}/{job_id}.out")

    except subprocess.CalledProcessError as e:
        print(f"\nError submitting job: {e.stderr}")
        raise
    finally:
        # Clean up temp file
        os.unlink(sbatch_file)


if __name__ == "__main__":
    main()
