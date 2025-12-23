#!/usr/bin/env python3
"""
Submit training jobs to Slurm via submitit.

This script runs on the login node (no torch required) and submits jobs
that execute inside Enroot containers on compute nodes.

Usage:
    # Submit single job
    python scripts/submit_job.py experiment=laq_oxe_debug

    # With cluster profile
    python scripts/submit_job.py experiment=laq_oxe_debug cluster=lrz_h100

    # Override parameters
    python scripts/submit_job.py experiment=laq_oxe_debug training.epochs=10

    # Dry run (print config, don't submit)
    python scripts/submit_job.py --dry-run experiment=laq_oxe_debug
"""

import argparse
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import submitit


# Project root (resolved at import time on login node)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PROJECT_ROOT_STR = str(PROJECT_ROOT)  # Serialize as string for pickle


def run_training(script: str, overrides: list, project_root: str) -> int:
    """
    Training function that runs on the compute node inside the container.
    Executes the actual training script via subprocess.
    """
    from pathlib import Path

    project_path = Path(project_root)
    script_path = project_path / "scripts" / f"{script}.py"

    # Build command
    cmd = [sys.executable, str(script_path)] + overrides

    print("=" * 80)
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {project_root}")
    print("=" * 80)

    # Execute training script from project root
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Submit training jobs via submitit",
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
        help="Print config and submission command, don't submit"
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

    args, overrides = parser.parse_known_args()

    # Load config to show experiment info
    config_dir = str(PROJECT_ROOT / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    print("=" * 80)
    print("HLRP Job Submission")
    print("=" * 80)
    print(f"\nScript: {args.script}.py")
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Description: {cfg.experiment.description}")
    print(f"\nSlurm settings:")
    print(f"  Partition: {args.partition}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Time: {args.time}")
    print(f"  Memory: {args.mem}")
    print(f"  CPUs: {args.cpus}")

    if args.dry_run:
        print("\n[DRY RUN] Would submit with overrides:")
        for o in overrides:
            print(f"  {o}")
        print("\nFull config:")
        print(OmegaConf.to_yaml(cfg))
        return

    # Setup submitit executor
    wrapper_script = PROJECT_ROOT / "scripts" / "enroot_wrapper.sh"
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    executor = submitit.SlurmExecutor(folder=str(outputs_dir / "%j"))
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_qos="mcml",
        slurm_gpus_per_node=args.gpus,
        slurm_cpus_per_task=args.cpus,
        slurm_mem=args.mem,
        slurm_time=args.time,
        slurm_setup=[
            f"chmod +x {wrapper_script}",
            "export NCCL_SOCKET_IFNAME=ib0",
            "export NCCL_DEBUG=WARN",
        ],
        slurm_additional_parameters={
            "chdir": str(PROJECT_ROOT),
        },
        # Use wrapper script as Python interpreter (runs inside container)
        python=str(wrapper_script),
    )

    # Submit job (pass project root as string for pickle compatibility)
    print("\nSubmitting job...")
    job = executor.submit(run_training, args.script, overrides, PROJECT_ROOT_STR)

    print(f"\nJob submitted!")
    print(f"  Job ID: {job.job_id}")
    print(f"  Output: {PROJECT_ROOT}/outputs/{job.job_id}/")
    print(f"\nMonitor with:")
    print(f"  squeue --me")
    print(f"  tail -f {PROJECT_ROOT}/outputs/{job.job_id}/*_log.out")


if __name__ == "__main__":
    main()
