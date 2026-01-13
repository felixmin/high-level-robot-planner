#!/usr/bin/env python3
"""
Submit training jobs to Slurm by generating sbatch scripts.

This script runs on the login node (no torch required) and generates
sbatch scripts that run inside Enroot containers on compute nodes.

Note:
    This is intended for cluster submissions (Slurm + container execution).
    For local runs, invoke the training script directly, e.g.:
        python scripts/2_train_laq.py experiment=...

Usage:
    # Submit single job
    python scripts/submit_job.py experiment=laq_oxe_debug

    # Override parameters
    python scripts/submit_job.py experiment=laq_oxe_debug training.epochs=10

    # Dry run (print script, don't submit)
    python scripts/submit_job.py --dry-run experiment=laq_oxe_debug

    # Custom resources
    python scripts/submit_job.py --time 01:00:00 --gpus 2 experiment=laq_full

    # Sweep (reads hydra.sweeper.params from experiment config)
    python scripts/submit_job.py experiment=laq_lr_sweep
    # Submits one job per parameter combination
"""

import argparse
import itertools
import os
import subprocess
import tempfile
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


# Project root (resolved at import time on login node)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def parse_sweep_params(cfg) -> dict[str, list[str]]:
    """Extract sweep parameters from sweep.params config.

    Returns dict mapping parameter names to lists of values.
    Example: {'training.optimizer.lr': ['1e-4', '5e-5'], 'seed': ['42', '123']}
    """
    sweep_params = {}

    # Check if sweep.params exists
    if not OmegaConf.select(cfg, "sweep.params"):
        return sweep_params

    params = cfg.sweep.params
    for key, value in params.items():
        # Value can be a string with comma-separated values or a list
        if isinstance(value, str):
            # Parse comma-separated values, strip whitespace
            values = [v.strip() for v in value.split(",")]
        elif isinstance(value, (list, tuple)):
            values = [str(v) for v in value]
        else:
            # Single value
            values = [str(value)]
        sweep_params[key] = values

    return sweep_params


def generate_sweep_combinations(sweep_params: dict[str, list[str]]) -> list[list[str]]:
    """Generate all combinations of sweep parameters as Hydra overrides.

    Returns list of override lists, e.g.:
    [['training.optimizer.lr=1e-4', 'seed=42'], ['training.optimizer.lr=1e-4', 'seed=123'], ...]
    """
    if not sweep_params:
        return [[]]  # Single empty combination (no sweep)

    # Get keys and values in consistent order
    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]

    # Generate Cartesian product
    combinations = []
    for combo in itertools.product(*value_lists):
        overrides = [f"{key}={val}" for key, val in zip(keys, combo)]
        combinations.append(overrides)

    return combinations


def generate_sbatch_script(
    script: str,
    overrides: list,
    partition: str,
    qos: str | None,
    account: str | None,
    gpus: int,
    time: str,
    mem: str,
    cpus: int,
    container_image: str,
    job_name: str,
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
{f"#SBATCH --qos={qos}" if qos else ""}
{f"#SBATCH --account={account}" if account else ""}
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
        default=None,
        help="Slurm partition (default: from cluster config)"
    )
    parser.add_argument(
        "--qos",
        default=None,
        help="Slurm QoS (default: from cluster config)"
    )
    parser.add_argument(
        "--account",
        default=None,
        help="Slurm account (default: from cluster config)"
    )
    parser.add_argument(
        "--gpus", "-g",
        type=int,
        default=None,
        help="Number of GPUs (default: from cluster config)"
    )
    parser.add_argument(
        "--time", "-t",
        default=None,
        help="Time limit (HH:MM:SS) (default: from cluster config)"
    )
    parser.add_argument(
        "--mem",
        default=None,
        help="Memory per node (default: from cluster config or 200G)"
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="CPUs per task (default: from cluster config)"
    )
    parser.add_argument(
        "--container",
        default=None,
        help="Container image path (overrides cluster.container.image)"
    )

    args, overrides = parser.parse_known_args()

    # Load config to show experiment info and get job name
    config_dir = str(PROJECT_ROOT / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    slurm_enabled = bool(OmegaConf.select(cfg, "cluster.slurm.enabled"))
    if not slurm_enabled:
        cluster_name = OmegaConf.select(cfg, "cluster.name") or "<unknown>"
        raise SystemExit(
            "Cluster config has `cluster.slurm.enabled: false` "
            f"(cluster={cluster_name}).\n"
            "Run the training script locally, or submit with a Slurm-enabled cluster, e.g.:\n"
            "  python scripts/submit_job.py experiment=... cluster=lrz_h100"
        )

    # Resolve Slurm defaults from Hydra config unless explicitly set via CLI.
    partition = args.partition or OmegaConf.select(cfg, "cluster.slurm.partition") or "mcml-hgx-h100-94x4"
    qos = args.qos if args.qos is not None else OmegaConf.select(cfg, "cluster.slurm.qos")
    account = args.account if args.account is not None else OmegaConf.select(cfg, "cluster.slurm.account")

    # Resolve compute defaults from Hydra config unless explicitly set via CLI.
    gpus = args.gpus if args.gpus is not None else (OmegaConf.select(cfg, "cluster.compute.gpus_per_node") or 1)
    cpus = args.cpus if args.cpus is not None else (OmegaConf.select(cfg, "cluster.compute.cpus_per_task") or 8)
    time_limit = args.time or OmegaConf.select(cfg, "cluster.compute.time_limit") or "24:00:00"

    # Container image is required for Slurm submissions.
    container_image = args.container or OmegaConf.select(cfg, "cluster.container.image")
    if not container_image:
        cluster_name = OmegaConf.select(cfg, "cluster.name") or "<unknown>"
        raise SystemExit(
            "Missing container image. Set `cluster.container.image` in the cluster config "
            f"(cluster={cluster_name}) or pass `--container /path/to/image.sqsh`."
        )

    # Memory: use CLI arg > cluster config > default 200G
    if args.mem is not None:
        mem = args.mem
    elif OmegaConf.select(cfg, "cluster.compute.mem_gb"):
        mem = f"{cfg.cluster.compute.mem_gb}G"
    else:
        mem = "200G"  # Safe default for OXE streaming

    # Check for sweep parameters
    sweep_params = parse_sweep_params(cfg)
    sweep_combinations = generate_sweep_combinations(sweep_params)
    is_sweep = len(sweep_combinations) > 1

    print("=" * 60)
    print("HLRP Job Submission")
    print("=" * 60)
    print(f"\nScript: {args.script}.py")
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Description: {cfg.experiment.description}")

    if is_sweep:
        print(f"\n🔄 SWEEP MODE: {len(sweep_combinations)} jobs")
        print(f"  Sweep parameters:")
        for param, values in sweep_params.items():
            print(f"    {param}: {values}")

    print(f"\nSlurm settings:")
    print(f"  Partition: {partition}")
    if qos:
        print(f"  QoS: {qos}")
    if account:
        print(f"  Account: {account}")
    print(f"  GPUs: {gpus}")
    print(f"  Time: {time_limit}")
    print(f"  Memory: {mem}")
    print(f"  CPUs: {cpus}")
    print(f"  Container: {container_image}")

    # Create logs directory
    logs_dir = PROJECT_ROOT / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Submit jobs for each sweep combination
    submitted_jobs = []

    for i, sweep_overrides in enumerate(sweep_combinations):
        # Combine base overrides with sweep overrides
        # Sweep overrides come last to take precedence
        combined_overrides = list(overrides) + sweep_overrides

        # Generate unique job name for sweeps
        if is_sweep:
            # Create a short suffix from sweep params (e.g., "lr1e-4_seed42")
            suffix_parts = []
            for override in sweep_overrides:
                key, val = override.split("=", 1)
                # Use last part of key (e.g., "lr" from "training.optimizer.lr")
                short_key = key.split(".")[-1]
                # Shorten value if needed
                short_val = val.replace("-", "").replace(".", "")[:8]
                suffix_parts.append(f"{short_key}{short_val}")
            job_name = f"hlrp_{cfg.experiment.name}_{'_'.join(suffix_parts)}"
        else:
            job_name = f"hlrp_{cfg.experiment.name}"

        # Generate sbatch script
        sbatch_content = generate_sbatch_script(
            script=args.script,
            overrides=combined_overrides,
            partition=partition,
            qos=qos,
            account=account,
            gpus=gpus,
            time=time_limit,
            mem=mem,
            cpus=cpus,
            container_image=container_image,
            job_name=job_name,
        )

        if args.dry_run:
            if is_sweep:
                print(f"\n[DRY RUN] Job {i+1}/{len(sweep_combinations)}: {sweep_overrides}")
            print("-" * 60)
            print(sbatch_content)
            print("-" * 60)
            continue

        # Write and submit sbatch script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sbatch', delete=False) as f:
            f.write(sbatch_content)
            sbatch_file = f.name

        try:
            if is_sweep:
                print(f"\nSubmitting job {i+1}/{len(sweep_combinations)}: {sweep_overrides}")
            else:
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
            submitted_jobs.append((job_id, sweep_overrides))

            print(f"  {output}")

        except subprocess.CalledProcessError as e:
            print(f"\nError submitting job: {e.stderr}")
            raise
        finally:
            # Clean up temp file
            os.unlink(sbatch_file)

    # Print summary
    if not args.dry_run and submitted_jobs:
        print("\n" + "=" * 60)
        print(f"Submitted {len(submitted_jobs)} job(s)")
        print("=" * 60)
        print("\nMonitor with:")
        print("  squeue --me")
        if len(submitted_jobs) == 1:
            job_id = submitted_jobs[0][0]
            print(f"  tail -f {logs_dir}/{job_id}.out")
        else:
            print(f"  tail -f {logs_dir}/<job_id>.out")
            print("\nJob IDs:")
            for job_id, sweep_ov in submitted_jobs:
                print(f"  {job_id}: {sweep_ov if sweep_ov else 'base config'}")


if __name__ == "__main__":
    main()
