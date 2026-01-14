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
    python scripts/submit_job.py submit.dry_run=true experiment=laq_oxe_debug

    # Custom resources (Hydra overrides)
    python scripts/submit_job.py cluster.compute.time_limit=01:00:00 cluster.compute.gpus_per_node=2 experiment=laq_full

    # Sweep (reads sweep.params from experiment config)
    python scripts/submit_job.py experiment=laq_lr_sweep
    # Submits one job per parameter combination
"""

import itertools
import os
import shlex
import subprocess
import sys
import tempfile
import time
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
    container_mounts: str,
    slurm_logs_dir: Path,
    cache_dir: Path,
) -> str:
    """Generate sbatch script content."""

    # Build the python command with overrides.
    # Quote each override so bash doesn't expand Hydra interpolations like `${now:...}`
    # or `${hydra.job.num}` inside the sbatch script.
    python_args = ["python", f"scripts/{script}.py", *overrides]
    python_cmd = " ".join(shlex.quote(str(arg)) for arg in python_args).strip()

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{f"#SBATCH --qos={qos}" if qos else ""}
{f"#SBATCH --account={account}" if account else ""}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={slurm_logs_dir}/%j.out
#SBATCH --error={slurm_logs_dir}/%j.err
#SBATCH --container-image={container_image}
#SBATCH --container-mounts={container_mounts}
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

# Persist caches on the mounted filesystem (avoid downloading models every job).
# Important: do not override HF_HOME here, otherwise Hugging Face auth (token) may no longer be
# discovered if it was previously stored under the default HF_HOME on $HOME.
mkdir -p "{cache_dir}/huggingface/hub" "{cache_dir}/torch"
export HF_HUB_CACHE="{cache_dir}/huggingface/hub"
export TORCH_HOME="{cache_dir}/torch"

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
    overrides = sys.argv[1:]

    # Load config to show experiment info and get job name
    config_dir = str(PROJECT_ROOT / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # Submission defaults (can be overridden via config/experiment/*.yaml or CLI)
    script = OmegaConf.select(cfg, "submit.script") or "2_train_laq"
    dry_run = bool(OmegaConf.select(cfg, "submit.dry_run") or False)

    script_path = PROJECT_ROOT / "scripts" / f"{script}.py"
    if not script_path.exists():
        raise SystemExit(f"Training script not found: {script_path}")

    slurm_enabled = bool(OmegaConf.select(cfg, "cluster.slurm.enabled"))

    # Resolve root + run-group directories.
    logging_root_dir = OmegaConf.select(cfg, "logging.root_dir")
    resolved_logging_root = Path(logging_root_dir) if logging_root_dir else None
    if resolved_logging_root and not resolved_logging_root.is_absolute():
        resolved_logging_root = PROJECT_ROOT / resolved_logging_root
    resolved_logging_root = resolved_logging_root or PROJECT_ROOT

    logging_runs_dir = OmegaConf.select(cfg, "logging.runs_dir")
    if logging_runs_dir:
        runs_dir = Path(logging_runs_dir)
        if not runs_dir.is_absolute():
            runs_dir = resolved_logging_root / runs_dir
    else:
        date_part = time.strftime("%Y-%m-%d")
        time_part = time.strftime("%H-%M-%S")
        runs_dir = resolved_logging_root / "runs" / date_part / time_part

    # Cache dir is stable across runs (by default under logging_root_dir).
    cache_dir = Path(OmegaConf.select(cfg, "submit.cache_dir") or "cache")
    if not cache_dir.is_absolute():
        cache_dir = resolved_logging_root / cache_dir

    # Slurm stdout/err: relative to the run-group directory by default.
    slurm_logs_dir = Path(OmegaConf.select(cfg, "submit.slurm_logs_dir") or "slurm")
    if not slurm_logs_dir.is_absolute():
        slurm_logs_dir = runs_dir / slurm_logs_dir

    # Check for sweep parameters
    sweep_params = parse_sweep_params(cfg)
    sweep_combinations = generate_sweep_combinations(sweep_params)
    is_sweep = len(sweep_combinations) > 1

    if not slurm_enabled:
        print("=" * 60)
        print("HLRP Local Run (no Slurm)")
        print("=" * 60)
        print(f"\nScript: {script}.py")
        print(f"Experiment: {cfg.experiment.name}")
        print(f"Description: {cfg.experiment.description}")
        print(f"\nPaths:")
        print(f"  Runs dir: {runs_dir}")
        print(f"  Cache dir: {cache_dir}")
        runs_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if is_sweep:
            print(f"\n🔄 SWEEP MODE: {len(sweep_combinations)} runs")
            print("  (Runs sequentially on the local machine)")

        base_env = os.environ.copy()
        base_env["PYTHONPATH"] = f"{PROJECT_ROOT}/packages:" + base_env.get("PYTHONPATH", "")
        base_env["HF_HUB_CACHE"] = str(cache_dir / "huggingface" / "hub")
        base_env["TORCH_HOME"] = str(cache_dir / "torch")

        run_prefix = time.strftime("%Y%m%d-%H%M%S")
        for i, sweep_overrides in enumerate(sweep_combinations):
            combined_overrides = list(overrides) + sweep_overrides
            override_str = " ".join(combined_overrides).strip()

            # Ensure a stable output root for this submission group.
            runs_override = f"logging.runs_dir={runs_dir}"

            # Give each run a unique id so unified logging doesn't collide.
            run_id = f"local-{run_prefix}-{i+1:03d}"
            job_override = f"logging.job_id={run_id}"
            cmd_overrides = list(combined_overrides)
            if not any(ov.startswith("logging.runs_dir=") for ov in cmd_overrides):
                cmd_overrides.append(runs_override)
            if not any(ov.startswith("logging.job_id=") for ov in cmd_overrides):
                cmd_overrides.append(job_override)
            cmd = [sys.executable, str(script_path)] + cmd_overrides

            if dry_run:
                print(f"\n[DRY RUN] {run_id}: {override_str}")
                print("  " + " ".join(cmd))
                continue

            print(f"\nRunning {run_id}: {override_str}" if override_str else f"\nRunning {run_id}")
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=base_env, check=True)

        return

    # Resolve Slurm/compute settings from Hydra config (override via CLI, e.g. cluster.compute.time_limit=...).
    partition = OmegaConf.select(cfg, "cluster.slurm.partition") or "mcml-hgx-h100-94x4"
    qos = OmegaConf.select(cfg, "cluster.slurm.qos")
    account = OmegaConf.select(cfg, "cluster.slurm.account")
    gpus = int(OmegaConf.select(cfg, "cluster.compute.gpus_per_node") or 1)
    cpus = int(OmegaConf.select(cfg, "cluster.compute.cpus_per_task") or 8)
    time_limit = OmegaConf.select(cfg, "cluster.compute.time_limit") or "24:00:00"

    # Container image is required for Slurm submissions.
    container_image = OmegaConf.select(cfg, "cluster.container.image")
    if not container_image:
        cluster_name = OmegaConf.select(cfg, "cluster.name") or "<unknown>"
        raise SystemExit(
            "Missing container image. Set `cluster.container.image` in the cluster config "
            f"(cluster={cluster_name})."
        )

    # Memory: use cluster config > default 200G
    if OmegaConf.select(cfg, "cluster.compute.mem_gb"):
        mem = f"{cfg.cluster.compute.mem_gb}G"
    else:
        mem = "200G"  # Safe default for OXE streaming

    # Ensure directories exist on the shared filesystem.
    runs_dir.mkdir(parents=True, exist_ok=True)
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build container mounts: always mount the project root, plus any external run/cache roots.
    mount_roots: list[Path] = [PROJECT_ROOT, runs_dir, cache_dir]

    # Ensure unique mount roots while preserving order.
    seen: set[Path] = set()
    unique_mounts: list[Path] = []
    for p in mount_roots:
        if p in seen:
            continue
        seen.add(p)
        unique_mounts.append(p)

    container_mounts = ",".join(f"{p}:{p}" for p in unique_mounts)

    print("=" * 60)
    print("HLRP Job Submission")
    print("=" * 60)
    print(f"\nScript: {script}.py")
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
    print(f"  Runs dir: {runs_dir}")
    print(f"  Slurm logs: {slurm_logs_dir}")
    print(f"  Cache dir: {cache_dir}")

    # Submit jobs for each sweep combination
    submitted_jobs = []

    for i, sweep_overrides in enumerate(sweep_combinations):
        # Combine base overrides with sweep overrides
        # Sweep overrides come last to take precedence
        combined_overrides = list(overrides) + sweep_overrides

        # Force run outputs into this submission group's run directory unless user overrides.
        if not any(ov.startswith("logging.runs_dir=") for ov in combined_overrides):
            combined_overrides.append(f"logging.runs_dir={runs_dir}")

        # Put Hydra config snapshots under this run group too unless user overrides.
        if not any(ov.startswith("hydra.run.dir=") for ov in combined_overrides):
            combined_overrides.append(f"hydra.run.dir={runs_dir}/outputs/hydra/${{now:%H-%M-%S}}")
        if not any(ov.startswith("hydra.sweep.dir=") for ov in combined_overrides):
            combined_overrides.append(f"hydra.sweep.dir={runs_dir}/outputs/hydra/${{now:%H-%M-%S}}")
        if not any(ov.startswith("hydra.sweep.subdir=") for ov in combined_overrides):
            combined_overrides.append("hydra.sweep.subdir=${hydra.job.num}")

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
            script=script,
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
            container_mounts=container_mounts,
            slurm_logs_dir=slurm_logs_dir,
            cache_dir=cache_dir,
        )

        if dry_run:
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
    if not dry_run and submitted_jobs:
        print("\n" + "=" * 60)
        print(f"Submitted {len(submitted_jobs)} job(s)")
        print("=" * 60)
        print("\nMonitor with:")
        print("  squeue --me")
        if len(submitted_jobs) == 1:
            job_id = submitted_jobs[0][0]
            print(f"  tail -f {slurm_logs_dir}/{job_id}.out")
        else:
            print(f"  tail -f {slurm_logs_dir}/<job_id>.out")
            print("\nJob IDs:")
            for job_id, sweep_ov in submitted_jobs:
                print(f"  {job_id}: {sweep_ov if sweep_ov else 'base config'}")


if __name__ == "__main__":
    main()
