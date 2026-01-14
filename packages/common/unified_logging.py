#!/usr/bin/env python3
"""
Unified logging setup for cluster and local runs.

This module provides:
1. Consolidated logging to runs/ folder
2. SLURM job ID integration
3. Stdout/stderr capture to log files
4. WandB integration with proper paths
5. Hydra output directory configuration
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import contextlib


def get_job_id() -> str:
    """
    Get the job identifier (SLURM_JOB_ID or wandb run ID).

    Priority:
    1. SLURM_JOB_ID (cluster runs)
    2. WANDB_RUN_ID (if WandB is initialized)
    3. 'local' (local development)

    Returns:
        Job identifier string
    """
    # Check for SLURM job ID first
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return job_id

    # Check for WandB run ID (set during wandb.init())
    wandb_id = os.environ.get("WANDB_RUN_ID")
    if wandb_id:
        return wandb_id

    # Default to 'local' for development
    return "local"


def setup_unified_logging(
    runs_dir: Path,
    job_id: Optional[str] = None,
    log_level: str = "INFO",
    capture_stdout: bool = True,
) -> tuple[logging.Logger, Path]:
    """
    Setup unified logging that captures all output to a single run-group directory.

    Creates:
    - <runs_dir>/outputs/<job_id>/unified.log - Complete training log
    - <runs_dir>/outputs/<job_id>/ - Per-job output directory

    Args:
        runs_dir: Path to the run-group directory (contains outputs/)
        job_id: Optional job ID (auto-detected if None)
        log_level: Logging level (INFO, DEBUG, etc.)
        capture_stdout: If True, redirect stdout/stderr to log file

    Returns:
        (logger, output_dir) tuple
    """
    if job_id is None:
        job_id = get_job_id()

    # Create directory structure
    outputs_dir = runs_dir / "outputs"
    output_dir = outputs_dir / job_id

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Failed to create directories: {e}")
        print(f"  - Attempted to create: {output_dir}")
        raise

    # Log file path
    log_file = output_dir / "unified.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (log file)
    try:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        print(f"WARNING: Failed to create log file handler: {e}")
        print(f"  - Attempted to write to: {log_file}")
        print(f"  - Logging will continue to console only")

    # Get module-specific logger
    logger = logging.getLogger("laq.training")

    # Log setup info
    logger.info("=" * 80)
    logger.info("Unified Logging Initialized")
    logger.info("=" * 80)
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log level: {log_level}")
    logger.info("=" * 80)

    # Note: Hydra creates its own output directory for config backups (hydra.run.dir).
    # Our unified logging handles the important outputs under <runs_dir>/outputs/<job_id>/:
    #   - Checkpoints → checkpoints/
    #   - WandB → wandb/
    #   - Logs → unified.log

    # Note: stdout/stderr capture is handled by WandB when enabled
    # Our file handler above captures all logging.* calls
    # WandB's stdout wrapper captures all print() calls
    # This creates two complementary logs:
    #   - <runs_dir>/outputs/<job_id>/unified.log: logger.info() calls (timestamped)
    #   - <runs_dir>/outputs/<job_id>/wandb/files/output.log: print() calls (WandB capture)

    return logger, output_dir


def setup_wandb_with_unified_paths(
    logger: logging.Logger,
    output_dir: Path,
    project: str,
    name: str,
    tags: list,
    use_wandb: bool = True,
):
    """
    Setup WandB logger with paths integrated into unified logging structure.

    Args:
        logger: Logger instance
        output_dir: Output directory from setup_unified_logging
        project: WandB project name
        name: Run name
        tags: WandB tags
        use_wandb: Whether to enable WandB

    Returns:
        WandbLogger or None
    """
    if not use_wandb:
        logger.info("WandB disabled")
        return None

    from lightning.pytorch.loggers import WandbLogger

    # WandB saves to output_dir/wandb
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(exist_ok=True)

    wandb_logger = WandbLogger(
        project=project,
        name=name,
        save_dir=str(wandb_dir),
        tags=tags,
    )

    logger.info(f"✓ WandB logger initialized (project={project})")
    logger.info(f"  - WandB directory: {wandb_dir}")

    return wandb_logger


@contextlib.contextmanager
def logging_context(workspace_root: Path, job_id: Optional[str] = None, log_level: str = "INFO"):
    """
    Context manager for unified logging setup.

    Usage:
        with logging_context(workspace_root) as (logger, output_dir):
            logger.info("Training started")
            # ... training code ...
    """
    logger, output_dir = setup_unified_logging(
        runs_dir=workspace_root,
        job_id=job_id,
        log_level=log_level,
    )

    try:
        yield logger, output_dir
    finally:
        # Cleanup: flush and close handlers
        for handler in logging.getLogger().handlers:
            handler.flush()
            if isinstance(handler, logging.FileHandler):
                handler.close()
