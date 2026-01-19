#!/usr/bin/env python3
"""
Script 2: Train LAQ (Latent Action Quantization)

Train the VQ-VAE model to compress frame-to-frame transitions into discrete latent codes.

Usage:
    # Local debug
    python scripts/2_train_laq.py experiment=laq_debug

    # Full training on LRZ
    sbatch slurm/train.sbatch scripts/2_train_laq.py experiment=laq_full
"""

import sys
from pathlib import Path

# Add packages to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback


from common.data import LAQDataModule, OXEDataModule
from common.callbacks import DatasetUsageLoggerCallback, ProgressLoggerCallback
from foundation.callbacks import ThroughputLoggingCallback, ThroughputLoggingConfig
from common.logging import set_seed, count_parameters
from common.unified_logging import resolve_runs_dir, setup_unified_logging, setup_wandb_with_unified_paths
from laq import (
    LAQTask,
    EMACallback,
    ValidationStrategyCallback,
    create_validation_strategies,
)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function.

    Steps:
    1. Setup logging and seed
    2. Initialize data module
    3. Initialize LAQ task
    4. Setup Lightning trainer with callbacks
    5. Train the model
    """
    # Setup unified logging
    runs_dir = resolve_runs_dir(
        logging_root_dir=cfg.logging.get("root_dir"),
        logging_runs_dir=cfg.logging.get("runs_dir"),
        workspace_root=workspace_root,
        experiment_name=OmegaConf.select(cfg, "experiment.name"),
    )

    logger, output_dir = setup_unified_logging(
        runs_dir=runs_dir,
        job_id=cfg.logging.get("job_id"),
        log_level=cfg.logging.get("level", "INFO"),
        logger_name="laq.training",
    )

    logger.info("=" * 80)
    logger.info("LAPA Stage 1: LAQ Training")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # Set random seed for reproducibility
    if hasattr(cfg, "seed"):
        set_seed(cfg.seed)
        logger.info(f"✓ Random seed set to {cfg.seed}")
    else:
        set_seed(42)
        logger.info(f"✓ Random seed set to 42 (default)")

    # Initialize data module
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Data Module")
    logger.info("=" * 80)

    # Detect data module type based on config
    data_config = {k: v for k, v in cfg.data.items() if k not in ["name", "task"]}

    if "dataset_name" in cfg.data or "datasets" in cfg.data:
        # OXE streaming dataset (single or multi-dataset)
        datamodule = OXEDataModule(**data_config)
        datamodule.setup()
        logger.info(f"  - Batch size: {cfg.data.batch_size}")
        logger.info(f"  - Estimated train pairs: ~{len(datamodule.train_dataset):,}")
    else:
        # Multi-source file-based dataset (YouTube, Bridge, etc.)
        datamodule = LAQDataModule(**data_config)
        datamodule.setup()

        source_info = [f"{s['type']}: {s['root']}" for s in cfg.data.sources]
        logger.info(f"✓ DataModule initialized (multi-source)")
        for s in source_info:
            logger.info(f"  - Source: {s}")
        logger.info(f"  - Image size: {cfg.data.image_size}")
        logger.info(f"  - Batch size: {cfg.data.batch_size}")
        logger.info(f"  - Total scenes available: {datamodule.total_available}")
        logger.info(f"  - Train frame pairs: {len(datamodule.train_dataset)}")
        logger.info(f"  - Val frame pairs: {len(datamodule.val_dataset)}")

        # Print per-dataset frame pair breakdown
        pairs_per_dataset = datamodule.get_pairs_per_dataset()
        if pairs_per_dataset["train"]:
            logger.info(f"  - Train pairs by dataset: {pairs_per_dataset['train']}")
        if pairs_per_dataset["val"]:
            logger.info(f"  - Val pairs by dataset: {pairs_per_dataset['val']}")

    # Initialize LAQ task
    logger.info("\n" + "=" * 80)
    logger.info("Initializing LAQ Task")
    logger.info("=" * 80)

    model_config = cfg.model
    training_config = cfg.training

    task = LAQTask(
        model_config=model_config,
        training_config=training_config,
        use_ema=training_config.get("use_ema", False),
    )

    num_params = count_parameters(task.model)
    logger.info(f"✓ LAQ model initialized")
    logger.info(f"  - Total parameters: {num_params:,}")
    logger.info(f"  - Codebook size: {model_config.codebook_size}")
    logger.info(f"  - Code sequence length: {model_config.code_seq_len}")

    # Setup callbacks
    logger.info("\n" + "=" * 80)
    logger.info("Setting up Callbacks")
    logger.info("=" * 80)

    callbacks = []

    # Checkpointing (save to unified output directory)
    checkpoint_config = training_config.checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    every_n_train_steps = checkpoint_config.every_n_train_steps
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor=checkpoint_config.monitor,
        mode=checkpoint_config.mode,
        save_top_k=checkpoint_config.save_top_k,
        save_last=checkpoint_config.save_last,
        every_n_train_steps=every_n_train_steps,
        # Avoid using metric keys like `val/loss` in the filename template: Lightning's
        # formatting and filesystem sanitization can be version-dependent, and step-based
        # checkpointing may run when no validation metrics are available.
        filename="laq-step{step:06d}",
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    logger.info(f"✓ Checkpoint callback added (monitor={checkpoint_config.monitor})")
    logger.info(f"  - Checkpoint directory: {checkpoint_dir}")
    logger.info(f"  - Checkpointing every {every_n_train_steps} steps")

    # Learning rate monitoring requires a Lightning logger and is added after logger setup.

    # Progress logging (for cluster jobs where tqdm doesn't work in log files)
    progress_logger = ProgressLoggerCallback(log_every_n_steps=100)
    callbacks.append(progress_logger)
    logger.info("✓ Progress logger added (logs every 100 steps)")

    # Throughput logging: logs perf/steps_per_sec and perf/samples_per_sec to wandb
    perf_cfg = training_config.get("throughput")
    if perf_cfg and bool(perf_cfg.get("enabled", True)):
        callbacks.append(
            ThroughputLoggingCallback(
                ThroughputLoggingConfig(
                    enabled=True,
                    log_every_n_steps=int(perf_cfg.get("log_every_n_steps", 10)),
                )
            )
        )
        logger.info("✓ Throughput logger added")

    # Dataset usage logging: print dataset mix consumed between validations.
    usage_cfg = training_config.get("dataset_usage_logger")
    if usage_cfg and bool(usage_cfg.get("enabled", True)):
        callbacks.append(
            DatasetUsageLoggerCallback(
                enabled=True,
                log_on_validation_end=bool(usage_cfg.get("log_on_validation_end", True)),
                log_every_n_steps=usage_cfg.get("log_every_n_steps"),
                key=str(usage_cfg.get("key", "dataset_name")),
                top_k=int(usage_cfg.get("top_k", 12)),
            )
        )
        logger.info("✓ Dataset usage logger added")

    # Setup validation strategies
    val_config = training_config.validation
    strategies_config = val_config.get("strategies", {})
    if strategies_config:
        strategies_config = OmegaConf.to_container(strategies_config, resolve=True)

    # Get bucket configs (new "buckets" key, with "val_buckets" as fallback)
    bucket_configs = val_config.get("buckets", val_config.get("val_buckets", None))
    if bucket_configs:
        bucket_configs = OmegaConf.to_container(bucket_configs, resolve=True)

    strategies = create_validation_strategies(strategies_config, val_buckets=bucket_configs)

    val_strategy_callback = ValidationStrategyCallback(
        strategies=strategies,
        bucket_configs=bucket_configs,  # Pass bucket configs for routing
        num_fixed_samples=val_config.get("num_fixed_samples", 8),
        num_random_samples=val_config.get("num_random_samples", 8),
        max_cached_samples=val_config.get("max_cached_samples", 256),
    )
    callbacks.append(val_strategy_callback)
    logger.info(f"✓ Validation strategy callback added ({len(strategies)} strategies)")
    logger.info(f"  - Max cached samples: {val_config.get('max_cached_samples', 256)}")
    if bucket_configs:
        logger.info(f"  - Buckets: {list(bucket_configs.keys())}")
    for strategy in strategies:
        bucket_info = f", buckets={strategy.buckets}" if strategy.buckets else ""
        logger.info(f"  - {strategy.name}: every {strategy.every_n_validations} validations{bucket_info}")

    # Optional EMA
    if training_config.get("use_ema", False):
        ema_callback = EMACallback(
            decay=training_config.get("ema_decay", 0.999),
            update_every=training_config.get("ema_update_every", 1),
            update_after_step=training_config.get("ema_update_after_step", 0),
        )
        callbacks.append(ema_callback)
        logger.info("✓ EMA callback added")

    # Setup WandB logger (use unified logging paths)
    logger.info("\n" + "=" * 80)
    logger.info("Setting up WandB Logger")
    logger.info("=" * 80)

    if hasattr(cfg, "logging") and cfg.logging.get("use_wandb", True):
        wandb_logger = setup_wandb_with_unified_paths(
            logger=logger,
            output_dir=output_dir,
            project=cfg.logging.get("project", "hlrp"),
            name=cfg.experiment.name,
            tags=cfg.logging.get("tags", []),
            use_wandb=True,
        )
    else:
        wandb_logger = None
        logger.info("✓ WandB disabled")

    # Learning rate monitoring (requires a Lightning logger)
    if wandb_logger is not None:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        logger.info("✓ Learning rate monitor added")
    else:
        logger.info("✓ Learning rate monitor disabled (no logger)")

    # Setup profiler
    logger.info("\n" + "=" * 80)
    logger.info("Setting up Profiler")
    logger.info("=" * 80)

    profiler = None
    profiler_type = None
    if training_config.get("profiler", {}).get("enabled", False):
        profiler_type = training_config.profiler.get("type", "simple")
        dirpath = str(output_dir / "profiles")
        filename = training_config.profiler.get("filename", "profile")

        if profiler_type == "simple":
            from lightning.pytorch.profilers import SimpleProfiler
            profiler = SimpleProfiler(dirpath=dirpath, filename=filename)
            logger.info(f"✓ SimpleProfiler enabled")
        elif profiler_type == "advanced":
            from lightning.pytorch.profilers import AdvancedProfiler
            profiler = AdvancedProfiler(dirpath=dirpath, filename=filename)
            logger.info(f"✓ AdvancedProfiler enabled")
        elif profiler_type == "pytorch":
            from lightning.pytorch.profilers import PyTorchProfiler
            profiler = PyTorchProfiler(
                dirpath=dirpath,
                filename=filename,
                emit_nvtx=False,
                export_to_chrome=True,
                row_limit=20,
            )
            logger.info(f"✓ PyTorchProfiler enabled (high overhead!)")
        logger.info(f"  - Output: {dirpath}/{filename}")
    else:
        logger.info("✓ Profiler disabled")

    # Setup trainer
    logger.info("\n" + "=" * 80)
    logger.info("Setting up Trainer")
    logger.info("=" * 80)

    # Get validation check interval from config
    val_check_interval = training_config.validation.get("check_interval", 10000)
    # Limit validation batches to save time (default: run all, set to fraction or int)
    limit_val_batches = training_config.validation.get("limit_batches", 1.0)

    trainer = pl.Trainer(
        max_epochs=training_config.epochs,
        max_steps=training_config.get("max_steps") or -1,  # Convert None to -1
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision=cfg.get("precision", "32-true"),
        gradient_clip_val=training_config.gradient.clip_val,
        gradient_clip_algorithm=training_config.gradient.clip_algorithm,
        callbacks=callbacks,
        logger=wandb_logger if wandb_logger is not None else False,
        default_root_dir=str(output_dir),
        profiler=profiler,
        log_every_n_steps=10,
        val_check_interval=val_check_interval,  # Configurable validation frequency
        limit_val_batches=limit_val_batches,  # Limit validation batches
        enable_progress_bar=bool(training_config.get("enable_progress_bar", True)),
        enable_model_summary=bool(training_config.get("enable_model_summary", True)),
    )

    logger.info(f"✓ Trainer initialized")
    logger.info(f"  - Max epochs: {training_config.epochs}")
    logger.info(f"  - Val check interval: {val_check_interval}")
    logger.info(f"  - Limit val batches: {limit_val_batches}")
    logger.info(f"  - Precision: {cfg.get('precision', '32-true')}")
    logger.info(f"  - Accelerator: auto")
    logger.info(f"  - Devices: auto")
    logger.info(f"  - Profiler: {profiler_type if profiler else 'disabled'}")

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    # Check for resume checkpoint
    ckpt_path = training_config.get("resume_from_checkpoint", None)
    if ckpt_path:
        logger.info(f"✓ Resuming from checkpoint: {ckpt_path}")

    trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)

    # Print best checkpoint
    if checkpoint_callback.best_model_path:
        logger.info(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
        logger.info(f"  - Best val/loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
