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
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from common.data import LAQDataModule
from common.logging import set_seed, count_parameters
from laq import (
    LAQTask,
    ReconstructionVisualizationCallback,
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
    print("=" * 80)
    print("LAPA Stage 1: LAQ Training")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed for reproducibility
    if hasattr(cfg, "seed"):
        set_seed(cfg.seed)
        print(f"✓ Random seed set to {cfg.seed}")
    else:
        set_seed(42)
        print(f"✓ Random seed set to 42 (default)")

    # Initialize data module
    print("\n" + "=" * 80)
    print("Initializing Data Module")
    print("=" * 80)

    # Unpack config directly, filtering out metadata keys
    datamodule = LAQDataModule(
        **{k: v for k, v in cfg.data.items() if k not in ["name", "task"]}
    )

    # Setup to create datasets and get sizes
    datamodule.setup()

    # Dataset mode
    pair_level = cfg.data.get("pair_level", False)
    mode_str = "pair-level" if pair_level else "scene-level"

    # Handle both folder and sources mode in logging
    if cfg.data.get("sources"):
        source_info = [f"{s['type']}: {s['root']}" for s in cfg.data.sources]
        print(f"✓ DataModule initialized ({mode_str}, multi-source)")
        for s in source_info:
            print(f"  - Source: {s}")
    else:
        print(f"✓ DataModule initialized ({mode_str})")
        print(f"  - Folder: {cfg.data.folder}")
    print(f"  - Image size: {cfg.data.image_size}")
    print(f"  - Batch size: {cfg.data.batch_size}")
    print(f"  - Total samples available: {datamodule.total_available}")
    print(f"  - Train samples: {len(datamodule.train_dataset)}")
    print(f"  - Val samples: {len(datamodule.val_dataset)}")

    # Initialize LAQ task
    print("\n" + "=" * 80)
    print("Initializing LAQ Task")
    print("=" * 80)

    model_config = cfg.model
    training_config = cfg.training

    task = LAQTask(
        model_config=model_config,
        training_config=training_config,
        use_ema=training_config.get("use_ema", False),
    )

    num_params = count_parameters(task.model)
    print(f"✓ LAQ model initialized")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Codebook size: {model_config.codebook_size}")
    print(f"  - Code sequence length: {model_config.code_seq_len}")

    # Setup callbacks
    print("\n" + "=" * 80)
    print("Setting up Callbacks")
    print("=" * 80)

    callbacks = []

    # Checkpointing
    checkpoint_config = training_config.checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_config.monitor,
        mode=checkpoint_config.mode,
        save_top_k=checkpoint_config.save_top_k,
        save_last=checkpoint_config.save_last,
        every_n_epochs=checkpoint_config.every_n_epochs,
        filename="laq-{epoch:02d}-{val/loss:.4f}",
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    print(f"✓ Checkpoint callback added (monitor={checkpoint_config.monitor})")

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    print("✓ Learning rate monitor added")

    # Setup validation strategies
    val_config = training_config.validation
    strategies_config = val_config.get("strategies", {})
    
    # Check if using new validation strategy system
    if strategies_config:
        # New validation system with configurable strategies
        strategies = create_validation_strategies(strategies_config)
        
        val_strategy_callback = ValidationStrategyCallback(
            strategies=strategies,
            num_fixed_samples=val_config.get("num_fixed_samples", 8),
            num_random_samples=val_config.get("num_random_samples", 8),
            max_cached_samples=val_config.get("max_cached_samples", 256),
        )
        callbacks.append(val_strategy_callback)
        print(f"✓ Validation strategy callback added ({len(strategies)} strategies)")
        print(f"  - Max cached samples: {val_config.get('max_cached_samples', 256)}")
        for strategy in strategies:
            print(f"  - {strategy.name}: every {strategy.every_n_validations} validations")
    else:
        # Legacy visualization callback
        viz_callback = ReconstructionVisualizationCallback(
            num_samples=val_config.get("num_vis_samples", 8),
            log_every_n_epochs=1,
            visualize_train=strategies_config.get("basic", {}).get("visualize_train", True),
            visualize_val=strategies_config.get("basic", {}).get("visualize_val", True),
        )
        callbacks.append(viz_callback)
        print(f"✓ Reconstruction visualization callback added (legacy mode)")

    # Optional EMA
    if training_config.get("use_ema", False):
        ema_callback = EMACallback(
            decay=training_config.get("ema_decay", 0.999),
            update_every=training_config.get("ema_update_every", 1),
            update_after_step=training_config.get("ema_update_after_step", 0),
        )
        callbacks.append(ema_callback)
        print("✓ EMA callback added")

    # Setup logger
    print("\n" + "=" * 80)
    print("Setting up Logger")
    print("=" * 80)

    if hasattr(cfg, "logging") and cfg.logging.get("use_wandb", True):
        logger = WandbLogger(
            project=cfg.logging.get("project", "laq-training"),
            name=cfg.experiment.name,
            save_dir=cfg.logging.get("save_dir", "logs"),
            tags=cfg.logging.get("tags", ["laq"]),
        )
        print(f"✓ WandB logger initialized (project={cfg.logging.get('project', 'laq-training')})")
    else:
        logger = None
        print("✓ No logger (WandB disabled)")

    # Setup profiler
    print("\n" + "=" * 80)
    print("Setting up Profiler")
    print("=" * 80)

    profiler = None
    profiler_type = None
    if training_config.get("profiler", {}).get("enabled", False):
        profiler_type = training_config.profiler.get("type", "simple")
        dirpath = training_config.profiler.get("dirpath", "./profiles")
        filename = training_config.profiler.get("filename", "profile")

        if profiler_type == "simple":
            from lightning.pytorch.profilers import SimpleProfiler
            profiler = SimpleProfiler(dirpath=dirpath, filename=filename)
            print(f"✓ SimpleProfiler enabled")
        elif profiler_type == "advanced":
            from lightning.pytorch.profilers import AdvancedProfiler
            profiler = AdvancedProfiler(dirpath=dirpath, filename=filename)
            print(f"✓ AdvancedProfiler enabled")
        elif profiler_type == "pytorch":
            from lightning.pytorch.profilers import PyTorchProfiler
            profiler = PyTorchProfiler(
                dirpath=dirpath,
                filename=filename,
                emit_nvtx=False,
                export_to_chrome=True,
                row_limit=20,
            )
            print(f"✓ PyTorchProfiler enabled (high overhead!)")
        print(f"  - Output: {dirpath}/{filename}")
    else:
        print("✓ Profiler disabled")

    # Setup trainer
    print("\n" + "=" * 80)
    print("Setting up Trainer")
    print("=" * 80)

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
        logger=logger,
        profiler=profiler,
        log_every_n_steps=10,
        val_check_interval=val_check_interval,  # Configurable validation frequency
        limit_val_batches=limit_val_batches,  # Limit validation batches
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print(f"✓ Trainer initialized")
    print(f"  - Max epochs: {training_config.epochs}")
    print(f"  - Val check interval: {val_check_interval}")
    print(f"  - Limit val batches: {limit_val_batches}")
    print(f"  - Precision: {cfg.get('precision', '32-true')}")
    print(f"  - Accelerator: auto")
    print(f"  - Devices: auto")
    print(f"  - Profiler: {profiler_type if profiler else 'disabled'}")

    # Train
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    trainer.fit(task, datamodule=datamodule)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Print best checkpoint
    if checkpoint_callback.best_model_path:
        print(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"  - Best val/loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
