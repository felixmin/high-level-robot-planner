#!/usr/bin/env python3
"""
LAQ Training Script

Train LAQ (Latent Action Quantization) model on video datasets.
Supports both real datasets and dummy test data for validation.

Usage:
    # Train with dummy data (overfitting test)
    python scripts/2_train_laq.py experiment=laq_debug data.data_dir=./datasets/dummy_videos training.overfit_batches=5
    
    # Train on full dataset
    python scripts/2_train_laq.py experiment=laq_full data.data_dir=/dss/.../datasets/openx_frames
    
    # Override specific parameters
    python scripts/2_train_laq.py experiment=laq_debug training.lr=1e-3 data.batch_size=16
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from packages.laq.task import create_laq_module_from_config
from packages.laq.data import create_dataloader


def train_laq(cfg: DictConfig):
    """
    Train LAQ model using Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        bool: True if training completed successfully
    """
    
    print(f"Config: {OmegaConf.to_yaml(cfg)}")
    print("=" * 60)
    print("LAQ Training")
    print("=" * 60)
    print(f"Experiment: {cfg.experiment.name}")
    data_dir = cfg.data.get('data_dir', cfg.data.get('train_shards', 'NOT SPECIFIED'))
    print(f"Data directory: {data_dir}")
    print(f"Max epochs: {cfg.training.epochs}")
    print(f"Batch size: {cfg.data.batch_size}")
    lr = cfg.get('optimizer', {}).get('lr', cfg.training.get('optimizer', {}).get('lr', 1e-4))
    print(f"Learning rate: {lr}")
    print(f"Accelerator: auto")
    print(f"Devices: 1")
    if cfg.training.get('overfit_batches'):
        print(f"Overfit batches: {cfg.training.overfit_batches}")
    if cfg.training.get('fast_dev_run'):
        print("Fast dev run: True")
    print()
    
    # Check if data exists
    # Handle both data_dir (for video files) and train_shards (for WebDataset)
    data_dir = cfg.data.get('data_dir', None)
    if data_dir:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"ERROR: Data directory not found: {data_dir}")
            print("Please ensure video files are in the specified directory")
            return False
    else:
        print("WARNING: No data_dir specified. Using train_shards if available.")
    
    # Create model
    try:
        # New LAPA config structure: model params are merged at top level via @_here_
        # Check if model config exists at top level or under cfg.model
        if hasattr(cfg, 'model'):
            model_config = OmegaConf.to_container(cfg.model, resolve=True)
        else:
            # Model config merged at top level - extract model params
            model_config = {}
            model_keys = ['image_size', 'patch_size', 'channels', 'dim', 'quant_dim',
                         'spatial_depth', 'temporal_depth', 'decoder_depth', 'heads',
                         'dim_head', 'mlp_ratio', 'dropout', 'codebook_size', 'code_seq_len']
            for key in model_keys:
                if hasattr(cfg, key):
                    model_config[key] = getattr(cfg, key)
        
        # Create LAQ module with model config and training params
        config_dict = {
            'model': model_config,
            'training': {
                'lr': cfg.get('optimizer', {}).get('lr', cfg.training.get('optimizer', {}).get('lr', 1e-4)),
                'weight_decay': cfg.get('optimizer', {}).get('weight_decay', cfg.training.get('optimizer', {}).get('weight_decay', 0.01)),
                'warmup_steps': cfg.training.get('scheduler', {}).get('warmup_steps', cfg.training.get('warmup_steps', 1000))
            }
        }
        
        model = create_laq_module_from_config(config_dict)
        print("✓ LAQ model created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create data loaders
    try:
        # For single sample testing, don't drop incomplete batches
        drop_last = not cfg.training.get('overfit_batches', False)
        
        # Get data directory (for video files) or use train_shards path
        video_dir = cfg.data.get('data_dir', None)
        if not video_dir:
            # If using WebDataset, extract directory from shard path
            train_shards = cfg.data.get('train_shards', '')
            if train_shards:
                # Extract directory from shard path (e.g., /path/to/dir/train_shard_00000.tar -> /path/to/dir)
                video_dir = str(Path(train_shards).parent)
            else:
                raise ValueError("Either data_dir or train_shards must be specified in config.data")
        
        train_loader = create_dataloader(
            video_dir=video_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=True,
            drop_last=drop_last,
            frame_spacing=cfg.data.get('frame_spacing', 1),
            max_frames_per_video=cfg.data.get('max_frames_per_video', 5)
        )
        
        # For validation, use same data (or create separate val set)
        val_loader = create_dataloader(
            video_dir=video_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=False,
            drop_last=drop_last,
            frame_spacing=cfg.data.get('frame_spacing', 1),
            max_frames_per_video=cfg.data.get('max_frames_per_video', 5)
        )
        
        print(f"✓ Data loaders created successfully")
        print(f"  - Training batches: {len(train_loader)}")
        print(f"  - Validation batches: {len(val_loader)}")
    except Exception as e:
        print(f"ERROR: Failed to create data loaders: {e}")
        return False
    
    # Setup logging
    if cfg.training.get('use_wandb', False):
        logger = WandbLogger(
            project=cfg.training.get('project_name', 'lapa-laq'),
            name=f"{cfg.experiment.name}-{Path(cfg.data.data_dir).name}",
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    else:
        logger = TensorBoardLogger("logs", name=f"{cfg.experiment.name}-{Path(cfg.data.data_dir).name}")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val/loss_recon',  # LAPA uses loss_recon, not loss_total
            mode='min',
            save_top_k=3,
            save_last=True,
            filename='best-{epoch:02d}-{val/loss_recon:.4f}',
            every_n_epochs=1
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Add early stopping for normal training (not overfitting)
    if not cfg.training.get('overfit_batches'):
        callbacks.append(
            EarlyStopping(
                monitor='val/loss_recon',  # LAPA uses loss_recon
                mode='min',
                patience=cfg.training.get('early_stopping_patience', 20),
                min_delta=1e-6
            )
        )
    
    # Create trainer
    trainer_kwargs = {
        'max_epochs': cfg.training.epochs,
        'logger': logger,
        'callbacks': callbacks,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'log_every_n_steps': cfg.training.get('log_every_n_steps', 10),
        'val_check_interval': cfg.training.get('val_check_interval', 1.0),
        'check_val_every_n_epoch': cfg.training.get('check_val_every_n_epoch', 1),
        'gradient_clip_val': cfg.training.gradient.clip_val,
        'accelerator': 'auto',
        'devices': 1
    }
    
    # Add overfitting or fast dev run options
    if cfg.training.get('overfit_batches'):
        trainer_kwargs['overfit_batches'] = cfg.training.overfit_batches
        print(f"✓ Overfitting mode: {cfg.training.overfit_batches} batches")
    elif cfg.training.get('fast_dev_run'):
        trainer_kwargs['fast_dev_run'] = True
        print("✓ Fast dev run mode enabled")
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    print("✓ Trainer configured successfully")
    print()
    
    # Train the model
    try:
        print("Starting training...")
        trainer.fit(model, train_loader, val_loader)
        
        # Get final validation loss (LAPA uses loss_recon, not loss_total)
        final_loss = trainer.callback_metrics.get('val/loss_recon', 
                                                  trainer.callback_metrics.get('val/loss_total', float('inf')))
        
        print()
        print("=" * 60)
        print("Training Results")
        print("=" * 60)
        print(f"Final validation loss: {final_loss:.6f}")
        print(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
        
        if cfg.training.get('overfit_batches'):
            # For overfitting test, check if loss is low enough
            target_loss = cfg.training.get('overfit_target_loss', 0.01)
            if final_loss <= target_loss:
                print("✓ OVERFITTING TEST PASSED!")
                print(f"  Loss converged to {final_loss:.6f} <= {target_loss}")
            else:
                print("✗ OVERFITTING TEST FAILED")
                print(f"  Loss {final_loss:.6f} > {target_loss}")
        
        print("✓ Training completed successfully!")
        return True
            
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        return False


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for LAQ training.
    
    Uses Hydra for configuration management as specified in PLAN.md.
    """
    success = train_laq(cfg)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())