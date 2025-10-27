"""
Logging utilities for LAPA project.

Provides consistent logging across all training stages with WandB integration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger


def setup_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO) -> logging.Logger:
    """
    Setup a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_wandb_logger(cfg: DictConfig) -> WandbLogger:
    """
    Setup WandB logger with configuration.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Configured WandbLogger instance
    """
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    wandb_logger = WandbLogger(
        project=cfg.logging.project,
        name=cfg.experiment_name,
        config=wandb_config,
        save_dir=cfg.logging.save_dir,
        log_model=cfg.logging.get("log_model", False),
        tags=cfg.logging.get("tags", []),
    )
    
    return wandb_logger


def log_hyperparameters(logger: logging.Logger, cfg: DictConfig) -> None:
    """
    Log hyperparameters to console.
    
    Args:
        logger: Logger instance
        cfg: Hydra configuration
    """
    logger.info("=" * 80)
    logger.info("Hyperparameters:")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

