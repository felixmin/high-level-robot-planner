#!/usr/bin/env python3
# ruff: noqa: E402
"""
Script 4: Train Foundation VLA Model

Train the foundation VLA model (Stage 2): image + language -> latent action tokens.

Usage:
    # Debug (local / single GPU):
    python scripts/4_train_foundation.py experiment=vla_cosmos2_tokens_debug model.laq.checkpoint=/path/to/laq.ckpt
"""

import logging
import sys
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

from common.data import OXEDataModule
from common.logging import set_seed
from common.unified_logging import setup_unified_logging, setup_wandb_with_unified_paths
from foundation.action_tokens import ActionTokenConfig
from foundation.constrained_decode import ActionTokenIds
from foundation.callbacks import (
    ThroughputLoggingCallback,
    ThroughputLoggingConfig,
    VLASampleVizConfig,
    VLASampleVisualizationCallback,
)
from foundation.image_adapters import oxe_first_frames_to_pil
from foundation.online_laq import LAQTaskCodeProvider
from foundation.qwen3vl_setup import prepare_action_token_training
from foundation.vla_inputs import ChatConfig
from foundation.vla_module import VLATokenLightningModule, VLAOptimizerConfig


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Unified logging (mirrors Stage 1)
    use_unified_logging = cfg.logging.get("unified", True)

    logging_root_dir = cfg.logging.get("root_dir")
    logging_runs_dir = cfg.logging.get("runs_dir")

    if logging_runs_dir:
        runs_dir = Path(logging_runs_dir)
    elif logging_root_dir:
        runs_dir = Path(logging_root_dir) / "runs" / "local"
    else:
        runs_dir = workspace_root / "runs" / "local"

    if use_unified_logging:
        logger, output_dir = setup_unified_logging(
            runs_dir=runs_dir,
            job_id=cfg.logging.get("job_id"),
            log_level=cfg.logging.get("level", "INFO"),
            capture_stdout=cfg.logging.get("capture_stdout", True),
        )
    else:
        logger = logging.getLogger("foundation.training")
        logging.basicConfig(level=logging.INFO)
        output_dir = runs_dir / "outputs" / "local"
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("LAPA Stage 2: Foundation VLA Training (Action Tokens)")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    set_seed(int(getattr(cfg, "seed", 42)))

    # Setup WandB logger (use unified logging paths). If disabled, avoid Lightning's default logger
    # to prevent creating extra `lightning_logs/` directories.
    wandb_logger = setup_wandb_with_unified_paths(
        logger=logger,
        output_dir=output_dir,
        project=cfg.logging.get("project", "hlrp"),
        name=cfg.experiment.name,
        tags=cfg.logging.get("tags", []),
        use_wandb=bool(cfg.logging.get("use_wandb", True)),
    )

    # Data: OXE streaming frame pairs + language
    if not hasattr(cfg.data, "dataset_name") and not hasattr(cfg.data, "datasets"):
        raise ValueError(
            "Stage 2 currently expects OXE-style data config (dataset_name or datasets)."
        )

    data_config = {k: v for k, v in cfg.data.items() if k not in ["name", "task"]}
    datamodule = OXEDataModule(**data_config)
    datamodule.setup()

    # LAQ: frozen label generator
    laq_ckpt = cfg.model.laq.checkpoint
    if not laq_ckpt:
        raise ValueError(
            "Set `model.laq.checkpoint=/path/to/laq.ckpt` for online LAQ labeling."
        )
    from laq import LAQTask

    def load_laq_task_from_checkpoint(checkpoint_path: str) -> LAQTask:
        """
        Load an LAQ Lightning checkpoint across Torch/Lightning versions.

        PyTorch 2.6 changed `torch.load` default to `weights_only=True`, but
        Lightning checkpoints include non-tensor objects (e.g., OmegaConf).
        """
        ckpt_path = str(checkpoint_path)

        # Newer Lightning versions may plumb `weights_only` through.
        try:
            return LAQTask.load_from_checkpoint(ckpt_path, weights_only=False)
        except TypeError:
            pass
        except RuntimeError as exc:
            # Fall back to a manual loader when the failure is due to weights-only loading.
            if "weights_only" not in str(exc).lower():
                raise

        import inspect

        load_kwargs: dict[str, object] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False

        checkpoint = torch.load(ckpt_path, **load_kwargs)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")

        hparams = checkpoint.get("hyper_parameters")
        if not isinstance(hparams, dict):
            raise KeyError("Checkpoint missing 'hyper_parameters' dict")

        model_config = hparams.get("model_config")
        training_config = hparams.get("training_config")
        use_ema = bool(hparams.get("use_ema", False))

        if isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        if isinstance(training_config, dict):
            training_config = OmegaConf.create(training_config)

        if model_config is None or training_config is None:
            raise KeyError("Checkpoint hyperparameters missing model/training config")

        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise KeyError("Checkpoint missing 'state_dict'")

        task = LAQTask(
            model_config=model_config, training_config=training_config, use_ema=use_ema
        )
        task.load_state_dict(state_dict, strict=True)
        return task

    try:
        laq_task = load_laq_task_from_checkpoint(laq_ckpt)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load LAQ checkpoint '{laq_ckpt}'. "
            "Provide a checkpoint trained with the current LAQ codebase."
        ) from exc
    laq_provider = LAQTaskCodeProvider(laq_task)

    # VLA model: Qwen3-VL (Cosmos-Reason2 weights)
    model_name = cfg.model.vla.model_name
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    torch_dtype = str(cfg.model.vla.get("torch_dtype", "bf16")).lower()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    vla_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=cfg.model.vla.get("attn_implementation", "sdpa"),
    )
    processor = Qwen3VLProcessor.from_pretrained(model_name)

    action_cfg = ActionTokenConfig(
        **OmegaConf.to_container(cfg.model.action_tokens, resolve=True)
    )
    token_id_map = prepare_action_token_training(
        model=vla_model, processor=processor, action_tokens=action_cfg
    )

    action_token_ids = ActionTokenIds(
        action_start_id=token_id_map[action_cfg.action_start],
        action_end_id=token_id_map[action_cfg.action_end],
        action_code_ids=[
            token_id_map[action_cfg.token_fmt.format(i=i)]
            for i in range(action_cfg.codebook_size)
        ],
        eos_token_id=int(getattr(processor.tokenizer, "eos_token_id", 0)),
        code_seq_len=action_cfg.code_seq_len,
    )

    module = VLATokenLightningModule(
        vla_model=vla_model,
        processor=processor,
        code_provider=laq_provider,
        action_tokens=action_cfg,
        chat=ChatConfig(system_prompt=cfg.model.chat.get("system_prompt")),
        optimizer=VLAOptimizerConfig(
            lr=float(cfg.training.optimizer.lr),
            weight_decay=float(cfg.training.optimizer.weight_decay),
        ),
        action_token_ids=action_token_ids,
        frames_to_images=oxe_first_frames_to_pil,
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=cfg.training.checkpoint.monitor,
            mode=cfg.training.checkpoint.mode,
            save_top_k=int(cfg.training.checkpoint.save_top_k),
            save_last=bool(cfg.training.checkpoint.save_last),
            filename="vla-step{step:06d}",
        ),
    ]

    viz_cfg = cfg.training.validation.get("visualization")
    if viz_cfg and bool(viz_cfg.get("enabled", True)):
        callbacks.append(
            VLASampleVisualizationCallback(
                VLASampleVizConfig(
                    enabled=bool(viz_cfg.get("enabled", True)),
                    num_samples=int(viz_cfg.get("num_samples", 4)),
                    every_n_val=int(viz_cfg.get("every_n_val", 1)),
                )
            )
        )
    perf_cfg = cfg.training.get("throughput")
    if perf_cfg and bool(perf_cfg.get("enabled", True)):
        callbacks.append(
            ThroughputLoggingCallback(
                ThroughputLoggingConfig(
                    enabled=bool(perf_cfg.get("enabled", True)),
                    log_every_n_steps=int(perf_cfg.get("log_every_n_steps", 10)),
                )
            )
        )
    if wandb_logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    else:
        logger.info("WandB disabled; skipping LearningRateMonitor (no logger).")

    # Validation defaults in Lightning run at the end of the (potentially huge) epoch.
    # For short max_steps runs with large IterableDatasets, validation may never run unless
    # we validate every N steps (like Stage 1) and/or limit validation batches.
    trainer_extra_kwargs: dict[str, object] = {}
    val_check_interval = cfg.training.validation.get(
        "check_interval", cfg.training.validation.get("val_check_interval")
    )
    if val_check_interval is not None:
        trainer_extra_kwargs["val_check_interval"] = val_check_interval
    limit_val_batches = cfg.training.validation.get(
        "limit_batches", cfg.training.validation.get("limit_val_batches")
    )
    if limit_val_batches is not None:
        trainer_extra_kwargs["limit_val_batches"] = limit_val_batches
    num_sanity_val_steps = cfg.training.validation.get("num_sanity_val_steps")
    if num_sanity_val_steps is not None:
        trainer_extra_kwargs["num_sanity_val_steps"] = int(num_sanity_val_steps)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        max_steps=cfg.training.max_steps,
        max_epochs=cfg.training.max_epochs,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=10,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.training.validation.check_val_every_n_epoch,
        logger=wandb_logger if wandb_logger is not None else False,
        default_root_dir=str(output_dir),
        **trainer_extra_kwargs,
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
