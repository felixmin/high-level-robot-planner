#!/usr/bin/env python3
# ruff: noqa: E402
"""
Script 4: Train Foundation VLA Model

Train the foundation VLA model (Stage 2): image + language -> latent action tokens.

Usage:
    # Debug (local / single GPU):
    python scripts/4_train_foundation.py experiment=vla_cosmos2_tokens_debug model.laq.checkpoint=/path/to/laq.ckpt
"""

import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

from common.callbacks import DatasetUsageLoggerCallback, ProgressLoggerCallback
from common.data_factory import create_datamodule
from common.logging import set_seed
from common.unified_logging import resolve_runs_dir, setup_unified_logging, setup_wandb_with_unified_paths
from foundation.action_tokens import ActionTokenConfig
from foundation.constrained_decode import ActionTokenIds
from foundation.callbacks import (
    ThroughputLoggingCallback,
    ThroughputLoggingConfig,
    VLATrainSampleVizConfig,
    VLATrainSampleVisualizationCallback,
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
    # Force new W&B run per Hydra job (fix for sweeps)
    if wandb.run:
        wandb.finish()

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
        logger_name="foundation.training",
    )

    logger.info("=" * 80)
    logger.info("LAPA Stage 2: Foundation VLA Training (Action Tokens)")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    set_seed(int(getattr(cfg, "seed", 42)))

    # Setup WandB logger (use unified logging paths). If disabled, avoid Lightning's default logger
    # to prevent creating extra `lightning_logs/` directories.
    # Calculate unique run name for sweeps
    run_name = cfg.experiment.name
    try:
        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode == "MULTIRUN":
            run_name = f"{run_name}_{hydra_cfg.job.num}"
    except (ValueError, Exception):
        pass

    wandb_logger = setup_wandb_with_unified_paths(
        logger=logger,
        output_dir=output_dir,
        project=cfg.logging.get("project", "hlrp"),
        name=run_name,
        group=cfg.experiment.name,
        tags=cfg.logging.get("tags", []),
        use_wandb=bool(cfg.logging.get("use_wandb", True)),
        settings=wandb.Settings(start_method="thread", reinit=True),
    )

    # Data: frame pairs + language (OXE backends only).
    if cfg.data.backend not in ("oxe_tf", "oxe_hf"):
        raise ValueError(
            f"Stage 2 expects an OXE backend (oxe_tf/oxe_hf), got {cfg.data.backend!r}"
        )

    datamodule = create_datamodule(cfg.data)
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
    if torch_dtype not in dtype_map:
        raise ValueError(
            f"Unknown model.vla.torch_dtype={torch_dtype!r}. "
            f"Supported: {sorted(dtype_map.keys())}"
        )
    dtype = dtype_map[torch_dtype]

    vla_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=cfg.model.vla.get("attn_implementation", "sdpa"),
    )
    # HuggingFace models default to eval mode after from_pretrained(); set train mode explicitly
    vla_model.train()
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

    # Sanity checks: action token ids must be distinct and must not map to UNK.
    code_ids = list(action_token_ids.action_code_ids)
    if len(set(code_ids)) != len(code_ids):
        raise RuntimeError(
            "Action code token ids are not unique. "
            "This typically means the action tokens were not properly added to the tokenizer. "
            f"code_ids={code_ids}"
        )
    if action_token_ids.action_start_id in code_ids or action_token_ids.action_end_id in code_ids:
        raise RuntimeError(
            "Action wrapper token id overlaps with an action code token id. "
            f"start_id={action_token_ids.action_start_id} end_id={action_token_ids.action_end_id} code_ids={code_ids}"
        )
    unk_id = getattr(processor.tokenizer, "unk_token_id", None)
    if unk_id is not None and unk_id in code_ids:
        raise RuntimeError(
            "One or more action code tokens mapped to unk_token_id; tokenization will be broken. "
            f"unk_token_id={unk_id} code_ids={code_ids}"
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
        train_teacher_forced_metrics_every_n_steps=(
            int(cfg.training.train_teacher_forced_metrics_every_n_steps)
            if cfg.training.get("train_teacher_forced_metrics_every_n_steps") is not None
            else None
        ),
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_cfg = cfg.training.checkpoint
    every_n_train_steps = checkpoint_cfg.get("every_n_train_steps")
    save_weights_only = bool(checkpoint_cfg.get("save_weights_only", False))

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=checkpoint_cfg.monitor,
            mode=checkpoint_cfg.mode,
            save_top_k=int(checkpoint_cfg.save_top_k),
            save_last=bool(checkpoint_cfg.save_last),
            save_weights_only=save_weights_only,
            every_n_train_steps=int(every_n_train_steps) if every_n_train_steps is not None else None,
            filename="vla-step{step:06d}",
            verbose=True,
        ),
    ]

    viz_cfg = cfg.training.validation.get("visualization")
    if viz_cfg and bool(viz_cfg.get("enabled", True)):
        callbacks.append(
            VLASampleVisualizationCallback(
                VLASampleVizConfig(
                    enabled=True,
                    num_samples=int(viz_cfg.get("num_samples", 4)),
                    every_n_val=int(viz_cfg.get("every_n_val", 1)),
                    include_freeform_pred=bool(viz_cfg.get("include_freeform_pred", False)),
                    freeform_max_new_tokens=int(viz_cfg.get("freeform_max_new_tokens", 32)),
                )
            )
        )
    train_viz_cfg = cfg.training.get("train_visualization")
    if train_viz_cfg and bool(train_viz_cfg.get("enabled", True)):
        callbacks.append(
            VLATrainSampleVisualizationCallback(
                VLATrainSampleVizConfig(
                    enabled=True,
                    num_samples=int(train_viz_cfg.get("num_samples", 4)),
                    every_n_steps=int(train_viz_cfg.get("every_n_steps", 500)),
                    include_freeform_pred=bool(
                        train_viz_cfg.get("include_freeform_pred", False)
                    ),
                    freeform_max_new_tokens=int(
                        train_viz_cfg.get("freeform_max_new_tokens", 32)
                    ),
                )
            )
        )
    perf_cfg = cfg.training.get("throughput")
    if perf_cfg and bool(perf_cfg.get("enabled", True)):
        callbacks.append(
            ThroughputLoggingCallback(
                ThroughputLoggingConfig(
                    enabled=True,
                    log_every_n_steps=int(perf_cfg.get("log_every_n_steps", 10)),
                )
            )
        )

    # Progress logging (useful on clusters where tqdm doesn't render nicely in logs).
    # Default: enable on Slurm, disabled for local runs unless explicitly configured.
    progress_cfg = cfg.training.get("progress_logger")
    enable_progress = bool(cfg.cluster.slurm.enabled) if progress_cfg is None else bool(
        progress_cfg.get("enabled", True)
    )
    if enable_progress:
        log_every = 100 if progress_cfg is None else int(progress_cfg.get("log_every_n_steps", 100))
        callbacks.append(ProgressLoggerCallback(log_every_n_steps=log_every))

    # Dataset usage logging (prints how much of each dataset was *actually consumed*).
    # Recommended to align with step-based validation cadence by logging on validation end.
    usage_cfg = cfg.training.get("dataset_usage_logger")
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
    if wandb_logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    else:
        logger.info("WandB disabled; skipping LearningRateMonitor (no logger).")

    # Validation defaults in Lightning run at the end of the (potentially huge) epoch.
    # For short max_steps runs with large IterableDatasets, validation may never run unless
    # we validate every N steps (like Stage 1) and/or limit validation batches.
    trainer_extra_kwargs: dict[str, object] = {}
    val_check_interval = cfg.training.validation.get("check_interval")
    if val_check_interval is not None:
        trainer_extra_kwargs["val_check_interval"] = val_check_interval
    limit_val_batches = cfg.training.validation.get("limit_batches")
    if limit_val_batches is not None:
        trainer_extra_kwargs["limit_val_batches"] = limit_val_batches
    num_sanity_val_steps = cfg.training.validation.get("num_sanity_val_steps")
    if num_sanity_val_steps is not None:
        trainer_extra_kwargs["num_sanity_val_steps"] = int(num_sanity_val_steps)
    overfit_batches = cfg.training.get("overfit_batches")
    if overfit_batches is not None:
        trainer_extra_kwargs["overfit_batches"] = overfit_batches
    limit_train_batches = cfg.training.get("limit_train_batches")
    if limit_train_batches is not None:
        trainer_extra_kwargs["limit_train_batches"] = limit_train_batches

    # Optional profiler (matches Stage 1 conventions).
    profiler = None
    profiler_cfg = cfg.training.get("profiler")
    if profiler_cfg and bool(profiler_cfg.get("enabled", False)):
        profiler_type = str(profiler_cfg.get("type", "simple"))
        dirpath = str(profiler_cfg.get("dirpath", output_dir / "profiles"))
        dirpath_path = Path(dirpath)
        if not dirpath_path.is_absolute():
            # Match Stage 1 behavior: resolve relative profiler paths inside the run directory.
            dirpath_path = output_dir / dirpath_path
        dirpath = str(dirpath_path)
        filename = str(profiler_cfg.get("filename", "profile"))
        if profiler_type == "simple":
            from lightning.pytorch.profilers import SimpleProfiler

            profiler = SimpleProfiler(dirpath=dirpath, filename=filename)
        elif profiler_type == "advanced":
            from lightning.pytorch.profilers import AdvancedProfiler

            profiler = AdvancedProfiler(dirpath=dirpath, filename=filename)
        elif profiler_type == "pytorch":
            from lightning.pytorch.profilers import PyTorchProfiler

            profiler = PyTorchProfiler(
                dirpath=dirpath,
                filename=filename,
                emit_nvtx=False,
                export_to_chrome=True,
                row_limit=20,
            )
        else:
            raise ValueError(f"Unknown profiler type: {profiler_type}")

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_steps=cfg.training.max_steps,
        max_epochs=cfg.training.max_epochs,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=int(cfg.training.get("log_every_n_steps", 10)),
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.training.validation.check_val_every_n_epoch,
        logger=wandb_logger if wandb_logger is not None else False,
        default_root_dir=str(output_dir),
        profiler=profiler,
        **trainer_extra_kwargs,
    )

    ckpt_path = cfg.training.get("resume_from_checkpoint")
    if ckpt_path:
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
