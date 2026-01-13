#!/usr/bin/env python3
"""
Script 4: Train Foundation VLA Model

Train the foundation VLA model (Stage 2): image + language -> latent action tokens.

Usage:
    # Debug (local / single GPU):
    python scripts/4_train_foundation.py experiment=vla_cosmos2_tokens_debug model.laq.checkpoint=/path/to/laq.ckpt
"""

import sys
import logging
from pathlib import Path

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from common.unified_logging import setup_unified_logging
from common.logging import set_seed
from common.data import OXEDataModule

from foundation.action_tokens import ActionTokenConfig
from foundation.constrained_decode import ActionTokenIds
from foundation.online_laq import LAQTaskCodeProvider
from foundation.qwen3vl_setup import prepare_action_token_training
from foundation.vla_inputs import ChatConfig
from foundation.vla_module import VLATokenLightningModule, VLAOptimizerConfig


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Unified logging (mirrors Stage 1)
    use_unified_logging = cfg.logging.get("unified", True)
    if use_unified_logging:
        logger, output_dir = setup_unified_logging(
            workspace_root=workspace_root,
            job_id=None,
            log_level=cfg.logging.get("level", "INFO"),
            capture_stdout=cfg.logging.get("capture_stdout", True),
        )
    else:
        logger = logging.getLogger("foundation.training")
        logging.basicConfig(level=logging.INFO)
        output_dir = Path("./outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("LAPA Stage 2: Foundation VLA Training (Action Tokens)")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    set_seed(int(getattr(cfg, "seed", 42)))

    # Data: OXE streaming frame pairs + language
    if not hasattr(cfg.data, "dataset_name") and not hasattr(cfg.data, "datasets"):
        raise ValueError("Stage 2 currently expects OXE-style data config (dataset_name or datasets).")

    data_config = {k: v for k, v in cfg.data.items() if k not in ["name", "task"]}
    datamodule = OXEDataModule(**data_config)
    datamodule.setup()

    # LAQ: frozen label generator
    laq_ckpt = cfg.model.laq.checkpoint
    if not laq_ckpt:
        raise ValueError("Set `model.laq.checkpoint=/path/to/laq.ckpt` for online LAQ labeling.")
    from laq import LAQTask

    laq_task = LAQTask.load_from_checkpoint(laq_ckpt)
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

    action_cfg = ActionTokenConfig(**OmegaConf.to_container(cfg.model.action_tokens, resolve=True))
    token_id_map = prepare_action_token_training(
        model=vla_model, processor=processor, action_tokens=action_cfg
    )

    action_token_ids = ActionTokenIds(
        action_start_id=token_id_map[action_cfg.action_start],
        action_end_id=token_id_map[action_cfg.action_end],
        action_code_ids=[token_id_map[action_cfg.token_fmt.format(i=i)] for i in range(action_cfg.codebook_size)],
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
            filename="vla-{step}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

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
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
