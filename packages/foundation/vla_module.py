"""
Stage 2 (Foundation) LightningModule for token-based latent-action prediction.

Training loop:
1) Take an OXE/OpenX batch containing frame pairs + language.
2) Run frozen LAQ (Stage 1) to produce discrete codes [B, code_seq_len].
3) Format codes into an action-token completion string.
4) Use Qwen3-VL processor to build multimodal inputs + prompt-masked labels.
5) Optimize standard LM loss on the completion tokens (Approach A).

This module is written to be unit-testable: the VLA model, processor, and code
provider can be injected (use fakes for CPU tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import lightning.pytorch as pl
import torch

from foundation.action_tokens import ActionTokenConfig
from foundation.online_laq import LatentCodeProvider, extract_oxe_language, oxe_frames_to_laq_video
from foundation.vla_inputs import ChatConfig, build_inputs_with_prompt_mask


@dataclass
class VLAOptimizerConfig:
    lr: float = 1e-5
    weight_decay: float = 0.01


class VLATokenLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        vla_model: torch.nn.Module,
        processor: Any,
        code_provider: LatentCodeProvider,
        action_tokens: ActionTokenConfig,
        chat: Optional[ChatConfig] = None,
        optimizer: Optional[VLAOptimizerConfig] = None,
        frames_to_images: Optional[Callable[[torch.Tensor], List[Any]]] = None,
    ):
        super().__init__()
        self.vla_model = vla_model
        self.processor = processor
        self.code_provider = code_provider
        self.action_tokens = action_tokens
        self.chat = chat or ChatConfig(system_prompt=None)
        self.optimizer_cfg = optimizer or VLAOptimizerConfig()

        # Convert OXE batch frames into image objects for the VLM processor.
        # In production this will likely return PIL Images; in tests we can inject a stub.
        self.frames_to_images = frames_to_images or (lambda frames: [object() for _ in range(frames.shape[0])])

        if self.code_provider.codebook_size != self.action_tokens.codebook_size:
            raise ValueError(
                f"LAQ codebook_size ({self.code_provider.codebook_size}) != "
                f"ActionTokenConfig ({self.action_tokens.codebook_size})"
            )
        if self.code_provider.code_seq_len != self.action_tokens.code_seq_len:
            raise ValueError(
                f"LAQ code_seq_len ({self.code_provider.code_seq_len}) != "
                f"ActionTokenConfig ({self.action_tokens.code_seq_len})"
            )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if isinstance(batch, dict):
            frames = batch["frames"]
            instructions = extract_oxe_language(batch)
        else:
            raise TypeError("Expected dict batch with keys from OXEDataModule (frames, language, ...)")

        video = oxe_frames_to_laq_video(frames)
        codes = self.code_provider.codes_from_video(video)  # [B, S]

        if codes.shape[0] != frames.shape[0]:
            raise ValueError("Batch size mismatch between frames and codes")

        targets = [self.action_tokens.format_target(row.tolist()) for row in codes]
        images = self.frames_to_images(frames)

        inputs = build_inputs_with_prompt_mask(
            processor=self.processor,
            images=images,
            instructions=instructions,
            targets=targets,
            chat=self.chat,
            device=self.device,
        )

        outputs = self.vla_model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay
        )
        return optimizer

