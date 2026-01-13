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
from foundation.constrained_decode import ActionTokenIds, make_prefix_allowed_tokens_fn
from foundation.online_laq import LatentCodeProvider, extract_oxe_language, oxe_frames_to_laq_video
from foundation.vla_inputs import ChatConfig, build_inputs_with_prompt_mask, build_prompt_inputs


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
        action_token_ids: Optional[ActionTokenIds] = None,
    ):
        super().__init__()
        self.vla_model = vla_model
        self.processor = processor
        self.code_provider = code_provider
        self.action_tokens = action_tokens
        self.chat = chat or ChatConfig(system_prompt=None)
        self.optimizer_cfg = optimizer or VLAOptimizerConfig()
        self.action_token_ids = action_token_ids

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

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if isinstance(batch, dict):
            frames = batch["frames"]
            instructions = extract_oxe_language(batch)
        else:
            raise TypeError("Expected dict batch with keys from OXEDataModule (frames, language, ...)")

        video = oxe_frames_to_laq_video(frames)
        codes = self.code_provider.codes_from_video(video)

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

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        if (
            self.action_token_ids is not None
            and hasattr(self.vla_model, "generate")
            and batch_idx == 0
        ):
            acc = self._compute_token_accuracy(frames=frames, instructions=instructions, gt_codes=codes)
            self.log("val/token_accuracy", acc, prog_bar=True, sync_dist=True)

        return loss

    @torch.no_grad()
    def _compute_token_accuracy(
        self, *, frames: torch.Tensor, instructions: list[str], gt_codes: torch.Tensor
    ) -> torch.Tensor:
        images = self.frames_to_images(frames)
        prompt_inputs = build_prompt_inputs(
            processor=self.processor,
            images=images,
            instructions=instructions,
            chat=self.chat,
            device=self.device,
        )

        attention_mask = prompt_inputs.get("attention_mask")
        if attention_mask is None:
            raise KeyError("processor output must include attention_mask")
        prompt_lens = attention_mask.sum(dim=1).to(torch.long)

        token_ids = self.action_token_ids
        assert token_ids is not None
        prefix_fn = make_prefix_allowed_tokens_fn(token_ids)

        max_new = token_ids.code_seq_len + 3  # <ACTION> + codes + </ACTION>
        generated = self.vla_model.generate(
            **prompt_inputs,
            max_new_tokens=max_new,
            do_sample=False,
            prefix_allowed_tokens_fn=prefix_fn,
        )

        # Map code token id -> code index
        code_id_to_index = {tid: i for i, tid in enumerate(token_ids.action_code_ids)}

        correct = 0
        total = 0

        for i in range(generated.shape[0]):
            start = int(prompt_lens[i].item())
            gen_suffix = generated[i, start:].tolist()
            pred_code_ids = [t for t in gen_suffix if t in code_id_to_index]
            pred_codes = [code_id_to_index[t] for t in pred_code_ids[: token_ids.code_seq_len]]

            gt = gt_codes[i].tolist()
            # pad missing predictions with -1
            if len(pred_codes) < len(gt):
                pred_codes = pred_codes + ([-1] * (len(gt) - len(pred_codes)))

            for p, g in zip(pred_codes[: len(gt)], gt, strict=True):
                total += 1
                if p == g:
                    correct += 1

        acc = 0.0 if total == 0 else (correct / total)
        return torch.tensor(acc, device=self.device, dtype=torch.float32)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay
        )
        return optimizer
