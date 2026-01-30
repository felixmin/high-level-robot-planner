"""
Stage 2 (Foundation) LightningModule driven by a VLA backend.

Flow:
1) Take an OXE/OpenX batch containing frames + language.
2) Run frozen LAQ (Stage 1) to produce target codes [B, S].
3) Delegate prompting/masking + LM loss + constrained generation/parsing to the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import torch

from foundation.backends.interfaces import BackendMode, FoundationBatch, VLABackend
from foundation.constrained_decode import ActionTokenIds
from foundation.online_laq import LatentCodeProvider, extract_oxe_language, oxe_frames_to_laq_video


@dataclass
class VLAOptimizerConfig:
    lr: float = 1e-5
    weight_decay: float = 0.01


class VLATokenBackendLightningModule(pl.LightningModule):
    """
    Stage 2 module: image + language -> LAQ code tokens.

    Notes:
    - `code_provider` is frozen and used only to compute supervision.
    - The backend owns model-specific details (processor, chat templating, masking, decoding/parsing).
    """

    def __init__(
        self,
        *,
        backend: VLABackend,
        code_provider: LatentCodeProvider,
        optimizer: VLAOptimizerConfig | None = None,
        action_token_ids: ActionTokenIds | None = None,
        train_teacher_forced_metrics_every_n_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend  # should be an nn.Module so Lightning can optimize it
        self.code_provider = code_provider
        self.optimizer_cfg = optimizer or VLAOptimizerConfig()
        self.action_token_ids = action_token_ids
        self.train_teacher_forced_metrics_every_n_steps = train_teacher_forced_metrics_every_n_steps

        # Stashed for visualization callback (rank0 only reads it).
        self._last_val_sample: dict[str, Any] | None = None

    @property
    def vla_model(self) -> Any:
        return getattr(self.backend, "vla_model", None)

    @property
    def processor(self) -> Any:
        return getattr(self.backend, "processor", None)

    @property
    def action_tokens(self) -> Any:
        cfg = getattr(self.backend, "cfg", None)
        return getattr(cfg, "action_tokens", None)

    @property
    def chat(self) -> Any:
        cfg = getattr(self.backend, "cfg", None)
        return getattr(cfg, "chat", None)

    @property
    def frames_to_images(self) -> Any:
        return getattr(self.backend, "frames_to_images", None)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out = self._loss_from_oxe_batch(batch)
        self.log("train/loss", out.loss, prog_bar=True, sync_dist=True)
        return out.loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out, codes, frames, instructions = self._loss_and_targets_from_oxe_batch(batch)
        self.log("val/loss", out.loss, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            latent = self.backend.latent_from_batch(
                FoundationBatch(frames=frames, instructions=instructions),
                mode=BackendMode.CODES,
            )
            pred = latent.tokens
            if pred is not None:
                gt = codes.to(device=pred.device, dtype=torch.long)
                pred = pred.to(torch.long)
                if pred.shape == gt.shape and pred.numel() > 0:
                    matches = (pred == gt).to(torch.float32)
                    self.log(
                        "val/token_accuracy",
                        matches.mean().to(self.device),
                        prog_bar=True,
                        sync_dist=True,
                    )
                    self.log(
                        "val/sequence_accuracy",
                        matches.all(dim=1).to(torch.float32).mean().to(self.device),
                        prog_bar=False,
                        sync_dist=True,
                    )

            meta = latent.meta if isinstance(latent.meta, dict) else {}
            gen_debug = meta.get("parse_debug") if isinstance(meta.get("parse_debug"), list) else None
            if gen_debug:
                start_frac = torch.tensor(
                    sum(1 for r in gen_debug if isinstance(r, dict) and r.get("has_action_start"))
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                end_frac = torch.tensor(
                    sum(1 for r in gen_debug if isinstance(r, dict) and r.get("has_action_end"))
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                mean_codes = torch.tensor(
                    sum(int(r.get("num_codes_parsed", 0)) for r in gen_debug if isinstance(r, dict))
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                self.log("val/gen_has_action_start_frac", start_frac, prog_bar=False, sync_dist=True)
                self.log("val/gen_has_action_end_frac", end_frac, prog_bar=False, sync_dist=True)
                self.log("val/gen_num_codes_parsed_mean", mean_codes, prog_bar=False, sync_dist=True)

            # Save a small sample for visualization callbacks.
            try:
                max_items = min(64, len(instructions), int(codes.shape[0]))
                episode_id = batch.get("episode_id") if isinstance(batch, dict) else None
                frame_idx = batch.get("frame_idx") if isinstance(batch, dict) else None
                pred_list = (
                    pred[:max_items].detach().cpu().tolist()
                    if isinstance(pred, torch.Tensor)
                    else [[-1] * int(codes.shape[1]) for _ in range(max_items)]
                )
                self._last_val_sample = {
                    "frames": frames[:max_items].detach().cpu(),
                    "instructions": list(instructions[:max_items]),
                    "gt_codes": [row.tolist() for row in codes[:max_items].detach().cpu()],
                    "pred_codes": [list(row) for row in pred_list],
                    "gen_debug": gen_debug[:max_items] if isinstance(gen_debug, list) else None,
                    "episode_id": list(episode_id[:max_items]) if episode_id is not None else None,
                    "frame_idx": list(frame_idx[:max_items]) if frame_idx is not None else None,
                }
            except Exception:
                self._last_val_sample = None

        return out.loss

    def _loss_from_oxe_batch(self, batch: Any):
        out, _codes, _frames, _instructions = self._loss_and_targets_from_oxe_batch(batch)
        return out

    def _loss_and_targets_from_oxe_batch(
        self, batch: Any
    ) -> tuple[Any, torch.Tensor, torch.Tensor, list[str]]:
        if not isinstance(batch, dict):
            raise TypeError("Expected OXE batch dict with keys including 'frames' and 'language'.")

        frames = batch["frames"]
        instructions = extract_oxe_language(batch)

        video = oxe_frames_to_laq_video(frames)
        codes = self.code_provider.codes_from_video(video).to(torch.long).detach().cpu()  # [B, S]

        out = self.backend.loss_from_batch(
            FoundationBatch(frames=frames, instructions=instructions, target_codes=codes),
            mode=BackendMode.CODES,
        )
        return out, codes, frames, instructions

    @torch.no_grad()
    def _predict_freeform_text(
        self,
        *,
        frames: torch.Tensor,
        instructions: list[str],
        max_new_tokens: int = 32,
    ) -> list[str]:
        model = self.vla_model
        processor = self.processor
        chat = self.chat
        frames_to_images = self.frames_to_images
        if model is None or processor is None or chat is None or frames_to_images is None:
            raise RuntimeError("Backend must expose vla_model, processor, chat, and frames_to_images for freeform.")

        from foundation.vla_inputs import build_prompt_inputs

        images = frames_to_images(frames)
        prompt_inputs = build_prompt_inputs(
            processor=processor,
            images=images,
            instructions=instructions,
            chat=chat,
            device=self.device,
        )

        input_ids = prompt_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise TypeError("processor output input_ids must be a 2D tensor")
        prompt_len = int(input_ids.shape[1])

        generated = model.generate(
            **prompt_inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

        tok = getattr(processor, "tokenizer", None)
        decode = getattr(tok, "decode", None) if tok is not None else None
        if decode is None:
            raise TypeError("processor.tokenizer must implement decode(...)")

        texts: list[str] = []
        for i in range(int(generated.shape[0])):
            suffix_ids = generated[i, prompt_len:].tolist()
            texts.append(str(decode(suffix_ids, skip_special_tokens=False)))
        return texts

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.weight_decay,
        )
