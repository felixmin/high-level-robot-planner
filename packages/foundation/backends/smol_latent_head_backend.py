from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import torch
import torch.nn.functional as F

from foundation.action_tokens import ActionTokenConfig
from foundation.backends.interfaces import BackendMode, FoundationBatch, LatentOutput, LossOutput
from foundation.image_adapters import oxe_first_frames_to_pil
from foundation.vla_inputs import ChatConfig


def _masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)


def _infer_hidden_size(model: Any) -> int:
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return int(text_cfg.hidden_size)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    raise AttributeError("Could not infer hidden size from model.config")


@dataclass
class SmolLatentHeadBackendConfig:
    model_name: str
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = False
    chat: ChatConfig = ChatConfig(system_prompt=None)
    action_tokens: ActionTokenConfig = ActionTokenConfig(codebook_size=8, code_seq_len=4)


class SmolLatentHeadBackend(torch.nn.Module):
    """
    SmolVLM latent-head backend (LeRobot `latent_smol`-style latent mode).

    - Conditions on frame t only (first frame) to avoid leakage from (t+Δ).
    - Predicts LAQ codes via CE loss from pooled prefix hidden state.
    """

    def __init__(
        self,
        *,
        config: SmolLatentHeadBackendConfig,
        vlm: torch.nn.Module | None = None,
        processor: Any | None = None,
        frames_to_images: Callable[[torch.Tensor], List[Any]] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.vlm = vlm
        self.processor = processor
        self.frames_to_images = frames_to_images or oxe_first_frames_to_pil

        self.codebook_size = int(self.cfg.action_tokens.codebook_size)
        self.code_seq_len = int(self.cfg.action_tokens.code_seq_len)

        self.laq_head: torch.nn.Linear | None = None

    def setup(self, *, device: torch.device) -> None:
        if self.vlm is None or self.processor is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.vlm = AutoModelForImageTextToText.from_pretrained(
                self.cfg.model_name,
                torch_dtype=self.cfg.torch_dtype,
                trust_remote_code=bool(self.cfg.trust_remote_code),
            )
            self.processor = AutoProcessor.from_pretrained(
                self.cfg.model_name,
                trust_remote_code=bool(self.cfg.trust_remote_code),
            )

        self.vlm.to(device)

        if self.laq_head is None:
            hidden = _infer_hidden_size(self.vlm)
            try:
                vlm_dtype = next(self.vlm.parameters()).dtype
            except StopIteration:
                vlm_dtype = self.cfg.torch_dtype
            self.laq_head = torch.nn.Linear(hidden, self.code_seq_len * self.codebook_size).to(
                device=device, dtype=vlm_dtype
            )

    def _require_ready(self) -> tuple[torch.device, torch.nn.Module, Any, torch.nn.Linear]:
        if self.vlm is None or self.processor is None or self.laq_head is None:
            raise RuntimeError("Backend not initialized. Call setup(device=...) first.")
        try:
            device = next(self.vlm.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return device, self.vlm, self.processor, self.laq_head

    def _build_texts(self, instructions: Sequence[str]) -> list[str]:
        proc = self.processor
        sys = self.cfg.chat.system_prompt

        apply_chat = getattr(proc, "apply_chat_template", None) if proc is not None else None
        if apply_chat is None:
            if sys:
                return [f"{sys}\n{instr}" for instr in instructions]
            return [str(instr) for instr in instructions]

        texts: list[str] = []
        for instr in instructions:
            messages: list[dict[str, Any]] = []
            if sys:
                messages.append({"role": "system", "content": [{"type": "text", "text": str(sys)}]})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},
                        {"type": "text", "text": str(instr)},
                    ],
                }
            )
            texts.append(str(apply_chat(messages, tokenize=False, add_generation_prompt=False)))
        return texts

    def _forward_logits(self, batch: FoundationBatch) -> torch.Tensor:
        device, vlm, processor, head = self._require_ready()

        images_1 = self.frames_to_images(batch.frames)
        # SmolVLMProcessor expects nested images: one sublist per sample.
        images = [[img] for img in images_1]
        texts = self._build_texts(batch.instructions)

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        if "attention_mask" not in inputs:
            raise KeyError("processor output must include attention_mask")

        out = vlm(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states (output_hidden_states=True may be unsupported).")

        last = hidden_states[-1]
        attn = inputs["attention_mask"].to(dtype=torch.bool)
        pooled = _masked_mean_pool(last, attn)
        if pooled.dtype != head.weight.dtype:
            pooled = pooled.to(dtype=head.weight.dtype)
        logits = head(pooled).view(-1, self.code_seq_len, self.codebook_size)
        return logits

    def loss_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LossOutput:
        if mode is not BackendMode.CODES:
            raise NotImplementedError(f"{type(self).__name__} only supports mode={BackendMode.CODES.value!r}")
        if batch.target_codes is None:
            raise ValueError("batch.target_codes is required for latent-head training.")

        logits = self._forward_logits(batch)
        codes = batch.target_codes.to(device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits.reshape(-1, self.codebook_size), codes.reshape(-1))
        return LossOutput(loss=loss, metrics={"loss": float(loss.detach().cpu().item())})

    @torch.no_grad()
    def latent_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LatentOutput:
        if mode is not BackendMode.CODES:
            raise NotImplementedError(f"{type(self).__name__} only supports mode={BackendMode.CODES.value!r}")
        logits = self._forward_logits(batch)
        tokens = logits.argmax(dim=-1)
        return LatentOutput(logits=logits, tokens=tokens, vector=None, meta=None)
