from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from foundation.backends.interfaces import FoundationBatch
from foundation.backends.smolvla_shared.config import SmolVLASharedCoreConfig
from foundation.backends.smolvla_shared.flow import (
    create_sinusoidal_pos_embedding,
    make_noisy_target,
    reverse_euler_integration,
    sample_beta_time,
)
from foundation.backends.smolvla_shared.preprocess import (
    get_last_layer_module,
    gpu_preprocess_images,
    infer_hidden_size,
    masked_mean_pool,
)
from foundation.image_adapters import oxe_first_frames_to_pil


class SmolVLASharedCore(torch.nn.Module):
    """Shared SmolVLA-style trunk with latent flow and optional action heads."""

    def __init__(
        self,
        *,
        config: SmolVLASharedCoreConfig,
        vlm: torch.nn.Module | None = None,
        processor: Any | None = None,
        frames_to_images: Callable[[torch.Tensor], list[Any]] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.vlm = vlm
        self.processor = processor
        self.frames_to_images = frames_to_images or oxe_first_frames_to_pil

        self.codebook_size = int(self.cfg.action_tokens.codebook_size)
        self.code_seq_len = int(self.cfg.action_tokens.code_seq_len)

        self.latent_in_proj: torch.nn.Linear | None = None
        self.latent_time_mlp_in: torch.nn.Linear | None = None
        self.latent_time_mlp_out: torch.nn.Linear | None = None
        self.latent_fusion_in: torch.nn.Linear | None = None
        self.latent_fusion_out: torch.nn.Linear | None = None
        self.action_head: torch.nn.Linear | None = None

        self._last_hidden_state: torch.Tensor | None = None
        self._hook_handle: Any = None

        self._vision_model: torch.nn.Module | None = None
        self._connector: torch.nn.Module | None = None
        self._text_model: torch.nn.Module | None = None
        self._text_embeddings: torch.nn.Module | None = None

    def _capture_hidden_state_hook(self, module: torch.nn.Module, input: Any, output: Any) -> None:
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self._last_hidden_state = hidden

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
        self.vlm.train()

        last_layer = get_last_layer_module(self.vlm, optimized=self.cfg.use_gpu_preprocessing)
        if last_layer is not None and self._hook_handle is None:
            self._hook_handle = last_layer.register_forward_hook(self._capture_hidden_state_hook)

        hidden = infer_hidden_size(self.vlm)
        try:
            vlm_dtype = next(self.vlm.parameters()).dtype
        except StopIteration:
            vlm_dtype = self.cfg.torch_dtype

        if self.latent_in_proj is None:
            self.latent_in_proj = torch.nn.Linear(self.cfg.latent_vector_dim, hidden).to(device=device, dtype=vlm_dtype)
            self.latent_time_mlp_in = torch.nn.Linear(hidden * 2, hidden).to(device=device, dtype=vlm_dtype)
            self.latent_time_mlp_out = torch.nn.Linear(hidden, hidden).to(device=device, dtype=vlm_dtype)
            self.latent_fusion_in = torch.nn.Linear(hidden * 2, hidden).to(device=device, dtype=vlm_dtype)
            self.latent_fusion_out = torch.nn.Linear(hidden, self.cfg.latent_vector_dim).to(device=device, dtype=vlm_dtype)

        if self.cfg.action_dim is not None and int(self.cfg.action_dim) > 0 and self.action_head is None:
            self.action_head = torch.nn.Linear(hidden, int(self.cfg.action_dim)).to(device=device, dtype=vlm_dtype)

    def _require_ready(self) -> tuple[torch.device, torch.nn.Module, Any]:
        if self.vlm is None or self.processor is None:
            raise RuntimeError("Core not initialized. Call setup(device=...) first.")
        try:
            device = next(self.vlm.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return device, self.vlm, self.processor

    def _require_latent_head(self) -> tuple[torch.nn.Linear, torch.nn.Linear, torch.nn.Linear, torch.nn.Linear, torch.nn.Linear]:
        if (
            self.latent_in_proj is None
            or self.latent_time_mlp_in is None
            or self.latent_time_mlp_out is None
            or self.latent_fusion_in is None
            or self.latent_fusion_out is None
        ):
            raise RuntimeError("Core latent head not initialized. Call setup(device=...) first.")
        return (
            self.latent_in_proj,
            self.latent_time_mlp_in,
            self.latent_time_mlp_out,
            self.latent_fusion_in,
            self.latent_fusion_out,
        )

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

    def _build_texts_simple(self, instructions: Sequence[str]) -> list[str]:
        sys = self.cfg.chat.system_prompt
        if sys:
            return [f"{sys}\n{instr}" for instr in instructions]
        return [str(instr) for instr in instructions]

    @staticmethod
    def _extract_first_frame(frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"Expected frames with ndim=5, got {frames.ndim}")

        if frames.shape[-1] == 3:
            first = frames[:, 0]
            return first.permute(0, 3, 1, 2)
        if frames.shape[2] == 3:
            return frames[:, 0]
        if frames.shape[1] == 3:
            return frames[:, :, 0]

        raise ValueError(f"Unrecognized frame layout: {tuple(frames.shape)}")

    def _cache_model_components(self, vlm: torch.nn.Module) -> None:
        model = vlm.model if hasattr(vlm, "model") else vlm
        self._vision_model = model.vision_model
        self._connector = model.connector
        self._text_model = model.text_model
        self._text_embeddings = model.text_model.embed_tokens

    def _forward_pooled_original(self, batch: FoundationBatch) -> torch.Tensor:
        device, vlm, processor = self._require_ready()

        images_1 = self.frames_to_images(batch.frames)
        images = [[img] for img in images_1]
        texts = self._build_texts(batch.instructions)

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        if "attention_mask" not in inputs:
            raise KeyError("processor output must include attention_mask")

        self._last_hidden_state = None
        if self._hook_handle is not None:
            _ = vlm(**inputs, output_hidden_states=False, return_dict=True)
            if self._last_hidden_state is None:
                raise RuntimeError("Forward hook did not capture hidden state.")
            last = self._last_hidden_state
        else:
            out = vlm(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = getattr(out, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states.")
            last = hidden_states[-1]

        attn = inputs["attention_mask"].to(dtype=torch.bool)
        return masked_mean_pool(last, attn)

    def _forward_pooled_optimized(self, batch: FoundationBatch) -> torch.Tensor:
        device, vlm, processor = self._require_ready()

        if self._vision_model is None:
            self._cache_model_components(vlm)

        first = self._extract_first_frame(batch.frames)
        batch_size = int(first.shape[0])

        pixel_values = gpu_preprocess_images(
            first.to(device=device),
            target_size=self.cfg.image_size,
            normalize=True,
        ).to(dtype=self.cfg.torch_dtype)

        image_hidden_states = self._vision_model(
            pixel_values=pixel_values.to(dtype=self._vision_model.embeddings.patch_embedding.weight.dtype),
            patch_attention_mask=None,
        ).last_hidden_state

        image_embeds = self._connector(image_hidden_states)
        num_image_tokens = int(image_embeds.shape[1])

        texts = self._build_texts_simple(batch.instructions)
        tokenizer = processor.tokenizer
        text_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = text_inputs["input_ids"].to(device)
        text_attention_mask = text_inputs["attention_mask"].to(device)

        text_embeds = self._text_embeddings(input_ids)

        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        image_attention_mask = torch.ones(batch_size, num_image_tokens, device=device, dtype=torch.bool)
        combined_attention_mask = torch.cat([image_attention_mask, text_attention_mask.bool()], dim=1)
        position_ids = torch.arange(combined_embeds.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)

        self._last_hidden_state = None
        if self._hook_handle is not None:
            outputs = self._text_model(
                inputs_embeds=combined_embeds.to(dtype=self._text_model.embed_tokens.weight.dtype),
                attention_mask=combined_attention_mask,
                position_ids=position_ids,
                output_hidden_states=False,
                return_dict=True,
            )
            last = outputs.last_hidden_state if self._last_hidden_state is None else self._last_hidden_state
        else:
            outputs = self._text_model(
                inputs_embeds=combined_embeds.to(dtype=self._text_model.embed_tokens.weight.dtype),
                attention_mask=combined_attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            last = outputs.last_hidden_state

        return masked_mean_pool(last, combined_attention_mask)

    def encode(self, batch: FoundationBatch) -> torch.Tensor:
        if self.cfg.use_gpu_preprocessing:
            return self._forward_pooled_optimized(batch)
        return self._forward_pooled_original(batch)

    def _predict_velocity(self, pooled: torch.Tensor, x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        latent_in_proj, latent_time_mlp_in, latent_time_mlp_out, latent_fusion_in, latent_fusion_out = self._require_latent_head()

        if pooled.dtype != latent_in_proj.weight.dtype:
            pooled = pooled.to(dtype=latent_in_proj.weight.dtype)
        if x_t.dtype != latent_in_proj.weight.dtype:
            x_t = x_t.to(dtype=latent_in_proj.weight.dtype)

        latent_emb = latent_in_proj(x_t)
        time_emb = create_sinusoidal_pos_embedding(
            time,
            latent_emb.shape[-1],
            self.cfg.min_period,
            self.cfg.max_period,
            device=latent_emb.device,
        ).to(dtype=latent_emb.dtype)

        latent_time = torch.cat([latent_emb, time_emb], dim=-1)
        latent_time = latent_time_mlp_in(latent_time)
        latent_time = F.silu(latent_time)
        latent_time = latent_time_mlp_out(latent_time)

        fused = torch.cat([pooled, latent_time], dim=-1)
        fused = latent_fusion_in(fused)
        fused = F.silu(fused)
        return latent_fusion_out(fused)

    def latent_flow_loss(
        self,
        *,
        batch: FoundationBatch,
        target_vectors: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.encode(batch)

        if target_vectors.ndim != 2:
            raise ValueError(f"Expected target_vectors [B,D], got {tuple(target_vectors.shape)}")

        if pooled.shape[0] != target_vectors.shape[0]:
            raise ValueError("Batch size mismatch between pooled features and target vectors")

        target = target_vectors.to(device=pooled.device, dtype=pooled.dtype)
        if noise is None:
            noise = torch.randn_like(target)
        if time is None:
            time = sample_beta_time(
                batch_size=target.shape[0],
                device=target.device,
                dtype=target.dtype,
                alpha=float(self.cfg.time_beta_alpha),
                beta=float(self.cfg.time_beta_beta),
            )

        x_t, u_t = make_noisy_target(target=target, noise=noise, time=time)
        v_t = self._predict_velocity(pooled=pooled, x_t=x_t, time=time)
        return F.mse_loss(v_t, u_t)

    @torch.no_grad()
    def sample_latent_vectors(
        self,
        *,
        batch: FoundationBatch,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.encode(batch)

        if noise is None:
            noise = torch.randn(
                (pooled.shape[0], int(self.cfg.latent_vector_dim)),
                device=pooled.device,
                dtype=pooled.dtype,
            )

        return reverse_euler_integration(
            initial=noise,
            num_steps=int(self.cfg.flow_steps),
            velocity_fn=lambda x_t, time: self._predict_velocity(pooled=pooled, x_t=x_t, time=time),
        )

    def predict_actions(self, *, batch: FoundationBatch) -> torch.Tensor:
        if self.action_head is None:
            raise RuntimeError("Action head is not initialized.")
        pooled = self.encode(batch)
        if pooled.dtype != self.action_head.weight.dtype:
            pooled = pooled.to(dtype=self.action_head.weight.dtype)
        return self.action_head(pooled)
