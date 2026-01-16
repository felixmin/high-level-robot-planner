# LeRobot Integration Plan for HLRP Foundation VLA

## Overview

This document details the integration of the High-Level Robot Planner (HLRP) Foundation VLA into the LeRobot framework, enabling standardized finetuning and benchmarking.

---

## 1. Architecture Mapping

### HLRP VLA Architecture
```
Input: Image + Language Instruction
    ↓
Qwen3VL (Cosmos-Reason2-2B)
    ↓
Discrete Action Tokens: <ACTION> <ACT_3> <ACT_1> <ACT_7> <ACT_0> </ACTION>
    ↓
LAQ Decoder (frozen)
    ↓
Continuous Actions: [7, action_dim]
```

### LeRobot Policy Interface
```
Input: batch["observation.images.*"], batch["observation.state"], batch["observation.language_tokens"]
    ↓
Policy.forward() → loss
Policy.select_action() → Tensor[batch, action_dim]
    ↓
Continuous Actions to Environment
```

### Integration Strategy

**Key Insight**: HLRP outputs discrete latent codes that must be decoded to continuous actions for LeRobot compatibility.

**Solution**: Create a wrapper policy that:
1. Takes LeRobot batch format as input
2. Runs HLRP VLA to get discrete codes
3. Decodes codes to continuous actions via LAQ decoder
4. Returns continuous actions in LeRobot format

---

## 2. File Structure

```
lerobot/src/lerobot/policies/hlrp_vla/
├── __init__.py
├── configuration_hlrp_vla.py      # LeRobot config wrapping HLRP settings
├── modeling_hlrp_vla.py           # Policy class with VLA + LAQ decoder
├── processor_hlrp_vla.py          # Pre/post processors
└── action_decoder.py              # LAQ code → continuous action decoder
```

**Additionally modify:**
```
lerobot/src/lerobot/policies/factory.py  # Add registration
```

---

## 3. Configuration Design

### `configuration_hlrp_vla.py`

```python
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig, PolicyFeature, FeatureType
from lerobot.optim.optimizers import AdamWConfig

@PreTrainedConfig.register_subclass("hlrp_vla")
@dataclass
class HLRPVLAConfig(PreTrainedConfig):
    """Configuration for HLRP Foundation VLA policy."""

    # ══════════════════════════════════════════════════════════════════
    # Temporal Structure (LeRobot standard)
    # ══════════════════════════════════════════════════════════════════
    n_obs_steps: int = 1                    # Single frame input (HLRP uses frame pairs internally)
    chunk_size: int = 1                     # HLRP predicts 1 action chunk per forward
    n_action_steps: int = 1                 # Execute 1 action per step

    # ══════════════════════════════════════════════════════════════════
    # HLRP VLA Model Settings
    # ══════════════════════════════════════════════════════════════════
    vla_model_name: str = "nvidia/Cosmos-Reason2-2B"
    vla_checkpoint_path: str | None = None  # Path to finetuned VLA checkpoint
    torch_dtype: str = "bf16"               # bf16, fp16, fp32
    attn_implementation: str = "sdpa"

    # ══════════════════════════════════════════════════════════════════
    # Action Token Settings (must match LAQ training)
    # ══════════════════════════════════════════════════════════════════
    codebook_size: int = 8                  # Number of discrete codes
    code_seq_len: int = 4                   # Codes per action
    action_start_token: str = "<ACTION>"
    action_end_token: str = "</ACTION>"
    action_token_fmt: str = "<ACT_{i}>"

    # ══════════════════════════════════════════════════════════════════
    # LAQ Decoder Settings
    # ══════════════════════════════════════════════════════════════════
    laq_checkpoint_path: str = ""           # Required: path to LAQ checkpoint
    laq_decoder_type: str = "learned"       # "learned" (full LAQ) or "codebook_lookup"

    # ══════════════════════════════════════════════════════════════════
    # Image Processing
    # ══════════════════════════════════════════════════════════════════
    image_size: tuple[int, int] = (224, 224)  # Resize target for VLA

    # ══════════════════════════════════════════════════════════════════
    # State/Action Dimensions (set from dataset)
    # ══════════════════════════════════════════════════════════════════
    max_state_dim: int = 32
    max_action_dim: int = 32

    # ══════════════════════════════════════════════════════════════════
    # Chat Template
    # ══════════════════════════════════════════════════════════════════
    system_prompt: str = "You are a robot policy. Reply only with action tokens."

    # ══════════════════════════════════════════════════════════════════
    # Training Settings
    # ══════════════════════════════════════════════════════════════════
    freeze_vla_backbone: bool = False       # Freeze vision-language model
    freeze_laq_decoder: bool = True         # Always freeze LAQ decoder

    # Optimizer defaults
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 0.01
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler
    use_scheduler: bool = True
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-6

    # ══════════════════════════════════════════════════════════════════
    # Normalization (LeRobot standard)
    # ══════════════════════════════════════════════════════════════════
    normalization_mapping: dict = field(default_factory=lambda: {
        "VISUAL": "IDENTITY",       # Images normalized in processor
        "STATE": "MEAN_STD",        # Proprioceptive state
        "ACTION": "MEAN_STD",       # Continuous actions
    })

    # ══════════════════════════════════════════════════════════════════
    # Abstract Property Implementations
    # ══════════════════════════════════════════════════════════════════
    @property
    def observation_delta_indices(self) -> list | None:
        return None  # HLRP doesn't use delta observations

    @property
    def action_delta_indices(self) -> list | None:
        return None  # HLRP doesn't use delta actions

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def validate_features(self) -> None:
        """Validate input/output features are compatible with HLRP VLA."""
        # Require at least one image input
        if not self.image_features:
            raise ValueError("HLRP VLA requires at least one visual input feature")

        # Validate action dimension
        if self.action_feature is not None:
            action_dim = self.action_feature.shape[0]
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}"
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        if not self.use_scheduler:
            return None
        from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
```

---

## 4. Policy Implementation

### `modeling_hlrp_vla.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
from collections import deque
from typing import Any
from pathlib import Path

from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_hlrp_vla import HLRPVLAConfig
from .action_decoder import LAQActionDecoder


class HLRPVLAPolicy(PreTrainedPolicy):
    """
    LeRobot policy wrapper for HLRP Foundation VLA.

    Architecture:
        1. Qwen3VL (Cosmos-Reason2) generates discrete action tokens
        2. LAQ decoder converts discrete codes to continuous actions
    """

    config_class = HLRPVLAConfig
    name = "hlrp_vla"

    def __init__(self, config: HLRPVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # ══════════════════════════════════════════════════════════════
        # Load VLA Model (Qwen3VL / Cosmos-Reason2)
        # ══════════════════════════════════════════════════════════════
        dtype = self._get_torch_dtype(config.torch_dtype)

        self.processor = Qwen3VLProcessor.from_pretrained(config.vla_model_name)
        self.vla_model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vla_model_name,
            torch_dtype=dtype,
            attn_implementation=config.attn_implementation,
        )

        # Load finetuned weights if provided
        if config.vla_checkpoint_path:
            self._load_vla_checkpoint(config.vla_checkpoint_path)

        # ══════════════════════════════════════════════════════════════
        # Register Action Tokens
        # ══════════════════════════════════════════════════════════════
        self._setup_action_tokens()

        # ══════════════════════════════════════════════════════════════
        # Load LAQ Decoder (for code → continuous action)
        # ══════════════════════════════════════════════════════════════
        if not config.laq_checkpoint_path:
            raise ValueError("laq_checkpoint_path is required for action decoding")

        self.action_decoder = LAQActionDecoder.from_checkpoint(
            config.laq_checkpoint_path,
            decoder_type=config.laq_decoder_type,
        )
        self.action_decoder.eval()
        for p in self.action_decoder.parameters():
            p.requires_grad = False

        # ══════════════════════════════════════════════════════════════
        # Freezing Strategy
        # ══════════════════════════════════════════════════════════════
        if config.freeze_vla_backbone:
            for p in self.vla_model.parameters():
                p.requires_grad = False
            # Unfreeze only the new action token embeddings
            # (they're at the end of the embedding table)
            emb = self.vla_model.get_input_embeddings()
            n_action_tokens = config.codebook_size + 2  # codes + start/end
            emb.weight[-n_action_tokens:].requires_grad = True

        # ══════════════════════════════════════════════════════════════
        # Action Queue for Temporal Consistency
        # ══════════════════════════════════════════════════════════════
        self.reset()

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

    def _setup_action_tokens(self):
        """Add action tokens to tokenizer and resize embeddings."""
        cfg = self.config

        # Build token list
        tokens = [cfg.action_start_token, cfg.action_end_token]
        tokens += [cfg.action_token_fmt.format(i=i) for i in range(cfg.codebook_size)]

        # Add to tokenizer
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self.vla_model.resize_token_embeddings(len(self.processor.tokenizer))

        # Cache token IDs for constrained decoding
        self.action_start_id = self.processor.tokenizer.convert_tokens_to_ids(cfg.action_start_token)
        self.action_end_id = self.processor.tokenizer.convert_tokens_to_ids(cfg.action_end_token)
        self.action_code_ids = [
            self.processor.tokenizer.convert_tokens_to_ids(cfg.action_token_fmt.format(i=i))
            for i in range(cfg.codebook_size)
        ]
        self.eos_token_id = self.processor.tokenizer.eos_token_id

    def _load_vla_checkpoint(self, path: str):
        """Load finetuned VLA weights."""
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Filter to VLA model keys
        vla_state = {k.replace("vla_model.", ""): v for k, v in state_dict.items()
                     if k.startswith("vla_model.")}
        self.vla_model.load_state_dict(vla_state, strict=False)

    # ══════════════════════════════════════════════════════════════════════
    # LeRobot Interface Implementation
    # ══════════════════════════════════════════════════════════════════════

    def reset(self):
        """Called on environment reset."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._prev_frame = None  # For frame pair construction

    def get_optim_params(self) -> dict:
        """Return parameters for optimizer."""
        if self.config.freeze_vla_backbone:
            # Only train action token embeddings
            emb = self.vla_model.get_input_embeddings()
            n_action_tokens = self.config.codebook_size + 2
            return [{"params": [emb.weight[-n_action_tokens:]]}]
        else:
            # Train everything
            return [{"params": [p for p in self.vla_model.parameters() if p.requires_grad]}]

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """
        Training forward pass.

        Args:
            batch: LeRobot batch with images, state, actions, language

        Returns:
            loss: Scalar loss tensor
            info: Optional logging dict
        """
        # Prepare inputs
        images = self._extract_images(batch)
        instructions = self._extract_language(batch)
        target_actions = batch[ACTION]  # (B, chunk_size, action_dim)

        # Encode target actions to discrete codes via LAQ
        with torch.no_grad():
            target_codes = self.action_decoder.encode(target_actions)  # (B, code_seq_len)

        # Format target codes as token strings
        target_strings = self._format_codes_as_tokens(target_codes)

        # Build inputs with prompt masking
        inputs = self._build_training_inputs(images, instructions, target_strings)

        # Forward through VLA
        outputs = self.vla_model(**inputs)
        loss = outputs.loss

        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Predict action chunk at inference time.

        Returns:
            actions: (batch_size, chunk_size, action_dim) continuous actions
        """
        self.eval()

        images = self._extract_images(batch)
        instructions = self._extract_language(batch)

        # Build prompt-only inputs
        prompt_inputs = self._build_inference_inputs(images, instructions)

        # Generate with constrained decoding
        generated = self.vla_model.generate(
            **prompt_inputs,
            max_new_tokens=self.config.code_seq_len + 3,  # <ACTION> + codes + </ACTION>
            do_sample=False,
            prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn,
        )

        # Extract discrete codes from generated tokens
        codes = self._extract_codes_from_generation(generated, prompt_inputs)

        # Decode codes to continuous actions
        actions = self.action_decoder.decode(codes)  # (B, action_dim)

        # Reshape to (B, 1, action_dim) for chunk format
        return actions.unsqueeze(1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Select single action for environment step.

        Returns:
            action: (batch_size, action_dim) single action
        """
        self.eval()

        # Use cached action if available
        if len(self._action_queue) > 0:
            return self._action_queue.popleft()

        # Predict new chunk
        chunk = self.predict_action_chunk(batch)  # (B, chunk_size, action_dim)

        # Enqueue actions
        for t in range(chunk.shape[1]):
            self._action_queue.append(chunk[:, t, :])

        return self._action_queue.popleft()

    # ══════════════════════════════════════════════════════════════════════
    # Helper Methods
    # ══════════════════════════════════════════════════════════════════════

    def _extract_images(self, batch: dict) -> list:
        """Extract and preprocess images from batch."""
        images = []
        for key in self.config.image_features:
            img = batch[key]
            if img.ndim == 5:  # (B, T, C, H, W)
                img = img[:, -1]  # Take latest frame
            images.append(img)
        return images

    def _extract_language(self, batch: dict) -> list[str]:
        """Extract language instructions from batch."""
        # Try different possible keys
        for key in ["observation.language", "task", "instruction"]:
            if key in batch:
                return batch[key]
        return ["Pick up the object."] * len(batch[next(iter(batch))])

    def _format_codes_as_tokens(self, codes: torch.Tensor) -> list[str]:
        """Format discrete codes as action token strings."""
        cfg = self.config
        results = []
        for row in codes:
            tokens = [cfg.action_start_token]
            tokens += [cfg.action_token_fmt.format(i=int(c)) for c in row]
            tokens += [cfg.action_end_token]
            results.append(" ".join(tokens))
        return results

    def _build_training_inputs(self, images, instructions, targets):
        """Build training inputs with prompt masking."""
        # Build chat messages
        messages_batch = []
        for img, instr, tgt in zip(images, instructions, targets):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.config.system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": instr}]},
                {"role": "assistant", "content": [{"type": "text", "text": tgt}]},
            ]
            messages_batch.append(messages)

        # Process with tokenizer
        texts = [self.processor.apply_chat_template(m, tokenize=False) for m in messages_batch]
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # Create labels with prompt masking
        # (mask everything except the assistant's action tokens)
        labels = inputs["input_ids"].clone()
        # ... (detailed masking logic)

        inputs["labels"] = labels
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _build_inference_inputs(self, images, instructions):
        """Build inference inputs (prompt only, no target)."""
        messages_batch = []
        for img, instr in zip(images, instructions):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.config.system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": instr}]},
            ]
            messages_batch.append(messages)

        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages_batch]
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _prefix_allowed_tokens_fn(self, batch_id: int, input_ids: torch.Tensor) -> list[int]:
        """Constrained decoding: only allow valid action token sequences."""
        generated = input_ids.tolist()

        # Find where generation started (after prompt)
        # Count action tokens generated so far
        n_codes = sum(1 for t in generated if t in self.action_code_ids)
        has_start = self.action_start_id in generated
        has_end = self.action_end_id in generated

        if not has_start:
            return [self.action_start_id]
        elif n_codes < self.config.code_seq_len:
            return self.action_code_ids
        elif not has_end:
            return [self.action_end_id]
        else:
            return [self.eos_token_id]

    def _extract_codes_from_generation(self, generated, prompt_inputs) -> torch.Tensor:
        """Extract discrete code indices from generated token sequence."""
        batch_size = generated.shape[0]
        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)

        code_id_to_idx = {tid: i for i, tid in enumerate(self.action_code_ids)}

        results = []
        for b in range(batch_size):
            start = int(prompt_lens[b])
            gen_tokens = generated[b, start:].tolist()
            codes = [code_id_to_idx[t] for t in gen_tokens if t in code_id_to_idx]
            codes = codes[:self.config.code_seq_len]  # Truncate to expected length
            # Pad if needed
            while len(codes) < self.config.code_seq_len:
                codes.append(0)
            results.append(codes)

        return torch.tensor(results, device=self.device, dtype=torch.long)
```

---

## 5. LAQ Action Decoder

### `action_decoder.py`

The LAQ decoder converts discrete codes back to continuous actions. Two approaches:

**Option A: Full LAQ Decoder (Learned)**
- Uses the LAQ model's decoder path
- More accurate, handles temporal context

**Option B: Codebook Lookup (Simple)**
- Direct lookup in codebook embeddings
- Linear projection to action space
- Faster but less expressive

```python
import torch
import torch.nn as nn
from pathlib import Path


class LAQActionDecoder(nn.Module):
    """
    Decodes discrete LAQ codes to continuous robot actions.

    This bridges the gap between HLRP's discrete action tokens
    and LeRobot's continuous action interface.
    """

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, decoder_type: str = "learned"):
        """Load decoder from LAQ checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Extract config and state dict
        config = ckpt.get("hyper_parameters", {}).get("model", {})
        state_dict = ckpt.get("state_dict", ckpt)

        decoder = cls(
            codebook_size=config.get("codebook_size", 8),
            code_seq_len=config.get("code_seq_len", 4),
            embedding_dim=config.get("quant_dim", 32),
            action_dim=config.get("action_dim", 7),
            decoder_type=decoder_type,
        )

        # Load relevant weights
        decoder.load_laq_weights(state_dict)
        return decoder

    def __init__(
        self,
        codebook_size: int = 8,
        code_seq_len: int = 4,
        embedding_dim: int = 32,
        action_dim: int = 7,
        decoder_type: str = "learned",
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_seq_len = code_seq_len
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.decoder_type = decoder_type

        # Codebook embeddings (loaded from LAQ)
        self.codebook = nn.Embedding(codebook_size, embedding_dim)

        if decoder_type == "learned":
            # Full decoder MLP
            hidden_dim = embedding_dim * code_seq_len
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, action_dim),
            )
        else:
            # Simple linear projection
            self.decoder = nn.Linear(embedding_dim * code_seq_len, action_dim)

    def load_laq_weights(self, state_dict: dict):
        """Load codebook weights from LAQ checkpoint."""
        # Find codebook weights (naming varies)
        for key in ["model.vq.codebooks", "vq.codebooks", "codebooks"]:
            if key in state_dict:
                self.codebook.weight.data = state_dict[key].squeeze()
                break

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous actions to discrete codes.

        NOTE: This requires the full LAQ encoder, which is expensive.
        For training, we typically get codes from the dataset directly.

        Args:
            actions: (B, action_dim) or (B, T, action_dim)

        Returns:
            codes: (B, code_seq_len) discrete indices
        """
        raise NotImplementedError(
            "Action encoding requires the full LAQ model. "
            "Use pre-computed codes from the dataset instead."
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes to continuous actions.

        Args:
            codes: (B, code_seq_len) discrete indices in [0, codebook_size)

        Returns:
            actions: (B, action_dim) continuous actions
        """
        # Lookup embeddings
        embeddings = self.codebook(codes)  # (B, code_seq_len, embedding_dim)

        # Flatten code sequence
        flat = embeddings.view(embeddings.shape[0], -1)  # (B, code_seq_len * embedding_dim)

        # Decode to continuous actions
        actions = self.decoder(flat)  # (B, action_dim)

        return actions
```

---

## 6. Processor Pipeline

### `processor_hlrp_vla.py`

```python
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.processor.steps import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from .configuration_hlrp_vla import HLRPVLAConfig


def make_hlrp_vla_pre_post_processors(
    config: HLRPVLAConfig,
    dataset_stats: dict | None = None,
):
    """
    Create preprocessor and postprocessor pipelines for HLRP VLA.

    Preprocessor: raw observations → policy input format
    Postprocessor: policy output → environment action format
    """

    # ══════════════════════════════════════════════════════════════════
    # Preprocessor Steps
    # ══════════════════════════════════════════════════════════════════
    pre_steps = [
        # Add batch dimension for single samples
        AddBatchDimensionProcessorStep(),

        # Normalize inputs using dataset statistics
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),

        # Move to device
        DeviceProcessorStep(device=config.device),
    ]

    # ══════════════════════════════════════════════════════════════════
    # Postprocessor Steps
    # ══════════════════════════════════════════════════════════════════
    post_steps = [
        # Unnormalize actions back to original scale
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),

        # Move to CPU for environment
        DeviceProcessorStep(device="cpu"),
    ]

    preprocessor = PolicyProcessorPipeline(steps=pre_steps, name="hlrp_vla_preprocessor")
    postprocessor = PolicyProcessorPipeline(steps=post_steps, name="hlrp_vla_postprocessor")

    return preprocessor, postprocessor
```

---

## 7. Factory Registration

### Modify `lerobot/src/lerobot/policies/factory.py`

```python
# Add import at top
from lerobot.policies.hlrp_vla.configuration_hlrp_vla import HLRPVLAConfig

# In get_policy_class()
def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    if name == "hlrp_vla":
        from lerobot.policies.hlrp_vla.modeling_hlrp_vla import HLRPVLAPolicy
        return HLRPVLAPolicy
    # ... existing cases ...

# In make_policy_config()
def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "hlrp_vla":
        return HLRPVLAConfig(**kwargs)
    # ... existing cases ...

# In make_pre_post_processors()
def make_pre_post_processors(policy_cfg, ...):
    # ... existing code ...
    elif isinstance(policy_cfg, HLRPVLAConfig):
        from lerobot.policies.hlrp_vla.processor_hlrp_vla import make_hlrp_vla_pre_post_processors
        return make_hlrp_vla_pre_post_processors(config=policy_cfg, dataset_stats=dataset_stats)
```

---

## 8. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING FLOW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LeRobot Dataset                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ observation.images.cam0: (B, H, W, 3)                            │   │
│  │ observation.state: (B, state_dim)                                │   │
│  │ action: (B, action_dim)  ← continuous robot actions              │   │
│  │ task: ["Pick up red block", ...]                                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    HLRPVLAPolicy.forward()                        │   │
│  │                                                                   │   │
│  │  1. Extract images, language                                     │   │
│  │  2. Encode target actions → discrete codes (LAQ encoder)         │   │
│  │  3. Format codes as tokens: "<ACTION> <ACT_3> <ACT_1> ..."       │   │
│  │  4. Build VLM inputs with prompt masking                         │   │
│  │  5. Forward through Qwen3VL → cross-entropy loss                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│                          loss.backward()                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE FLOW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Environment Observation                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ observation.images.cam0: (1, H, W, 3)                            │   │
│  │ observation.state: (1, state_dim)                                │   │
│  │ task: "Pick up red block"                                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                HLRPVLAPolicy.select_action()                      │   │
│  │                                                                   │   │
│  │  1. Build prompt inputs (image + language)                       │   │
│  │  2. Generate with constrained decoding                           │   │
│  │     → "<ACTION> <ACT_3> <ACT_1> <ACT_7> <ACT_0> </ACTION>"       │   │
│  │  3. Extract discrete codes: [3, 1, 7, 0]                         │   │
│  │  4. Decode via LAQ: codes → (1, action_dim) continuous           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│                   action: (1, 7) e.g. [dx, dy, dz, drx, dry, drz, grip] │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Key Implementation Challenges

### Challenge 1: Discrete → Continuous Action Mapping

**Problem**: HLRP outputs discrete tokens, LeRobot expects continuous actions.

**Solution**: LAQ decoder that maps codebook indices → continuous actions.

**Training Requirement**: Need a trained LAQ decoder or learn it during finetuning.

### Challenge 2: Frame Pair Requirement

**Problem**: HLRP VLA was trained on frame pairs (t, t+k) for latent action learning.

**Solution Options**:
1. **Buffer previous frame**: Store last observation, construct pair
2. **Single frame mode**: Finetune VLA to work with single frames
3. **Temporal stacking**: Use `n_obs_steps=2` in LeRobot config

**Recommended**: Option 1 (buffer) for evaluation, Option 3 for training.

### Challenge 3: Language Input Format

**Problem**: LeRobot uses tokenized language, HLRP uses raw strings with chat template.

**Solution**: Extract raw strings from batch or decode tokens back to text.

### Challenge 4: Action Encoding for Training

**Problem**: LeRobot provides continuous actions, but VLA trains on discrete tokens.

**Solution Options**:
1. **Online LAQ encoding**: Run LAQ encoder on target actions (expensive)
2. **Pre-computed codes**: Store discrete codes in dataset alongside continuous actions
3. **Learned encoder**: Train lightweight encoder during finetuning

**Recommended**: Option 2 (pre-compute) for efficiency.

---

## 10. Usage Examples

### Finetuning on LeRobot Dataset

```bash
lerobot-train \
    --policy.type=hlrp_vla \
    --policy.vla_checkpoint_path=/path/to/hlrp_vla_checkpoint.ckpt \
    --policy.laq_checkpoint_path=/path/to/laq_checkpoint.ckpt \
    --policy.freeze_vla_backbone=true \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --batch_size=32 \
    --steps=50000 \
    --save_freq=5000
```

### Evaluation on Benchmark

```bash
lerobot-eval \
    --policy.path=outputs/hlrp_vla/checkpoints/last \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0 \
    --eval.n_episodes=50 \
    --eval.batch_size=10
```

### Python API

```python
from lerobot.policies.hlrp_vla import HLRPVLAPolicy, HLRPVLAConfig

# Load pretrained
policy = HLRPVLAPolicy.from_pretrained("your-hub-id/hlrp_vla_aloha")

# Or create from config
config = HLRPVLAConfig(
    vla_checkpoint_path="/path/to/vla.ckpt",
    laq_checkpoint_path="/path/to/laq.ckpt",
)
policy = HLRPVLAPolicy(config)

# Inference
obs = env.reset()
action = policy.select_action(preprocess(obs))
```

---

## 11. Testing Strategy

### Unit Tests

```python
# tests/policies/test_hlrp_vla.py

def test_config_validation():
    """Test config validates features correctly."""
    config = HLRPVLAConfig(...)
    config.validate_features()

def test_action_token_setup():
    """Test action tokens are added to tokenizer."""
    policy = HLRPVLAPolicy(config)
    assert policy.action_start_id is not None
    assert len(policy.action_code_ids) == config.codebook_size

def test_constrained_decoding():
    """Test generation only produces valid action sequences."""
    # ...

def test_forward_loss():
    """Test training forward pass returns valid loss."""
    batch = create_dummy_batch()
    loss, info = policy.forward(batch)
    assert loss.requires_grad
    assert loss.item() > 0

def test_select_action_shape():
    """Test inference returns correct action shape."""
    batch = create_dummy_batch()
    action = policy.select_action(batch)
    assert action.shape == (1, config.action_feature.shape[0])
```

### Integration Tests

```python
def test_lerobot_training_loop():
    """Test policy works with LeRobot training loop."""
    # Load small dataset
    # Run 10 training steps
    # Verify loss decreases

def test_lerobot_eval_rollout():
    """Test policy works with LeRobot evaluation."""
    # Create mock environment
    # Run 1 episode rollout
    # Verify actions are valid
```

---

## 12. Implementation Order

1. **Phase 1: Core Policy** (Priority: High)
   - [ ] `configuration_hlrp_vla.py` - Config class
   - [ ] `action_decoder.py` - LAQ code decoder
   - [ ] `modeling_hlrp_vla.py` - Policy class (inference only first)

2. **Phase 2: Training Support** (Priority: High)
   - [ ] Add `forward()` method with loss computation
   - [ ] Implement prompt masking for action tokens
   - [ ] Add freezing/unfreezing logic

3. **Phase 3: Integration** (Priority: Medium)
   - [ ] `processor_hlrp_vla.py` - Pre/post processors
   - [ ] Factory registration
   - [ ] Unit tests

4. **Phase 4: Validation** (Priority: Medium)
   - [ ] Integration tests with LeRobot training
   - [ ] Benchmark evaluation (PushT, Aloha)
   - [ ] Performance profiling

5. **Phase 5: Polish** (Priority: Low)
   - [ ] Hub integration (save/load)
   - [ ] Documentation
   - [ ] Example notebooks

---

## 13. Open Questions

1. **LAQ Decoder Training**: Should the LAQ decoder be finetuned during LeRobot training, or kept frozen?
   - Recommendation: Frozen initially, optional unfreezing for advanced users

2. **Action Encoding**: How to encode continuous actions to discrete codes for training?
   - Option A: Pre-compute and store in dataset (recommended)
   - Option B: Online encoding with frozen LAQ encoder (expensive)

3. **Frame Pair Handling**: How to handle HLRP's frame pair requirement?
   - Option A: Buffer previous frame (recommended for eval)
   - Option B: Finetune for single-frame input

4. **Chunk Size**: HLRP predicts 1 action per forward, but action chunking helps smoothness.
   - Option: Run VLA multiple times to build chunk, or modify VLA for chunk prediction

---

## Summary

This plan provides a complete pathway to integrate the HLRP Foundation VLA into LeRobot:

1. **Wrapper policy** that implements LeRobot's `PreTrainedPolicy` interface
2. **LAQ decoder** that converts discrete action tokens to continuous actions
3. **Constrained decoding** ensuring valid action token sequences
4. **Full training support** with prompt masking and flexible freezing
5. **Standard LeRobot workflow** compatibility (train, eval, hub)

The key insight is treating the VLA as a **discrete action predictor** and the LAQ decoder as a **discretization-to-continuous bridge**, making the combined system compatible with LeRobot's continuous action interface.

---

## 14. Alternative: Hierarchical Architecture with Diffusion Policy

This section explores an alternative architecture where the HLRP VLA provides **high-level latent conditioning** for a **low-level diffusion policy**.

### 14.1 Motivation

Instead of decoding discrete latent codes directly to continuous actions via a learned MLP, we can use the latent codes to **condition a diffusion policy**. This provides:

| Benefit | Explanation |
|---------|-------------|
| **Smoother actions** | Diffusion generates temporally coherent action trajectories |
| **Hierarchical decomposition** | VLA handles "what to do" (semantic), diffusion handles "how to do it" (motor) |
| **Language grounding** | Latent codes carry language intent from VLA to low-level policy |
| **Temporal abstraction** | 4 latent codes can represent a multi-step plan that diffusion executes |
| **Action chunking** | Diffusion naturally predicts action sequences (horizon > 1) |

### 14.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Hierarchical HLRP + Diffusion                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    HIGH-LEVEL: HLRP VLA (Stage 2)                   │ │
│  │                                                                      │ │
│  │  Image + Language ──► Qwen3VL ──► Discrete Codes [B, code_seq_len] │ │
│  │                                   e.g., [3, 1, 7, 0]                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│                        Latent Code Embedding                             │
│                        nn.Embedding(codebook_size, embed_dim)           │
│                                    │                                     │
│                                    ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │               LOW-LEVEL: Latent-Conditioned Diffusion               │ │
│  │                                                                      │ │
│  │  global_cond = concat([                                             │ │
│  │      image_features,        # ResNet + SpatialSoftmax              │ │
│  │      robot_state,           # Joint positions, etc.                 │ │
│  │      latent_embedding,      # From VLA output ← NEW                 │ │
│  │  ])                                                                 │ │
│  │                                                                      │ │
│  │  Diffusion UNet + FiLM conditioning ──► Actions [B, horizon, dim]  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 14.3 Language Conditioning in LeRobot Policies

**Key Finding**: LeRobot's vanilla Diffusion Policy does **NOT** have language conditioning out of the box.

| Policy | Language Support | How |
|--------|-----------------|-----|
| **DiffusionPolicy** | ❌ No | Only state + images |
| **SmolVLA** | ✅ Yes | `observation.language.tokens` via VLM |
| **Pi0 / Pi0_fast** | ✅ Yes | PaliGemma embeds language |
| **Pi05** | ✅ Yes | Same as Pi0 |
| **Groot** | ✅ Yes | Eagle2 VLM processes language |

**Implication**: For language-conditioned tasks (like LIBERO), you need either:
1. A VLA-style policy (SmolVLA, Pi0, etc.)
2. A modified Diffusion Policy with language conditioning
3. **Our approach**: Use HLRP latent codes AS the language conditioning

### 14.4 Why Latent Codes Are Better Than Raw Language

Using HLRP's latent codes instead of raw language tokens provides advantages:

```
Traditional:    "Pick up red block" ──► Tokenizer ──► [variable length tokens] ──► Policy

HLRP Approach:  "Pick up red block" + Image ──► VLA ──► [3, 1, 7, 0] (fixed 4 codes)
                                                              │
                     Already grounded ◄──────────────────────┘
                     in visual context
```

| Aspect | Raw Language Tokens | HLRP Latent Codes |
|--------|---------------------|-------------------|
| **Length** | Variable (5-50+ tokens) | Fixed (4 codes) |
| **Grounded?** | No (text only) | Yes (image + text → code) |
| **Semantic compression** | Low | High |
| **Temporal structure** | None | Implicit sub-goals |
| **Vocabulary size** | 32k-100k | 8 (codebook_size) |

### 14.5 Implementation: LatentConditionedDiffusionPolicy

#### Configuration

```python
@PreTrainedConfig.register_subclass("hlrp_diffusion")
@dataclass
class HLRPDiffusionConfig(DiffusionConfig):
    """Diffusion policy conditioned on HLRP latent codes."""

    # HLRP VLA settings
    vla_model_name: str = "nvidia/Cosmos-Reason2-2B"
    vla_checkpoint_path: str | None = None
    laq_checkpoint_path: str = ""  # For encoding during training

    # Latent conditioning
    codebook_size: int = 8
    code_seq_len: int = 4
    latent_embed_dim: int = 64  # Embedding dimension per code

    # Freeze VLA during diffusion training
    freeze_vla: bool = True

    # Inherited from DiffusionConfig
    # horizon, n_obs_steps, n_action_steps, etc.
```

#### Model Architecture

```python
class HLRPDiffusionPolicy(PreTrainedPolicy):
    """
    Hierarchical policy: HLRP VLA for latents + Diffusion for actions.
    """

    config_class = HLRPDiffusionConfig
    name = "hlrp_diffusion"

    def __init__(self, config: HLRPDiffusionConfig, **kwargs):
        super().__init__(config)

        # ══════════════════════════════════════════════════════════════
        # High-Level: HLRP VLA (frozen by default)
        # ══════════════════════════════════════════════════════════════
        self.vla = self._load_hlrp_vla(config)
        if config.freeze_vla:
            for p in self.vla.parameters():
                p.requires_grad = False

        # Latent code embedding
        self.latent_embedding = nn.Embedding(
            config.codebook_size,
            config.latent_embed_dim
        )

        # ══════════════════════════════════════════════════════════════
        # Low-Level: Modified Diffusion Model
        # ══════════════════════════════════════════════════════════════

        # Calculate global conditioning dimension
        # Original: state_dim + image_features
        # New: state_dim + image_features + latent_embed_dim * code_seq_len
        latent_cond_dim = config.latent_embed_dim * config.code_seq_len

        self.diffusion = LatentConditionedDiffusionModel(
            config,
            extra_cond_dim=latent_cond_dim
        )

        self.reset()

    def _load_hlrp_vla(self, config):
        """Load HLRP VLA model."""
        # Import from HLRP packages
        from foundation.vla_module import VLATokenLightningModule
        # ... load and return

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass."""

        # Get latent codes from VLA (or use pre-computed from dataset)
        with torch.no_grad() if self.config.freeze_vla else nullcontext():
            latent_codes = self._get_latent_codes(batch)  # [B, code_seq_len]

        # Embed latent codes
        latent_embeds = self.latent_embedding(latent_codes)  # [B, code_seq_len, embed_dim]
        latent_cond = latent_embeds.flatten(1)  # [B, code_seq_len * embed_dim]

        # Add to batch for diffusion conditioning
        batch["latent_conditioning"] = latent_cond

        # Compute diffusion loss
        loss = self.diffusion.compute_loss(batch)

        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Inference: VLA → latent codes → Diffusion → actions."""

        # Action queue for temporal consistency
        if len(self._action_queue) > 0:
            return self._action_queue.popleft()

        # Get latent codes from VLA
        latent_codes = self._get_latent_codes(batch)
        latent_embeds = self.latent_embedding(latent_codes)
        latent_cond = latent_embeds.flatten(1)

        # Add to batch
        batch["latent_conditioning"] = latent_cond

        # Generate actions via diffusion
        actions = self.diffusion.generate_actions(batch)  # [B, horizon, action_dim]

        # Enqueue actions
        for t in range(min(self.config.n_action_steps, actions.shape[1])):
            self._action_queue.append(actions[:, t])

        return self._action_queue.popleft()

    def _get_latent_codes(self, batch) -> Tensor:
        """Get latent codes from VLA or dataset."""
        # Option 1: Pre-computed codes in dataset
        if "latent_codes" in batch:
            return batch["latent_codes"]

        # Option 2: Online VLA inference
        images = self._extract_images(batch)
        instructions = self._extract_language(batch)
        return self.vla.predict_codes(images, instructions)
```

#### Modified Diffusion Model

```python
class LatentConditionedDiffusionModel(nn.Module):
    """DiffusionModel with additional latent code conditioning."""

    def __init__(self, config: HLRPDiffusionConfig, extra_cond_dim: int):
        super().__init__()
        self.config = config

        # Standard image encoder
        self.rgb_encoder = DiffusionRgbEncoder(config)

        # Global conditioning dimension
        # = state_dim + image_features + latent_cond_dim
        base_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            base_cond_dim += self.rgb_encoder.feature_dim * len(config.image_features)

        global_cond_dim = (base_cond_dim + extra_cond_dim) * config.n_obs_steps

        # UNet with FiLM conditioning
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(...)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode observations + latent codes into global conditioning."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]

        global_cond_feats = [batch[OBS_STATE]]

        # Image features
        if self.config.image_features:
            img_features = self.rgb_encoder(batch[OBS_IMAGES])
            global_cond_feats.append(img_features)

        # ══════════════════════════════════════════════════════════════
        # NEW: Latent code conditioning
        # ══════════════════════════════════════════════════════════════
        if "latent_conditioning" in batch:
            latent_cond = batch["latent_conditioning"]  # [B, latent_dim]
            # Expand to match n_obs_steps if needed
            if latent_cond.ndim == 2:
                latent_cond = latent_cond.unsqueeze(1).expand(-1, n_obs_steps, -1)
            global_cond_feats.append(latent_cond)

        # Concatenate and flatten
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute diffusion loss with latent conditioning."""
        global_cond = self._prepare_global_conditioning(batch)

        trajectory = batch[ACTION]
        eps = torch.randn_like(trajectory)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps,
                                   (trajectory.shape[0],), device=trajectory.device)

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        return F.mse_loss(pred, eps)

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate action trajectory via diffusion."""
        global_cond = self._prepare_global_conditioning(batch)
        batch_size = global_cond.shape[0]

        # Sample from noise
        sample = torch.randn(batch_size, self.config.horizon,
                             self.config.action_feature.shape[0],
                             device=global_cond.device)

        # Denoise
        self.noise_scheduler.set_timesteps(self.config.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            pred = self.unet(sample, t, global_cond=global_cond)
            sample = self.noise_scheduler.step(pred, t, sample).prev_sample

        return sample[:, :self.config.n_action_steps]
```

### 14.6 Training Data Requirements

For the hierarchical approach, training data needs:

| Field | Source | Description |
|-------|--------|-------------|
| `observation.images.*` | Standard | Camera images |
| `observation.state` | Standard | Robot proprioception |
| `action` | Standard | Continuous actions [B, horizon, action_dim] |
| `latent_codes` | **Pre-computed** | HLRP latent codes [B, code_seq_len] |
| `task` / `instruction` | Standard | Language instruction |

**Pre-computing latent codes**:
```bash
# Script to add latent codes to existing dataset
python scripts/precompute_latent_codes.py \
    --dataset lerobot/aloha_sim_insertion \
    --vla_checkpoint /path/to/vla.ckpt \
    --laq_checkpoint /path/to/laq.ckpt \
    --output_dataset lerobot/aloha_sim_insertion_with_latents
```

### 14.7 Comparison: Direct Decoding vs. Diffusion

| Aspect | Direct LAQ Decoding | Latent-Conditioned Diffusion |
|--------|---------------------|------------------------------|
| **Architecture** | VLA → MLP → actions | VLA → Diffusion → actions |
| **Action smoothness** | Depends on MLP | Naturally smooth (diffusion) |
| **Chunk size** | 1 (or learned) | Configurable horizon |
| **Training** | Simpler | Requires diffusion training |
| **Latency** | Fast (single forward) | Slower (N denoising steps) |
| **Temporal consistency** | Requires action queue | Built into diffusion |
| **Best for** | Quick deployment | High-quality trajectories |

### 14.8 When to Use Each Approach

**Use Direct LAQ Decoding (Section 1-13) when:**
- Fast inference is critical
- You have a well-trained LAQ decoder
- Single-step actions are sufficient

**Use Latent-Conditioned Diffusion (Section 14) when:**
- Smooth action trajectories are important
- You want action chunking
- You're targeting manipulation tasks with contact
- Training time/compute is available

### 14.9 LIBERO Benchmark Integration

For LIBERO specifically:

```python
# LIBERO expects language-conditioned policies
# Our hierarchical approach provides this via:
#   Language → VLA → Latent Codes → Diffusion → Actions

lerobot-eval \
    --policy.type=hlrp_diffusion \
    --policy.vla_checkpoint_path=/path/to/vla.ckpt \
    --env.type=libero \
    --env.task=LIBERO_SPATIAL \
    --eval.n_episodes=50
```

The `LiberoProcessorStep` handles observation format conversion, and our policy handles language conditioning via the VLA's latent codes.

---

## 15. Summary of Integration Options

| Option | Description | Complexity | Best For |
|--------|-------------|------------|----------|
| **A: Direct VLA + LAQ Decoder** | VLA predicts codes, MLP decodes to actions | Low | Fast inference, simple deployment |
| **B: VLA + Diffusion** | VLA provides latent conditioning for diffusion | Medium | Smooth trajectories, manipulation |
| **C: External Plugin** | Keep HLRP code external, register via `discover_packages_path` | Low | Clean separation, easy updates |
| **D: Fork LeRobot** | Add HLRP directly to lerobot policies | Medium | Tight integration, contributions |

**Recommended path**: Start with **Option A** for quick validation, then explore **Option B** for benchmark performance.
