from __future__ import annotations

from dataclasses import dataclass, field

import torch

from foundation.action_tokens import ActionTokenConfig
from foundation.vla_inputs import ChatConfig


@dataclass
class SmolVLASharedCoreConfig:
    model_name: str
    latent_vector_dim: int
    action_dim: int | None = None
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = False
    chat: ChatConfig = field(default_factory=lambda: ChatConfig(system_prompt=None))
    action_tokens: ActionTokenConfig = field(
        default_factory=lambda: ActionTokenConfig(codebook_size=8, code_seq_len=4)
    )
    use_gpu_preprocessing: bool = True
    image_size: tuple[int, int] = (384, 384)
    flow_hidden_dim: int = 1024
    flow_steps: int = 8
    min_period: float = 4e-3
    max_period: float = 4.0
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0


@dataclass
class SmolVLASharedBackendConfig:
    model_name: str
    latent_vector_dim: int
    action_dim: int | None = None
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = False
    chat: ChatConfig = field(default_factory=lambda: ChatConfig(system_prompt=None))
    action_tokens: ActionTokenConfig = field(
        default_factory=lambda: ActionTokenConfig(codebook_size=8, code_seq_len=4)
    )
    use_gpu_preprocessing: bool = True
    image_size: tuple[int, int] = (384, 384)
    flow_hidden_dim: int = 1024
    flow_steps: int = 8
    latent_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    min_period: float = 4e-3
    max_period: float = 4.0
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0

    def to_core_config(self) -> SmolVLASharedCoreConfig:
        return SmolVLASharedCoreConfig(
            model_name=self.model_name,
            latent_vector_dim=self.latent_vector_dim,
            action_dim=self.action_dim,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            chat=self.chat,
            action_tokens=self.action_tokens,
            use_gpu_preprocessing=self.use_gpu_preprocessing,
            image_size=self.image_size,
            flow_hidden_dim=self.flow_hidden_dim,
            flow_steps=self.flow_steps,
            min_period=self.min_period,
            max_period=self.max_period,
            time_beta_alpha=self.time_beta_alpha,
            time_beta_beta=self.time_beta_beta,
        )
