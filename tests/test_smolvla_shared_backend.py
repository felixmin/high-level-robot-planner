from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import torch

from foundation.action_tokens import ActionTokenConfig
from foundation.backends.interfaces import BackendMode, FoundationBatch
from foundation.backends.smolvla_shared.config import SmolVLASharedBackendConfig
from foundation.backends.smolvla_shared_backend import SmolVLASharedBackend
from foundation.vla_inputs import ChatConfig


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is False
        parts: List[str] = []
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    parts.append("<image>")
                elif item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join([p for p in parts if p])

    def __call__(self, *, text, images, return_tensors: str, padding: bool):
        assert return_tensors == "pt"
        assert padding is True
        assert isinstance(text, list)
        assert len(text) == len(images)
        assert all(isinstance(x, list) and len(x) == 1 for x in images)
        b = len(text)
        lengths = [max(1, len(str(t).split())) for t in text]
        max_len = max(lengths)
        input_ids = torch.zeros((b, max_len), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, l in enumerate(lengths):
            input_ids[i, :l] = torch.arange(1, l + 1, dtype=torch.long)
            attention_mask[i, :l] = 1
        pixel_values = torch.zeros((b, 3, 8, 8), dtype=torch.float32)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}


class DummyVLM(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden_size))
        self.proj = torch.nn.Linear(1, hidden_size, bias=False)

    def forward(self, input_ids, attention_mask, pixel_values, output_hidden_states: bool, return_dict: bool):
        assert output_hidden_states is True
        assert return_dict is True
        b, l = input_ids.shape
        x = torch.ones((b, l, 1), dtype=torch.float32, device=input_ids.device)
        last = self.proj(x)
        return SimpleNamespace(hidden_states=(last,))


def _make_backend() -> SmolVLASharedBackend:
    return SmolVLASharedBackend(
        config=SmolVLASharedBackendConfig(
            model_name="dummy",
            latent_vector_dim=8,
            action_dim=3,
            torch_dtype=torch.float32,
            trust_remote_code=False,
            chat=ChatConfig(system_prompt="sys"),
            action_tokens=ActionTokenConfig(codebook_size=8, code_seq_len=4),
            use_gpu_preprocessing=False,
            image_size=(384, 384),
            flow_hidden_dim=32,
            flow_steps=4,
            latent_loss_weight=1.0,
            action_loss_weight=1.0,
        ),
        vlm=DummyVLM(hidden_size=16),
        processor=FakeProcessor(),
        frames_to_images=lambda frames: [object() for _ in range(frames.shape[0])],
    )


def _make_batch() -> FoundationBatch:
    return FoundationBatch(
        frames=torch.randint(0, 256, (2, 2, 8, 8, 3), dtype=torch.uint8),
        instructions=["pick", "place"],
        target_latent_vectors=torch.randn(2, 4, 2),
        target_actions=torch.randn(2, 3),
    )


def test_smolvla_shared_backend_latent_flow() -> None:
    backend = _make_backend()
    backend.setup(device=torch.device("cpu"))

    batch = _make_batch()
    out = backend.loss_from_batch(batch, mode=BackendMode.LATENT_FLOW)
    assert torch.is_tensor(out.loss)

    latent = backend.latent_from_batch(batch, mode=BackendMode.LATENT_FLOW)
    assert latent.vector is not None
    assert latent.vector.shape == (2, 8)


def test_smolvla_shared_backend_actions_and_multitask() -> None:
    backend = _make_backend()
    backend.setup(device=torch.device("cpu"))

    batch = _make_batch()

    actions_out = backend.loss_from_batch(batch, mode=BackendMode.ACTIONS)
    assert torch.is_tensor(actions_out.loss)

    both_out = backend.loss_from_batch(batch, mode=BackendMode.MULTITASK)
    assert torch.is_tensor(both_out.loss)

    latent = backend.latent_from_batch(batch, mode=BackendMode.MULTITASK)
    assert latent.vector is not None
    assert latent.actions is not None
    assert latent.vector.shape == (2, 8)
    assert latent.actions.shape == (2, 3)
