from __future__ import annotations

from types import SimpleNamespace

import json
import torch
from PIL import Image

from foundation.action_tokens import ActionTokenConfig
from foundation.callbacks import (
    VLALatentFlowDecodeVisualizationCallback,
    VLALatentFlowDecodeVizConfig,
    VLASampleVizConfig,
    VLASampleVisualizationCallback,
)


def test_vla_sample_visualization_callback_writes_files(tmp_path):
    callback = VLASampleVisualizationCallback(
        VLASampleVizConfig(enabled=True, num_samples=2, every_n_val=1)
    )

    trainer = SimpleNamespace(
        is_global_zero=True,
        default_root_dir=str(tmp_path),
        global_step=123,
        logger=False,
    )

    frames = torch.randint(0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8)
    pl_module = SimpleNamespace(
        action_tokens=ActionTokenConfig(codebook_size=8, code_seq_len=4),
        action_token_ids=SimpleNamespace(code_seq_len=4),
        vla_model=SimpleNamespace(generate=lambda **_: None),
        frames_to_images=lambda _frames: [
            Image.new("RGB", (16, 16), color=(10, 20, 30)) for _ in range(_frames.shape[0])
        ],
        _last_val_sample={
            "frames": frames,
            "instructions": ["pick up block", "push button"],
            "gt_codes": [[3, 1, 7, 0], [3, 1, 7, 0]],
            "pred_codes": [[3, 1, 7, 0], [-1, -1, -1, -1]],
            "episode_id": ["ep1", "ep2"],
            "frame_idx": [0, 5],
        },
    )

    callback.on_validation_epoch_end(trainer, pl_module)

    viz_dir = tmp_path / "visualizations"
    assert (viz_dir / "val_samples_step000123.png").exists()
    assert (viz_dir / "val_samples_step000123.json").exists()


def test_vla_sample_visualization_keeps_gt_codes_in_latent_flow(tmp_path):
    callback = VLASampleVisualizationCallback(
        VLASampleVizConfig(enabled=True, num_samples=1, every_n_val=1)
    )

    trainer = SimpleNamespace(
        is_global_zero=True,
        default_root_dir=str(tmp_path),
        global_step=7,
        logger=False,
    )

    frames = torch.randint(0, 256, (1, 2, 16, 16, 3), dtype=torch.uint8)
    pl_module = SimpleNamespace(
        action_tokens=ActionTokenConfig(codebook_size=8, code_seq_len=4),
        action_token_ids=SimpleNamespace(code_seq_len=4),
        vla_model=SimpleNamespace(generate=lambda **_: None),
        frames_to_images=lambda _frames: [
            Image.new("RGB", (16, 16), color=(10, 20, 30)) for _ in range(_frames.shape[0])
        ],
        _last_val_sample={
            "frames": frames,
            "instructions": ["pick up block"],
            "mode": "latent_flow",
            "gt_codes": [[3, 1, 7, 0]],
            "pred_codes": None,
            "gt_vectors": [[0.1, 0.2, 0.3]],
            "pred_vectors": [[0.0, 0.25, 0.35]],
        },
    )

    callback.on_validation_epoch_end(trainer, pl_module)
    records = json.loads((tmp_path / "visualizations" / "val_samples_step000007.json").read_text())
    assert records[0]["gt_codes"] == [3, 1, 7, 0]


def test_vla_latent_flow_decode_visualization_accepts_batched_temporal_frames(tmp_path):
    class _FakeVQ:
        def __init__(self) -> None:
            self.codebooks = torch.zeros((8, 2), dtype=torch.float32)

        def project_out(self, vec: torch.Tensor) -> torch.Tensor:
            return vec

    class _FakeFlowConfig:
        summary_static_eps = 1e-6

    class _FakeLaqModel:
        def __init__(self) -> None:
            self.code_seq_len = 1
            self.action_shape = (1, 1)
            self.patch_height_width = (1, 1)
            self.vq = _FakeVQ()
            self.flow_config = _FakeFlowConfig()

        def decoder_context_projection(self, first_frame: torch.Tensor) -> torch.Tensor:
            assert tuple(first_frame.shape[1:3]) == (3, 1)
            bsz = int(first_frame.shape[0])
            return torch.zeros((bsz, 1, 1, 1, 2), dtype=torch.float32, device=first_frame.device)

        def spatial_rel_pos_bias(self, h: int, w: int, *, device: torch.device):
            assert (h, w) == (1, 1)
            return torch.zeros((1,), dtype=torch.float32, device=device)

        def flow_decoder(self, pixel_context: torch.Tensor, actions: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
            del actions, attn_bias
            bsz = int(pixel_context.shape[0])
            return torch.zeros((bsz, 2, 8, 8), dtype=torch.float32, device=pixel_context.device)

    callback = VLALatentFlowDecodeVisualizationCallback(
        laq_checkpoint_path="unused.ckpt",
        cfg=VLALatentFlowDecodeVizConfig(enabled=True, num_samples=1, every_n_val=1, max_decode_batch_size=1),
    )
    callback._ensure_laq_model = lambda _device: _FakeLaqModel()  # type: ignore[method-assign]

    trainer = SimpleNamespace(
        is_global_zero=True,
        default_root_dir=str(tmp_path),
        global_step=11,
        logger=False,
    )

    pl_module = SimpleNamespace(
        device=torch.device("cpu"),
        _last_val_sample={
            "mode": "latent_flow",
            "frames": torch.randint(0, 256, (1, 2, 3, 16, 16), dtype=torch.uint8),
            "instructions": ["pick up block"],
            "gt_vectors": [[0.1, 0.2]],
            "pred_vectors": [[0.0, 0.25]],
            "episode_id": ["ep1"],
            "frame_idx": [0],
            "dataset_name": ["mix"],
        },
    )

    callback.on_validation_epoch_end(trainer, pl_module)

    viz_dir = tmp_path / "visualizations"
    assert (viz_dir / "val_latent_flow_decode_step000011.png").exists()
    assert (viz_dir / "val_latent_flow_decode_step000011.json").exists()
