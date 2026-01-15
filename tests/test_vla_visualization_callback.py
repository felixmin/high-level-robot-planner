from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from foundation.action_tokens import ActionTokenConfig
from foundation.callbacks import VLASampleVizConfig, VLASampleVisualizationCallback


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
