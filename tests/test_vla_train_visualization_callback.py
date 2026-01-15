from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from foundation.callbacks import VLATrainSampleVizConfig, VLATrainSampleVisualizationCallback


class DummyCodeProvider:
    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        b = video.shape[0]
        return torch.tensor([[3, 1, 7, 0]] * b, dtype=torch.long)


def test_vla_train_visualization_callback_writes_files(tmp_path):
    callback = VLATrainSampleVisualizationCallback(
        VLATrainSampleVizConfig(enabled=True, num_samples=2, every_n_steps=10)
    )

    trainer = SimpleNamespace(
        is_global_zero=True,
        default_root_dir=str(tmp_path),
        global_step=10,
        logger=False,
    )

    pl_module = SimpleNamespace(
        action_tokens=SimpleNamespace(
            format_target=lambda codes: f"<ACTION> {codes} </ACTION>"
        ),
        action_token_ids=SimpleNamespace(code_seq_len=4),
        vla_model=SimpleNamespace(generate=lambda **_: None),
        frames_to_images=lambda frames: [
            Image.new("RGB", (16, 16), color=(10, 20, 30)) for _ in range(frames.shape[0])
        ],
        code_provider=DummyCodeProvider(),
        _predict_codes=lambda *, frames, instructions: [[3, 1, 7, 0] for _ in instructions],
    )

    batch = {
        "frames": torch.randint(0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8),
        "language": ["pick up block", "push button"],
        "episode_id": ["ep1", "ep2"],
        "frame_idx": [0, 5],
    }

    callback.on_train_batch_end(trainer, pl_module, outputs=None, batch=batch, batch_idx=0)

    viz_dir = tmp_path / "visualizations"
    assert (viz_dir / "train_samples_step000010.png").exists()
    assert (viz_dir / "train_samples_step000010.json").exists()

