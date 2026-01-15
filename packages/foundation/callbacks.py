"""
Stage 2 (Foundation) Lightning callbacks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from PIL import Image, ImageDraw, ImageFont


@dataclass
class VLASampleVizConfig:
    enabled: bool = True
    num_samples: int = 4
    every_n_val: int = 1


@dataclass
class ThroughputLoggingConfig:
    enabled: bool = True
    log_every_n_steps: int = 10


def _default_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _wrap_text(text: str, *, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        if sum(len(x) for x in current) + len(current) + len(w) > width and current:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return lines


def _render_panel(
    *,
    image: Image.Image,
    instruction: str,
    gt: str,
    pred: str,
    width: int = 384,
) -> Image.Image:
    font = _default_font()
    image = image.convert("RGB").resize((width, width))

    text_lines = []
    text_lines += ["instruction:"] + _wrap_text(instruction, width=56)
    text_lines += ["", "gt:"] + _wrap_text(gt, width=56)
    text_lines += ["", "pred:"] + _wrap_text(pred, width=56)

    line_h = 14
    pad = 10
    text_h = pad * 2 + line_h * max(1, len(text_lines))
    panel = Image.new("RGB", (width, width + text_h), color=(255, 255, 255))
    panel.paste(image, (0, 0))

    draw = ImageDraw.Draw(panel)
    y = width + pad
    for line in text_lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    return panel


class VLASampleVisualizationCallback(Callback):
    def __init__(self, cfg: Optional[VLASampleVizConfig] = None):
        super().__init__()
        self.cfg = cfg or VLASampleVizConfig()
        self._val_count = 0

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self.cfg.enabled:
            return
        if not trainer.is_global_zero:
            return

        self._val_count += 1
        if self.cfg.every_n_val > 1 and (self._val_count % self.cfg.every_n_val) != 0:
            return

        action_tokens = getattr(pl_module, "action_tokens", None)
        token_ids = getattr(pl_module, "action_token_ids", None)
        if action_tokens is None or token_ids is None:
            return

        # Only visualize if the model can generate.
        vla_model = getattr(pl_module, "vla_model", None)
        if vla_model is None or not hasattr(vla_model, "generate"):
            return

        sample = getattr(pl_module, "_last_val_sample", None)
        if not isinstance(sample, dict):
            return

        frames = sample.get("frames")
        instructions = sample.get("instructions")
        gt_codes = sample.get("gt_codes")
        pred_codes = sample.get("pred_codes")
        if frames is None or instructions is None or gt_codes is None or pred_codes is None:
            return

        try:
            images = pl_module.frames_to_images(frames)
        except Exception:
            return

        num = min(self.cfg.num_samples, len(instructions), len(images), len(pred_codes))
        if num <= 0:
            return

        out_dir = Path(str(trainer.default_root_dir)) / "visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        step = int(getattr(trainer, "global_step", 0))

        panels: list[Image.Image] = []
        records: list[dict[str, Any]] = []
        for i in range(num):
            gt_str = action_tokens.format_target(gt_codes[i])
            try:
                pred_str = action_tokens.format_target(pred_codes[i])
            except Exception:
                pred_str = f"<INVALID> {pred_codes[i]}"
            panels.append(
                _render_panel(
                    image=images[i],
                    instruction=str(instructions[i]),
                    gt=gt_str,
                    pred=pred_str,
                )
            )
            records.append(
                {
                    "step": step,
                    "index": i,
                    "instruction": str(instructions[i]),
                    "gt_codes": gt_codes[i],
                    "pred_codes": pred_codes[i],
                }
            )

        # Save a horizontal strip (simple + reliable).
        w, h = panels[0].size
        grid = Image.new("RGB", (w * len(panels), h), color=(255, 255, 255))
        for i, p in enumerate(panels):
            grid.paste(p, (i * w, 0))

        png_path = out_dir / f"val_samples_step{step:06d}.png"
        json_path = out_dir / f"val_samples_step{step:06d}.json"
        grid.save(png_path)
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

        # If using WandB, log the image too.
        try:
            import wandb  # type: ignore

            if hasattr(trainer, "logger") and getattr(trainer.logger, "experiment", None):
                trainer.logger.experiment.log(  # type: ignore[attr-defined]
                    {"val/samples": wandb.Image(str(png_path))}, step=step
                )
        except Exception:
            pass


class ThroughputLoggingCallback(Callback):
    def __init__(self, cfg: Optional[ThroughputLoggingConfig] = None):
        super().__init__()
        self.cfg = cfg or ThroughputLoggingConfig()
        self._last_time: Optional[float] = None
        self._last_step: Optional[int] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.cfg.enabled:
            return
        import time

        self._last_time = time.perf_counter()
        self._last_step = int(getattr(trainer, "global_step", 0))

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not self.cfg.enabled:
            return

        step = int(getattr(trainer, "global_step", 0))
        if step <= 0:
            return
        if self.cfg.log_every_n_steps <= 0 or (step % self.cfg.log_every_n_steps) != 0:
            return

        import time

        now = time.perf_counter()
        if self._last_time is None or self._last_step is None:
            self._last_time = now
            self._last_step = step
            return

        dt = now - self._last_time
        ds = step - self._last_step
        if dt <= 0.0 or ds <= 0:
            self._last_time = now
            self._last_step = step
            return

        batch_size: Optional[int] = None
        try:
            frames = batch["frames"] if isinstance(batch, dict) else batch
            batch_size = int(getattr(frames, "shape", [None])[0])
        except Exception:
            batch_size = None

        steps_per_sec = float(ds) / float(dt)
        pl_module.log(
            "perf/steps_per_sec",
            steps_per_sec,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        if batch_size is not None and batch_size > 0:
            pl_module.log(
                "perf/samples_per_sec",
                float(batch_size) * steps_per_sec,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        self._last_time = now
        self._last_step = step
