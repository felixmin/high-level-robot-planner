"""
Stage 2 (Foundation) Lightning callbacks.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


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
    meta: str,
    instruction: str,
    gt: str,
    pred: str,
    width: int = 384,
) -> Image.Image:
    font = _default_font()
    image = image.convert("RGB").resize((width, width))

    text_lines = []
    if meta:
        text_lines += _wrap_text(meta, width=56) + [""]
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


def _select_diverse_indices(
    *,
    episode_id: Any,
    frame_idx: Any,
    instructions: list[Any],
    max_items: int,
) -> list[int]:
    if max_items <= 0:
        return []

    n = min(len(instructions), max_items)
    if n <= 0:
        return []

    episode_ids: Optional[list[Any]] = episode_id if isinstance(episode_id, list) else None
    frame_idxs: Optional[list[Any]] = frame_idx if isinstance(frame_idx, list) else None

    def get_ep(i: int) -> str:
        if episode_ids is None or i >= len(episode_ids) or episode_ids[i] is None:
            return ""
        return str(episode_ids[i])

    def get_frame(i: int) -> str:
        if frame_idxs is None or i >= len(frame_idxs) or frame_idxs[i] is None:
            return ""
        return str(frame_idxs[i])

    chosen: list[int] = []

    # Prefer distinct (episode, instruction) pairs.
    seen_ep_instr: set[tuple[str, str]] = set()
    for i in range(len(instructions)):
        if len(chosen) >= n:
            break
        key = (get_ep(i), str(instructions[i]))
        if key in seen_ep_instr:
            continue
        seen_ep_instr.add(key)
        chosen.append(i)

    # If still short, prefer distinct (episode, frame_idx) within the same instruction.
    if len(chosen) < n:
        seen_ep_frame: set[tuple[str, str]] = set()
        for i in chosen:
            seen_ep_frame.add((get_ep(i), get_frame(i)))
        for i in range(len(instructions)):
            if len(chosen) >= n:
                break
            if i in chosen:
                continue
            key = (get_ep(i), get_frame(i))
            if key in seen_ep_frame:
                continue
            seen_ep_frame.add(key)
            chosen.append(i)

    # Fill remaining slots deterministically.
    if len(chosen) < n:
        for i in range(len(instructions)):
            if len(chosen) >= n:
                break
            if i not in chosen:
                chosen.append(i)

    return chosen[:n]


def _save_grid_and_records(
    *,
    panels: list[Image.Image],
    records: list[dict[str, Any]],
    out_dir: Path,
    prefix: str,
    step: int,
    trainer: pl.Trainer,
    wandb_key: str,
) -> None:
    if not panels:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    w, h = panels[0].size
    grid = Image.new("RGB", (w * len(panels), h), color=(255, 255, 255))
    for i, p in enumerate(panels):
        grid.paste(p, (i * w, 0))

    png_path = out_dir / f"{prefix}_step{step:06d}.png"
    json_path = out_dir / f"{prefix}_step{step:06d}.json"
    grid.save(png_path)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    try:
        import wandb  # type: ignore

        if hasattr(trainer, "logger") and getattr(trainer.logger, "experiment", None):
            trainer.logger.experiment.log(  # type: ignore[attr-defined]
                {wandb_key: wandb.Image(str(png_path))}, step=step
            )
    except Exception:
        pass


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
        episode_id = sample.get("episode_id")
        frame_idx = sample.get("frame_idx")
        if frames is None or instructions is None or gt_codes is None or pred_codes is None:
            return

        try:
            images = pl_module.frames_to_images(frames)
        except Exception:
            logger.debug("val sample viz: frames_to_images failed", exc_info=True)
            return

        num = min(self.cfg.num_samples, len(instructions), len(images), len(pred_codes))
        if num <= 0:
            return

        indices = _select_diverse_indices(
            episode_id=episode_id,
            frame_idx=frame_idx,
            instructions=list(instructions),
            max_items=num,
        )

        out_dir = Path(str(trainer.default_root_dir)) / "visualizations"
        step = int(getattr(trainer, "global_step", 0))

        panels: list[Image.Image] = []
        records: list[dict[str, Any]] = []
        for j, i in enumerate(indices):
            gt_str = action_tokens.format_target(gt_codes[i])
            try:
                pred_str = action_tokens.format_target(pred_codes[i])
            except Exception:
                pred_str = f"<INVALID> {pred_codes[i]}"
            panels.append(
                _render_panel(
                    image=images[i],
                    meta=f"episode_id: {episode_id[i]}  frame_idx: {frame_idx[i]}"
                    if isinstance(episode_id, list)
                    and isinstance(frame_idx, list)
                    and i < len(episode_id)
                    and i < len(frame_idx)
                    else "",
                    instruction=str(instructions[i]),
                    gt=gt_str,
                    pred=pred_str,
                )
            )
            records.append(
                {
                    "step": step,
                    "index": int(i),
                    "rank": int(j),
                    "instruction": str(instructions[i]),
                    "gt_codes": gt_codes[i],
                    "pred_codes": pred_codes[i],
                    "episode_id": episode_id[i] if isinstance(episode_id, list) and i < len(episode_id) else None,
                    "frame_idx": frame_idx[i] if isinstance(frame_idx, list) and i < len(frame_idx) else None,
                }
            )

        _save_grid_and_records(
            panels=panels,
            records=records,
            out_dir=out_dir,
            prefix="val_samples",
            step=step,
            trainer=trainer,
            wandb_key="val/samples",
        )


class ThroughputLoggingCallback(Callback):
    def __init__(self, cfg: Optional[ThroughputLoggingConfig] = None):
        super().__init__()
        self.cfg = cfg or ThroughputLoggingConfig()
        self._last_time: Optional[float] = None
        self._last_step: Optional[int] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.cfg.enabled:
            return

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


@dataclass
class VLATrainSampleVizConfig:
    enabled: bool = True
    num_samples: int = 4
    every_n_steps: int = 500


class VLATrainSampleVisualizationCallback(Callback):
    def __init__(self, cfg: Optional[VLATrainSampleVizConfig] = None):
        super().__init__()
        self.cfg = cfg or VLATrainSampleVizConfig()

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
        if not trainer.is_global_zero:
            return

        step = int(getattr(trainer, "global_step", 0))
        if self.cfg.every_n_steps <= 0 or (step % self.cfg.every_n_steps) != 0:
            return
        if not isinstance(batch, dict):
            return

        action_tokens = getattr(pl_module, "action_tokens", None)
        token_ids = getattr(pl_module, "action_token_ids", None)
        if action_tokens is None or token_ids is None:
            return

        vla_model = getattr(pl_module, "vla_model", None)
        if vla_model is None or not hasattr(vla_model, "generate"):
            return

        try:
            from foundation.online_laq import extract_oxe_language, oxe_frames_to_laq_video
        except Exception:
            logger.debug("train sample viz: failed to import online_laq helpers", exc_info=True)
            return

        frames = batch.get("frames")
        if frames is None:
            return

        try:
            instructions = extract_oxe_language(batch)
        except Exception:
            logger.debug("train sample viz: failed to extract language", exc_info=True)
            return

        try:
            images = pl_module.frames_to_images(frames)
        except Exception:
            logger.debug("train sample viz: frames_to_images failed", exc_info=True)
            return

        # Compute GT codes via frozen LAQ.
        try:
            video = oxe_frames_to_laq_video(frames)
            gt_codes_t = pl_module.code_provider.codes_from_video(video)
            gt_codes = [row.tolist() for row in gt_codes_t.detach().cpu()]
        except Exception:
            logger.debug("train sample viz: LAQ GT code extraction failed", exc_info=True)
            return

        # Predict codes via constrained generation.
        try:
            pred_codes = pl_module._predict_codes(frames=frames, instructions=instructions)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("train sample viz: constrained generation failed", exc_info=True)
            return

        episode_id = batch.get("episode_id")
        if isinstance(episode_id, list):
            episode_id_list = episode_id
        else:
            episode_id_list = None

        frame_idx = batch.get("frame_idx")
        frame_idx_list = frame_idx if isinstance(frame_idx, list) else None

        # Select diverse samples from this batch.
        max_items = min(self.cfg.num_samples, len(instructions), len(images), len(pred_codes), len(gt_codes))
        if max_items <= 0:
            return

        chosen = _select_diverse_indices(
            episode_id=episode_id_list,
            frame_idx=frame_idx_list,
            instructions=list(instructions),
            max_items=max_items,
        )

        out_dir = Path(str(trainer.default_root_dir)) / "visualizations"

        panels: list[Image.Image] = []
        records: list[dict[str, Any]] = []
        for rank, i in enumerate(chosen):
            gt_str = action_tokens.format_target(gt_codes[i])
            try:
                pred_str = action_tokens.format_target(pred_codes[i])
            except Exception:
                pred_str = f"<INVALID> {pred_codes[i]}"

            panels.append(
                _render_panel(
                    image=images[i],
                    meta=f"episode_id: {episode_id_list[i]}  frame_idx: {frame_idx_list[i]}"
                    if episode_id_list is not None
                    and frame_idx_list is not None
                    and i < len(episode_id_list)
                    and i < len(frame_idx_list)
                    else "",
                    instruction=str(instructions[i]),
                    gt=gt_str,
                    pred=pred_str,
                )
            )
            records.append(
                {
                    "step": step,
                    "index": int(i),
                    "rank": int(rank),
                    "instruction": str(instructions[i]),
                    "gt_codes": gt_codes[i],
                    "pred_codes": pred_codes[i],
                    "episode_id": episode_id_list[i] if episode_id_list and i < len(episode_id_list) else None,
                    "frame_idx": frame_idx_list[i] if frame_idx_list and i < len(frame_idx_list) else None,
                }
            )

        _save_grid_and_records(
            panels=panels,
            records=records,
            out_dir=out_dir,
            prefix="train_samples",
            step=step,
            trainer=trainer,
            wandb_key="train/samples",
        )
