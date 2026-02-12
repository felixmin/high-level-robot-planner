#!/usr/bin/env python3
"""
Offline LAQ action-token analysis.

Runs a standalone evaluation loop (no Lightning Trainer) over the OXE val loader,
collects token + flow metadata, and writes analysis plots/files.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import numpy as np
import torch
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import flow_to_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

from common.cache_env import configure_cache_env, resolve_cache_dir  # noqa: E402
from common.data_factory import create_datamodule  # noqa: E402
from common.logging import set_seed  # noqa: E402
from common.unified_logging import resolve_runs_dir, setup_unified_logging  # noqa: E402
from laq import LAQTask  # noqa: E402


def _seq_str(seq: tuple[int, ...]) -> str:
    return "-".join(str(x) for x in seq)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _tensor_to_img(frame_chw: torch.Tensor) -> np.ndarray:
    frame = frame_chw.detach().cpu()
    if frame.dtype == torch.uint8:
        frame = frame.float().div(255.0)
    else:
        frame = frame.float()
        if float(frame.max()) > 1.5:
            frame = frame.div(255.0)
    frame = frame.clamp(0.0, 1.0).numpy()
    if frame.shape[0] == 1:
        frame = np.repeat(frame, 3, axis=0)
    return np.transpose(frame, (1, 2, 0))


def _frames_to_model_float(frames: torch.Tensor) -> torch.Tensor:
    if frames.dtype == torch.uint8:
        return frames.float().div(255.0)
    frames = frames.float()
    if float(frames.max()) > 1.5:
        frames = frames.div(255.0)
    return frames


def _flow_to_rgb(
    flow_b2hw: torch.Tensor,
    q: float = 0.95,
    min_keep_ratio: float = 0.15,
    denoise: bool = True,
) -> torch.Tensor:
    """Convert flow to RGB. Optionally suppress low-magnitude noise for readability."""
    flow = flow_b2hw.detach().cpu().float()
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)
    if flow.shape[0] == 0:
        return torch.empty((0, 3, *flow.shape[-2:]), dtype=torch.float32)
    if not denoise:
        # Render each sample independently to avoid batch-wise max normalization
        # in torchvision.flow_to_image washing out low-motion samples.
        imgs = [flow_to_image(flow[i]).float().div(255.0) for i in range(flow.shape[0])]
        return torch.stack(imgs, dim=0)
    normed = torch.zeros_like(flow)
    for i in range(flow.shape[0]):
        fx = flow[i, 0]
        fy = flow[i, 1]
        mag = torch.sqrt(fx * fx + fy * fy)
        flat = mag.flatten()
        scale = float(torch.quantile(flat, q).item()) if flat.numel() > 0 else 0.0
        if not math.isfinite(scale) or scale < 1e-6:
            scale = float(flat.max().item()) if flat.numel() > 0 else 1.0
        scale = max(scale, 1e-6)
        keep = (mag >= (min_keep_ratio * scale)).float()
        normed[i, 0] = (fx / scale) * keep
        normed[i, 1] = (fy / scale) * keep
    imgs = [flow_to_image(normed[i]).float().div(255.0) for i in range(normed.shape[0])]
    return torch.stack(imgs, dim=0)


def _to_state_vec(x: Any, dim: int = 2) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size < dim:
        return None
    return arr[:dim]


def _load_laq_task(checkpoint_path: str, cfg: DictConfig, logger) -> LAQTask:
    ckpt_path = str(checkpoint_path)
    try:
        task = LAQTask.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
        logger.info("Loaded checkpoint with LAQTask.load_from_checkpoint(weights_only=False)")
        return task
    except TypeError:
        task = LAQTask.load_from_checkpoint(ckpt_path, map_location="cpu")
        logger.info("Loaded checkpoint with LAQTask.load_from_checkpoint()")
        return task
    except Exception as exc:
        logger.warning(f"Primary checkpoint load failed ({exc}); using strict fallback loader.")

    task = LAQTask(
        model_config=cfg.model,
        training_config=cfg.training,
        use_ema=bool(cfg.training.get("use_ema", False)),
    )

    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    ckpt = torch.load(ckpt_path, **load_kwargs)
    state_dict = ckpt.get("state_dict", ckpt)
    model_state = {
        k.replace("model.", "", 1): v
        for k, v in state_dict.items()
        if isinstance(k, str) and k.startswith("model.")
    }
    missing, unexpected = task.model.load_state_dict(model_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Strict fallback load failed. missing={len(missing)}, unexpected={len(unexpected)}"
        )
    logger.info("Loaded checkpoint via strict fallback path")
    return task


def _compute_mean_flow(
    flow_b2hw: torch.Tensor,
    static_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fx = flow_b2hw[:, 0]
    fy = flow_b2hw[:, 1]
    mag = torch.sqrt(fx * fx + fy * fy)
    total_mag = mag.flatten(1).sum(dim=1)
    weights = mag / (total_mag[:, None, None] + static_eps)
    mean_dx = (fx * weights).flatten(1).sum(dim=1)
    mean_dy = (fy * weights).flatten(1).sum(dim=1)
    is_static = total_mag <= static_eps
    fallback_dx = fx.flatten(1).mean(dim=1)
    fallback_dy = fy.flatten(1).mean(dim=1)
    mean_dx = torch.where(is_static, fallback_dx, mean_dx)
    mean_dy = torch.where(is_static, fallback_dy, mean_dy)
    return mean_dx, mean_dy, is_static


def _dataset_names_from_cfg(cfg: DictConfig) -> list[str]:
    names: list[str] = []
    datasets = OmegaConf.select(cfg, "data.dataset.oxe.datasets")
    if datasets is None:
        return names
    for d in datasets:
        name = d.get("name") if isinstance(d, dict) else getattr(d, "name", None)
        if name:
            names.append(str(name))
    return names


def _setup_output_dir(base_output_dir: Path, cfg: DictConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(str(cfg.analysis.output_root))
    if not root.is_absolute():
        root = base_output_dir / root
    out_dir = root / f"laq_token_analysis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    return out_dir


def collect_samples(task: LAQTask, dataloader, cfg: DictConfig, logger):
    if task.model.flow_teacher is None:
        raise RuntimeError("Checkpoint model has no flow_teacher; this analysis requires flow supervision.")

    samples_per_dataset = int(cfg.analysis.samples_per_dataset)
    max_batches = int(cfg.analysis.max_batches)
    visual_subset_size = int(cfg.analysis.visual_subset_size)
    static_eps = float(cfg.analysis.static_eps)
    target_datasets = _dataset_names_from_cfg(cfg)

    per_dataset_counts: dict[str, int] = {name: 0 for name in target_datasets}
    records: list[dict[str, Any]] = []
    visual_samples: list[dict[str, Any]] = []
    seen_kept = 0

    device = task.device
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        if not isinstance(batch, dict):
            continue
        frames = batch["frames"]
        batch_size = int(frames.shape[0])
        dataset_names = list(batch.get("dataset_name", ["unknown"] * batch_size))
        languages = list(batch.get("language", [""] * batch_size))
        episode_ids = list(batch.get("episode_id", [""] * batch_size))
        frame_idxs = list(batch.get("frame_idx", [0] * batch_size))
        initial_states = list(batch.get("initial_state", [None] * batch_size))

        with torch.no_grad():
            frames_model = _frames_to_model_float(frames)
            frames_dev = frames_model.to(device, non_blocking=True)
            indices = task.model(frames_dev, return_only_codebook_ids=True)
            first_frame = frames_dev[:, :, :1]
            last_frame = frames_dev[:, :, -1:]
            flow = task.model.flow_teacher.compute_flow(first_frame, last_frame)
            mean_dx, mean_dy, is_static = _compute_mean_flow(flow, static_eps=static_eps)

        indices_cpu = indices.detach().cpu()
        mean_dx_cpu = mean_dx.detach().cpu()
        mean_dy_cpu = mean_dy.detach().cpu()
        is_static_cpu = is_static.detach().cpu()

        for i in range(batch_size):
            ds = str(dataset_names[i])
            if ds not in per_dataset_counts:
                per_dataset_counts[ds] = 0
            if per_dataset_counts[ds] >= samples_per_dataset:
                continue

            seq = tuple(int(x) for x in indices_cpu[i].tolist())
            rec = {
                "indices": seq,
                "mean_dx": float(mean_dx_cpu[i].item()),
                "mean_dy": float(mean_dy_cpu[i].item()),
                "dataset_name": ds,
                "language": str(languages[i]) if i < len(languages) else "",
                "is_static": bool(is_static_cpu[i].item()),
            }
            records.append(rec)
            per_dataset_counts[ds] += 1

            seen_kept += 1
            vis_item = {
                "indices": seq,
                "dataset_name": ds,
                "language": rec["language"],
                "episode_id": str(episode_ids[i]) if i < len(episode_ids) else "",
                "frame_idx": int(frame_idxs[i]) if i < len(frame_idxs) else 0,
                "initial_state": _to_state_vec(initial_states[i]) if i < len(initial_states) else None,
                "frames": frames_model[i].detach().cpu(),
            }
            if len(visual_samples) < visual_subset_size:
                visual_samples.append(vis_item)
            else:
                j = random.randrange(seen_kept)
                if j < visual_subset_size:
                    visual_samples[j] = vis_item

        if (batch_idx + 1) % 20 == 0:
            logger.info(f"collect batch={batch_idx + 1} counts={per_dataset_counts}")

        if target_datasets and all(per_dataset_counts.get(ds, 0) >= samples_per_dataset for ds in target_datasets):
            break

    return records, visual_samples, per_dataset_counts


def run_sanity_checks(records: list[dict[str, Any]], per_dataset_counts: dict[str, int], cfg: DictConfig) -> dict[str, Any]:
    min_dataset_samples = int(cfg.analysis.min_dataset_samples)

    total = max(1, len(records))
    bad = 0
    static_counts = defaultdict(int)
    total_counts = defaultdict(int)
    for r in records:
        dx = r["mean_dx"]
        dy = r["mean_dy"]
        if not (math.isfinite(dx) and math.isfinite(dy)):
            bad += 1
        ds = r["dataset_name"]
        total_counts[ds] += 1
        if r["is_static"]:
            static_counts[ds] += 1

    seq_counter = Counter(tuple(r["indices"]) for r in records)
    low_support = sum(1 for _, c in seq_counter.items() if c < 10)
    static_rate = {
        ds: (static_counts[ds] / max(1, total_counts[ds]))
        for ds in sorted(total_counts.keys())
    }
    return {
        "total_samples": len(records),
        "nan_or_inf_rate": bad / total,
        "per_dataset_counts": {k: int(v) for k, v in sorted(per_dataset_counts.items())},
        "dataset_under_min_samples": [k for k, v in per_dataset_counts.items() if v < min_dataset_samples],
        "static_rate_per_dataset": static_rate,
        "num_unique_sequences": len(seq_counter),
        "num_sequences_below_10_support": low_support,
    }


def select_inspected_sequences(
    records: list[dict[str, Any]],
    k: int,
) -> tuple[list[tuple[int, ...]], Counter, dict[tuple[int, ...], Counter]]:
    seq_counter: Counter = Counter(tuple(r["indices"]) for r in records)
    seq_ds_counts: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    for r in records:
        seq = tuple(r["indices"])
        seq_ds_counts[seq][r["dataset_name"]] += 1
    ranked = sorted(
        seq_counter.keys(),
        key=lambda s: (len(seq_ds_counts[s]), seq_counter[s]),
        reverse=True,
    )
    return ranked[:k], seq_counter, seq_ds_counts


def save_inspected_sequences(
    out_dir: Path,
    inspected: list[tuple[int, ...]],
    seq_counter: Counter,
    seq_ds_counts: dict[tuple[int, ...], Counter],
) -> None:
    datasets = sorted({ds for c in seq_ds_counts.values() for ds in c.keys()})
    with (out_dir / "inspected_tokens.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "sequence", "total_n", "num_datasets", *[f"{d}_n" for d in datasets]])
        for i, seq in enumerate(inspected):
            row = [i, _seq_str(seq), int(seq_counter.get(seq, 0)), len(seq_ds_counts.get(seq, {}))]
            row.extend(int(seq_ds_counts.get(seq, {}).get(ds, 0)) for ds in datasets)
            writer.writerow(row)
    payload = []
    for i, seq in enumerate(inspected):
        payload.append(
            {
                "rank": i,
                "sequence": list(seq),
                "sequence_str": _seq_str(seq),
                "total_n": int(seq_counter.get(seq, 0)),
                "per_dataset": {k: int(v) for k, v in sorted(seq_ds_counts.get(seq, {}).items())},
            }
        )
    _save_json(out_dir / "inspected_tokens.json", {"tokens": payload})


def _pick_balanced_samples(
    samples: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if len(samples) <= limit:
        return list(samples)
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_ds[s["dataset_name"]].append(s)
    picked: list[dict[str, Any]] = []
    while len(picked) < limit:
        advanced = False
        for ds in sorted(by_ds.keys()):
            if by_ds[ds] and len(picked) < limit:
                picked.append(by_ds[ds].pop(0))
                advanced = True
        if not advanced:
            break
    return picked


def analysis_1_flow_token_scatter(records: list[dict[str, Any]], out_dir: Path, top_k: int) -> None:
    out = out_dir / "analysis_1_flow_scatter"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in records if not r["is_static"]]
    if not rows:
        return
    seq_counter = Counter(tuple(r["indices"]) for r in rows)
    top = [seq for seq, _ in seq_counter.most_common(top_k)]
    top_map = {seq: i for i, seq in enumerate(top)}
    cmap = plt.get_cmap("tab20", max(1, len(top)))

    def render(subset: list[dict[str, Any]], path: Path, title: str) -> None:
        if not subset:
            return
        fig, ax = plt.subplots(figsize=(8, 7))
        other_x = [r["mean_dx"] for r in subset if tuple(r["indices"]) not in top_map]
        other_y = [r["mean_dy"] for r in subset if tuple(r["indices"]) not in top_map]
        if other_x:
            ax.scatter(other_x, other_y, s=8, c="lightgray", alpha=0.5, label=f"others ({len(other_x)})")
        for seq, idx in top_map.items():
            xs = [r["mean_dx"] for r in subset if tuple(r["indices"]) == seq]
            ys = [r["mean_dy"] for r in subset if tuple(r["indices"]) == seq]
            if xs:
                ax.scatter(xs, ys, s=10, alpha=0.8, color=cmap(idx), label=f"[{_seq_str(seq)}] ({len(xs)})")
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ax.axvline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("mean_dx")
        ax.set_ylabel("mean_dy")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    render(rows, out / "flow_scatter_combined.png", "Flow-Token Scatter (combined)")
    for ds in sorted({r["dataset_name"] for r in rows}):
        ds_rows = [r for r in rows if r["dataset_name"] == ds]
        render(ds_rows, out / f"flow_scatter_{ds}.png", f"Flow-Token Scatter ({ds})")


def analysis_5_token_dataset_heatmap(records: list[dict[str, Any]], out_dir: Path, top_k: int) -> None:
    out = out_dir / "analysis_5_heatmap"
    out.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    datasets = sorted({r["dataset_name"] for r in records})
    seq_counter = Counter(tuple(r["indices"]) for r in records)
    top_seqs = [seq for seq, _ in seq_counter.most_common(top_k)]
    if not top_seqs:
        return

    mat = np.zeros((len(top_seqs), len(datasets)), dtype=np.float32)
    count_mat = np.zeros((len(top_seqs), len(datasets)), dtype=np.int32)
    total_per_seq: list[int] = []
    for i, seq in enumerate(top_seqs):
        row_records = [r for r in records if tuple(r["indices"]) == seq]
        total = max(1, len(row_records))
        total_per_seq.append(int(len(row_records)))
        ds_counter = Counter(r["dataset_name"] for r in row_records)
        for j, ds in enumerate(datasets):
            cnt = int(ds_counter.get(ds, 0))
            count_mat[i, j] = cnt
            mat[i, j] = 100.0 * cnt / total

    fig, ax = plt.subplots(figsize=(2 + 1.4 * len(datasets), max(6, 0.42 * len(top_seqs))))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(top_seqs)))
    ax.set_yticklabels([f"{_seq_str(s)} (n={n})" for s, n in zip(top_seqs, total_per_seq)], fontsize=7)
    ann_fs = 7 if len(top_seqs) <= 24 else 6
    for i in range(len(top_seqs)):
        for j in range(len(datasets)):
            pct = float(mat[i, j])
            cnt = int(count_mat[i, j])
            txt = f"{cnt}\n{pct:.1f}%"
            color = "white" if pct >= 55.0 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=ann_fs, color=color)
    ax.set_title("Token-Dataset Distribution (count + %)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% from dataset")
    fig.tight_layout()
    fig.savefig(out / "heatmap.png", dpi=150)
    plt.close(fig)

    with (out / "distribution.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        cols = ["sequence", "total_n"]
        for ds in datasets:
            cols.extend([f"{ds}_count", f"{ds}_pct"])
        writer.writerow(cols)
        for i, seq in enumerate(top_seqs):
            row = [_seq_str(seq), total_per_seq[i]]
            for j, _ in enumerate(datasets):
                row.extend([int(count_mat[i, j]), f"{float(mat[i, j]):.4f}"])
            writer.writerow(row)


def analysis_2_cross_dataset_consistency(
    records: list[dict[str, Any]],
    out_dir: Path,
    min_samples_per_dataset: int,
) -> dict[str, Any]:
    out = out_dir / "analysis_2_consistency"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in records if not r["is_static"]]
    if not rows:
        metrics = {
            "mean_weighted_angular_variance": float("nan"),
            "num_qualifying_tokens": 0,
            "min_samples_per_dataset_threshold": int(min_samples_per_dataset),
        }
        _save_json(out / "metrics.json", metrics)
        return metrics

    by_seq_ds: dict[tuple[int, ...], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        seq = tuple(r["indices"])
        by_seq_ds[seq][r["dataset_name"]].append((r["mean_dx"], r["mean_dy"]))

    token_rows: list[dict[str, Any]] = []
    for seq, ds_map in by_seq_ds.items():
        filtered = {ds: vals for ds, vals in ds_map.items() if len(vals) >= min_samples_per_dataset}
        if len(filtered) < 2:
            continue
        per_dataset: dict[str, dict[str, float]] = {}
        unit_angles = []
        total_support = 0
        for ds, vals in filtered.items():
            arr = np.asarray(vals, dtype=np.float32)
            angles = np.arctan2(arr[:, 1], arr[:, 0])
            ux = float(np.mean(np.cos(angles)))
            uy = float(np.mean(np.sin(angles)))
            mean_angle = float(np.arctan2(uy, ux))
            per_dataset[ds] = {"angle": mean_angle, "n": int(arr.shape[0])}
            unit_angles.append((math.cos(mean_angle), math.sin(mean_angle)))
            total_support += int(arr.shape[0])

        vec = np.asarray(unit_angles, dtype=np.float32)
        mean_unit = np.mean(vec, axis=0)
        circ_var = 1.0 - float(np.linalg.norm(mean_unit))
        token_rows.append(
            {
                "sequence": seq,
                "variance": circ_var,
                "support": total_support,
                "per_dataset": per_dataset,
            }
        )

    token_rows.sort(key=lambda x: x["support"], reverse=True)

    if token_rows:
        weighted_num = sum(r["variance"] * r["support"] for r in token_rows)
        weighted_den = sum(r["support"] for r in token_rows)
        mean_weighted = weighted_num / max(1, weighted_den)
    else:
        mean_weighted = float("nan")

    datasets = sorted({r["dataset_name"] for r in rows})
    with (out / "consistency.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        cols = []
        for ds in datasets:
            cols.extend([f"{ds}_angle_rad", f"{ds}_angle_deg", f"{ds}_n"])
        writer.writerow(["sequence", "variance", "support", "num_datasets", *cols])
        for r in token_rows:
            row = [_seq_str(r["sequence"]), f"{r['variance']:.6f}", r["support"], len(r["per_dataset"])]
            for ds in datasets:
                if ds in r["per_dataset"]:
                    angle_rad = float(r["per_dataset"][ds]["angle"])
                    row.append(f"{angle_rad:.6f}")
                    row.append(f"{np.degrees(angle_rad):.2f}")
                    row.append(r["per_dataset"][ds]["n"])
                else:
                    row.extend(["", "", ""])
            writer.writerow(row)

    plot_rows = token_rows[: min(20, len(token_rows))]
    if plot_rows:
        fig, ax1 = plt.subplots(figsize=(max(10, len(plot_rows) * 0.6), 6))
        x = np.arange(len(plot_rows))
        for ds in datasets:
            xs, ys = [], []
            for i, r in enumerate(plot_rows):
                d = r["per_dataset"].get(ds)
                if d is not None:
                    xs.append(i)
                    ys.append(np.degrees(d["angle"]))
            if xs:
                ax1.scatter(xs, ys, s=40, label=ds)
        ax1.set_ylabel("Mean angle (deg)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([_seq_str(r["sequence"]) for r in plot_rows], rotation=45, ha="right", fontsize=8)
        ax1.set_title("Cross-Dataset Token Consistency (top support sequences)")
        ax2 = ax1.twinx()
        ax2.plot(x, [r["variance"] for r in plot_rows], color="black", marker="o", linewidth=1.0, label="variance")
        ax2.set_ylabel("Circular variance")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(out / "consistency.png", dpi=150)
        plt.close(fig)

    metrics = {
        "mean_weighted_angular_variance": mean_weighted,
        "num_qualifying_tokens": len(token_rows),
        "min_samples_per_dataset_threshold": int(min_samples_per_dataset),
    }
    _save_json(out / "metrics.json", metrics)
    (out / "README.txt").write_text(
        "How to read this analysis:\n"
        "- Each sequence has one mean direction per dataset.\n"
        "- Circular variance near 0 means sequence direction is consistent across datasets.\n"
        "- Higher variance means sequence direction changes across datasets.\n"
        "- Use consistency.csv for exact per-dataset angles and counts.\n"
    )
    return metrics


def analysis_3_sequence_examples(
    records: list[dict[str, Any]],
    visual_samples: list[dict[str, Any]],
    out_dir: Path,
    inspected_sequences: list[tuple[int, ...]],
    examples_per_sequence: int,
) -> None:
    out = out_dir / "analysis_3_examples"
    out.mkdir(parents=True, exist_ok=True)
    if not records or not visual_samples:
        return

    seq_counter = Counter(tuple(r["indices"]) for r in records)
    top_seqs = list(inspected_sequences)
    for rank, seq in enumerate(top_seqs):
        matches = [v for v in visual_samples if tuple(v["indices"]) == seq]
        if not matches:
            continue
        random.shuffle(matches)
        n = min(int(examples_per_sequence), len(matches))
        if n == 0:
            continue

        seq_records = [r for r in records if tuple(r["indices"]) == seq]
        ds_counts = Counter(r["dataset_name"] for r in seq_records)
        breakdown = " | ".join(f"{k}:{v}" for k, v in ds_counts.items())

        fig, axes = plt.subplots(n, 2, figsize=(7.2, max(2, n * 1.7)))
        if n == 1:
            axes = np.asarray([axes])
        for i in range(n):
            sample = matches[i]
            fr = sample["frames"]  # [C, 2, H, W]
            img0 = _tensor_to_img(fr[:, 0])
            img1 = _tensor_to_img(fr[:, -1])
            axes[i, 0].imshow(img0)
            axes[i, 1].imshow(img1)
            axes[i, 0].set_ylabel(sample["dataset_name"], fontsize=8)
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
        axes[0, 0].set_title("frame_t")
        axes[0, 1].set_title("frame_t+k")
        fig.suptitle(
            f"Sequence [{_seq_str(seq)}] | n={seq_counter[seq]} | {breakdown}",
            fontsize=9,
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        fig.savefig(out / f"seq_{rank}_{_seq_str(seq)}.png", dpi=140)
        plt.close(fig)


def analysis_4_latent_transfer(
    task: LAQTask,
    records: list[dict[str, Any]],
    visual_samples: list[dict[str, Any]],
    out_dir: Path,
    inspected_sequences: list[tuple[int, ...]],
    top_k_transfer: int,
) -> None:
    out = out_dir / "analysis_4_transfer"
    out.mkdir(parents=True, exist_ok=True)
    if not records or not visual_samples:
        return

    seq_counter = Counter(tuple(r["indices"]) for r in records)
    top_seqs = list(inspected_sequences)[:top_k_transfer]
    if not top_seqs:
        top_seqs = [seq for seq, _ in seq_counter.most_common(top_k_transfer)]
    by_dataset = defaultdict(list)
    for i, s in enumerate(visual_samples):
        by_dataset[s["dataset_name"]].append((i, s))
    datasets = sorted(by_dataset.keys())

    for seq in top_seqs:
        source_candidates = [s for s in visual_samples if tuple(s["indices"]) == seq]
        if not source_candidates:
            continue
        source = max(
            source_candidates,
            key=lambda s: float((s["frames"][:, -1] - s["frames"][:, 0]).abs().mean().item()),
        )

        targets: list[dict[str, Any]] = []
        for ds in datasets:
            for _, item in by_dataset[ds][:2]:
                targets.append(item)
        targets = targets[:8]
        if not targets:
            continue

        with torch.no_grad():
            source_pair = source["frames"].unsqueeze(0).to(task.device)
            source_latents, _ = task.encode_latents(source_pair)
            source_first = source_pair[:, :, 0]
            source_pred = task.decode_with_latents(source_first, source_latents)
            if source_pred is not None and source_pred.ndim == 5:
                source_pred = source_pred[:, :, 0]
            first_targets = torch.stack([t["frames"][:, 0] for t in targets], dim=0).to(task.device)
            latents_rep = source_latents.expand(first_targets.shape[0], -1, -1).contiguous()
            pred = task.decode_with_latents(first_targets, latents_rep)
            if pred is None:
                continue
            if pred.ndim == 5:
                pred = pred[:, :, 0]
            source_flow = task.model.flow_teacher.compute_flow(source_pair[:, :, :1], source_pair[:, :, -1:])
            source_flow_rgb = _flow_to_rgb(source_flow, denoise=False)[0]
            source_pred_flow_rgb = source_flow_rgb
            if source_pred is not None:
                source_pred_flow = task.model.flow_teacher.compute_flow(
                    source_first.unsqueeze(2), source_pred.unsqueeze(2)
                )
                source_pred_flow_rgb = _flow_to_rgb(source_pred_flow)[0]
            pred_flow = task.model.flow_teacher.compute_flow(
                first_targets.unsqueeze(2), pred.unsqueeze(2)
            )
            pred_flow_rgb = _flow_to_rgb(pred_flow)

        fig, axes = plt.subplots(1 + len(targets), 4, figsize=(12, 2.2 * (1 + len(targets))))
        if len(targets) == 0:
            axes = np.asarray([axes])

        axes[0, 0].imshow(_tensor_to_img(source["frames"][:, 0]))
        axes[0, 1].imshow(_tensor_to_img(source["frames"][:, -1]))
        if source_pred is not None:
            axes[0, 2].imshow(_tensor_to_img(source_pred[0].detach().cpu()))
        else:
            axes[0, 2].imshow(_tensor_to_img(source["frames"][:, -1]))
        axes[0, 3].imshow(_tensor_to_img(source_pred_flow_rgb))
        axes[0, 0].set_ylabel("source", fontsize=9)
        for c in range(4):
            axes[0, c].axis("off")
        axes[0, 0].set_title("frame_t")
        axes[0, 1].set_title("real_t+k")
        axes[0, 2].set_title("pred_t+k (transfer)")
        axes[0, 3].set_title("flow(frame_t->pred)")

        pred_cpu = pred.detach().cpu()
        for i, tgt in enumerate(targets, start=1):
            axes[i, 0].imshow(_tensor_to_img(tgt["frames"][:, 0]))
            axes[i, 1].imshow(_tensor_to_img(tgt["frames"][:, -1]))
            axes[i, 2].imshow(_tensor_to_img(pred_cpu[i - 1]))
            axes[i, 3].imshow(_tensor_to_img(pred_flow_rgb[i - 1]))
            axes[i, 0].set_ylabel(f"{i}:{tgt['dataset_name']}", fontsize=8)
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
            axes[i, 2].axis("off")
            axes[i, 3].axis("off")

        fig.suptitle(
            f"Latent Transfer [{_seq_str(seq)}] | col0: target_t | col1: real_t+k | col2: predicted_t+k (transfer) | col3: flow(target_t->pred)",
            fontsize=9,
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        fig.savefig(out / f"transfer_seq_{_seq_str(seq)}.png", dpi=140)
        plt.close(fig)
        fig2, ax2 = plt.subplots(1, 2, figsize=(6, 3))
        ax2[0].imshow(_tensor_to_img(source["frames"][:, 0]))
        ax2[0].set_title("source_t")
        ax2[1].imshow(_tensor_to_img(source_flow_rgb))
        ax2[1].set_title("source_gt_flow")
        for a in ax2:
            a.axis("off")
        fig2.tight_layout()
        fig2.savefig(out / f"source_gt_flow_{_seq_str(seq)}.png", dpi=140)
        plt.close(fig2)


def analysis_6_per_token_compass(
    records: list[dict[str, Any]],
    out_dir: Path,
    inspected_sequences: list[tuple[int, ...]],
    codebook_size: int,
    min_samples: int,
) -> None:
    out = out_dir / "analysis_6_compass"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in records if not r["is_static"]]
    if not rows:
        return
    seq_counter = Counter(tuple(r["indices"]) for r in rows)
    top_seqs = list(inspected_sequences)
    if not top_seqs:
        top_seqs = [seq for seq, _ in seq_counter.most_common(24)]

    ds_vals = defaultdict(list)
    for r in rows:
        ds_vals[r["dataset_name"]].append((r["mean_dx"], r["mean_dy"]))
    ds_mean: dict[str, tuple[float, float]] = {}
    for ds, vals in ds_vals.items():
        arr = np.asarray(vals, dtype=np.float32)
        ds_mean[ds] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))

    with (out / "dataset_baseline.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "mean_dx", "mean_dy", "count"])
        for ds in sorted(ds_mean.keys()):
            mdx, mdy = ds_mean[ds]
            writer.writerow([ds, f"{mdx:.6f}", f"{mdy:.6f}", len(ds_vals[ds])])

    data_rows = []
    for seq in top_seqs:
        seq_rows = [r for r in rows if tuple(r["indices"]) == seq]
        if len(seq_rows) < min_samples:
            continue
        arr = np.asarray([(r["mean_dx"], r["mean_dy"]) for r in seq_rows], dtype=np.float32)
        mean_dx = float(arr[:, 0].mean())
        mean_dy = float(arr[:, 1].mean())
        centered = np.asarray(
            [
                (
                    r["mean_dx"] - ds_mean[r["dataset_name"]][0],
                    r["mean_dy"] - ds_mean[r["dataset_name"]][1],
                )
                for r in seq_rows
            ],
            dtype=np.float32,
        )
        mean_dx_centered = float(centered[:, 0].mean())
        mean_dy_centered = float(centered[:, 1].mean())
        ds_counts = Counter(r["dataset_name"] for r in seq_rows)
        data_rows.append(
            {
                "sequence": _seq_str(seq),
                "count": len(seq_rows),
                "num_datasets": len(ds_counts),
                "mean_dx": mean_dx,
                "mean_dy": mean_dy,
                "angle_deg": float(np.degrees(np.arctan2(mean_dy, mean_dx))),
                "mean_dx_centered": mean_dx_centered,
                "mean_dy_centered": mean_dy_centered,
                "angle_centered_deg": float(np.degrees(np.arctan2(mean_dy_centered, mean_dx_centered))),
                "dataset_breakdown": "|".join(f"{k}:{v}" for k, v in sorted(ds_counts.items())),
            }
        )

    with (out / "compass_data.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "count",
                "num_datasets",
                "mean_dx",
                "mean_dy",
                "angle_deg",
                "mean_dx_centered",
                "mean_dy_centered",
                "angle_centered_deg",
                "dataset_breakdown",
            ],
        )
        writer.writeheader()
        writer.writerows(data_rows)

    (out / "README.txt").write_text(
        "Compass interpretation:\n"
        "- Left panel: raw token mean motion (dataset biases included).\n"
        "- Middle panel: dataset-centered token mean motion (dataset baseline removed).\n"
        "- If many raw arrows share one direction, check centered panel + dataset_baseline.csv.\n"
    )

    if not data_rows:
        return

    max_raw = max(math.hypot(r["mean_dx"], r["mean_dy"]) for r in data_rows)
    max_ctr = max(math.hypot(r["mean_dx_centered"], r["mean_dy_centered"]) for r in data_rows)
    lim = max(0.05, max(max_raw, max_ctr) * 1.35)
    fig, (ax_raw, ax_ctr, ax_txt) = plt.subplots(
        1, 3, figsize=(18, 7), gridspec_kw={"width_ratios": [1.0, 1.0, 1.3]}
    )
    for i, r in enumerate(data_rows):
        ax_raw.arrow(
            0,
            0,
            r["mean_dx"],
            r["mean_dy"],
            head_width=lim * 0.03,
            alpha=0.8,
            length_includes_head=True,
        )
        ax_raw.text(r["mean_dx"] * 1.03, r["mean_dy"] * 1.03, str(i), fontsize=8)
        ax_ctr.arrow(
            0,
            0,
            r["mean_dx_centered"],
            r["mean_dy_centered"],
            head_width=lim * 0.03,
            alpha=0.8,
            length_includes_head=True,
        )
        ax_ctr.text(
            r["mean_dx_centered"] * 1.03,
            r["mean_dy_centered"] * 1.03,
            str(i),
            fontsize=8,
        )
    for ax, title in [
        (ax_raw, "Raw Mean Motion"),
        (ax_ctr, "Dataset-Centered Mean Motion"),
    ]:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ax.axvline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("mean_dx")
        ax.set_ylabel("mean_dy")

    ax_txt.axis("off")
    lines = ["Index -> sequence | n | #datasets | angle(raw/centered) | ds split"]
    for i, r in enumerate(data_rows):
        lines.append(
            f"{i:02d}: {r['sequence']} | {r['count']} | {r['num_datasets']} | "
            f"{r['angle_deg']:.1f}/{r['angle_centered_deg']:.1f} | {r['dataset_breakdown']}"
        )
    lines.append("")
    lines.append("Dataset baselines (all non-static samples):")
    for ds in sorted(ds_mean.keys()):
        mdx, mdy = ds_mean[ds]
        lines.append(f"- {ds}: ({mdx:.4f}, {mdy:.4f}), n={len(ds_vals[ds])}")
    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(out / "compass.png", dpi=150)
    plt.close(fig)


def _indices_to_flow_action_tokens(model, indices: torch.Tensor) -> torch.Tensor:
    raw = model.vq.codebooks[indices]
    tokens = model.vq.project_out(raw)
    action_h, action_w = model.action_shape
    return rearrange(tokens, "b (h w) d -> b 1 h w d", h=action_h, w=action_w)


def _predict_flow_from_indices(model, first_frames_bchw: torch.Tensor, indices_b4: torch.Tensor) -> torch.Tensor:
    first_frames = first_frames_bchw.unsqueeze(2)  # [B, C, 1, H, W]
    action_tokens = _indices_to_flow_action_tokens(model, indices_b4)
    pixel_context = model.decoder_context_projection(first_frames)
    h, w = model.patch_height_width
    attn_bias = model.spatial_rel_pos_bias(h, w, device=first_frames.device)
    pred_flow = model.flow_decoder(pixel_context, action_tokens, attn_bias)
    return pred_flow


def analysis_7_flow_decoder_transfer(
    task: LAQTask,
    records: list[dict[str, Any]],
    visual_samples: list[dict[str, Any]],
    out_dir: Path,
    inspected_sequences: list[tuple[int, ...]],
    top_k_transfer: int,
    top_k_tokens: int,
) -> None:
    out = out_dir / "analysis_7_flow_transfer"
    out.mkdir(parents=True, exist_ok=True)
    model = task.model
    if model.flow_decoder is None:
        return
    if not records or not visual_samples:
        return

    seq_counter = Counter(tuple(r["indices"]) for r in records)
    top_seqs = list(inspected_sequences)[:top_k_transfer]
    if not top_seqs:
        top_seqs = [seq for seq, _ in seq_counter.most_common(top_k_transfer)]
    datasets = sorted({v["dataset_name"] for v in visual_samples})
    (out / "INSPECTED_TOKENS.txt").write_text("\n".join(_seq_str(s) for s in top_seqs))

    # 7a: same token, different images
    for seq in top_seqs:
        targets = []
        for ds in datasets:
            ds_samples = [v for v in visual_samples if v["dataset_name"] == ds]
            targets.extend(ds_samples[:2])
        targets = targets[:8]
        if not targets:
            continue

        with torch.no_grad():
            first = torch.stack([t["frames"][:, 0] for t in targets], dim=0).to(task.device)
            idx = torch.tensor([list(seq)] * first.shape[0], dtype=torch.long, device=task.device)
            pred_flow = _predict_flow_from_indices(model, first, idx)
            flow_rgb = _flow_to_rgb(pred_flow)
            gt_pair = torch.stack([t["frames"] for t in targets], dim=0).to(task.device)
            gt_flow = model.flow_teacher.compute_flow(gt_pair[:, :, :1], gt_pair[:, :, -1:])
            gt_flow_rgb = _flow_to_rgb(gt_flow, denoise=False)

        fig, axes = plt.subplots(len(targets), 4, figsize=(12, max(3, 1.8 * len(targets))))
        if len(targets) == 1:
            axes = np.asarray([axes])
        for i, t in enumerate(targets):
            axes[i, 0].imshow(_tensor_to_img(t["frames"][:, 0]))
            axes[i, 1].imshow(_tensor_to_img(t["frames"][:, -1]))
            axes[i, 2].imshow(_tensor_to_img(gt_flow_rgb[i]))
            axes[i, 3].imshow(_tensor_to_img(flow_rgb[i]))
            axes[i, 0].set_ylabel(t["dataset_name"], fontsize=8)
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
            axes[i, 2].axis("off")
            axes[i, 3].axis("off")
        axes[0, 0].set_title("frame_t")
        axes[0, 1].set_title("real_t+k")
        axes[0, 2].set_title("gt_flow")
        axes[0, 3].set_title("pred_flow (fixed token)")
        fig.suptitle(
            f"Same Token [{_seq_str(seq)}], Different Images",
            fontsize=10,
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        fig.savefig(out / f"same_token_{_seq_str(seq)}.png", dpi=150)
        plt.close(fig)

    # 7b: different tokens, same image
    top_tokens = list(inspected_sequences)[:top_k_tokens]
    if not top_tokens:
        top_tokens = [seq for seq, _ in seq_counter.most_common(top_k_tokens)]
    if not top_tokens:
        return
    seeds = []
    for ds in datasets:
        ds_samples = [v for v in visual_samples if v["dataset_name"] == ds]
        if ds_samples:
            seeds.append(ds_samples[0])
    seeds = seeds[:4]
    for i, seed in enumerate(seeds):
        own_seq = tuple(seed["indices"])
        token_list = [own_seq] + [s for s in top_tokens if s != own_seq]
        token_list = token_list[:top_k_tokens]
        with torch.no_grad():
            first = seed["frames"][:, 0].unsqueeze(0).repeat(len(token_list), 1, 1, 1).to(task.device)
            idx = torch.tensor([list(s) for s in token_list], dtype=torch.long, device=task.device)
            pred_flow = _predict_flow_from_indices(model, first, idx)
            flow_rgb = _flow_to_rgb(pred_flow)
            seed_pair = seed["frames"].unsqueeze(0).to(task.device)
            gt_flow = model.flow_teacher.compute_flow(seed_pair[:, :, :1], seed_pair[:, :, -1:])
            gt_flow_rgb = _flow_to_rgb(gt_flow, denoise=False)[0]

        ncols = 3 + len(token_list)
        fig, axes = plt.subplots(1, ncols, figsize=(2.0 * ncols, 2.6))
        axes[0].imshow(_tensor_to_img(seed["frames"][:, 0]))
        axes[0].set_title("frame_t")
        axes[0].axis("off")
        axes[1].imshow(_tensor_to_img(seed["frames"][:, -1]))
        axes[1].set_title("frame_t+k")
        axes[1].axis("off")
        axes[2].imshow(_tensor_to_img(gt_flow_rgb))
        axes[2].set_title("gt_flow")
        axes[2].axis("off")
        for k, seq in enumerate(token_list):
            j = 3 + k
            axes[j].imshow(_tensor_to_img(flow_rgb[k]))
            label = _seq_str(seq)
            if tuple(seq) == own_seq:
                label = f"own:{label}"
            axes[j].set_title(label, fontsize=7)
            axes[j].axis("off")
        fig.suptitle(
            f"Different Tokens, Same Image ({seed['dataset_name']})",
            fontsize=10,
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        fig.savefig(out / f"diff_tokens_{seed['dataset_name']}_{i}.png", dpi=150)
        plt.close(fig)

    # 7c: similar start state (same dataset), token swap
    for ds in datasets:
        cand = [v for v in visual_samples if v["dataset_name"] == ds and v.get("initial_state") is not None]
        if len(cand) < 2:
            continue
        cand = cand[:80]
        best = None
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                a = cand[i]
                b = cand[j]
                sa = a["initial_state"]
                sb = b["initial_state"]
                if sa is None or sb is None:
                    continue
                dist = float(np.linalg.norm(sa - sb))
                same_tok = tuple(a["indices"]) == tuple(b["indices"])
                score = (1 if same_tok else 0, dist)
                if best is None or score < best[0]:
                    best = (score, a, b, dist)
        if best is None:
            continue
        _, a, b, dist = best
        tok_a = tuple(a["indices"])
        tok_b = tuple(b["indices"])

        with torch.no_grad():
            first_a = a["frames"][:, 0].unsqueeze(0).to(task.device)
            first_b = b["frames"][:, 0].unsqueeze(0).to(task.device)
            idx_a = torch.tensor([list(tok_a)], dtype=torch.long, device=task.device)
            idx_b = torch.tensor([list(tok_b)], dtype=torch.long, device=task.device)

            flow_a_own = _predict_flow_from_indices(model, first_a, idx_a)
            flow_a_swap = _predict_flow_from_indices(model, first_a, idx_b)
            flow_b_own = _predict_flow_from_indices(model, first_b, idx_b)
            flow_b_swap = _predict_flow_from_indices(model, first_b, idx_a)

            gt_flow_a = model.flow_teacher.compute_flow(
                a["frames"].unsqueeze(0).to(task.device)[:, :, :1],
                a["frames"].unsqueeze(0).to(task.device)[:, :, -1:],
            )
            gt_flow_b = model.flow_teacher.compute_flow(
                b["frames"].unsqueeze(0).to(task.device)[:, :, :1],
                b["frames"].unsqueeze(0).to(task.device)[:, :, -1:],
            )

            flow_a_own_rgb = _flow_to_rgb(flow_a_own)[0]
            flow_a_swap_rgb = _flow_to_rgb(flow_a_swap)[0]
            flow_b_own_rgb = _flow_to_rgb(flow_b_own)[0]
            flow_b_swap_rgb = _flow_to_rgb(flow_b_swap)[0]
            gt_flow_a_rgb = _flow_to_rgb(gt_flow_a, denoise=False)[0]
            gt_flow_b_rgb = _flow_to_rgb(gt_flow_b, denoise=False)[0]

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        # Row A
        axes[0, 0].imshow(_tensor_to_img(a["frames"][:, 0]))
        axes[0, 1].imshow(_tensor_to_img(a["frames"][:, -1]))
        axes[0, 2].imshow(_tensor_to_img(gt_flow_a_rgb))
        axes[0, 3].imshow(_tensor_to_img(flow_a_own_rgb))
        axes[0, 4].imshow(_tensor_to_img(flow_a_swap_rgb))
        axes[0, 0].set_ylabel(f"A tok={_seq_str(tok_a)}", fontsize=8)
        # Row B
        axes[1, 0].imshow(_tensor_to_img(b["frames"][:, 0]))
        axes[1, 1].imshow(_tensor_to_img(b["frames"][:, -1]))
        axes[1, 2].imshow(_tensor_to_img(gt_flow_b_rgb))
        axes[1, 3].imshow(_tensor_to_img(flow_b_own_rgb))
        axes[1, 4].imshow(_tensor_to_img(flow_b_swap_rgb))
        axes[1, 0].set_ylabel(f"B tok={_seq_str(tok_b)}", fontsize=8)
        for r in range(2):
            for c in range(5):
                axes[r, c].axis("off")
        titles = ["frame_t", "frame_t+k", "gt_flow", "pred own token", "pred swapped token"]
        for c, t in enumerate(titles):
            axes[0, c].set_title(t, fontsize=9)
        fig.suptitle(
            f"Similar-start token swap ({ds}) | ||state_A-state_B||={dist:.4f}",
            fontsize=10,
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        fig.savefig(out / f"similar_start_swap_{ds}.png", dpi=150)
        plt.close(fig)


def analysis_8_token_reports(
    task: LAQTask,
    records: list[dict[str, Any]],
    visual_samples: list[dict[str, Any]],
    out_dir: Path,
    inspected_sequences: list[tuple[int, ...]],
    top_k_reports: int,
    examples_per_token: int,
    targets_per_dataset: int,
) -> None:
    out = out_dir / "analysis_8_token_reports"
    out.mkdir(parents=True, exist_ok=True)
    if not records or not visual_samples:
        return

    model = task.model
    seq_counter = Counter(tuple(r["indices"]) for r in records)
    seqs = list(inspected_sequences)[:top_k_reports]
    if not seqs:
        seqs = [s for s, _ in seq_counter.most_common(top_k_reports)]
    if not seqs:
        return

    rows_non_static = [r for r in records if not r["is_static"]]
    ds_all = sorted({r["dataset_name"] for r in records})
    ds_baseline: dict[str, tuple[float, float]] = {}
    for ds in ds_all:
        vals = [(r["mean_dx"], r["mean_dy"]) for r in rows_non_static if r["dataset_name"] == ds]
        if vals:
            arr = np.asarray(vals, dtype=np.float32)
            ds_baseline[ds] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
        else:
            ds_baseline[ds] = (0.0, 0.0)

    _save_json(
        out / "dataset_baseline.json",
        {k: {"mean_dx": v[0], "mean_dy": v[1]} for k, v in sorted(ds_baseline.items())},
    )

    for rank, seq in enumerate(seqs):
        seq_str = _seq_str(seq)
        token_dir = out / f"token_{rank:02d}_{seq_str}"
        token_dir.mkdir(parents=True, exist_ok=True)

        token_records = [r for r in records if tuple(r["indices"]) == seq]
        token_visual = [v for v in visual_samples if tuple(v["indices"]) == seq]
        if not token_records:
            continue

        ds_counts = Counter(r["dataset_name"] for r in token_records)
        arr = np.asarray([(r["mean_dx"], r["mean_dy"]) for r in token_records], dtype=np.float32)
        mean_dx = float(arr[:, 0].mean())
        mean_dy = float(arr[:, 1].mean())
        centered_arr = np.asarray(
            [
                (
                    r["mean_dx"] - ds_baseline[r["dataset_name"]][0],
                    r["mean_dy"] - ds_baseline[r["dataset_name"]][1],
                )
                for r in token_records
            ],
            dtype=np.float32,
        )
        mean_dx_centered = float(centered_arr[:, 0].mean())
        mean_dy_centered = float(centered_arr[:, 1].mean())
        summary = {
            "sequence": seq_str,
            "total_count": int(len(token_records)),
            "visual_count": int(len(token_visual)),
            "num_datasets": int(len(ds_counts)),
            "per_dataset_counts": {k: int(v) for k, v in sorted(ds_counts.items())},
            "mean_shift_raw": {
                "dx": mean_dx,
                "dy": mean_dy,
                "angle_deg": float(np.degrees(np.arctan2(mean_dy, mean_dx))),
                "magnitude": float(np.hypot(mean_dx, mean_dy)),
            },
            "mean_shift_dataset_centered": {
                "dx": mean_dx_centered,
                "dy": mean_dy_centered,
                "angle_deg": float(np.degrees(np.arctan2(mean_dy_centered, mean_dx_centered))),
                "magnitude": float(np.hypot(mean_dx_centered, mean_dy_centered)),
            },
            "dataset_baseline_used": {
                k: {"mean_dx": ds_baseline[k][0], "mean_dy": ds_baseline[k][1]} for k in sorted(ds_baseline.keys())
            },
        }
        _save_json(token_dir / "summary.json", summary)

        # A) sample gallery for this token
        if token_visual:
            sample_pool = list(token_visual)
            random.shuffle(sample_pool)
            picks = _pick_balanced_samples(sample_pool, limit=int(examples_per_token))
            with torch.no_grad():
                pair = torch.stack([p["frames"] for p in picks], dim=0).to(task.device)
                gt_flow = model.flow_teacher.compute_flow(pair[:, :, :1], pair[:, :, -1:])
                gt_flow_rgb = _flow_to_rgb(gt_flow, denoise=False)
            n = len(picks)
            fig, axes = plt.subplots(n, 3, figsize=(9, max(3, 1.9 * n)))
            if n == 1:
                axes = np.asarray([axes])
            for i, s in enumerate(picks):
                axes[i, 0].imshow(_tensor_to_img(s["frames"][:, 0]))
                axes[i, 1].imshow(_tensor_to_img(s["frames"][:, -1]))
                axes[i, 2].imshow(_tensor_to_img(gt_flow_rgb[i]))
                axes[i, 0].set_ylabel(s["dataset_name"], fontsize=8)
                for c in range(3):
                    axes[i, c].axis("off")
            axes[0, 0].set_title("frame_t")
            axes[0, 1].set_title("frame_t+k")
            axes[0, 2].set_title("gt_flow")
            fig.suptitle(f"Token [{seq_str}] samples (n={len(token_records)})", fontsize=10, y=0.995)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
            fig.savefig(token_dir / "samples.png", dpi=150)
            plt.close(fig)

        # B) transfer report for this token
        if token_visual:
            source = max(
                token_visual,
                key=lambda s: float((s["frames"][:, -1] - s["frames"][:, 0]).abs().mean().item()),
            )
            targets: list[dict[str, Any]] = []
            for ds in ds_all:
                ds_pool = [v for v in visual_samples if v["dataset_name"] == ds]
                random.shuffle(ds_pool)
                targets.extend(ds_pool[: int(targets_per_dataset)])
            targets = targets[: max(1, int(targets_per_dataset) * max(1, len(ds_all)))]
            if targets:
                with torch.no_grad():
                    source_pair = source["frames"].unsqueeze(0).to(task.device)
                    source_latents, _ = task.encode_latents(source_pair)
                    source_first = source_pair[:, :, 0]
                    source_pred = task.decode_with_latents(source_first, source_latents)
                    if source_pred is not None and source_pred.ndim == 5:
                        source_pred = source_pred[:, :, 0]
                    source_gt_flow = model.flow_teacher.compute_flow(source_pair[:, :, :1], source_pair[:, :, -1:])
                    source_gt_flow_rgb = _flow_to_rgb(source_gt_flow, denoise=False)[0]
                    source_pred_flow_rgb = source_gt_flow_rgb
                    if source_pred is not None:
                        source_pred_flow = model.flow_teacher.compute_flow(
                            source_first.unsqueeze(2), source_pred.unsqueeze(2)
                        )
                        source_pred_flow_rgb = _flow_to_rgb(source_pred_flow)[0]

                    first_targets = torch.stack([t["frames"][:, 0] for t in targets], dim=0).to(task.device)
                    latents_rep = source_latents.expand(first_targets.shape[0], -1, -1).contiguous()
                    pred = task.decode_with_latents(first_targets, latents_rep)
                    if pred is not None and pred.ndim == 5:
                        pred = pred[:, :, 0]
                    pred_flow_rgb = None
                    gt_flow_rgb = None
                    if pred is not None:
                        pred_flow = model.flow_teacher.compute_flow(first_targets.unsqueeze(2), pred.unsqueeze(2))
                        pred_flow_rgb = _flow_to_rgb(pred_flow)
                    tgt_pair = torch.stack([t["frames"] for t in targets], dim=0).to(task.device)
                    gt_flow = model.flow_teacher.compute_flow(tgt_pair[:, :, :1], tgt_pair[:, :, -1:])
                    gt_flow_rgb = _flow_to_rgb(gt_flow, denoise=False)

                if pred is not None:
                    pred_cpu = pred.detach().cpu()
                    n = len(targets)
                    fig, axes = plt.subplots(1 + n, 5, figsize=(14, max(4, 2.0 * (1 + n))))
                    if n == 0:
                        axes = np.asarray([axes])
                    axes[0, 0].imshow(_tensor_to_img(source["frames"][:, 0]))
                    axes[0, 1].imshow(_tensor_to_img(source["frames"][:, -1]))
                    axes[0, 2].imshow(_tensor_to_img(source_pred[0].detach().cpu() if source_pred is not None else source["frames"][:, -1]))
                    axes[0, 3].imshow(_tensor_to_img(source_gt_flow_rgb))
                    axes[0, 4].imshow(_tensor_to_img(source_pred_flow_rgb))
                    axes[0, 0].set_ylabel("source", fontsize=9)
                    for c in range(5):
                        axes[0, c].axis("off")
                    for i, t in enumerate(targets, start=1):
                        axes[i, 0].imshow(_tensor_to_img(t["frames"][:, 0]))
                        axes[i, 1].imshow(_tensor_to_img(t["frames"][:, -1]))
                        axes[i, 2].imshow(_tensor_to_img(pred_cpu[i - 1]))
                        axes[i, 3].imshow(_tensor_to_img(gt_flow_rgb[i - 1]))
                        axes[i, 4].imshow(_tensor_to_img(pred_flow_rgb[i - 1]))
                        axes[i, 0].set_ylabel(t["dataset_name"], fontsize=8)
                        for c in range(5):
                            axes[i, c].axis("off")
                    axes[0, 0].set_title("frame_t")
                    axes[0, 1].set_title("real_t+k")
                    axes[0, 2].set_title("pred_t+k (transfer)")
                    axes[0, 3].set_title("gt_flow")
                    axes[0, 4].set_title("transfer_flow")
                    fig.suptitle(f"Token [{seq_str}] latent transfer report", fontsize=10, y=0.995)
                    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
                    fig.savefig(token_dir / "latent_transfer.png", dpi=150)
                    plt.close(fig)

        # C) mean-shift diagnostics (raw vs centered)
        fig, (ax_raw, ax_ctr) = plt.subplots(1, 2, figsize=(12, 5))
        colors = plt.get_cmap("tab10")
        ds_order = sorted(ds_counts.keys())
        for j, ds in enumerate(ds_order):
            ds_pts = [r for r in token_records if r["dataset_name"] == ds]
            xs = [r["mean_dx"] for r in ds_pts]
            ys = [r["mean_dy"] for r in ds_pts]
            ax_raw.scatter(xs, ys, s=14, alpha=0.7, color=colors(j % 10), label=f"{ds} ({len(ds_pts)})")
            xs_c = [r["mean_dx"] - ds_baseline[ds][0] for r in ds_pts]
            ys_c = [r["mean_dy"] - ds_baseline[ds][1] for r in ds_pts]
            ax_ctr.scatter(xs_c, ys_c, s=14, alpha=0.7, color=colors(j % 10), label=f"{ds} ({len(ds_pts)})")

        ax_raw.scatter([mean_dx], [mean_dy], marker="*", s=150, color="black", label="token mean")
        ax_ctr.scatter([mean_dx_centered], [mean_dy_centered], marker="*", s=150, color="black", label="token mean")
        for ax, title in [(ax_raw, "Raw mean flow points"), (ax_ctr, "Dataset-centered mean flow points")]:
            ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
            ax.axvline(0.0, color="black", linewidth=0.5, alpha=0.4)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("mean_dx")
            ax.set_ylabel("mean_dy")
            ax.set_title(title)
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(
            f"Token [{seq_str}] shift | raw angle={summary['mean_shift_raw']['angle_deg']:.1f} deg | "
            f"centered angle={summary['mean_shift_dataset_centered']['angle_deg']:.1f} deg",
            fontsize=10,
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        fig.savefig(token_dir / "mean_shift.png", dpi=150)
        plt.close(fig)

        # D) flow-decoder behavior (same token on varied images)
        if model.flow_decoder is not None:
            targets: list[dict[str, Any]] = []
            for ds in ds_all:
                ds_pool = [v for v in visual_samples if v["dataset_name"] == ds]
                random.shuffle(ds_pool)
                targets.extend(ds_pool[: int(targets_per_dataset)])
            targets = targets[: max(1, int(targets_per_dataset) * max(1, len(ds_all)))]
            if targets:
                with torch.no_grad():
                    first = torch.stack([t["frames"][:, 0] for t in targets], dim=0).to(task.device)
                    idx = torch.tensor([list(seq)] * first.shape[0], dtype=torch.long, device=task.device)
                    pred_flow = _predict_flow_from_indices(model, first, idx)
                    pred_flow_rgb = _flow_to_rgb(pred_flow)
                    gt_pair = torch.stack([t["frames"] for t in targets], dim=0).to(task.device)
                    gt_flow = model.flow_teacher.compute_flow(gt_pair[:, :, :1], gt_pair[:, :, -1:])
                    gt_flow_rgb = _flow_to_rgb(gt_flow, denoise=False)
                n = len(targets)
                fig, axes = plt.subplots(n, 4, figsize=(12, max(3, 1.9 * n)))
                if n == 1:
                    axes = np.asarray([axes])
                for i, t in enumerate(targets):
                    axes[i, 0].imshow(_tensor_to_img(t["frames"][:, 0]))
                    axes[i, 1].imshow(_tensor_to_img(t["frames"][:, -1]))
                    axes[i, 2].imshow(_tensor_to_img(gt_flow_rgb[i]))
                    axes[i, 3].imshow(_tensor_to_img(pred_flow_rgb[i]))
                    axes[i, 0].set_ylabel(t["dataset_name"], fontsize=8)
                    for c in range(4):
                        axes[i, c].axis("off")
                axes[0, 0].set_title("frame_t")
                axes[0, 1].set_title("real_t+k")
                axes[0, 2].set_title("gt_flow")
                axes[0, 3].set_title("pred_flow (fixed token)")
                fig.suptitle(f"Token [{seq_str}] flow decoder report", fontsize=10, y=0.995)
                fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
                fig.savefig(token_dir / "flow_decoder.png", dpi=150)
                plt.close(fig)


def _check_outputs(out_dir: Path) -> dict[str, Any]:
    checks = {}
    png_files = list(out_dir.rglob("*.png"))
    checks["num_png_files"] = len(png_files)
    checks["empty_png_files"] = [str(p) for p in png_files if p.stat().st_size == 0]
    checks["has_sanity_checks"] = (out_dir / "sanity_checks.json").exists()
    checks["has_collection_stats"] = (out_dir / "collection_stats.json").exists()
    return checks


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    if cfg.analysis.get("checkpoint_path") is None:
        raise ValueError("Set analysis.checkpoint_path")

    runs_dir = None
    try:
        if HydraConfig.initialized():
            runs_dir = Path(str(HydraConfig.get().runtime.output_dir))
    except Exception:
        runs_dir = None
    if runs_dir is None:
        runs_dir = resolve_runs_dir(
            logging_root_dir=cfg.logging.get("root_dir"),
            logging_runs_dir=cfg.logging.get("runs_dir"),
            workspace_root=workspace_root,
            experiment_name=OmegaConf.select(cfg, "experiment.name"),
        )
    logger, base_output_dir = setup_unified_logging(
        runs_dir=runs_dir,
        job_id=cfg.logging.get("job_id"),
        log_level=cfg.logging.get("level", "INFO"),
        logger_name="laq.token_analysis",
    )
    out_dir = _setup_output_dir(base_output_dir=base_output_dir, cfg=cfg)
    logger.info(f"Analysis output dir: {out_dir}")

    cache_dir = resolve_cache_dir(cfg=cfg, workspace_root=workspace_root)
    if cache_dir is not None:
        configure_cache_env(cache_dir=cache_dir, logger=logger)

    set_seed(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    task = _load_laq_task(str(cfg.analysis.checkpoint_path), cfg=cfg, logger=logger)
    task = task.to(device)
    task.eval()

    datamodule = create_datamodule(cfg.data)
    datamodule.setup(stage="fit")
    dataloader = datamodule.val_dataloader()

    records, visual_samples, per_dataset_counts = collect_samples(task, dataloader, cfg, logger)
    logger.info(f"Collected samples: {len(records)}")
    logger.info(f"Visual subset: {len(visual_samples)}")
    logger.info(f"Per-dataset counts: {per_dataset_counts}")

    collection_stats = {
        "num_records": len(records),
        "num_visual_samples": len(visual_samples),
        "per_dataset_counts": {k: int(v) for k, v in sorted(per_dataset_counts.items())},
    }
    _save_json(out_dir / "collection_stats.json", collection_stats)

    inspected_k = int(cfg.analysis.get("inspected_tokens_k", 12))
    inspected_sequences, seq_counter, seq_ds_counts = select_inspected_sequences(records, k=inspected_k)
    save_inspected_sequences(out_dir, inspected_sequences, seq_counter, seq_ds_counts)
    logger.info(f"Inspected tokens: {[ _seq_str(s) for s in inspected_sequences ]}")

    sanity = run_sanity_checks(records, per_dataset_counts, cfg)
    _save_json(out_dir / "sanity_checks.json", sanity)

    run_cfg = cfg.analysis.run
    if bool(run_cfg.analysis_1):
        analysis_1_flow_token_scatter(records, out_dir, top_k=int(cfg.analysis.top_k_sequences))
    if bool(run_cfg.analysis_5):
        analysis_5_token_dataset_heatmap(records, out_dir, top_k=int(cfg.analysis.top_k_heatmap))
    consistency_metrics = None
    if bool(run_cfg.analysis_2):
        consistency_metrics = analysis_2_cross_dataset_consistency(
            records, out_dir, min_samples_per_dataset=int(cfg.analysis.min_samples_per_dataset)
        )
    if bool(run_cfg.analysis_6):
        analysis_6_per_token_compass(
            records,
            out_dir,
            inspected_sequences=inspected_sequences,
            codebook_size=int(cfg.model.codebook_size),
            min_samples=int(cfg.analysis.min_compass_samples),
        )
    if bool(run_cfg.analysis_3):
        analysis_3_sequence_examples(
            records,
            visual_samples,
            out_dir,
            inspected_sequences=inspected_sequences[: int(cfg.analysis.top_k_sequences)],
            examples_per_sequence=int(cfg.analysis.examples_per_sequence),
        )
    if bool(run_cfg.analysis_7):
        analysis_7_flow_decoder_transfer(
            task,
            records,
            visual_samples,
            out_dir,
            inspected_sequences=inspected_sequences,
            top_k_transfer=int(cfg.analysis.top_k_transfer),
            top_k_tokens=int(cfg.analysis.top_k_transfer_tokens),
        )
    if bool(run_cfg.analysis_4):
        analysis_4_latent_transfer(
            task,
            records,
            visual_samples,
            out_dir,
            inspected_sequences=inspected_sequences,
            top_k_transfer=int(cfg.analysis.top_k_transfer),
        )
    if bool(run_cfg.get("analysis_8", True)):
        analysis_8_token_reports(
            task,
            records,
            visual_samples,
            out_dir,
            inspected_sequences=inspected_sequences,
            top_k_reports=int(cfg.analysis.top_k_token_reports),
            examples_per_token=int(cfg.analysis.token_report_examples),
            targets_per_dataset=int(cfg.analysis.token_report_targets_per_dataset),
        )

    checks = _check_outputs(out_dir)
    _save_json(out_dir / "output_checks.json", checks)
    logger.info(f"Output checks: {checks}")


if __name__ == "__main__":
    main()
