"""Action-frame filtering utilities for LeRobot-v3 anchor sampling.

This module computes per-anchor motion/action scores, trims low-activity episode
endpoints, and materializes keep/drop masks used by training-time samplers.
Scores and decisions are cached in dataset-local sidecars with config
fingerprinting to avoid recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchorFilterResult:
    kept_anchor_values: np.ndarray
    kept_offsets_start: np.ndarray
    kept_offsets_end: np.ndarray
    kept_counts: np.ndarray
    kept_range_start: np.ndarray
    kept_range_end: np.ndarray
    cache_path: str | None
    summary: dict[str, Any]


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _as_list_of_int(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    raise TypeError(f"Expected list/tuple for integer list, got {type(value)}")


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    if override is None:
        return out
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def normalize_filtering_config(
    *,
    global_filtering: Mapping[str, Any] | None,
    source_filtering: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge global/source filtering config into a normalized runtime schema."""
    base = {} if global_filtering is None else dict(global_filtering)
    merged = _deep_merge_dict(base, source_filtering)
    enabled = _to_bool(merged.get("enabled"), default=False)
    mode = str(merged.get("mode", "none"))
    if not enabled or mode == "none":
        return None

    motion_cfg = dict(merged.get("motion", {}))
    sparse_flow_cfg = dict(motion_cfg.get("sparse_flow", {}))
    motion_cfg["sparse_flow"] = sparse_flow_cfg

    action_cfg = dict(merged.get("action", {}))
    cache_cfg = dict(merged.get("cache", {}))

    normalized = {
        "enabled": True,
        "mode": mode,
        "apply_at_sampling": _to_bool(merged.get("apply_at_sampling"), True),
        "trim_episode_ends": _to_bool(merged.get("trim_episode_ends"), True),
        "motion": {
            "enabled": _to_bool(motion_cfg.get("enabled"), mode in {"motion", "both"}),
            "method": str(motion_cfg.get("method", "frame_diff")),
            "frame_gap": motion_cfg.get("frame_gap"),
            "aggregate_all_cameras": _to_bool(motion_cfg.get("aggregate_all_cameras"), False),
            "aggregate_reduce": str(motion_cfg.get("aggregate_reduce", "mean")),
            "resize_short_side": _to_int(motion_cfg.get("resize_short_side"), 224),
            "grayscale": _to_bool(motion_cfg.get("grayscale"), True),
            "blur_kernel": _to_int(motion_cfg.get("blur_kernel"), 5),
            "diff_pixel_threshold": _to_float(motion_cfg.get("diff_pixel_threshold"), 0.03),
            "smoothing_window": _to_int(motion_cfg.get("smoothing_window"), 5),
            "consecutive_active_k": _to_int(motion_cfg.get("consecutive_active_k"), 3),
            "low_threshold": _to_float(motion_cfg.get("low_threshold"), 0.01),
            "high_threshold": _to_float(motion_cfg.get("high_threshold"), 0.02),
            "use_hysteresis": _to_bool(motion_cfg.get("use_hysteresis"), True),
            "sparse_flow": {
                "enabled": _to_bool(sparse_flow_cfg.get("enabled"), False),
                "only_on_uncertain": _to_bool(sparse_flow_cfg.get("only_on_uncertain"), True),
                "max_corners": _to_int(sparse_flow_cfg.get("max_corners"), 200),
                "quality_level": _to_float(sparse_flow_cfg.get("quality_level"), 0.01),
                "min_distance": _to_float(sparse_flow_cfg.get("min_distance"), 5.0),
                "win_size": _to_int(sparse_flow_cfg.get("win_size"), 15),
                "max_level": _to_int(sparse_flow_cfg.get("max_level"), 2),
                "min_tracked_fraction": _to_float(
                    sparse_flow_cfg.get("min_tracked_fraction"), 0.25
                ),
                "median_flow_threshold": _to_float(
                    sparse_flow_cfg.get("median_flow_threshold"), 0.6
                ),
            },
        },
        "action": {
            "enabled": _to_bool(action_cfg.get("enabled"), mode in {"action", "both"}),
            "method": str(action_cfg.get("method", "norm")),
            "threshold": _to_float(action_cfg.get("threshold"), 0.02),
            "exclude_dims": _as_list_of_int(action_cfg.get("exclude_dims")),
            "chunk_size": _to_int(action_cfg.get("chunk_size"), 1),
            "chunk_reduce": str(action_cfg.get("chunk_reduce", "max")),
            "min_nonzero_ratio": _to_float(action_cfg.get("min_nonzero_ratio"), 0.0),
        },
        "cache": {
            "enabled": _to_bool(cache_cfg.get("enabled"), True),
            "reuse_if_config_unchanged": _to_bool(
                cache_cfg.get("reuse_if_config_unchanged"), True
            ),
            "force_recompute": _to_bool(cache_cfg.get("force_recompute"), False),
        },
    }
    return normalized


def infer_motion_frame_gap(
    *,
    request_image_deltas: tuple[int, ...],
    configured_frame_gap: int | None,
) -> int:
    """Infer temporal gap used for motion scoring from requested image deltas."""
    if configured_frame_gap is not None:
        return max(1, int(configured_frame_gap))
    if len(request_image_deltas) == 0:
        return 1
    return max(1, int(max(request_image_deltas) - min(request_image_deltas)))


def _to_gray_float_pair(
    pair: torch.Tensor,
    *,
    resize_short_side: int,
    blur_kernel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pair.ndim != 4:
        raise ValueError(f"Expected image pair [T,C,H,W] or [T,H,W,C], got {tuple(pair.shape)}")
    if int(pair.shape[0]) != 2:
        raise ValueError(f"Expected exactly 2 time steps for pair, got T={int(pair.shape[0])}")

    if int(pair.shape[1]) == 3:
        x = pair
    elif int(pair.shape[-1]) == 3:
        x = pair.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported pair layout for RGB conversion: {tuple(pair.shape)}")

    x = x.to(torch.float32)
    if x.max() > 1.0:
        x = x / 255.0

    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    h = int(gray.shape[-2])
    w = int(gray.shape[-1])
    short = min(h, w)
    if resize_short_side > 0 and short != resize_short_side:
        scale = float(resize_short_side) / float(short)
        new_h = max(8, int(round(h * scale)))
        new_w = max(8, int(round(w * scale)))
        gray = F.interpolate(gray, size=(new_h, new_w), mode="bilinear", align_corners=False)

    if blur_kernel > 1:
        k = int(blur_kernel)
        if k % 2 == 0:
            k += 1
        gray = F.avg_pool2d(gray, kernel_size=k, stride=1, padding=k // 2)

    return gray[0, 0], gray[1, 0]


def motion_score_from_pair(
    pair: torch.Tensor,
    *,
    resize_short_side: int,
    blur_kernel: int,
    diff_pixel_threshold: float,
) -> float:
    frame0, frame1 = _to_gray_float_pair(
        pair,
        resize_short_side=resize_short_side,
        blur_kernel=blur_kernel,
    )
    diff = torch.abs(frame1 - frame0)
    active = (diff >= float(diff_pixel_threshold)).to(torch.float32)
    return float(active.mean().item())


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = int(max(1, window))
    if window <= 1 or values.size <= 1:
        return values.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    padded = np.pad(values.astype(np.float32), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _first_consecutive_true(mask: np.ndarray, k: int) -> int | None:
    k = int(max(1, k))
    run = 0
    for idx, value in enumerate(mask.tolist()):
        run = run + 1 if value else 0
        if run >= k:
            return idx - k + 1
    return None


def _last_consecutive_true(mask: np.ndarray, k: int) -> int | None:
    if mask.size == 0:
        return None
    rev = mask[::-1]
    first_rev = _first_consecutive_true(rev, k)
    if first_rev is None:
        return None
    end_inclusive = int(mask.size) - first_rev - 1
    return end_inclusive


def _extract_action_tensor(raw: Mapping[str, Any], action_key: str) -> torch.Tensor | None:
    if action_key not in raw:
        return None
    value = raw[action_key]
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected action tensor rank 1/2, got {tuple(tensor.shape)}")
    return tensor.to(torch.float32)


def action_score_from_tensor(
    action: torch.Tensor,
    *,
    exclude_dims: list[int],
    reduce: str,
) -> tuple[float, float]:
    if action.ndim != 2:
        raise ValueError(f"Expected action shape [T,A], got {tuple(action.shape)}")
    keep_dims = [idx for idx in range(int(action.shape[1])) if idx not in set(exclude_dims)]
    if len(keep_dims) == 0:
        vec = action
    else:
        vec = action[:, keep_dims]
    norms = torch.linalg.norm(vec, dim=1)
    nonzero_ratio = float((norms > 1.0e-8).to(torch.float32).mean().item())
    reduce = str(reduce)
    if reduce == "mean":
        score = float(norms.mean().item())
    else:
        score = float(norms.max().item())
    return score, nonzero_ratio


def _compute_sparse_flow(
    pair: torch.Tensor,
    *,
    resize_short_side: int,
    blur_kernel: int,
    max_corners: int,
    quality_level: float,
    min_distance: float,
    win_size: int,
    max_level: int,
) -> tuple[float, float]:
    del pair
    del resize_short_side
    del blur_kernel
    del max_corners
    del quality_level
    del min_distance
    del win_size
    del max_level
    return 0.0, 0.0


def _build_filter_cache_path(
    *,
    dataset_root: Path,
    split: str,
    camera_tag: str,
    fingerprint: str,
) -> Path:
    cache_dir = dataset_root / "meta" / "hlrp_action_frame_filter_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{split}_{camera_tag}_{fingerprint}.npz"


def _camera_name_token(camera_key: str) -> str:
    token = str(camera_key).split(".")[-1].strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in token).strip("_")
    return safe or "camera"


def _camera_tag(camera_keys: tuple[str, ...]) -> str:
    if len(camera_keys) == 1:
        return _camera_name_token(camera_keys[0])
    return f"allcams{len(camera_keys)}"


def _fingerprint_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_anchor_filter(
    *,
    repo_id: str,
    root: str | None,
    revision: str | None,
    video_backend: str | None,
    tolerance_s: float | None,
    request_image_deltas: tuple[int, ...],
    camera_dataset_keys: tuple[str, ...],
    camera_aggregate_reduce: str,
    action_key: str | None,
    episode_ids: np.ndarray,
    candidate_start: np.ndarray,
    candidate_end: np.ndarray,
    filtering_cfg: Mapping[str, Any],
    split: str,
) -> AnchorFilterResult:
    """Compute and cache action-frame filtering decisions for one source split."""
    mode = str(filtering_cfg["mode"])
    motion_cfg = dict(filtering_cfg["motion"])
    action_cfg = dict(filtering_cfg["action"])
    cache_cfg = dict(filtering_cfg["cache"])

    motion_enabled = bool(motion_cfg.get("enabled", False)) and mode in {"motion", "both"}
    action_enabled = bool(action_cfg.get("enabled", False)) and mode in {"action", "both"}
    trim_enabled = bool(filtering_cfg.get("trim_episode_ends", True)) and motion_enabled

    frame_gap = infer_motion_frame_gap(
        request_image_deltas=request_image_deltas,
        configured_frame_gap=motion_cfg.get("frame_gap"),
    )

    action_chunk_size = int(action_cfg.get("chunk_size", 1))
    action_deltas = list(range(max(1, action_chunk_size))) if action_enabled and action_key else []

    if len(camera_dataset_keys) == 0:
        raise ValueError("camera_dataset_keys must contain at least one camera key")

    delta_timestamps: dict[str, list[float]] = {
        str(camera_key): [0.0, float(frame_gap)] for camera_key in camera_dataset_keys
    }
    if action_enabled and action_key is not None:
        delta_timestamps[action_key] = [float(delta) for delta in action_deltas]

    ds = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        revision=revision,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
        tolerance_s=(1.0e-4 if tolerance_s is None else float(tolerance_s)),
    )

    source_root = Path(ds.meta.root)
    fingerprint_payload = {
        "version": 1,
        "repo_id": str(repo_id),
        "revision": None if revision is None else str(revision),
        "split": str(split),
        "camera_dataset_keys": [str(key) for key in camera_dataset_keys],
        "camera_aggregate_reduce": str(camera_aggregate_reduce),
        "action_key": None if action_key is None else str(action_key),
        "request_image_deltas": [int(x) for x in request_image_deltas],
        "episode_ids": [int(x) for x in episode_ids.tolist()],
        "candidate_start": [int(x) for x in candidate_start.tolist()],
        "candidate_end": [int(x) for x in candidate_end.tolist()],
        "filtering": filtering_cfg,
    }
    fingerprint = _fingerprint_payload(fingerprint_payload)
    camera_tag = _camera_tag(camera_dataset_keys)
    cache_path = _build_filter_cache_path(
        dataset_root=source_root,
        split=split,
        camera_tag=camera_tag,
        fingerprint=fingerprint,
    )

    can_reuse = (
        bool(cache_cfg.get("enabled", True))
        and bool(cache_cfg.get("reuse_if_config_unchanged", True))
        and not bool(cache_cfg.get("force_recompute", False))
        and cache_path.is_file()
    )

    if can_reuse:
        payload = np.load(cache_path, allow_pickle=False)
        cached_fingerprint = str(payload["fingerprint"]) if "fingerprint" in payload else ""
        if cached_fingerprint == fingerprint:
            summary = {
                "cache": "hit",
                "cache_path": str(cache_path),
                "fingerprint": fingerprint,
                "anchors_before": int(payload["anchors_before"]),
                "anchors_after": int(payload["anchors_after"]),
                "trimmed_fraction": float(payload["trimmed_fraction"]),
                "motion_only_removed": int(payload["motion_only_removed"]),
                "action_only_removed": int(payload["action_only_removed"]),
            }
            return AnchorFilterResult(
                kept_anchor_values=payload["kept_anchor_values"].astype(np.int64),
                kept_offsets_start=payload["kept_offsets_start"].astype(np.int64),
                kept_offsets_end=payload["kept_offsets_end"].astype(np.int64),
                kept_counts=payload["kept_counts"].astype(np.int32),
                kept_range_start=payload["kept_range_start"].astype(np.int64),
                kept_range_end=payload["kept_range_end"].astype(np.int64),
                cache_path=str(cache_path),
                summary=summary,
            )

    total_before = 0
    total_after = 0
    total_trimmed = 0
    motion_only_removed = 0
    action_only_removed = 0

    kept_anchor_values_parts: list[np.ndarray] = []
    kept_offsets_start: list[int] = []
    kept_offsets_end: list[int] = []
    kept_counts: list[int] = []
    kept_range_start: list[int] = []
    kept_range_end: list[int] = []

    motion_raw_parts: list[np.ndarray] = []
    motion_smooth_parts: list[np.ndarray] = []
    motion_raw_per_camera_parts: list[np.ndarray] = []
    motion_smooth_per_camera_parts: list[np.ndarray] = []
    action_score_parts: list[np.ndarray] = []
    action_ratio_parts: list[np.ndarray] = []
    flow_score_parts: list[np.ndarray] = []
    flow_valid_parts: list[np.ndarray] = []
    keep_mask_parts: list[np.ndarray] = []
    candidate_offsets_start: list[int] = []
    candidate_offsets_end: list[int] = []
    trim_start_local: list[int] = []
    trim_end_local: list[int] = []

    running_kept_offset = 0
    running_candidate_offset = 0

    method = str(motion_cfg.get("method", "frame_diff"))
    aggregate_reduce = str(camera_aggregate_reduce).lower()
    if aggregate_reduce not in {"mean", "max"}:
        raise ValueError(f"Unsupported camera_aggregate_reduce={camera_aggregate_reduce!r}")
    low_thr = float(motion_cfg.get("low_threshold", 0.01))
    high_thr = float(motion_cfg.get("high_threshold", 0.02))
    use_hysteresis = bool(motion_cfg.get("use_hysteresis", True))

    for row_idx in range(int(episode_ids.shape[0])):
        start = int(candidate_start[row_idx])
        end = int(candidate_end[row_idx])
        n = max(0, end - start)

        candidate_offsets_start.append(running_candidate_offset)
        candidate_offsets_end.append(running_candidate_offset + n)
        running_candidate_offset += n

        if n == 0:
            kept_offsets_start.append(running_kept_offset)
            kept_offsets_end.append(running_kept_offset)
            kept_counts.append(0)
            kept_range_start.append(start)
            kept_range_end.append(start)
            trim_start_local.append(0)
            trim_end_local.append(0)

            motion_raw_parts.append(np.zeros((0,), dtype=np.float32))
            motion_smooth_parts.append(np.zeros((0,), dtype=np.float32))
            motion_raw_per_camera_parts.append(
                np.zeros((len(camera_dataset_keys), 0), dtype=np.float32)
            )
            motion_smooth_per_camera_parts.append(
                np.zeros((len(camera_dataset_keys), 0), dtype=np.float32)
            )
            action_score_parts.append(np.zeros((0,), dtype=np.float32))
            action_ratio_parts.append(np.zeros((0,), dtype=np.float32))
            flow_score_parts.append(np.zeros((0,), dtype=np.float32))
            flow_valid_parts.append(np.zeros((0,), dtype=np.float32))
            keep_mask_parts.append(np.zeros((0,), dtype=np.bool_))
            continue

        anchors = np.arange(start, end, dtype=np.int64)
        total_before += int(n)

        motion_raw = np.full((n,), np.nan, dtype=np.float32)
        motion_raw_per_camera = np.full(
            (len(camera_dataset_keys), n),
            np.nan,
            dtype=np.float32,
        )
        action_score = np.full((n,), np.nan, dtype=np.float32)
        action_ratio = np.full((n,), np.nan, dtype=np.float32)

        need_motion = motion_enabled or trim_enabled
        need_action = action_enabled and action_key is not None

        for local_idx, anchor in enumerate(anchors.tolist()):
            raw = ds[int(anchor)]
            if need_motion:
                per_camera_scores: list[float] = []
                for cam_idx, camera_dataset_key in enumerate(camera_dataset_keys):
                    pair = raw[camera_dataset_key]
                    pair_t = pair if torch.is_tensor(pair) else torch.as_tensor(pair)
                    score = motion_score_from_pair(
                        pair_t,
                        resize_short_side=int(motion_cfg.get("resize_short_side", 224)),
                        blur_kernel=int(motion_cfg.get("blur_kernel", 5)),
                        diff_pixel_threshold=float(motion_cfg.get("diff_pixel_threshold", 0.03)),
                    )
                    motion_raw_per_camera[cam_idx, local_idx] = score
                    per_camera_scores.append(score)
                if aggregate_reduce == "max":
                    motion_raw[local_idx] = float(np.max(per_camera_scores))
                else:
                    motion_raw[local_idx] = float(np.mean(per_camera_scores))

            if need_action:
                action_t = _extract_action_tensor(raw, action_key)
                if action_t is not None:
                    score, ratio = action_score_from_tensor(
                        action_t,
                        exclude_dims=list(action_cfg.get("exclude_dims", [])),
                        reduce=str(action_cfg.get("chunk_reduce", "max")),
                    )
                    action_score[local_idx] = score
                    action_ratio[local_idx] = ratio

        motion_smooth = _smooth_1d(
            np.nan_to_num(motion_raw, nan=0.0),
            int(motion_cfg.get("smoothing_window", 5)),
        )
        motion_smooth_per_camera = np.stack(
            [
                _smooth_1d(
                    np.nan_to_num(motion_raw_per_camera[cam_idx], nan=0.0),
                    int(motion_cfg.get("smoothing_window", 5)),
                )
                for cam_idx in range(len(camera_dataset_keys))
            ],
            axis=0,
        ).astype(np.float32)

        motion_keep = np.ones((n,), dtype=np.bool_)
        motion_uncertain = np.zeros((n,), dtype=np.bool_)
        flow_score = np.full((n,), np.nan, dtype=np.float32)
        flow_valid = np.full((n,), np.nan, dtype=np.float32)

        if motion_enabled:
            if method == "sparse_flow":
                motion_keep[:] = False
                for local_idx, anchor in enumerate(anchors.tolist()):
                    raw = ds[int(anchor)]
                    pair = raw[camera_dataset_keys[0]]
                    pair_t = pair if torch.is_tensor(pair) else torch.as_tensor(pair)
                    try:
                        score, valid_frac = _compute_sparse_flow(
                            pair_t,
                            resize_short_side=int(motion_cfg.get("resize_short_side", 224)),
                            blur_kernel=int(motion_cfg.get("blur_kernel", 5)),
                            max_corners=int(motion_cfg["sparse_flow"].get("max_corners", 200)),
                            quality_level=float(motion_cfg["sparse_flow"].get("quality_level", 0.01)),
                            min_distance=float(motion_cfg["sparse_flow"].get("min_distance", 5.0)),
                            win_size=int(motion_cfg["sparse_flow"].get("win_size", 15)),
                            max_level=int(motion_cfg["sparse_flow"].get("max_level", 2)),
                        )
                    except Exception:
                        score = 0.0
                        valid_frac = 0.0
                    flow_score[local_idx] = score
                    flow_valid[local_idx] = valid_frac
                flow_thr = float(motion_cfg["sparse_flow"].get("median_flow_threshold", 0.6))
                min_valid = float(motion_cfg["sparse_flow"].get("min_tracked_fraction", 0.25))
                motion_keep = (flow_score >= flow_thr) & (flow_valid >= min_valid)
            else:
                if use_hysteresis and high_thr > low_thr:
                    keep_hi = motion_smooth >= high_thr
                    drop_lo = motion_smooth < low_thr
                    motion_uncertain = (~keep_hi) & (~drop_lo)
                    motion_keep = keep_hi | motion_uncertain
                else:
                    thr = high_thr if high_thr > 0 else low_thr
                    motion_keep = motion_smooth >= thr

                if method == "two_stage" and bool(motion_cfg["sparse_flow"].get("enabled", False)):
                    if bool(motion_cfg["sparse_flow"].get("only_on_uncertain", True)):
                        eval_mask = motion_uncertain
                    else:
                        eval_mask = np.ones((n,), dtype=np.bool_)

                    if eval_mask.any():
                        logger.warning(
                            "Sparse-flow refinement requested but torch-only mode is active; skipping refinement."
                        )

        action_keep = np.ones((n,), dtype=np.bool_)
        if action_enabled and action_key is not None and np.isfinite(action_score).any():
            action_keep = (
                np.nan_to_num(action_score, nan=0.0) >= float(action_cfg.get("threshold", 0.02))
            ) & (
                np.nan_to_num(action_ratio, nan=0.0)
                >= float(action_cfg.get("min_nonzero_ratio", 0.0))
            )

        if mode == "motion":
            keep = motion_keep
        elif mode == "action":
            keep = action_keep
        else:
            keep = motion_keep & action_keep

        trim_start = 0
        trim_end = n
        if trim_enabled:
            active_thr = high_thr if high_thr > 0 else low_thr
            active_mask = motion_smooth >= active_thr
            k = int(motion_cfg.get("consecutive_active_k", 3))
            first_active = _first_consecutive_true(active_mask, k)
            last_active = _last_consecutive_true(active_mask, k)
            if first_active is not None and last_active is not None and last_active >= first_active:
                trim_start = int(first_active)
                trim_end = int(last_active) + 1
                outside = np.ones((n,), dtype=np.bool_)
                outside[trim_start:trim_end] = False
                total_trimmed += int(outside.sum())
                keep[outside] = False

        keep_mask = keep.astype(np.bool_)
        if mode == "both":
            dropped = ~keep_mask
            motion_only_removed += int((dropped & (~motion_keep) & action_keep).sum())
            action_only_removed += int((dropped & motion_keep & (~action_keep)).sum())

        kept = anchors[keep_mask]
        total_after += int(kept.shape[0])

        kept_offsets_start.append(running_kept_offset)
        running_kept_offset += int(kept.shape[0])
        kept_offsets_end.append(running_kept_offset)
        kept_counts.append(int(kept.shape[0]))
        kept_range_start.append(int(kept[0]) if kept.shape[0] > 0 else start)
        kept_range_end.append(int(kept[-1]) + 1 if kept.shape[0] > 0 else start)
        trim_start_local.append(int(trim_start))
        trim_end_local.append(int(trim_end))

        kept_anchor_values_parts.append(kept)
        motion_raw_parts.append(motion_raw)
        motion_smooth_parts.append(motion_smooth)
        motion_raw_per_camera_parts.append(motion_raw_per_camera)
        motion_smooth_per_camera_parts.append(motion_smooth_per_camera)
        action_score_parts.append(action_score)
        action_ratio_parts.append(action_ratio)
        flow_score_parts.append(flow_score)
        flow_valid_parts.append(flow_valid)
        keep_mask_parts.append(keep_mask)

    kept_anchor_values = (
        np.concatenate(kept_anchor_values_parts, axis=0)
        if kept_anchor_values_parts
        else np.zeros((0,), dtype=np.int64)
    )

    anchors_before = int(total_before)
    anchors_after = int(total_after)
    trimmed_fraction = float(total_trimmed) / float(max(1, anchors_before))

    if bool(cache_cfg.get("enabled", True)):
        np.savez_compressed(
            cache_path,
            fingerprint=np.asarray(fingerprint),
            camera_dataset_key=np.asarray(str(camera_dataset_keys[0])),
            camera_dataset_keys=np.asarray(list(camera_dataset_keys)),
            motion_aggregate_reduce=np.asarray(aggregate_reduce),
            kept_anchor_values=kept_anchor_values.astype(np.int64),
            kept_offsets_start=np.asarray(kept_offsets_start, dtype=np.int64),
            kept_offsets_end=np.asarray(kept_offsets_end, dtype=np.int64),
            kept_counts=np.asarray(kept_counts, dtype=np.int32),
            kept_range_start=np.asarray(kept_range_start, dtype=np.int64),
            kept_range_end=np.asarray(kept_range_end, dtype=np.int64),
            episode_ids=episode_ids.astype(np.int32),
            candidate_start=candidate_start.astype(np.int64),
            candidate_end=candidate_end.astype(np.int64),
            candidate_offsets_start=np.asarray(candidate_offsets_start, dtype=np.int64),
            candidate_offsets_end=np.asarray(candidate_offsets_end, dtype=np.int64),
            motion_raw=np.concatenate(motion_raw_parts, axis=0).astype(np.float32),
            motion_smooth=np.concatenate(motion_smooth_parts, axis=0).astype(np.float32),
            motion_raw_per_camera=np.concatenate(motion_raw_per_camera_parts, axis=1).astype(np.float32),
            motion_smooth_per_camera=np.concatenate(motion_smooth_per_camera_parts, axis=1).astype(
                np.float32
            ),
            action_score=np.concatenate(action_score_parts, axis=0).astype(np.float32),
            action_nonzero_ratio=np.concatenate(action_ratio_parts, axis=0).astype(np.float32),
            flow_score=np.concatenate(flow_score_parts, axis=0).astype(np.float32),
            flow_valid_fraction=np.concatenate(flow_valid_parts, axis=0).astype(np.float32),
            keep_mask=np.concatenate(keep_mask_parts, axis=0).astype(np.bool_),
            trim_start_local=np.asarray(trim_start_local, dtype=np.int32),
            trim_end_local=np.asarray(trim_end_local, dtype=np.int32),
            anchors_before=np.asarray(anchors_before, dtype=np.int64),
            anchors_after=np.asarray(anchors_after, dtype=np.int64),
            trimmed_fraction=np.asarray(trimmed_fraction, dtype=np.float32),
            motion_only_removed=np.asarray(motion_only_removed, dtype=np.int64),
            action_only_removed=np.asarray(action_only_removed, dtype=np.int64),
            motion_low_threshold=np.asarray(low_thr, dtype=np.float32),
            motion_high_threshold=np.asarray(high_thr, dtype=np.float32),
            action_threshold=np.asarray(float(action_cfg.get("threshold", 0.02)), dtype=np.float32),
        )

    summary = {
        "cache": "miss",
        "cache_path": str(cache_path),
        "fingerprint": fingerprint,
        "anchors_before": anchors_before,
        "anchors_after": anchors_after,
        "trimmed_fraction": trimmed_fraction,
        "motion_only_removed": int(motion_only_removed),
        "action_only_removed": int(action_only_removed),
    }

    return AnchorFilterResult(
        kept_anchor_values=kept_anchor_values,
        kept_offsets_start=np.asarray(kept_offsets_start, dtype=np.int64),
        kept_offsets_end=np.asarray(kept_offsets_end, dtype=np.int64),
        kept_counts=np.asarray(kept_counts, dtype=np.int32),
        kept_range_start=np.asarray(kept_range_start, dtype=np.int64),
        kept_range_end=np.asarray(kept_range_end, dtype=np.int64),
        cache_path=str(cache_path),
        summary=summary,
    )
