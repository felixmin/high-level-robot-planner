"""Helper functions for local OpenX indexed-full loading."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from common.adapters.oxe_shared import OXEDatasetConfig, resolve_nested_key

logger = logging.getLogger(__name__)

_TRAIN_SPLIT_RE = re.compile(r"^train(?:\[(.*)\])?$")
_IMAGE_KEY_PRIORITY = (
    "image",
    "hand_image",
    "wrist_image",
    "highres_image",
    "image_with_depth",
    "top_image",
    "rgb",
    "front_rgb",
    "rgb_static",
    "rgb_gripper",
    "agentview_rgb",
    "eye_in_hand_rgb",
)


def _fallback_dataset_config(dataset_name: str) -> OXEDatasetConfig:
    logger.warning(
        "Dataset %s is not in OXE_DATASETS; using generic local fallback config",
        dataset_name,
    )
    return OXEDatasetConfig(
        name=dataset_name,
        gcs_path="",
        image_key="image",
        instruction_key="natural_language_instruction",
        state_key="state",
        action_key="world_vector",
        action_is_dict=True,
        action_dim=3,
        state_dim=0,
        allow_missing_state=True,
    )


def _parse_split_index(token: str, n_items: int, default: int) -> int:
    token = token.strip()
    if token == "":
        return default
    if token.endswith("%"):
        frac = float(token[:-1]) / 100.0
        return int(round(frac * float(n_items)))
    return int(token)


def _parse_train_split(split: str, n_items: int) -> tuple[int, int]:
    split = str(split).strip()
    match = _TRAIN_SPLIT_RE.fullmatch(split)
    if not match:
        raise ValueError(
            f"Unsupported split {split!r}. Expected 'train' or 'train[start:end]'."
        )

    body = match.group(1)
    if body is None or body.strip() == "":
        return 0, n_items

    if ":" in body:
        start_raw, end_raw = body.split(":", 1)
    else:
        start_raw, end_raw = body, ""

    start = _parse_split_index(start_raw, n_items, default=0)
    end = _parse_split_index(end_raw, n_items, default=n_items)

    start = max(0, min(n_items, start))
    end = max(0, min(n_items, end))
    if end < start:
        end = start
    return start, end


def _to_float_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).reshape(-1)
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            return None
    if isinstance(value, (float, int, bool, np.number)):
        return np.asarray([value], dtype=np.float32)
    if hasattr(value, "numpy"):
        arr = value.numpy()
        return np.asarray(arr, dtype=np.float32).reshape(-1)
    return None


def _decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _resolve_optional(container: Any, keypath: Optional[str]) -> Any:
    if keypath is None:
        return None
    try:
        return resolve_nested_key(container, keypath)
    except Exception:
        return None


def _to_pil_rgb(image_value: Any) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, bytes):
        return Image.open(io.BytesIO(image_value)).convert("RGB")
    if isinstance(image_value, np.ndarray):
        arr = image_value
    elif hasattr(image_value, "numpy"):
        arr = np.asarray(image_value.numpy())
    elif isinstance(image_value, (list, tuple)):
        arr = np.asarray(image_value)
    else:
        raise TypeError(f"Unsupported image type: {type(image_value)}")

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def _decode_image_to_tensor(image_value: Any, image_size: int) -> torch.Tensor:
    image = _to_pil_rgb(image_value)
    if image_size > 0 and (image.width != image_size or image.height != image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _extract_image(obs: Dict[str, Any], config: OXEDatasetConfig) -> Any:
    value = _resolve_optional(obs, config.image_key)
    if value is not None:
        return value
    for key in _IMAGE_KEY_PRIORITY:
        if key in obs and obs[key] is not None:
            return obs[key]
    for k, v in obs.items():
        key = str(k).lower()
        if ("image" in key or "rgb" in key) and v is not None:
            return v
    return None


def discover_local_subdatasets(root: str) -> List[str]:
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return []
    names: List[str] = []
    for path in sorted(root_path.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if any(path.glob("*.tar")):
            names.append(path.name)
    return names


def _extract_instruction(
    step: Dict[str, Any], obs: Dict[str, Any], config: OXEDatasetConfig
) -> str:
    src = step if config.instruction_in_step else obs
    value = _resolve_optional(src, config.instruction_key)
    return _decode_text(value)


def _extract_state(
    obs: Dict[str, Any], config: OXEDatasetConfig, output_state_dim: int
) -> np.ndarray:
    out = np.zeros(output_state_dim, dtype=np.float32)
    if output_state_dim == 0:
        return out
    if config.state_key is None or config.state_dim <= 0:
        return out

    value = _resolve_optional(obs, config.state_key)
    vec = _to_float_vector(value)
    if vec is None:
        return out

    local_dim = int(max(0, config.state_dim))
    vec = vec[:local_dim]
    if vec.shape[0] < local_dim:
        vec = np.pad(vec, (0, local_dim - vec.shape[0]))
    out[: min(local_dim, output_state_dim)] = vec[:output_state_dim]
    return out


def _extract_action_step(
    step: Dict[str, Any], config: OXEDatasetConfig, output_action_dim: int
) -> np.ndarray:
    out = np.zeros(output_action_dim, dtype=np.float32)
    if output_action_dim == 0 or config.action_dim <= 0:
        return out

    action_obj = step.get("action")
    value = None
    if config.action_is_dict and config.action_key:
        if isinstance(action_obj, dict):
            value = _resolve_optional(action_obj, config.action_key)
    elif config.action_key:
        value = _resolve_optional(step, config.action_key)
        if value is None and isinstance(action_obj, dict):
            value = _resolve_optional(action_obj, config.action_key)
    else:
        value = action_obj

    vec = _to_float_vector(value)
    if vec is None:
        vec = np.zeros(config.action_dim, dtype=np.float32)

    local_dim = int(max(0, config.action_dim))
    vec = vec[:local_dim]
    if vec.shape[0] < local_dim:
        vec = np.pad(vec, (0, local_dim - vec.shape[0]))
    out[: min(local_dim, output_action_dim)] = vec[:output_action_dim]
    return out
