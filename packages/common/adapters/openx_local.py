"""
Local OpenX adapter for non-network OXE loading from downloaded HF tar shards.

This backend reads episodes from local `<root>/<dataset_name>/*.tar` shards and
emits frame-pair samples compatible with existing OXE training batches.
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import random
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from common.adapters.oxe_shared import OXE_DATASETS, OXEDatasetConfig, resolve_nested_key

logger = logging.getLogger(__name__)

_TRAIN_SPLIT_RE = re.compile(r"^train(?:\[(.*)\])?$")
_EPISODE_INDEX_CACHE: Dict[tuple[str, str, int, Optional[int], Optional[int]], List[EpisodeRef]] = {}
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


@dataclass(frozen=True)
class EpisodeRef:
    shard_path: str
    member_name: str
    pair_count: int


@dataclass
class IndexedDatasetSlice:
    dataset_id: int
    dataset_name: str
    weight: float
    offset: int
    config: OXEDatasetConfig
    shard_paths: List[str]
    selected_members_by_shard: Dict[str, set[str]]
    pair_count: int


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    vals = [max(0.0, float(w)) for w in weights]
    total = float(sum(vals))
    if total <= 0.0:
        return [1.0 / float(len(vals)) for _ in vals]
    return [v / total for v in vals]


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
        if config.allow_missing_state:
            return out
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


def _iter_selected_episodes_from_shard(
    shard_path: str, selected_members: Optional[set[str]]
) -> Iterator[tuple[str, Dict[str, Any]]]:
    try:
        with tarfile.open(shard_path, "r") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".data.pickle"):
                    continue
                if selected_members is not None and member.name not in selected_members:
                    continue
                fileobj = tf.extractfile(member)
                if fileobj is None:
                    continue
                try:
                    episode = pickle.load(fileobj)
                except Exception:
                    continue
                if isinstance(episode, dict) and isinstance(episode.get("steps"), list):
                    yield member.name, episode
    except (tarfile.ReadError, OSError):
        return


def _scan_shard_episode_refs(
    shard_path: str, offset: int, pairs_per_episode: Optional[int]
) -> List[tuple[str, int]]:
    refs: List[tuple[str, int]] = []
    for member_name, episode in _iter_selected_episodes_from_shard(
        shard_path, selected_members=None
    ):
        steps = episode.get("steps")
        if not isinstance(steps, list):
            continue
        n_pairs = max(0, len(steps) - int(offset))
        if pairs_per_episode is not None:
            n_pairs = min(n_pairs, int(pairs_per_episode))
        if n_pairs <= 0:
            continue
        refs.append((member_name, int(n_pairs)))
    return refs


def _index_one_shard_task(
    shard_path: str, offset: int, pairs_per_episode: Optional[int]
) -> tuple[str, List[tuple[str, int]]]:
    return shard_path, _scan_shard_episode_refs(shard_path, offset, pairs_per_episode)


def _shard_cache_path(
    cache_dir: Path,
    dataset_name: str,
    shard_path: str,
    offset: int,
    pairs_per_episode: Optional[int],
) -> Path:
    p = Path(shard_path)
    stat = p.stat()
    token = (
        f"{dataset_name}__{p.name}__{stat.st_size}__{stat.st_mtime_ns}"
        f"__o{int(offset)}__ppe{int(pairs_per_episode) if pairs_per_episode is not None else -1}.pkl"
    )
    return cache_dir / token


def _read_cached_shard_index(path: Path) -> Optional[List[tuple[str, int]]]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None
    if not isinstance(obj, list):
        return None
    out: List[tuple[str, int]] = []
    for item in obj:
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], int)
        ):
            out.append((item[0], int(item[1])))
    return out


def _write_cached_shard_index(path: Path, refs: List[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(refs, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _sample_t_indices(
    max_t: int, pairs_per_episode: Optional[int], rng: random.Random
) -> List[int]:
    if max_t <= 0:
        return []
    if pairs_per_episode is None or pairs_per_episode <= 0 or pairs_per_episode >= max_t:
        return list(range(max_t))
    return sorted(rng.sample(range(max_t), k=int(pairs_per_episode)))


class OpenXLocalIndexedPairIterable(IterableDataset):
    """
    Local indexed OpenX pair dataset.

    Indexes episodes from local tar shards once at startup, then iterates pairs
    from selected train/val split ranges without network streaming.
    """

    def __init__(
        self,
        *,
        root: str,
        dataset_entries: List[Dict[str, Any]],
        split_key: str,
        image_size: int,
        return_metadata: bool,
        max_shards_per_dataset: Optional[int],
        pairs_per_episode: Optional[int],
        index_workers: int,
        index_cache_dir: Optional[str],
        seed: int,
        resample_each_epoch: bool,
        stopping_strategy: str,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.dataset_entries = list(dataset_entries)
        self.split_key = split_key
        self.image_size = int(image_size)
        self.return_metadata = bool(return_metadata)
        self.max_shards_per_dataset = (
            int(max_shards_per_dataset)
            if max_shards_per_dataset is not None and int(max_shards_per_dataset) > 0
            else None
        )
        self.pairs_per_episode = (
            int(pairs_per_episode)
            if pairs_per_episode is not None and int(pairs_per_episode) > 0
            else None
        )
        self.index_workers = int(index_workers)
        if self.index_workers <= 0:
            self.index_workers = max(1, os.cpu_count() or 1)
        self.index_cache_dir = (
            Path(index_cache_dir).expanduser().resolve()
            if index_cache_dir is not None and str(index_cache_dir).strip() != ""
            else None
        )
        self.seed = int(seed)
        self.resample_each_epoch = bool(resample_each_epoch)
        self.stopping_strategy = str(stopping_strategy)
        if self.stopping_strategy not in {"all_exhausted", "first_exhausted"}:
            raise ValueError(
                "stopping_strategy must be 'all_exhausted' or 'first_exhausted', "
                f"got {self.stopping_strategy!r}"
            )
        self.epoch = 0

        self._datasets = self._build_index()
        if not self._datasets:
            raise ValueError("OpenX local index has no selected datasets")

        self._weights = _normalize_weights([d.weight for d in self._datasets])
        self._max_action_dim = max((d.config.action_dim for d in self._datasets), default=0)
        self._max_state_dim = max((d.config.state_dim for d in self._datasets), default=0)
        self._total_pairs = int(sum(d.pair_count for d in self._datasets))

    def __len__(self) -> int:
        return self._total_pairs

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_index(self) -> List[IndexedDatasetSlice]:
        if not self.root.exists():
            raise FileNotFoundError(f"OpenX local root does not exist: {self.root}")

        slices: List[IndexedDatasetSlice] = []
        for dataset_id, entry in enumerate(self.dataset_entries):
            dataset_name = str(entry["name"])
            config = OXE_DATASETS.get(dataset_name)
            if config is None:
                config = _fallback_dataset_config(dataset_name)
            offset = int(entry["pair_offset_steps"])
            weight = float(entry.get("weight", 1.0))
            split = str(entry[self.split_key])

            dataset_dir = self.root / dataset_name
            if not dataset_dir.exists():
                logger.debug(f"Skipping missing local dataset directory: {dataset_dir}")
                continue

            shard_paths = sorted(str(p) for p in dataset_dir.glob("*.tar"))
            if self.max_shards_per_dataset is not None:
                shard_paths = shard_paths[: self.max_shards_per_dataset]
            if not shard_paths:
                logger.debug(f"Skipping dataset with no shards: {dataset_name}")
                continue

            cache_key = (
                str(self.root),
                dataset_name,
                offset,
                self.max_shards_per_dataset,
                self.pairs_per_episode,
            )
            all_eps = _EPISODE_INDEX_CACHE.get(cache_key)
            if all_eps is None:
                refs_by_shard: Dict[str, List[tuple[str, int]]] = {}
                to_scan: List[str] = []

                for shard_path in shard_paths:
                    if self.index_cache_dir is None:
                        to_scan.append(shard_path)
                        continue
                    cache_path = _shard_cache_path(
                        self.index_cache_dir,
                        dataset_name,
                        shard_path,
                        offset,
                        self.pairs_per_episode,
                    )
                    cached = _read_cached_shard_index(cache_path)
                    if cached is None:
                        to_scan.append(shard_path)
                    else:
                        refs_by_shard[shard_path] = cached

                if to_scan:
                    if self.index_workers == 1 or len(to_scan) == 1:
                        for shard_path in to_scan:
                            refs = _scan_shard_episode_refs(
                                shard_path, offset, self.pairs_per_episode
                            )
                            refs_by_shard[shard_path] = refs
                    else:
                        with ProcessPoolExecutor(
                            max_workers=min(self.index_workers, len(to_scan))
                        ) as ex:
                            futures = [
                                ex.submit(
                                    _index_one_shard_task,
                                    shard_path,
                                    offset,
                                    self.pairs_per_episode,
                                )
                                for shard_path in to_scan
                            ]
                            for fut in as_completed(futures):
                                shard_path, refs = fut.result()
                                refs_by_shard[shard_path] = refs

                    if self.index_cache_dir is not None:
                        for shard_path in to_scan:
                            refs = refs_by_shard.get(shard_path, [])
                            cache_path = _shard_cache_path(
                                self.index_cache_dir,
                                dataset_name,
                                shard_path,
                                offset,
                                self.pairs_per_episode,
                            )
                            try:
                                _write_cached_shard_index(cache_path, refs)
                            except Exception:
                                logger.debug(
                                    "Failed to write shard index cache: %s",
                                    cache_path,
                                )

                all_eps = []
                for shard_path in shard_paths:
                    refs = refs_by_shard.get(shard_path, [])
                    for member_name, pair_count in refs:
                        all_eps.append(
                            EpisodeRef(
                                shard_path=shard_path,
                                member_name=member_name,
                                pair_count=pair_count,
                            )
                        )
                _EPISODE_INDEX_CACHE[cache_key] = list(all_eps)

            if not all_eps:
                logger.warning(f"Skipping dataset with no valid episodes: {dataset_name}")
                continue

            start, end = _parse_train_split(split, len(all_eps))
            selected_eps = all_eps[start:end]
            if not selected_eps:
                logger.warning(
                    "Skipping dataset after split selection: %s split=%s episodes=%d",
                    dataset_name,
                    split,
                    len(all_eps),
                )
                continue

            selected_members_by_shard: Dict[str, set[str]] = {}
            ordered_selected_shards: List[str] = []
            for ep in selected_eps:
                if ep.shard_path not in selected_members_by_shard:
                    selected_members_by_shard[ep.shard_path] = set()
                    ordered_selected_shards.append(ep.shard_path)
                selected_members_by_shard[ep.shard_path].add(ep.member_name)

            pair_count = int(sum(ep.pair_count for ep in selected_eps))
            slices.append(
                IndexedDatasetSlice(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    weight=weight,
                    offset=offset,
                    config=config,
                    shard_paths=ordered_selected_shards,
                    selected_members_by_shard=selected_members_by_shard,
                    pair_count=pair_count,
                )
            )
            logger.info(
                "✓ Indexed local dataset %s split=%s shards=%d episodes=%d pairs=%d",
                dataset_name,
                split,
                len(ordered_selected_shards),
                len(selected_eps),
                pair_count,
            )

        return slices

    def _iter_dataset_for_worker(
        self, dataset_idx: int, worker_id: int, num_workers: int
    ) -> Iterator[Any]:
        ds = self._datasets[dataset_idx]
        base_seed = self.seed + worker_id * 100_003 + dataset_idx * 9973
        if self.resample_each_epoch:
            base_seed += self.epoch * 10_000_019
        rng = random.Random(base_seed)

        selected_shards = ds.shard_paths[worker_id::num_workers]
        for shard_path in selected_shards:
            selected_members = ds.selected_members_by_shard.get(shard_path, set())
            if not selected_members:
                continue
            for member_name, episode in _iter_selected_episodes_from_shard(
                shard_path, selected_members=selected_members
            ):
                steps = episode.get("steps")
                if not isinstance(steps, list):
                    continue
                max_t = len(steps) - ds.offset
                t_indices = _sample_t_indices(max_t, self.pairs_per_episode, rng)
                if not t_indices:
                    continue

                for t in t_indices:
                    step_t = steps[t]
                    step_h = steps[t + ds.offset]
                    if not isinstance(step_t, dict) or not isinstance(step_h, dict):
                        continue

                    obs_t = step_t.get("observation", {})
                    obs_h = step_h.get("observation", {})
                    if not isinstance(obs_t, dict) or not isinstance(obs_h, dict):
                        continue

                    raw_img_t = _extract_image(obs_t, ds.config)
                    raw_img_h = _extract_image(obs_h, ds.config)
                    if raw_img_t is None or raw_img_h is None:
                        continue

                    try:
                        img_t = _decode_image_to_tensor(raw_img_t, self.image_size)
                        img_h = _decode_image_to_tensor(raw_img_h, self.image_size)
                    except Exception:
                        continue

                    frames = torch.stack([img_t, img_h], dim=0).permute(1, 0, 2, 3)
                    if not self.return_metadata:
                        yield frames
                        continue

                    action = np.zeros(self._max_action_dim, dtype=np.float32)
                    for j in range(t, t + ds.offset):
                        action += _extract_action_step(
                            steps[j], ds.config, self._max_action_dim
                        )
                    initial_state = _extract_state(
                        obs_t, ds.config, self._max_state_dim
                    )
                    language = _extract_instruction(step_t, obs_t, ds.config)
                    episode_id = f"{ds.dataset_name}:{Path(shard_path).name}:{member_name}"

                    yield {
                        "frames": frames,
                        "episode_id": episode_id,
                        "frame_idx": int(t),
                        "dataset_name": ds.dataset_name,
                        "dataset_type": ds.dataset_name,
                        "language": language,
                        "offset": int(ds.offset),
                        "action": action,
                        "initial_state": initial_state,
                    }

    def __iter__(self) -> Iterator[Any]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)

        if len(self._datasets) == 1:
            yield from self._iter_dataset_for_worker(0, worker_id, num_workers)
            return

        select_seed = self.seed + worker_id * 1_000_003
        if self.resample_each_epoch:
            select_seed += self.epoch * 1_000_000_007
        selector = random.Random(select_seed)

        iterators = {
            i: self._iter_dataset_for_worker(i, worker_id, num_workers)
            for i in range(len(self._datasets))
        }
        active = list(iterators.keys())
        weights = {i: self._weights[i] for i in active}

        while active:
            choice = selector.choices(active, weights=[weights[i] for i in active], k=1)[0]
            try:
                yield next(iterators[choice])
            except StopIteration:
                if self.stopping_strategy == "first_exhausted":
                    return
                active.remove(choice)
                weights.pop(choice, None)
                if not active:
                    return
                norm = float(sum(weights.values()))
                if norm <= 0.0:
                    for i in active:
                        weights[i] = 1.0 / float(len(active))
                else:
                    for i in active:
                        weights[i] = weights[i] / norm


class OpenXLocalFullPairDataset(Dataset):
    """Materialized local OpenX pair dataset for small full-RAM experiments."""

    def __init__(self, indexed_ds: OpenXLocalIndexedPairIterable, max_pairs: Optional[int]):
        self.samples: List[Any] = []
        limit = int(max_pairs) if max_pairs is not None and int(max_pairs) > 0 else None
        for sample in indexed_ds:
            self.samples.append(sample)
            if limit is not None and len(self.samples) >= limit:
                break
        if not self.samples:
            raise ValueError("OpenX full mode produced zero samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        return self.samples[idx]
