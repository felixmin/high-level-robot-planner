"""
OXE adapter v2: gather-based pairs, unified tf.data mixing, deferred JPEG decode.

Follows the Octo/dlimp pattern:
  SkipDecoding -> gather pairs -> from_tensor_slices -> sample_from_datasets
  -> shuffle -> batch -> decode_and_resize -> ram_budget -> prefetch
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from common.adapters.oxe_shared import (
    OXE_DATASETS,
    OXEDatasetConfig,
    _import_tensorflow_cpu_only,
    _resolve_tfds_builder_dir,
    compute_pair_frame_indices,
    pad_or_truncate_1d,
    resolve_nested_key,
    strip_null_bytes,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thread budget allocation (from Octo pattern)
# ---------------------------------------------------------------------------


def _allocate_threads(total: int, weights: List[float]) -> List[int]:
    """Distribute total threads across datasets proportionally by weight, min 1 each."""
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()
    raw = weights * total
    alloc = np.maximum(1, np.round(raw)).astype(int)
    # Clamp to total
    while alloc.sum() > total and total >= len(weights):
        idx = np.argmax(alloc)
        alloc[idx] = max(1, alloc[idx] - 1)
    return alloc.tolist()


# ---------------------------------------------------------------------------
# Per-episode gather transform (replaces v1 ring-buffer scan)
# ---------------------------------------------------------------------------


def _make_episode_to_pairs_fn(
    *,
    config: OXEDatasetConfig,
    offset: int,
    pair_frame_indices: List[int],
    image_size: int,
    output_action_dim: int,
    output_state_dim: int,
):
    """
    Returns a tf.data map function that converts an episode dict into a
    tf.data.Dataset of frame pairs (all T-K valid pairs).

    Images stay as tf.string (JPEG) — decode happens later in the pipeline.
    Actions and states are padded/truncated to output_action_dim / output_state_dim
    so all datasets produce uniform shapes for batching.
    """
    tf = _import_tensorflow_cpu_only()

    num_frames = len(pair_frame_indices)
    action_dim = config.action_dim
    state_dim = config.state_dim
    pair_indices_const = tf.constant(pair_frame_indices, dtype=tf.int32)

    out_action_dim = max(output_action_dim, 1)
    out_state_dim = max(output_state_dim, 1)

    def episode_to_pairs(episode):
        steps = episode["steps"]

        # --- Images (RLDS: steps["observation"][image_key]) ---
        image_key = config.image_key
        images = resolve_nested_key(steps["observation"], image_key)

        # Some datasets store raw uint8 instead of JPEG — encode for uniform pipeline
        if images.dtype != tf.string:
            images = tf.map_fn(
                lambda img: tf.io.encode_jpeg(tf.cast(img, tf.uint8)),
                images, fn_output_signature=tf.string,
            )

        T = tf.shape(images)[0]
        n_pairs = tf.maximum(T - offset, 0)
        start_indices = tf.range(n_pairs)

        # --- Gather frames (still JPEG bytes): (n_pairs, num_frames) ---
        abs_indices = (
            tf.expand_dims(start_indices, 1)
            + tf.expand_dims(pair_indices_const, 0)
        )
        flat_indices = tf.reshape(abs_indices, [-1])
        flat_frames = tf.gather(images, flat_indices)
        frames = tf.reshape(flat_frames, [n_pairs, num_frames])

        # --- Actions: cumulative sum between start and end ---
        if config.action_is_dict and config.action_key:
            raw_actions = resolve_nested_key(steps["action"], config.action_key)
        elif config.action_key:
            raw_actions = resolve_nested_key(steps, config.action_key)
        else:
            raw_actions = steps["action"]

        raw_actions = tf.cast(raw_actions, tf.float32)
        raw_actions = tf.reshape(raw_actions, [T, -1])  # flatten extra dims

        if action_dim > 0:
            raw_actions = raw_actions[:, :action_dim]
            action_cumsum = tf.cumsum(raw_actions, axis=0)
            end_cum = tf.gather(action_cumsum, start_indices + offset)
            start_cum = tf.gather(action_cumsum, start_indices)
            cum_actions = end_cum - start_cum
            # Pad to uniform output dim for cross-dataset batching
            if action_dim < out_action_dim:
                cum_actions = tf.pad(
                    cum_actions, [[0, 0], [0, out_action_dim - action_dim]]
                )
            elif action_dim > out_action_dim:
                cum_actions = cum_actions[:, :out_action_dim]
        else:
            cum_actions = tf.zeros([n_pairs, out_action_dim], dtype=tf.float32)

        # --- State: initial state at frame 0 ---
        if config.state_key and state_dim > 0:
            raw_state = resolve_nested_key(steps["observation"], config.state_key)
            raw_state = tf.cast(raw_state, tf.float32)
            state0 = tf.cond(
                T > 0,
                lambda: pad_or_truncate_1d(tf.reshape(raw_state[0], [-1]), out_state_dim),
                lambda: tf.zeros([out_state_dim], dtype=tf.float32),
            )
            initial_states = tf.tile(
                tf.expand_dims(state0, 0), [n_pairs, 1]
            )
        else:
            initial_states = tf.zeros([n_pairs, out_state_dim], dtype=tf.float32)

        # --- Language instruction ---
        if config.instruction_in_step:
            lang_raw = steps[config.instruction_key]
        else:
            lang_raw = steps["observation"][config.instruction_key]

        if lang_raw.dtype == tf.string:
            lang_scalar = tf.cond(
                T > 0,
                lambda: strip_null_bytes(lang_raw[0]),
                lambda: tf.constant("", dtype=tf.string),
            )
        else:
            lang_scalar = tf.constant("", dtype=tf.string)
        languages = tf.fill([n_pairs], lang_scalar)

        # --- Metadata ---
        ep_id_raw = episode.get("episode_id", tf.constant("", dtype=tf.string))
        if ep_id_raw.dtype != tf.string:
            ep_id_raw = tf.strings.as_string(ep_id_raw)

        dataset_name_const = tf.constant(config.name, dtype=tf.string)

        if config.robot_key and "episode_metadata" in episode:
            robot_raw = episode["episode_metadata"].get(
                config.robot_key, tf.constant("unknown", dtype=tf.string)
            )
            if robot_raw.dtype != tf.string:
                robot_raw = tf.strings.as_string(robot_raw)
        else:
            robot_raw = tf.constant("unknown", dtype=tf.string)

        return tf.data.Dataset.from_tensor_slices({
            "frames": frames,
            "action": cum_actions,
            "initial_state": initial_states,
            "language": languages,
            "frame_idx": start_indices,
            "episode_id": tf.fill([n_pairs], ep_id_raw),
            "dataset_name": tf.fill([n_pairs], dataset_name_const),
            "robot": tf.fill([n_pairs], robot_raw),
        })

    return episode_to_pairs


# ---------------------------------------------------------------------------
# Single-dataset pipeline builder
# ---------------------------------------------------------------------------


def build_single_dataset_pipeline(
    *,
    config: OXEDatasetConfig,
    split: str,
    offset: int,
    pair_frame_indices: List[int],
    image_size: int,
    num_parallel_calls: int,
    output_action_dim: int,
    output_state_dim: int,
    train: bool = True,
    tfds_source: str = "auto",
    tfds_local_root: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Build a tf.data pipeline for a single OXE dataset.

    Returns an infinite stream of frame-pair dicts (images as tf.string JPEG).
    """
    tf = _import_tensorflow_cpu_only()
    import tensorflow_datasets as tfds

    builder_dir = _resolve_tfds_builder_dir(
        gcs_builder_dir=config.gcs_path,
        source=tfds_source,
        local_root=tfds_local_root,
    )
    builder = tfds.builder_from_directory(builder_dir)

    episode_to_pairs = _make_episode_to_pairs_fn(
        config=config,
        offset=offset,
        pair_frame_indices=pair_frame_indices,
        image_size=image_size,
        output_action_dim=output_action_dim,
        output_state_dim=output_state_dim,
    )

    # Octo pattern: file-level shuffle only (not episode-level dataset.shuffle
    # which materializes full episodes in memory). Post-mix sample shuffle
    # provides cross-episode diversity.
    dataset = builder.as_dataset(
        split=split,
        shuffle_files=train,
        decoders={"steps": tfds.decode.SkipDecoding()},
        read_config=tfds.ReadConfig(
            skip_prefetch=True,
            num_parallel_calls_for_interleave_files=num_parallel_calls,
            interleave_cycle_length=num_parallel_calls,
        ),
    )

    dataset = dataset.repeat()

    dataset = dataset.interleave(
        episode_to_pairs,
        cycle_length=num_parallel_calls,
        num_parallel_calls=num_parallel_calls,
        deterministic=False,
    )

    return dataset


# ---------------------------------------------------------------------------
# Multi-dataset mixed pipeline builder
# ---------------------------------------------------------------------------


def _make_decode_and_resize_fn(image_size: int, num_frames: int):
    """Returns a batched map function that decodes JPEG frames and resizes."""
    tf = _import_tensorflow_cpu_only()

    def decode_and_resize(batch):
        # batch["frames"]: (B, num_frames) tf.string
        flat = tf.reshape(batch["frames"], [-1])

        def _decode_one(jpeg_bytes):
            img = tf.io.decode_jpeg(jpeg_bytes, channels=3)
            img = tf.image.resize(img, [image_size, image_size])
            return tf.cast(img, tf.uint8)

        decoded = tf.map_fn(_decode_one, flat, fn_output_signature=tf.uint8)
        # decoded: (B*num_frames, H, W, 3)
        B = tf.shape(batch["frames"])[0]
        decoded = tf.reshape(decoded, [B, num_frames, image_size, image_size, 3])
        batch["frames"] = decoded
        return batch

    return decode_and_resize


def build_mixed_pipeline(
    *,
    dataset_entries: List[Dict[str, Any]],
    image_size: int,
    batch_size: int,
    pair_frames_mode: str = "endpoints",
    pair_frames_stride: int = 1,
    pair_frames_n: int = 2,
    total_threads: int = 48,
    ram_budget_gb: int = 1,
    shuffle_sample_buffer: int = 5000,
    tfds_source: str = "auto",
    tfds_local_root: Optional[str] = None,
    seed: Optional[int] = None,
    train: bool = True,
    use_synthetic_data: bool = False,
    synthetic_num_samples: int = 1000,
):
    """
    Build the complete mixed pipeline for N OXE datasets.

    Returns a batched tf.data pipeline with decoded uint8 images.
    """
    tf = _import_tensorflow_cpu_only()

    weights = []
    configs = []
    offsets = []
    for entry in dataset_entries:
        name = entry["name"]
        if name not in OXE_DATASETS:
            raise ValueError(f"Unknown OXE dataset: {name!r}")
        configs.append(OXE_DATASETS[name])
        weights.append(float(entry.get("weight", 1.0)))
        offsets.append(int(entry["pair_offset_steps"]))

    if use_synthetic_data:
        return _build_synthetic_pipeline(
            configs=configs,
            offsets=offsets,
            image_size=image_size,
            batch_size=batch_size,
            pair_frames_mode=pair_frames_mode,
            pair_frames_stride=pair_frames_stride,
            pair_frames_n=pair_frames_n,
            num_samples=synthetic_num_samples,
        )

    threads_per_dataset = _allocate_threads(total_threads, weights)
    sample_shuffle = shuffle_sample_buffer if train else 0

    # Compute max action/state dims for uniform batching across datasets
    max_action_dim = max((c.action_dim for c in configs), default=0)
    max_state_dim = max((c.state_dim for c in configs), default=0)

    # Build per-dataset pipelines
    datasets = []
    all_num_frames = set()
    for cfg, entry, offset, threads in zip(
        configs, dataset_entries, offsets, threads_per_dataset
    ):
        split = entry.get("train_split", "train[:90%]") if train else entry.get("val_split", "train[90%:]")
        pair_indices = compute_pair_frame_indices(offset, pair_frames_mode, pair_frames_stride, pair_frames_n)
        all_num_frames.add(len(pair_indices))

        ds = build_single_dataset_pipeline(
            config=cfg,
            split=split,
            offset=offset,
            pair_frame_indices=pair_indices,
            image_size=image_size,
            num_parallel_calls=max(1, threads),
            output_action_dim=max_action_dim,
            output_state_dim=max_state_dim,
            train=train,
            tfds_source=tfds_source,
            tfds_local_root=tfds_local_root,
            seed=seed,
        )
        datasets.append(ds)

    if len(all_num_frames) != 1:
        raise ValueError(
            f"All datasets must produce same num_frames but got {all_num_frames}. "
            "Check pair_frames_mode and offsets."
        )
    num_frames = all_num_frames.pop()

    # Mix
    norm_weights = np.array(weights, dtype=np.float64)
    norm_weights = norm_weights / norm_weights.sum()

    dataset = tf.data.Dataset.sample_from_datasets(
        datasets,
        weights=norm_weights.tolist(),
        seed=seed,
        stop_on_empty_dataset=False,
    )

    if sample_shuffle > 0:
        dataset = dataset.shuffle(sample_shuffle, seed=seed)

    dataset = dataset.batch(batch_size)

    # Decode and resize (the single decode point)
    decode_fn = _make_decode_and_resize_fn(image_size, num_frames)
    dataset = dataset.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # RAM budget
    options = tf.data.Options()
    options.autotune.ram_budget = ram_budget_gb * 1024 * 1024 * 1024
    options.autotune.enabled = True
    options.deterministic = False
    dataset = dataset.with_options(options)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_frames


def _build_synthetic_pipeline(
    *,
    configs,
    offsets,
    image_size,
    batch_size,
    pair_frames_mode,
    pair_frames_stride,
    pair_frames_n,
    num_samples,
):
    """Build a synthetic pipeline for testing without real data."""
    tf = _import_tensorflow_cpu_only()

    offset = offsets[0]
    pair_indices = compute_pair_frame_indices(offset, pair_frames_mode, pair_frames_stride, pair_frames_n)
    num_frames = len(pair_indices)
    cfg = configs[0]

    def gen():
        rng = np.random.default_rng(42)
        for i in range(num_samples):
            # Create fake JPEG bytes
            fake_img = rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(fake_img).save(buf, format="JPEG")
            jpeg_bytes = buf.getvalue()

            yield {
                "frames": [jpeg_bytes] * num_frames,
                "action": rng.standard_normal(cfg.action_dim).astype(np.float32),
                "initial_state": rng.standard_normal(max(cfg.state_dim, 1)).astype(np.float32),
                "language": f"do something {i}",
                "frame_idx": i % 100,
                "episode_id": f"ep_{i // 10}",
                "dataset_name": cfg.name,
                "robot": "synthetic",
            }

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "frames": tf.TensorSpec([num_frames], tf.string),
            "action": tf.TensorSpec([cfg.action_dim], tf.float32),
            "initial_state": tf.TensorSpec([max(cfg.state_dim, 1)], tf.float32),
            "language": tf.TensorSpec([], tf.string),
            "frame_idx": tf.TensorSpec([], tf.int32),
            "episode_id": tf.TensorSpec([], tf.string),
            "dataset_name": tf.TensorSpec([], tf.string),
            "robot": tf.TensorSpec([], tf.string),
        },
    ).repeat()

    dataset = dataset.batch(batch_size)

    decode_fn = _make_decode_and_resize_fn(image_size, num_frames)
    dataset = dataset.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_frames


# ---------------------------------------------------------------------------
# PyTorch IterableDataset wrapper
# ---------------------------------------------------------------------------


class OXEFramePairDatasetV2(IterableDataset):
    """
    Thin PyTorch wrapper around a v2 tf.data pipeline.

    Converts tf tensors to torch and permutes to (B, C, num_frames, H, W).
    """

    def __init__(
        self,
        *,
        dataset_entries: List[Dict[str, Any]],
        image_size: int,
        batch_size: int,
        pair_frames_mode: str = "endpoints",
        pair_frames_stride: int = 1,
        pair_frames_n: int = 2,
        total_threads: int = 48,
        ram_budget_gb: int = 1,
        shuffle_sample_buffer: int = 5000,
        tfds_source: str = "auto",
        tfds_local_root: Optional[str] = None,
        seed: Optional[int] = None,
        train: bool = True,
        use_synthetic_data: bool = False,
        synthetic_num_samples: int = 1000,
    ):
        super().__init__()
        self._build_kwargs = dict(
            dataset_entries=dataset_entries,
            image_size=image_size,
            batch_size=batch_size,
            pair_frames_mode=pair_frames_mode,
            pair_frames_stride=pair_frames_stride,
            pair_frames_n=pair_frames_n,
            total_threads=total_threads,
            ram_budget_gb=ram_budget_gb,
            shuffle_sample_buffer=shuffle_sample_buffer,
            tfds_source=tfds_source,
            tfds_local_root=tfds_local_root,
            seed=seed,
            train=train,
            use_synthetic_data=use_synthetic_data,
            synthetic_num_samples=synthetic_num_samples,
        )
        self.image_size = image_size
        self.batch_size = batch_size
        self._pipeline = None
        self._iterator = None
        self._num_frames = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            self._pipeline, self._num_frames = build_mixed_pipeline(**self._build_kwargs)
            self._iterator = iter(self._pipeline)

    def __iter__(self):
        self._ensure_pipeline()
        return self

    def __next__(self):
        self._ensure_pipeline()
        batch = next(self._iterator)

        # Convert tf tensors to numpy
        frames_np = batch["frames"].numpy()  # (B, num_frames, H, W, 3) uint8
        action_np = batch["action"].numpy()  # (B, action_dim)
        state_np = batch["initial_state"].numpy()  # (B, state_dim)
        language = [s.decode("utf-8", errors="replace") for s in batch["language"].numpy()]
        frame_idx = batch["frame_idx"].numpy().tolist()
        episode_id = [s.decode("utf-8", errors="replace") for s in batch["episode_id"].numpy()]
        dataset_name = [s.decode("utf-8", errors="replace") for s in batch["dataset_name"].numpy()]
        robot = [s.decode("utf-8", errors="replace") for s in batch["robot"].numpy()]

        # (B, num_frames, H, W, 3) -> (B, 3, num_frames, H, W)
        frames_torch = torch.from_numpy(frames_np).permute(0, 4, 2, 1, 3)
        # Actually: (B, num_frames, H, W, 3) -> want (B, 3, num_frames, H, W)
        # permute: 0=B, 1=num_frames, 2=H, 3=W, 4=C -> (0, 4, 1, 2, 3)
        frames_torch = torch.from_numpy(frames_np).permute(0, 4, 1, 2, 3)

        return {
            "frames": frames_torch,
            "episode_id": episode_id,
            "frame_idx": frame_idx,
            "offset": self._build_kwargs["dataset_entries"][0]["pair_offset_steps"],
            "language": language,
            "dataset_name": dataset_name,
            "action": [action_np[i] for i in range(action_np.shape[0])],
            "initial_state": [state_np[i] for i in range(state_np.shape[0])],
            "robot": robot,
        }

    def cleanup(self):
        self._iterator = None
        self._pipeline = None
        import gc
        gc.collect()

    def __del__(self):
        self.cleanup()
