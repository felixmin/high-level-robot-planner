import logging

"""
Open X-Embodiment (OXE) adapter for streaming RLDS datasets from GCS.

This adapter is different from file-based adapters (YouTube, Bridge) because:
1. Data streams from Google Cloud Storage, not local files
2. Uses tf.data pipelines for efficient prefetching
3. Returns tensors directly, not file paths
4. Handles heterogeneous action/observation formats across datasets

Currently supports:
- language_table: gs://gresearch/robotics/language_table/0.0.1
  * 442k episodes, 2D tabletop manipulation
  * Action: 2D (x, y), State: 2D effector_translation
  * Language instructions as encoded tensors

- language_table_blocktorelative_oracle_sim: gs://gresearch/robotics/language_table_blocktorelative_oracle_sim/0.0.1
  * 200k episodes, oracle agent with longer trajectories (27-46 steps)
  * Same format as language_table

- bridge: gs://gresearch/robotics/bridge/0.1.0
  * 25,460 train + 3,475 test episodes, WidowX kitchen manipulation
  * Action: 3D world_vector (xyz translation) - uses first 2 dims for 2D plots
  * State: 7D robot state - uses first 2 dims for visualization
  * Language instructions as string tensors

- rt1: gs://gresearch/robotics/fractal20220817_data/0.1.0
  * 87k episodes, Google Robot mobile manipulator
  * Action: 3D world_vector (EEF displacement, dict-based like Bridge)
  * State: 7D base_pose_tool_reached (position + quaternion)
  * Language instructions as string tensors

- robonet: gs://gresearch/robotics/robo_net/1.0.0
  * 83k episodes, multi-robot (widowx, franka, baxter, sawyer)
  * Action: 5D flat array (3D EEF delta + wrist rotation + gripper)
  * State: 5D (EEF position + theta + gripper)
  * Language instructions at step level (not in observation)
  * Robot type available in episode_metadata for filtering

Key features:
- Automatic handling of dict-based actions (Bridge, RT-1) vs flat arrays (language_table, RoboNet)
- Automatic handling of string vs encoded instruction formats
- Support for step-level instructions (RoboNet) vs observation-level (others)
- Robot type extraction for multi-robot datasets
- Cumulative action computation between frame pairs
- TFDS split syntax support (e.g., "train[:90%]", "train[1000:2000]")
"""

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset
import random
import os
import zlib


@dataclass
class OXEDatasetConfig:
    """Configuration for an OXE dataset."""

    name: str
    gcs_path: str
    image_key: str = "rgb"  # Key in observation dict for images
    instruction_key: str = "instruction"  # Key for language instruction
    state_key: str = "effector_translation"  # Key for robot state (e.g. gripper pos)
    image_shape: Tuple[int, int, int] = (360, 640, 3)  # H, W, C
    control_frequency_hz: float = 10.0  # For time calculations
    action_dim: int = 2  # Dimensionality of action space (2 for language_table)
    state_dim: int = 2  # Dimensionality of state space to extract
    # For datasets with dict-based actions (like Bridge)
    action_key: Optional[str] = None  # If None, action is flat array; if set, extract this key
    action_is_dict: bool = False  # True if action is a dict with multiple keys
    # For datasets where instruction is at step level, not in observation (e.g., RoboNet)
    instruction_in_step: bool = False
    # For datasets with episode-level metadata (e.g., RoboNet has robot type)
    robot_key: Optional[str] = None  # Key in episode_metadata for robot type
    # Approx avg episode length for __len__ calculation
    avg_episode_length: int = 30


# Registry of supported OXE datasets
OXE_DATASETS = {
    "language_table": OXEDatasetConfig(
        name="language_table",
        gcs_path="gs://gresearch/robotics/language_table/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
        avg_episode_length=40,  # ~40 steps on average
    ),
    # Oracle sim datasets (scripted agent, longer episodes, cleaner data)
    "language_table_blocktorelative_oracle_sim": OXEDatasetConfig(
        name="language_table_blocktorelative_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_blocktorelative_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
        avg_episode_length=40,
    ),
    "language_table_blocktoblock_oracle_sim": OXEDatasetConfig(
        name="language_table_blocktoblock_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_blocktoblock_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
        avg_episode_length=40,
    ),
    "language_table_blocktoabsolute_oracle_sim": OXEDatasetConfig(
        name="language_table_blocktoabsolute_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
        avg_episode_length=40,
    ),
    "language_table_separate_oracle_sim": OXEDatasetConfig(
        name="language_table_separate_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_separate_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
        avg_episode_length=40,
    ),
    # Bridge dataset (WidowX kitchen manipulation)
    "bridge": OXEDatasetConfig(
        name="bridge",
        gcs_path="gs://gresearch/robotics/bridge/0.1.0",
        image_key="image",  # Bridge uses "image" not "rgb"
        instruction_key="natural_language_instruction",
        state_key="state",  # 7D robot state
        image_shape=(480, 640, 3),
        control_frequency_hz=5.0,  # Bridge is ~5Hz
        action_dim=3,  # world_vector is 3D (we use first 2 for 2D plots)
        state_dim=2,  # Use first 2 dims of 7D state for visualization
        action_key="world_vector",  # Extract world_vector from action dict
        action_is_dict=True,
        avg_episode_length=50,  # Bridge episodes are typically longer
    ),
    # RT-1 dataset (Google Robot, mobile manipulator)
    # 87k episodes of table-top manipulation with 17 objects
    "rt1": OXEDatasetConfig(
        name="rt1",
        gcs_path="gs://gresearch/robotics/fractal20220817_data/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        state_key="base_pose_tool_reached",  # 7D: position (3) + quaternion (4)
        image_shape=(256, 320, 3),
        control_frequency_hz=3.0,  # ~3Hz based on episode lengths
        action_dim=3,  # world_vector is 3D EEF displacement
        state_dim=3,  # Use first 3 dims (position) for visualization
        action_key="world_vector",  # Extract from action dict
        action_is_dict=True,
        avg_episode_length=30,  # RT-1 episodes are variable but often ~30 steps
    ),
    # RoboNet dataset (multi-robot, random interactions)
    # 83k episodes across widowx, franka, baxter, sawyer robots
    "robonet": OXEDatasetConfig(
        name="robonet",
        gcs_path="gs://gresearch/robotics/robo_net/1.0.0",
        image_key="image",  # Main camera (also has image1, image2)
        instruction_key="language_instruction",  # Step-level, not in observation
        state_key="state",  # 5D: [eef_x, eef_y, eef_z, theta, gripper]
        image_shape=(240, 320, 3),
        control_frequency_hz=10.0,  # Estimate
        action_dim=3,  # First 3 dims of 5D action (EEF delta)
        state_dim=3,  # First 3 dims of 5D state (EEF position)
        action_key=None,  # Flat action array
        action_is_dict=False,
        instruction_in_step=True,  # Instruction at step level, not observation
        robot_key="robot",  # episode_metadata.robot for filtering
        avg_episode_length=30,  # Estimate
    ),
}


class OXEFramePairDataset(IterableDataset):
    """
    PyTorch IterableDataset that streams frame pairs from OXE datasets on GCS.

    Uses tf.data internally for efficient streaming and prefetching,
    converts to PyTorch tensors on iteration.

    IMPORTANT: This dataset uses persistent tf.data pipelines to avoid memory leaks.
    The pipeline is created once and reused across epochs. Call cleanup() explicitly
    if you need to release TensorFlow resources before the object is garbage collected.

    Args:
        dataset_name: Name of OXE dataset (e.g., "language_table")
        split: TFDS split string (e.g., "train", "train[:1000]")
        offset: Frame offset for pairs (in steps, not seconds)
        image_size: Target image size (will resize if different)
        shuffle_buffer: Size of shuffle buffer (0 to disable)
        prefetch_buffer: tf.data prefetch buffer size
        num_parallel_calls: Parallelism for tf.data operations (default: AUTOTUNE)
        return_metadata: If True, return dict with metadata
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        offset: int,
        prefetch_buffer: int,
        episode_shuffle_buffer: int,
        pair_shuffle_buffer: int,
        image_size: int = 256,
        num_parallel_calls: Optional[int] = None,  # None = AUTOTUNE
        return_metadata: bool = False,
        gcs_path: Optional[str] = None,
        persistent_iterator: bool = True,  # Keep iterator alive to avoid shuffle buffer refill
        samples_per_episode: int = 0,
        seed: Optional[int] = None,
        precomputed_size: Optional[int] = None,  # Avoid TF init in __len__
        episode_prefetch_buffer: int = 1,  # Phase 3: overlap episode fetch/decode
        num_parallel_episodes: int = 4,  # Phase 4: parallel episode processing via interleave
    ):
        super().__init__()

        if dataset_name not in OXE_DATASETS and gcs_path is None:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(OXE_DATASETS.keys())} or provide gcs_path"
            )

        self.config = OXE_DATASETS.get(dataset_name)
        if gcs_path is not None:
            # Override GCS path if provided
            if self.config is None:
                self.config = OXEDatasetConfig(name=dataset_name, gcs_path=gcs_path)
            else:
                self.config = OXEDatasetConfig(
                    name=self.config.name,
                    gcs_path=gcs_path,
                    image_key=self.config.image_key,
                    instruction_key=self.config.instruction_key,
                    image_shape=self.config.image_shape,
                    control_frequency_hz=self.config.control_frequency_hz,
                )

        self.split = split
        self.offset = offset
        self.image_size = image_size
        self.prefetch_buffer = prefetch_buffer
        self.num_parallel_calls = num_parallel_calls
        self.return_metadata = return_metadata
        self.persistent_iterator = persistent_iterator
        self.samples_per_episode = samples_per_episode
        self.seed = seed
        self._rng = random.Random(seed)
        self._tf_seed: Optional[int] = None
        self._precomputed_size = precomputed_size
        self.episode_prefetch_buffer = episode_prefetch_buffer
        self.num_parallel_episodes = num_parallel_episodes
        self.episode_shuffle_buffer = episode_shuffle_buffer
        self.pair_shuffle_buffer = pair_shuffle_buffer

        # Lazy initialization of tf.data pipeline
        self._builder = None
        self._num_episodes = None
        # Persistent pipeline - created once, reused across epochs
        self._persistent_pipeline = None
        # Persistent iterator - avoids shuffle buffer refill on each epoch
        self._pipeline_iterator = None
        # Idempotency flag for cleanup
        self._cleaned_up = False

    def _init_rng_for_worker(self) -> None:
        """
        Ensure sampling RNG is unique per DataLoader worker (and reproducible when seed is set).

        Note: IterableDataset + num_workers>0 does not shard automatically; we also shard the
        TFDS episode dataset in `_create_tf_pipeline()` to avoid duplicate episodes.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if self.seed is not None:
            combined_seed = (int(self.seed) + 1000003 * int(worker_id)) & 0x7FFFFFFF
        else:
            # Avoid identical RNG state across forked workers when seed is not provided.
            entropy = int.from_bytes(os.urandom(8), "little")
            combined_seed = (entropy ^ (os.getpid() << 16) ^ int(worker_id)) & 0x7FFFFFFF

        self._rng = random.Random(combined_seed)
        self._tf_seed = combined_seed

    @staticmethod
    def _decode_tf_string(val: Any) -> str:
        if val is None:
            raise ValueError("Expected a string value, got None")
        if hasattr(val, "numpy"):
            val = val.numpy()
        if isinstance(val, bytes):
            return val.decode("utf-8").rstrip("\x00")
        if isinstance(val, str):
            return val.rstrip("\x00")
        raise TypeError(f"Expected bytes/str for string value, got {type(val)}")

    def _init_tfds(self):
        """Initialize TFDS builder and dataset (lazy)."""
        if self._builder is not None:
            return

        # Import TF only when needed
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Disable GPU for tf.data (we only use CPU for data loading)
        tf.config.set_visible_devices([], "GPU")

        self._builder = tfds.builder_from_directory(self.config.gcs_path)
        self._num_episodes = self._builder.info.splits[
            self.split.split("[")[0]
        ].num_examples

    def __len__(self):
        """
        Approximate length based on episodes * avg pairs per episode.

        IMPORTANT: If precomputed_size is set, returns that value directly to avoid
        TensorFlow initialization. This prevents startup hangs and OOM issues when
        DataLoader workers fork before TF is initialized.
        """
        if self._precomputed_size is not None:
            return self._precomputed_size
        raise ValueError(
            "OXEFramePairDataset.__len__ requires precomputed_size to be provided "
            "(to avoid initializing TensorFlow/TFDS inside DataLoader workers)."
        )

    def _create_tf_pipeline(self):
        """
        Create tf.data pipeline for streaming frame pairs.

        Phase 4: Use `Dataset.interleave()` with a per-episode tf.data pipeline.
        The per-episode pipeline must be TF-native (no `.numpy()`, no Python lists)
        so it can be traced safely by tf.data.
        """
        import tensorflow as tf

        self._init_tfds()
        self._init_rng_for_worker()
        ds = self._builder.as_dataset(split=self.split)

        # Shard episodes across DataLoader workers to avoid duplicates when num_workers > 0.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        # Shuffle episodes
        if self.episode_shuffle_buffer > 0:
            ds = ds.shuffle(self.episode_shuffle_buffer, seed=self._tf_seed)

        # Prefetch episodes (keep small; each episode can be large)
        if self.episode_prefetch_buffer > 0:
            ds = ds.prefetch(self.episode_prefetch_buffer)

        image_key = self.config.image_key
        instruction_key = self.config.instruction_key
        state_key = self.config.state_key
        offset = int(self.offset)
        image_size = int(self.image_size)
        return_metadata = bool(self.return_metadata)
        dataset_name = self.config.name
        action_dim = int(self.config.action_dim)
        state_dim = int(self.config.state_dim)
        action_key = self.config.action_key
        action_is_dict = bool(self.config.action_is_dict)
        instruction_in_step = bool(self.config.instruction_in_step)
        robot_key = self.config.robot_key
        samples_per_episode = int(self.samples_per_episode) if self.samples_per_episode else 0
        num_parallel_episodes = int(self.num_parallel_episodes)
        num_parallel_calls = (
            self.num_parallel_calls if self.num_parallel_calls is not None else tf.data.AUTOTUNE
        )

        offset_tf = tf.constant(offset, dtype=tf.int32)
        dataset_name_tf = tf.constant(dataset_name, dtype=tf.string)
        per_episode_sample_shuffle = int(self.pair_shuffle_buffer) if self.pair_shuffle_buffer > 0 else 1000

        def _strip_null_bytes(s: tf.Tensor) -> tf.Tensor:
            s = tf.convert_to_tensor(s, dtype=tf.string)
            return tf.strings.regex_replace(s, "\x00+$", "")

        def _resize_pair(pair: tf.Tensor) -> tf.Tensor:
            shape = tf.shape(pair)
            h = shape[1]
            w = shape[2]
            needs_resize = tf.logical_or(tf.not_equal(h, image_size), tf.not_equal(w, image_size))

            def _do_resize():
                resized = tf.image.resize(pair, [image_size, image_size])
                return tf.cast(resized, tf.uint8)

            pair = tf.cond(needs_resize, _do_resize, lambda: tf.cast(pair, tf.uint8))
            return tf.ensure_shape(pair, [2, image_size, image_size, 3])

        def _resize_frame(frame: tf.Tensor) -> tf.Tensor:
            shape = tf.shape(frame)
            h = shape[0]
            w = shape[1]
            needs_resize = tf.logical_or(tf.not_equal(h, image_size), tf.not_equal(w, image_size))

            def _do_resize():
                resized = tf.image.resize(frame, [image_size, image_size])
                return tf.cast(resized, tf.uint8)

            frame = tf.cond(needs_resize, _do_resize, lambda: tf.cast(frame, tf.uint8))
            return tf.ensure_shape(frame, [image_size, image_size, 3])

        def process_episode_to_pairs(episode):
            steps_ds = episode["steps"]

            frames_ds = steps_ds.map(
                lambda s: s["observation"][image_key],
                num_parallel_calls=num_parallel_calls,
            )
            frames_ds = frames_ds.map(_resize_frame, num_parallel_calls=num_parallel_calls)
            pairs_ds = tf.data.Dataset.zip((frames_ds, frames_ds.skip(offset))).map(
                lambda f_t, f_tp: tf.ensure_shape(tf.stack([f_t, f_tp], axis=0), [2, image_size, image_size, 3]),
                num_parallel_calls=num_parallel_calls,
            )

            if not return_metadata:
                if samples_per_episode > 0:
                    pairs_ds = pairs_ds.shuffle(per_episode_sample_shuffle, seed=self._tf_seed).take(
                        samples_per_episode
                    )
                return pairs_ds

            episode_id_tf = episode["episode_id"]

            if robot_key:
                robot_raw = episode["episode_metadata"][robot_key]
                robot_tf = robot_raw if robot_raw.dtype == tf.string else tf.strings.as_string(robot_raw)
            else:
                robot_tf = tf.constant("", dtype=tf.string)

            def _extract_action(step):
                if action_is_dict:
                    if not action_key:
                        raise ValueError("Config error: action_is_dict=True requires action_key")
                    a = step["action"][action_key]
                else:
                    a = step["action"]
                a = tf.cast(tf.reshape(a, [-1])[:action_dim], tf.float32)
                return tf.ensure_shape(a, [action_dim])

            def _extract_state(step):
                s = step["observation"][state_key]
                s = tf.cast(tf.reshape(s, [-1])[:state_dim], tf.float32)
                return tf.ensure_shape(s, [state_dim])

            actions_ds = steps_ds.map(_extract_action, num_parallel_calls=num_parallel_calls)
            states_ds = steps_ds.map(_extract_state, num_parallel_calls=num_parallel_calls)

            # Sum actions[t : t+offset] (length offset) for each pair index t.
            cumulative_action_ds = (
                actions_ds.window(size=offset, shift=1, drop_remainder=True)
                .flat_map(lambda w: w.batch(offset))
                .map(lambda x: tf.reduce_sum(x, axis=0), num_parallel_calls=num_parallel_calls)
            )

            if instruction_in_step:
                language_ds = steps_ds.map(
                    lambda s: _strip_null_bytes(s[instruction_key]),
                    num_parallel_calls=num_parallel_calls,
                )
            else:
                def _extract_obs_language(step):
                    instr = step["observation"][instruction_key]
                    if dataset_name in {
                        "language_table",
                        "language_table_blocktorelative_oracle_sim",
                        "language_table_blocktoblock_oracle_sim",
                        "language_table_blocktoabsolute_oracle_sim",
                    }:
                        instr = tf.strings.unicode_encode(tf.cast(instr, tf.int32), "UTF-8")
                    return _strip_null_bytes(instr)

                language_ds = (
                    steps_ds.map(_extract_obs_language, num_parallel_calls=num_parallel_calls)
                    .take(1)
                    .repeat()
                )

            pairs_enum_ds = pairs_ds.enumerate()
            zipped = tf.data.Dataset.zip((pairs_enum_ds, cumulative_action_ds, states_ds, language_ds))

            def _attach_meta(pair_enum, cumulative_action, initial_state, language):
                frame_idx, pair = pair_enum
                meta = {
                    "episode_id": episode_id_tf,
                    "frame_idx": tf.cast(frame_idx, tf.int32),
                    "offset": offset_tf,
                    "language": language,
                    "dataset_type": dataset_name_tf,
                    "dataset_name": dataset_name_tf,
                    "action": cumulative_action,
                    "initial_state": initial_state,
                    "robot": robot_tf,
                }
                return pair, meta

            out_ds = zipped.map(_attach_meta, num_parallel_calls=num_parallel_calls)
            if samples_per_episode > 0:
                out_ds = out_ds.shuffle(per_episode_sample_shuffle, seed=self._tf_seed).take(samples_per_episode)
            return out_ds

        tf_ds = ds.interleave(
            process_episode_to_pairs,
            cycle_length=num_parallel_episodes,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        # Pair-level shuffle for better mixing
        if self.pair_shuffle_buffer > 0:
            tf_ds = tf_ds.shuffle(self.pair_shuffle_buffer, seed=self._tf_seed)

        # Prefetch for performance
        prefetch_val = self.prefetch_buffer if self.prefetch_buffer > 0 else tf.data.AUTOTUNE
        return tf_ds.prefetch(prefetch_val)

    def _get_or_create_pipeline(self):
        """Get existing pipeline or create new one (lazy, persistent)."""
        if self._persistent_pipeline is None:
            self._persistent_pipeline = self._create_tf_pipeline()
        return self._persistent_pipeline

    def cleanup(self, *, ignore_errors: bool = False) -> None:
        """
        Explicitly release TensorFlow resources.

        Call this when you are done with the dataset and want to free memory
        before the object is garbage collected.

        This method is idempotent - safe to call multiple times.

        """
        # Idempotency check: skip if already cleaned up
        if self._cleaned_up:
            return

        # Mark as cleaned up FIRST to prevent re-entry during exception handling
        self._cleaned_up = True

        def _run(fn):
            if not ignore_errors:
                fn()
                return
            try:
                fn()
            except Exception:
                logger.debug("Cleanup step failed", exc_info=True)

        _run(lambda: setattr(self, "_persistent_pipeline", None))
        _run(lambda: setattr(self, "_pipeline_iterator", None))
        _run(lambda: setattr(self, "_builder", None))

        import gc

        _run(gc.collect)

        import tensorflow as tf

        _run(tf.keras.backend.clear_session)

        logger.debug(f"OXEFramePairDataset cleanup completed for {self.config.name}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup(ignore_errors=True)
        except Exception:
            pass

    def reset_iterator(self):
        """Force reset the underlying iterator (useful for cyclic sampling)."""
        self._pipeline_iterator = None
        if self._persistent_pipeline is not None:
            # Just getting a new iterator from the existing pipeline works
            # because TF datasets are re-iterable
            self._pipeline_iterator = iter(self._persistent_pipeline)

    def _get_or_create_iterator(self):
        """Get existing iterator or create new one.

        When persistent_iterator=True (default), reuses the same iterator across
        epochs to avoid refilling the shuffle buffer from GCS each time.
        """
        tf_ds = self._get_or_create_pipeline()

        if self.persistent_iterator:
            if self._pipeline_iterator is None:
                self._pipeline_iterator = iter(tf_ds)
            return self._pipeline_iterator
        else:
            # Create fresh iterator each epoch (triggers shuffle buffer refill)
            return iter(tf_ds)

    def __iter__(self) -> Iterator:
        """Iterate over frame pairs."""
        # Get or create iterator (persistent by default to avoid shuffle buffer refill)
        tf_iter = self._get_or_create_iterator()

        # Helper to decode TF string tensors efficiently
        def _decode_str(val) -> str:
            """Decode bytes/str value, handling null bytes."""
            if isinstance(val, bytes):
                return val.decode("utf-8").rstrip("\x00")
            return str(val) if val else ""

        # Cache known values to avoid per-sample .numpy() calls
        dataset_name = self.config.name
        offset = self.offset

        for item in tf_iter:
            if self.return_metadata:
                pair_tf, meta_tf = item
                pair_np = pair_tf.numpy()
                pair_pt = (
                    torch.from_numpy(pair_np).permute(3, 0, 1, 2).float() / 255.0
                )

                episode_id = _decode_str(meta_tf["episode_id"].numpy())
                language = _decode_str(meta_tf["language"].numpy())
                robot = _decode_str(meta_tf["robot"].numpy())

                meta = {
                    "episode_id": episode_id,
                    "frame_idx": int(meta_tf["frame_idx"].numpy()),
                    "offset": offset,
                    "language": language,
                    "dataset_type": dataset_name,
                    "dataset_name": dataset_name,
                    "action": meta_tf["action"].numpy(),
                    "initial_state": meta_tf["initial_state"].numpy(),
                    "robot": robot,
                }
                yield {"frames": pair_pt, **meta}
            else:
                pair_np = item.numpy()
                yield torch.from_numpy(pair_np).permute(3, 0, 1, 2).float() / 255.0


class MultiOXEFramePairDataset(IterableDataset):
    """
    PyTorch IterableDataset that interleaves frame pairs from multiple OXE datasets.

    Alternates between datasets based on weights to ensure good mixing.
    Each dataset can have different configs (action format, image size, etc.)

    Args:
        datasets: List of dataset configs, each with:
            - name: Dataset name (e.g., "bridge", "language_table")
            - train_split: TFDS split string for training (REQUIRED)
            - val_split: TFDS split string for validation (REQUIRED)
            - weight: Sampling weight (default proportionate)
            - offset: Frame offset for pairs (REQUIRED)
            - size: Precomputed dataset size (REQUIRED)
        image_size: Target image size (shared)
        prefetch_buffer: tf.data prefetch buffer size
        episode_shuffle_buffer: Shuffle buffer for episodes (REQUIRED)
        pair_shuffle_buffer: Shuffle buffer for pairs/samples (REQUIRED)
        return_metadata: If True, return dict with metadata
        is_train: If True, use train_split; else use val_split
        num_parallel_episodes: Number of episodes to process in parallel (default: 4)
    """

    def __init__(
        self,
        datasets: list,
        prefetch_buffer: int,
        episode_shuffle_buffer: int,
        pair_shuffle_buffer: int,
        image_size: int = 256,
        return_metadata: bool = True,
        is_train: bool = True,
        persistent_iterator: bool = True,  # Keep iterators alive to avoid shuffle buffer refill
        samples_per_episode: int = 0,
        seed: Optional[int] = None,
        num_parallel_episodes: int = 4,
    ):
        super().__init__()

        self.dataset_configs = datasets
        self.image_size = image_size
        self.prefetch_buffer = prefetch_buffer
        self.return_metadata = return_metadata
        self.is_train = is_train
        self.persistent_iterator = persistent_iterator
        self.samples_per_episode = samples_per_episode
        self.seed = seed
        self.num_parallel_episodes = num_parallel_episodes
        self.episode_shuffle_buffer = episode_shuffle_buffer
        self.pair_shuffle_buffer = pair_shuffle_buffer

        # Will be populated lazily
        self._datasets = None
        self._weights = None

    def _init_datasets(self):
        """Initialize individual datasets lazily."""
        if self._datasets is not None:
            return

        self._datasets = []

        # Compute per-dataset shuffle buffers
        n_datasets = len(self.dataset_configs)
        episode_buffer_per_ds = max(10, self.episode_shuffle_buffer // n_datasets) if self.is_train else 0
        pair_buffer_per_ds = max(10, self.pair_shuffle_buffer // n_datasets) if self.is_train else 0

        for cfg in self.dataset_configs:
            # Require train_split and val_split explicitly
            if "train_split" not in cfg:
                raise ValueError(
                    f"Dataset '{cfg['name']}' is missing required 'train_split' field."
                )
            if "val_split" not in cfg:
                raise ValueError(
                    f"Dataset '{cfg['name']}' is missing required 'val_split' field."
                )

            # Get split based on train/val mode
            split = cfg["train_split"] if self.is_train else cfg["val_split"]

            # Offset is required for each dataset
            if "offset" not in cfg:
                raise ValueError(
                    f"Dataset '{cfg['name']}' is missing required 'offset' field. "
                    f"Please specify offset in config/data/oxe/{cfg['name']}.yaml"
                )

            if "size" not in cfg or cfg["size"] is None:
                raise ValueError(
                    f"Dataset '{cfg['name']}' is missing required 'size' field. "
                    f"Please specify size in config/data/oxe/{cfg['name']}.yaml"
                )
            precomputed_size = int(cfg["size"])

            ds = OXEFramePairDataset(
                dataset_name=cfg["name"],
                split=split,
                image_size=self.image_size,
                offset=cfg["offset"],
                prefetch_buffer=self.prefetch_buffer,
                return_metadata=self.return_metadata,
                persistent_iterator=self.persistent_iterator,
                samples_per_episode=cfg.get("samples_per_episode", self.samples_per_episode),
                seed=self._get_dataset_seed(cfg),
                precomputed_size=precomputed_size,
                episode_prefetch_buffer=cfg.get("episode_prefetch_buffer", 1),
                num_parallel_episodes=cfg.get("num_parallel_episodes", self.num_parallel_episodes),
                episode_shuffle_buffer=episode_buffer_per_ds,
                pair_shuffle_buffer=pair_buffer_per_ds,
            )
            self._datasets.append(ds)

        sizes = [int(cfg["size"]) for cfg in self.dataset_configs]
        self._weights = self._compute_weights(self.dataset_configs, sizes)

    def _get_dataset_seed(self, cfg: dict) -> Optional[int]:
        # If an explicit per-dataset seed is provided, use it as-is.
        if "seed" in cfg and cfg["seed"] is not None:
            return int(cfg["seed"])
        if self.seed is None:
            return None
        # Derive deterministic per-dataset seed from global seed + dataset name.
        name_hash = zlib.crc32(str(cfg.get("name", "")).encode("utf-8")) & 0x7FFFFFFF
        return (int(self.seed) + int(name_hash)) & 0x7FFFFFFF

    def _compute_weights(
        self, datasets_config: list, sizes: list[int]
    ) -> list[float]:
        """
        Compute normalized weights using mixed absolute mode.

        Weight specification options:
        - Omitted or None: Proportionate weighting (share remaining weight by size)
        - "proportionate": Explicit proportionate weighting
        - Numeric (int/float): Absolute sampling ratio (e.g., 0.3 = 30% of samples)

        Algorithm (mixed absolute mode):
        1. Identify explicit numeric weights vs proportionate
        2. Sum explicit weights (should be < 1.0; warns if >= 1.0)
        3. Remaining pool = 1.0 - sum(explicit)
        4. Proportionate datasets share remaining pool by relative size
        5. Final weights are normalized to sum to 1.0 for sampling correctness

        Note: If explicit weights sum to > 1.0, they are still normalized to ensure
        the sampling loop works correctly. A warning is logged in this case.

        Args:
            datasets_config: List of dataset config dicts
            sizes: List of estimated dataset sizes (from len(ds))

        Returns:
            List of normalized weights summing to 1.0
        """
        explicit_weights: dict[int, float] = {}  # index -> weight
        proportionate_indices: list[int] = []

        for i, cfg in enumerate(datasets_config):
            weight_spec = cfg.get("weight")
            if weight_spec is None or weight_spec == "proportionate":
                proportionate_indices.append(i)
            elif isinstance(weight_spec, (int, float)):
                explicit_weights[i] = float(weight_spec)
            else:
                raise ValueError(
                    f"Invalid weight specification for dataset '{cfg.get('name', i)}': "
                    f"{weight_spec!r}. Must be numeric, 'proportionate', or omitted."
                )

        # Validate explicit weights sum
        explicit_sum = sum(explicit_weights.values())
        if explicit_sum >= 1.0:
            logger.warning(
                f"Explicit weights sum to {explicit_sum:.3f} >= 1.0. "
                "Proportionate datasets will have zero weight."
            )

        # Calculate remaining weight pool for proportionate datasets
        remaining = max(0.0, 1.0 - explicit_sum)

        # Get sizes for proportionate datasets
        prop_sizes = [sizes[i] for i in proportionate_indices]
        total_prop_size = sum(prop_sizes)

        # Build final weights
        final_weights = [0.0] * len(datasets_config)

        # Assign explicit weights
        for i, w in explicit_weights.items():
            final_weights[i] = w

        # Distribute remaining weight to proportionate datasets by size
        if total_prop_size > 0 and remaining > 0:
            for i in proportionate_indices:
                final_weights[i] = remaining * (sizes[i] / total_prop_size)
        elif proportionate_indices and remaining > 0:
            # All proportionate datasets have zero size - equal split
            for i in proportionate_indices:
                final_weights[i] = remaining / len(proportionate_indices)

        # Normalize to sum to 1.0 (required for sampling loop correctness)
        # This handles cases where explicit weights sum to > 1.0
        total_weight = sum(final_weights)
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            final_weights = [w / total_weight for w in final_weights]

        # Log weight breakdown for debugging
        logger.info("Dataset weights computed (mixed absolute mode):")
        for i, cfg in enumerate(datasets_config):
            name = cfg.get("name", f"dataset_{i}")
            weight_spec = cfg.get("weight")
            mode = "explicit" if i in explicit_weights else "proportionate"
            logger.info(
                f"  {name}: {final_weights[i]:.3f} ({mode}, size={sizes[i]:,})"
            )

        return final_weights

    def __len__(self):
        """Approximate total length across all datasets."""
        self._init_datasets()
        return sum(len(ds) for ds in self._datasets)

    def cleanup(self, *, ignore_errors: bool = False) -> None:
        """
        Explicitly release TensorFlow resources from all underlying datasets.

        Call this when you are done with the dataset and want to free memory.
        This method is idempotent - safe to call multiple times.

        """
        datasets_to_cleanup = self._datasets
        self._datasets = None  # Clear reference first to prevent re-entry
        self._weights = None

        if datasets_to_cleanup is not None:
            for ds in datasets_to_cleanup:
                if not ignore_errors:
                    ds.cleanup()
                else:
                    try:
                        ds.cleanup(ignore_errors=True)
                    except Exception:
                        logger.debug("Dataset cleanup failed", exc_info=True)

        logger.debug("MultiOXEFramePairDataset cleanup completed")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup(ignore_errors=True)
        except Exception:
            pass

    def __iter__(self):
        """Interleave samples from all datasets based on weights with CYCLIC sampling."""
        import random as _random

        self._init_datasets()

        # Create iterators for each dataset
        # NOTE: If iterators are already active (persistent), this continues them.
        iterators = [iter(ds) for ds in self._datasets]
        rng = _random.Random(self.seed) if self.seed is not None else _random.Random()
        
        # We loop for exactly __len__ steps to define the epoch length.
        # This prevents distribution shift: even if a dataset runs out,
        # we restart it so the mixing weights remain constant.
        steps_total = len(self)
        
        for _ in range(steps_total):
            # Choose dataset based on weights
            # We select from ALL datasets, not just "active" ones
            r = rng.random()
            cumsum = 0
            dataset_idx = 0
            for i, w in enumerate(self._weights):
                cumsum += w
                if r <= cumsum:
                    dataset_idx = i
                    break
            
            # Fetch item from chosen dataset, handling exhaustion by cycling
            try:
                item = next(iterators[dataset_idx])
                yield item
            except StopIteration:
                # Dataset exhausted: restart it immediately (Cycle)
                # 1. Force reset the underlying iterator in the dataset object
                self._datasets[dataset_idx].reset_iterator()
                # 2. Get the new iterator
                iterators[dataset_idx] = iter(self._datasets[dataset_idx])
                # 3. Retry fetching (assumes dataset is not empty)
                try:
                    item = next(iterators[dataset_idx])
                    yield item
                except StopIteration as e:
                    dataset_name = self.dataset_configs[dataset_idx].get(
                        "name", f"dataset_{dataset_idx}"
                    )
                    raise RuntimeError(
                        f"Dataset {dataset_name!r} is empty; cannot cycle iterator."
                    ) from e


def get_oxe_dataset_info(dataset_name: str = "language_table") -> Dict[str, Any]:
    """Get information about an OXE dataset."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    if dataset_name not in OXE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = OXE_DATASETS[dataset_name]
    builder = tfds.builder_from_directory(config.gcs_path)

    return {
        "name": config.name,
        "gcs_path": config.gcs_path,
        "splits": {
            name: split.num_examples for name, split in builder.info.splits.items()
        },
        "image_shape": config.image_shape,
        "control_frequency_hz": config.control_frequency_hz,
        "features": str(builder.info.features),
    }
