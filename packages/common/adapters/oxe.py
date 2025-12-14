"""
Open X-Embodiment (OXE) adapter for streaming RLDS datasets from GCS.

This adapter is different from file-based adapters (YouTube, Bridge) because:
1. Data streams from Google Cloud Storage, not local files
2. Uses tf.data pipelines for efficient prefetching
3. Returns tensors directly, not file paths

Currently supports:
- language_table: gs://gresearch/robotics/language_table/0.0.1
- language_table_blocktorelative_oracle_sim (200k episodes, longer sequences)
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


@dataclass
class OXEDatasetConfig:
    """Configuration for an OXE dataset."""

    name: str
    gcs_path: str
    image_key: str = "rgb"  # Key in observation dict for images
    instruction_key: str = "instruction"  # Key for language instruction
    image_shape: Tuple[int, int, int] = (360, 640, 3)  # H, W, C
    control_frequency_hz: float = 10.0  # For time calculations
    action_dim: int = 2  # Dimensionality of action space (2 for language_table)


# Registry of supported OXE datasets
OXE_DATASETS = {
    "language_table": OXEDatasetConfig(
        name="language_table",
        gcs_path="gs://gresearch/robotics/language_table/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
    ),
    # Oracle sim datasets (scripted agent, longer episodes, cleaner data)
    "language_table_blocktorelative_oracle_sim": OXEDatasetConfig(
        name="language_table_blocktorelative_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_blocktorelative_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
    ),
    "language_table_blocktoblock_oracle_sim": OXEDatasetConfig(
        name="language_table_blocktoblock_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_blocktoblock_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
    ),
    "language_table_blocktoabsolute_oracle_sim": OXEDatasetConfig(
        name="language_table_blocktoabsolute_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
    ),
    "language_table_separate_oracle_sim": OXEDatasetConfig(
        name="language_table_separate_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_separate_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
    ),
}


class OXEFramePairDataset(IterableDataset):
    """
    PyTorch IterableDataset that streams frame pairs from OXE datasets on GCS.

    Uses tf.data internally for efficient streaming and prefetching,
    converts to PyTorch tensors on iteration.

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
        dataset_name: str = "language_table",
        split: str = "train",
        offset: int = 5,
        image_size: int = 256,
        shuffle_buffer: int = 1000,
        prefetch_buffer: int = 2,
        num_parallel_calls: Optional[int] = None,  # None = AUTOTUNE
        return_metadata: bool = False,
        gcs_path: Optional[str] = None,
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
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.num_parallel_calls = num_parallel_calls
        self.return_metadata = return_metadata

        # Lazy initialization of tf.data pipeline
        self._builder = None
        self._num_episodes = None

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
        """Approximate length based on episodes * avg pairs per episode."""
        self._init_tfds()
        # Estimate: avg 25 steps per episode for oracle_sim, offset=5 -> ~20 pairs
        avg_pairs_per_episode = max(1, 25 - self.offset)
        return self._num_episodes * avg_pairs_per_episode

    def _create_tf_pipeline(self):
        """Create tf.data pipeline for streaming frame pairs."""
        import tensorflow as tf

        self._init_tfds()
        ds = self._builder.as_dataset(split=self.split)

        # Shuffle episodes if enabled
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(min(self.shuffle_buffer, 1000))

        image_key = self.config.image_key
        instruction_key = self.config.instruction_key
        offset = self.offset
        image_size = self.image_size
        return_metadata = self.return_metadata
        dataset_name = self.config.name
        action_dim = self.config.action_dim

        def episode_to_pairs_generator():
            """Generator that yields frame pairs from episodes."""
            for episode in ds:
                episode_id = episode.get("episode_id", b"unknown")
                if isinstance(episode_id, tf.Tensor):
                    episode_id = episode_id.numpy()
                if isinstance(episode_id, bytes):
                    episode_id = episode_id.decode("utf-8")

                steps = list(episode["steps"])
                n_steps = len(steps)

                if n_steps < offset + 1:
                    continue

                # Get instruction (same for all steps in episode)
                instruction = ""
                if return_metadata and instruction_key in steps[0]["observation"]:
                    instr_tensor = steps[0]["observation"][instruction_key]
                    try:
                        instruction = (
                            tf.strings.unicode_encode(
                                tf.cast(instr_tensor, tf.int32), "UTF-8"
                            )
                            .numpy()
                            .decode("utf-8")
                            .rstrip("\x00")
                        )
                    except Exception:
                        instruction = ""

                # Extract all frames and resize in batch for efficiency
                frames_raw = [s["observation"][image_key] for s in steps]
                frames = tf.stack(frames_raw)  # [T, H, W, C]

                # Resize if needed (batch resize is faster)
                if frames.shape[1] != image_size or frames.shape[2] != image_size:
                    frames = tf.image.resize(frames, [image_size, image_size])
                    frames = tf.cast(frames, tf.uint8)

                frames = frames.numpy()

                # Extract actions if available (for visualization)
                actions = None
                if return_metadata and "action" in steps[0]:
                    try:
                        actions = np.stack([s["action"].numpy() for s in steps])
                    except Exception:
                        actions = None

                # Generate pairs
                for t in range(n_steps - offset):
                    # Stack: [2, H, W, C]
                    pair = np.stack([frames[t], frames[t + offset]], axis=0)

                    if return_metadata:
                        # Compute accumulated action (sum of actions from t to t+offset)
                        # This represents the total movement between the two frames
                        if actions is not None:
                            cumulative_action = actions[t : t + offset].sum(axis=0)
                        else:
                            cumulative_action = np.zeros(action_dim, dtype=np.float32)

                        meta = {
                            "episode_id": episode_id,
                            "frame_idx": t,
                            "offset": offset,
                            "instruction": instruction,
                            "dataset_type": "oxe",
                            "dataset_name": dataset_name,
                            "action": cumulative_action.astype(np.float32),
                        }
                        yield pair, meta
                    else:
                        yield pair

        # Create tf.data.Dataset from generator
        if return_metadata:
            output_signature = (
                tf.TensorSpec(
                    shape=(2, image_size, image_size, 3), dtype=tf.uint8
                ),
                {
                    "episode_id": tf.TensorSpec(shape=(), dtype=tf.string),
                    "frame_idx": tf.TensorSpec(shape=(), dtype=tf.int32),
                    "offset": tf.TensorSpec(shape=(), dtype=tf.int32),
                    "instruction": tf.TensorSpec(shape=(), dtype=tf.string),
                    "dataset_type": tf.TensorSpec(shape=(), dtype=tf.string),
                    "dataset_name": tf.TensorSpec(shape=(), dtype=tf.string),
                    # Action: cumulative action between frame pairs (e.g., 2D for language_table)
                    "action": tf.TensorSpec(shape=(action_dim,), dtype=tf.float32),
                },
            )
        else:
            output_signature = tf.TensorSpec(
                shape=(2, image_size, image_size, 3), dtype=tf.uint8
            )

        tf_ds = tf.data.Dataset.from_generator(
            episode_to_pairs_generator, output_signature=output_signature
        )

        # Apply shuffle if enabled (at pair level for better mixing)
        if self.shuffle_buffer > 0:
            tf_ds = tf_ds.shuffle(self.shuffle_buffer)

        # Prefetch for performance
        tf_ds = tf_ds.prefetch(tf.data.AUTOTUNE)

        return tf_ds

    def __iter__(self) -> Iterator:
        """Iterate over frame pairs."""
        import tensorflow as tf

        tf_ds = self._create_tf_pipeline()

        # Iterate and convert to PyTorch
        for item in tf_ds:
            try:
                if self.return_metadata:
                    pair_tf, meta_tf = item
                    pair_np = pair_tf.numpy()
                    # Convert [2, H, W, C] -> [C, 2, H, W] and normalize
                    pair_pt = (
                        torch.from_numpy(pair_np).permute(3, 0, 1, 2).float() / 255.0
                    )
                    # Convert metadata
                    meta = {
                        "episode_id": (
                            meta_tf["episode_id"].numpy().decode("utf-8")
                            if isinstance(meta_tf["episode_id"].numpy(), bytes)
                            else str(meta_tf["episode_id"].numpy())
                        ),
                        "frame_idx": int(meta_tf["frame_idx"].numpy()),
                        "offset": int(meta_tf["offset"].numpy()),
                        "instruction": (
                            meta_tf["instruction"].numpy().decode("utf-8").rstrip("\x00")
                            if isinstance(meta_tf["instruction"].numpy(), bytes)
                            else ""
                        ),
                        "dataset_type": (
                            meta_tf["dataset_type"].numpy().decode("utf-8")
                            if isinstance(meta_tf["dataset_type"].numpy(), bytes)
                            else "oxe"
                        ),
                        "dataset_name": (
                            meta_tf["dataset_name"].numpy().decode("utf-8")
                            if isinstance(meta_tf["dataset_name"].numpy(), bytes)
                            else self.config.name
                        ),
                        # Cumulative action between frames (for visualization)
                        "action": meta_tf["action"].numpy().tolist(),
                    }
                    yield {"frames": pair_pt, **meta}
                else:
                    pair_np = item.numpy()
                    # Convert [2, H, W, C] -> [C, 2, H, W] and normalize
                    pair_pt = (
                        torch.from_numpy(pair_np).permute(3, 0, 1, 2).float() / 255.0
                    )
                    yield pair_pt
            except tf.errors.OutOfRangeError:
                break
            except Exception as e:
                # Log but continue on errors (some episodes may be malformed)
                print(f"Warning: Error processing item: {e}")
                continue


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
