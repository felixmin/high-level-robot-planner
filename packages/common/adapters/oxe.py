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
    ),
    "language_table_blocktoblock_oracle_sim": OXEDatasetConfig(
        name="language_table_blocktoblock_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_blocktoblock_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
    ),
    "language_table_blocktoabsolute_oracle_sim": OXEDatasetConfig(
        name="language_table_blocktoabsolute_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
    ),
    "language_table_separate_oracle_sim": OXEDatasetConfig(
        name="language_table_separate_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_separate_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
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
        # Persistent pipeline - created once, reused across epochs
        self._persistent_pipeline = None
        self._pipeline_iterator = None
        self._epoch_count = 0
        # Idempotency flag for cleanup
        self._cleaned_up = False

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
        state_key = getattr(self.config, "state_key", "effector_translation")
        offset = self.offset
        image_size = self.image_size
        return_metadata = self.return_metadata
        dataset_name = self.config.name
        action_dim = self.config.action_dim
        state_dim = getattr(self.config, "state_dim", 2)
        action_key = getattr(self.config, "action_key", None)
        action_is_dict = getattr(self.config, "action_is_dict", False)
        instruction_in_step = getattr(self.config, "instruction_in_step", False)
        robot_key = getattr(self.config, "robot_key", None)

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

                # Extract robot type from episode metadata if available
                robot_type = ""
                if return_metadata and robot_key and "episode_metadata" in episode:
                    try:
                        robot_tensor = episode["episode_metadata"].get(robot_key)
                        if robot_tensor is not None:
                            robot_val = robot_tensor.numpy()
                            if isinstance(robot_val, bytes):
                                robot_type = robot_val.decode("utf-8")
                            else:
                                robot_type = str(robot_val)
                    except Exception:
                        robot_type = ""

                # Get instruction (same for all steps in episode)
                instruction = ""
                if return_metadata:
                    try:
                        if instruction_in_step:
                            # Instruction at step level (e.g., RoboNet)
                            instr_tensor = steps[0].get(instruction_key)
                            if instr_tensor is not None:
                                instr_val = instr_tensor.numpy()
                                if isinstance(instr_val, bytes):
                                    instruction = instr_val.decode("utf-8")
                                else:
                                    instruction = str(instr_val)
                        elif instruction_key in steps[0]["observation"]:
                            # Instruction in observation dict (Bridge, RT-1, language_table)
                            instr_tensor = steps[0]["observation"][instruction_key]
                            # Check if it is a string tensor (Bridge) or encoded ints (language_table)
                            if instr_tensor.dtype == tf.string:
                                # String tensor (e.g., Bridge natural_language_instruction)
                                instruction = instr_tensor.numpy().decode("utf-8")
                            else:
                                # Encoded int tensor (e.g., language_table)
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
                        if action_is_dict and action_key:
                            # Dict-based actions (e.g., Bridge has world_vector, rotation_delta)
                            actions = np.stack([s["action"][action_key].numpy() for s in steps])
                        else:
                            # Flat action array (e.g., language_table, RoboNet)
                            # Only take first action_dim dimensions
                            actions = np.stack([s["action"].numpy()[:action_dim] for s in steps])
                    except Exception:
                        actions = None
                        
                # Extract states if available
                states = None
                if return_metadata and state_key in steps[0]["observation"]:
                    try:
                        # Only take first N dims (e.g. 2 for 2D plot)
                        states = np.stack([s["observation"][state_key].numpy()[:state_dim] for s in steps])
                    except Exception:
                        states = None

                # Generate pairs
                for t in range(n_steps - offset):
                    # Stack: [2, H, W, C]
                    pair = np.stack([frames[t], frames[t + offset]], axis=0)

                    if return_metadata:
                        # Compute accumulated action (sum of actions from t to t+offset)
                        if actions is not None:
                            cumulative_action = actions[t : t + offset].sum(axis=0)
                        else:
                            cumulative_action = np.zeros(action_dim, dtype=np.float32)
                            
                        # Get initial state at start of pair
                        if states is not None:
                            initial_state = states[t]
                        else:
                            initial_state = np.zeros(state_dim, dtype=np.float32)

                        meta = {
                            "episode_id": episode_id,
                            "frame_idx": t,
                            "offset": offset,
                            "language": instruction,  # Standardized key (was: instruction)
                            "dataset_type": dataset_name,  # Use actual dataset name for filtering
                            "dataset_name": dataset_name,
                            "action": cumulative_action.astype(np.float32),
                            "initial_state": initial_state.astype(np.float32),
                            "robot": robot_type,  # Robot type for filtering (e.g., RoboNet)
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
                    "language": tf.TensorSpec(shape=(), dtype=tf.string),  # Standardized key
                    "dataset_type": tf.TensorSpec(shape=(), dtype=tf.string),
                    "dataset_name": tf.TensorSpec(shape=(), dtype=tf.string),
                    # Action: cumulative action between frame pairs (e.g., 2D for language_table)
                    "action": tf.TensorSpec(shape=(action_dim,), dtype=tf.float32),
                    # State: initial state
                    "initial_state": tf.TensorSpec(shape=(state_dim,), dtype=tf.float32),
                    # Robot type (for multi-robot datasets like RoboNet)
                    "robot": tf.TensorSpec(shape=(), dtype=tf.string),
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

    def _get_or_create_pipeline(self):
        """Get existing pipeline or create new one (lazy, persistent)."""
        if self._persistent_pipeline is None:
            self._persistent_pipeline = self._create_tf_pipeline()
        return self._persistent_pipeline

    def cleanup(self) -> None:
        """
        Explicitly release TensorFlow resources.

        Call this when you are done with the dataset and want to free memory
        before the object is garbage collected.

        This method is idempotent - safe to call multiple times.

        Uses extremely permissive error handling to ensure clean process exit
        even if TensorFlow/GCS is in a bad state (e.g., killed mid-step).
        """
        # Idempotency check: skip if already cleaned up
        if self._cleaned_up:
            return

        # Mark as cleaned up FIRST to prevent re-entry during exception handling
        self._cleaned_up = True

        # Use BaseException to catch SystemExit, KeyboardInterrupt, etc.
        # This ensures process can exit cleanly regardless of TF/GCS state
        try:
            if self._persistent_pipeline is not None:
                del self._persistent_pipeline
        except BaseException:
            pass
        finally:
            self._persistent_pipeline = None

        try:
            if self._pipeline_iterator is not None:
                del self._pipeline_iterator
        except BaseException:
            pass
        finally:
            self._pipeline_iterator = None

        try:
            if self._builder is not None:
                del self._builder
        except BaseException:
            pass
        finally:
            self._builder = None

        # Garbage collection - skip if interpreter is shutting down
        try:
            import gc
            gc.collect()
        except BaseException:
            pass

        # Try to clear TensorFlow caches (may fail during shutdown)
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except BaseException:
            pass

        try:
            logger.debug(f"OXEFramePairDataset cleanup completed for {self.config.name}")
        except BaseException:
            pass  # Logging may fail during interpreter shutdown

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass

    def __iter__(self) -> Iterator:
        """Iterate over frame pairs."""
        import tensorflow as tf

        self._epoch_count += 1

        # Get or create the persistent pipeline
        # The pipeline uses repeat() internally, so we just create a new iterator
        tf_ds = self._get_or_create_pipeline()

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
                    # Convert metadata - use standardized keys
                    dataset_name = (
                        meta_tf["dataset_name"].numpy().decode("utf-8")
                        if isinstance(meta_tf["dataset_name"].numpy(), bytes)
                        else self.config.name
                    )
                    # Extract robot type
                    robot = ""
                    if "robot" in meta_tf:
                        robot_val = meta_tf["robot"].numpy()
                        if isinstance(robot_val, bytes):
                            robot = robot_val.decode("utf-8")
                        else:
                            robot = str(robot_val) if robot_val else ""

                    meta = {
                        "episode_id": (
                            meta_tf["episode_id"].numpy().decode("utf-8")
                            if isinstance(meta_tf["episode_id"].numpy(), bytes)
                            else str(meta_tf["episode_id"].numpy())
                        ),
                        "frame_idx": int(meta_tf["frame_idx"].numpy()),
                        "offset": int(meta_tf["offset"].numpy()),
                        # Standardized key (was: instruction)
                        "language": (
                            meta_tf["language"].numpy().decode("utf-8").rstrip("\x00")
                            if isinstance(meta_tf["language"].numpy(), bytes)
                            else ""
                        ),
                        # Use dataset_name for both (enables bucket filtering by dataset)
                        "dataset_type": dataset_name,
                        "dataset_name": dataset_name,
                        # Cumulative action between frames (for visualization)
                        "action": meta_tf["action"].numpy().tolist(),
                        # Initial state (for visualization)
                        "initial_state": meta_tf["initial_state"].numpy().tolist(),
                        # Robot type (for multi-robot datasets)
                        "robot": robot,
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
                logger.warning(f"Error processing OXE item: {e}")
                continue


class MultiOXEFramePairDataset(IterableDataset):
    """
    PyTorch IterableDataset that interleaves frame pairs from multiple OXE datasets.

    Alternates between datasets based on weights to ensure good mixing.
    Each dataset can have different configs (action format, image size, etc.)

    Args:
        datasets: List of dataset configs, each with:
            - name: Dataset name (e.g., "bridge", "language_table")
            - train_split / val_split: TFDS split strings
            - weight: Sampling weight (default 1.0)
            - offset: Optional per-dataset offset override
        image_size: Target image size (shared)
        offset: Default frame offset for pairs
        shuffle_buffer: Size of shuffle buffer (split across datasets)
        prefetch_buffer: tf.data prefetch buffer size
        return_metadata: If True, return dict with metadata
        is_train: If True, use train_split; else use val_split
    """

    def __init__(
        self,
        datasets: list,
        image_size: int = 256,
        offset: int = 5,
        shuffle_buffer: int = 200,
        prefetch_buffer: int = 2,
        return_metadata: bool = True,
        is_train: bool = True,
    ):
        super().__init__()

        self.dataset_configs = datasets
        self.image_size = image_size
        self.default_offset = offset
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.return_metadata = return_metadata
        self.is_train = is_train

        # Will be populated lazily
        self._datasets = None
        self._weights = None

    def _init_datasets(self):
        """Initialize individual datasets lazily."""
        if self._datasets is not None:
            return

        self._datasets = []
        weights = []

        # Split shuffle buffer across datasets
        buffer_per_dataset = max(10, self.shuffle_buffer // len(self.dataset_configs))

        for cfg in self.dataset_configs:
            # Get split based on train/val mode
            if self.is_train:
                split = cfg.get("train_split", cfg.get("split", "train[:90%]"))
            else:
                split = cfg.get("val_split", "train[90%:]")

            ds = OXEFramePairDataset(
                dataset_name=cfg["name"],
                split=split,
                image_size=self.image_size,
                offset=cfg.get("offset", self.default_offset),
                shuffle_buffer=buffer_per_dataset if self.is_train else 0,
                prefetch_buffer=self.prefetch_buffer,
                return_metadata=self.return_metadata,
            )
            self._datasets.append(ds)
            weights.append(cfg.get("weight", 1.0))

        # Normalize weights
        total_weight = sum(weights)
        self._weights = [w / total_weight for w in weights]

    def __len__(self):
        """Approximate total length across all datasets."""
        self._init_datasets()
        return sum(len(ds) for ds in self._datasets)

    def cleanup(self) -> None:
        """
        Explicitly release TensorFlow resources from all underlying datasets.

        Call this when you are done with the dataset and want to free memory.
        This method is idempotent - safe to call multiple times.

        Uses extremely permissive error handling to ensure clean process exit
        even if TensorFlow/GCS is in a bad state.
        """
        datasets_to_cleanup = self._datasets
        self._datasets = None  # Clear reference first to prevent re-entry
        self._weights = None

        if datasets_to_cleanup is not None:
            for ds in datasets_to_cleanup:
                try:
                    ds.cleanup()
                except BaseException:
                    pass  # Ignore all errors during shutdown

        try:
            logger.debug("MultiOXEFramePairDataset cleanup completed")
        except BaseException:
            pass  # Logging may fail during interpreter shutdown

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass

    def __iter__(self):
        """Interleave samples from all datasets based on weights."""
        import random

        self._init_datasets()

        # Create iterators for each dataset
        iterators = [iter(ds) for ds in self._datasets]
        active = list(range(len(iterators)))  # Indices of active iterators

        while active:
            # Choose dataset based on weights (only from active ones)
            active_weights = [self._weights[i] for i in active]
            total = sum(active_weights)
            normalized = [w / total for w in active_weights]

            # Weighted random selection
            r = random.random()
            cumsum = 0
            chosen_idx = 0
            for i, w in enumerate(normalized):
                cumsum += w
                if r <= cumsum:
                    chosen_idx = i
                    break

            dataset_idx = active[chosen_idx]

            try:
                item = next(iterators[dataset_idx])
                yield item
            except StopIteration:
                # This dataset is exhausted, remove from active
                active.remove(dataset_idx)


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
