import logging
import math

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

# TensorFlow is used only for data input pipelines. When training with PyTorch on GPU,
# letting TF see GPUs can trigger expensive device initialization/JIT compilation and
# can contend for GPU memory. We disable TF GPU visibility once per process.
_TF_GPU_DISABLED = False
_TF_GPU_DISABLE_ERROR_LOGGED = False


def _import_tensorflow_cpu_only():
    import os
    import sys

    global _TF_GPU_DISABLED
    global _TF_GPU_DISABLE_ERROR_LOGGED

    tf = sys.modules.get("tensorflow")
    if tf is not None:
        if not _TF_GPU_DISABLED:
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception as e:
                if not _TF_GPU_DISABLE_ERROR_LOGGED:
                    logger.info(f"TensorFlow GPU disable failed (tensorflow already imported): {e}")
                    _TF_GPU_DISABLE_ERROR_LOGGED = True
                pass
            _TF_GPU_DISABLED = True
        return tf

    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import tensorflow as tf  # noqa: F401
    finally:
        if prev_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception as e:
        if not _TF_GPU_DISABLE_ERROR_LOGGED:
            logger.info(f"TensorFlow GPU disable failed (after import): {e}")
            _TF_GPU_DISABLE_ERROR_LOGGED = True
        pass
    _TF_GPU_DISABLED = True
    return tf

from dataclasses import dataclass
from pathlib import Path
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
    state_key: Optional[str] = "effector_translation"  # Key for robot state (e.g. gripper pos)
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
    # Some datasets do not expose a useful robot state (or any state tensor) in
    # `observation`. For those, allow state to be missing and emit zeros instead.
    allow_missing_state: bool = False


# Registry of supported OXE datasets
OXE_DATASETS = {
    "language_table": OXEDatasetConfig(
        name="language_table",
        gcs_path="gs://gresearch/robotics/language_table/0.1.0",
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
        gcs_path="gs://gresearch/robotics/robo_net/0.1.0",
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
    # --- Additional datasets present in the LRZ cluster mirror ---
    # ALOHA Mobile (multi-camera mobile manipulation)
    "aloha_mobile": OXEDatasetConfig(
        name="aloha_mobile",
        gcs_path="gs://gresearch/robotics/aloha_mobile/0.0.1",
        image_key="cam_high",
        instruction_key="language_instruction",  # step-level
        instruction_in_step=True,
        state_key="state",
        image_shape=(480, 640, 3),
        control_frequency_hz=10.0,  # unknown; not used in pipeline
        action_dim=16,
        state_dim=14,
        action_is_dict=False,
        avg_episode_length=200,  # long trajectories
    ),
    # DROID (large-scale real robot dataset)
    "droid": OXEDatasetConfig(
        name="droid",
        gcs_path="gs://gresearch/robotics/droid/1.0.1",
        image_key="exterior_image_1_left",
        instruction_key="language_instruction",  # step-level
        instruction_in_step=True,
        state_key="cartesian_position",
        image_shape=(180, 320, 3),
        control_frequency_hz=10.0,  # unknown; not used in pipeline
        action_dim=7,
        state_dim=6,
        action_is_dict=False,
        avg_episode_length=300,
    ),
    # Berkeley Autolab UR5 (vision + language + UR5)
    "berkeley_autolab_ur5": OXEDatasetConfig(
        name="berkeley_autolab_ur5",
        gcs_path="gs://gresearch/robotics/berkeley_autolab_ur5/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="robot_state",
        image_shape=(480, 640, 3),
        control_frequency_hz=10.0,  # unknown; not used in pipeline
        action_dim=3,
        state_dim=15,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=90,
    ),
    # Jaco Play (table-top manipulation with Jaco)
    "jaco_play": OXEDatasetConfig(
        name="jaco_play",
        gcs_path="gs://gresearch/robotics/jaco_play/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="end_effector_cartesian_pos",
        image_shape=(224, 224, 3),
        control_frequency_hz=10.0,  # unknown; not used in pipeline
        action_dim=3,
        state_dim=7,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=70,
    ),
    # KUKA (large-scale KUKA real robot dataset)
    "kuka": OXEDatasetConfig(
        name="kuka",
        gcs_path="gs://gresearch/robotics/kuka/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="clip_function_input/base_pose_tool_reached",
        image_shape=(512, 640, 3),
        control_frequency_hz=10.0,  # unknown; not used in pipeline
        action_dim=3,
        state_dim=7,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=15,
    ),
    # TACO Play (teleop manipulation with multiple cameras)
    "taco_play": OXEDatasetConfig(
        name="taco_play",
        gcs_path="gs://gresearch/robotics/taco_play/0.1.0",
        image_key="rgb_static",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="robot_obs",
        image_shape=(150, 200, 3),
        control_frequency_hz=10.0,  # unknown; not used in pipeline
        action_dim=7,
        state_dim=15,
        action_key="actions",
        action_is_dict=True,
        avg_episode_length=60,
    ),
    # Roboturk (human teleoperation; does not expose a robot state tensor)
    "roboturk": OXEDatasetConfig(
        name="roboturk",
        gcs_path="gs://gresearch/robotics/roboturk/0.1.0",
        image_key="front_rgb",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key=None,
        image_shape=(480, 640, 3),
        control_frequency_hz=10.0,  # unknown; not used in pipeline
        action_dim=3,
        state_dim=0,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=90,
        allow_missing_state=True,
    ),
    # --- More datasets from the LRZ cluster mirror (image-only training path) ---
    "bc_z": OXEDatasetConfig(
        name="bc_z",
        gcs_path="gs://gresearch/robotics/bc_z/0.1.0",
        image_key="image",
        image_shape=(171, 213, 3),
        avg_episode_length=30,
    ),
    "berkeley_cable_routing": OXEDatasetConfig(
        name="berkeley_cable_routing",
        gcs_path="gs://gresearch/robotics/berkeley_cable_routing/0.1.0",
        image_key="image",
        image_shape=(128, 128, 3),
        avg_episode_length=30,
    ),
    "columbia_cairlab_pusht_real": OXEDatasetConfig(
        name="columbia_cairlab_pusht_real",
        gcs_path="gs://gresearch/robotics/columbia_cairlab_pusht_real/0.1.0",
        image_key="image",
        image_shape=(240, 320, 3),
        avg_episode_length=30,
    ),
    "mimic_play": OXEDatasetConfig(
        name="mimic_play",
        gcs_path="gs://gresearch/robotics/mimic_play/0.0.1",
        image_key="image/front_image_1",
        image_shape=(120, 120, 3),
        avg_episode_length=200,
    ),
    "berkeley_fanuc_manipulation": OXEDatasetConfig(
        name="berkeley_fanuc_manipulation",
        gcs_path="gs://gresearch/robotics/berkeley_fanuc_manipulation/0.1.0",
        image_key="image",
        image_shape=(224, 224, 3),
        avg_episode_length=60,
    ),
    "dobbe": OXEDatasetConfig(
        name="dobbe",
        gcs_path="gs://gresearch/robotics/dobbe/0.0.1",
        image_key="wrist_image",
        image_shape=(256, 256, 3),
        avg_episode_length=60,
    ),
    "uiuc_d3field": OXEDatasetConfig(
        name="uiuc_d3field",
        gcs_path="gs://gresearch/robotics/uiuc_d3field/0.1.0",
        image_key="image_1",
        image_shape=(360, 640, 3),
        avg_episode_length=60,
    ),
    "ucsd_kitchen_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="ucsd_kitchen_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(480, 640, 3),
        avg_episode_length=60,
    ),
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="ucsd_pick_and_place_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/ucsd_pick_and_place_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(224, 224, 3),
        avg_episode_length=60,
    ),
    "furniture_bench_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="furniture_bench_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/furniture_bench_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(224, 224, 3),
        avg_episode_length=60,
    ),
    "maniskill_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="maniskill_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(256, 256, 3),
        avg_episode_length=60,
    ),
    "robo_set": OXEDatasetConfig(
        name="robo_set",
        gcs_path="gs://gresearch/robotics/robo_set/0.0.1",
        image_key="image_left",
        image_shape=(240, 424, 3),
        avg_episode_length=60,
    ),
    "stanford_hydra_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="stanford_hydra_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/stanford_hydra_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(240, 320, 3),
        avg_episode_length=60,
    ),
    "stanford_robocook_converted_externally_to_rlds": OXEDatasetConfig(
        name="stanford_robocook_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/stanford_robocook_converted_externally_to_rlds/0.1.0",
        image_key="image_1",
        image_shape=(256, 256, 3),
        avg_episode_length=60,
    ),
    "spoc": OXEDatasetConfig(
        name="spoc",
        gcs_path="gs://gresearch/robotics/spoc/0.0.1",
        image_key="image",
        image_shape=(224, 384, 3),
        avg_episode_length=60,
    ),
    "tidybot": OXEDatasetConfig(
        name="tidybot",
        gcs_path="gs://gresearch/robotics/tidybot/0.0.1",
        image_key="image",
        image_shape=(360, 640, 3),
        avg_episode_length=60,
    ),
    "toto": OXEDatasetConfig(
        name="toto",
        gcs_path="gs://gresearch/robotics/toto/0.1.0",
        image_key="image",
        image_shape=(480, 640, 3),
        avg_episode_length=60,
    ),
    "viola": OXEDatasetConfig(
        name="viola",
        gcs_path="gs://gresearch/robotics/viola/0.1.0",
        image_key="agentview_rgb",
        image_shape=(224, 224, 3),
        avg_episode_length=60,
    ),
    "vima_converted_externally_to_rlds": OXEDatasetConfig(
        name="vima_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/vima_converted_externally_to_rlds/0.0.1",
        image_key="image",
        image_shape=(128, 256, 3),
        avg_episode_length=60,
    ),
    "utaustin_mutex": OXEDatasetConfig(
        name="utaustin_mutex",
        gcs_path="gs://gresearch/robotics/utaustin_mutex/0.1.0",
        image_key="image",
        image_shape=(128, 128, 3),
        avg_episode_length=60,
    ),
    "fmb": OXEDatasetConfig(
        name="fmb",
        gcs_path="gs://gresearch/robotics/fmb/0.0.1",
        image_key="image_side_1",
        image_shape=(256, 256, 3),
        avg_episode_length=60,
    ),
}


def _parse_tfds_prepared_dataset_dir(builder_dir: str) -> tuple[str, str]:
    """
    Parse a TFDS prepared dataset directory path into (dataset_dirname, version).

    Works for both local paths and GCS paths like:
      - gs://gresearch/robotics/bridge/0.1.0
      - /some/root/bridge/0.1.0
    """
    parts = [p for p in str(builder_dir).rstrip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"Invalid TFDS builder dir (expected .../<dataset>/<version>): {builder_dir!r}"
        )
    return parts[-2], parts[-1]


def _resolve_tfds_builder_dir(
    *, gcs_builder_dir: str, source: str, local_root: Optional[str]
) -> str:
    """
    Resolve the TFDS prepared dataset directory based on `source`.

    `source`:
      - "gcs": always use `gcs_builder_dir`
      - "local": always use `<local_root>/<dataset_dirname>/<version>` (must exist)
      - "auto": prefer local if available, else fall back to `gcs_builder_dir`
    """
    source = str(source).lower().strip()
    if source not in {"gcs", "local", "auto"}:
        raise ValueError(
            f"tfds_source must be one of: 'gcs', 'local', 'auto' (got {source!r})"
        )

    if source == "gcs":
        return str(gcs_builder_dir)

    if not local_root:
        if source == "local":
            raise ValueError("tfds_source='local' requires tfds_local_root to be set")
        return str(gcs_builder_dir)

    dataset_dirname, version = _parse_tfds_prepared_dataset_dir(gcs_builder_dir)
    local_builder_dir = str(Path(str(local_root)) / dataset_dirname / version)

    if source == "local":
        if not Path(local_builder_dir).exists():
            raise FileNotFoundError(
                f"Local TFDS dataset not found: {local_builder_dir} (from local_root={local_root!r})"
            )
        return local_builder_dir

    # auto: local if present, else gcs
    return local_builder_dir if Path(local_builder_dir).exists() else str(gcs_builder_dir)



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
        episode_queue_shuffle_buffer: Shuffle buffer for incoming episodes (0 to disable)
        intra_episode_sample_shuffle_buffer: Shuffle buffer for per-episode samples (0 to disable)
        final_stream_prefetch_buffer: tf.data prefetch buffer size (after pair stream is formed)
        pipeline_transform_parallelism: Parallelism for map operations (default: AUTOTUNE)
        return_metadata: If True, return dict with metadata
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        offset: int,
        final_stream_prefetch_buffer: int,
        episode_queue_shuffle_buffer: int,
        intra_episode_sample_shuffle_buffer: int,
        image_size: int,
        return_metadata: bool,
        is_train: bool,
        output_batch_size: Optional[int],
        output_action_dim: Optional[int],
        output_state_dim: Optional[int],
        persistent_iterator: bool,  # Keep iterator alive to avoid shuffle buffer refill
        samples_per_episode: int,
        seed: Optional[int],
        debug_use_synthetic_data: bool,
        debug_synthetic_num_samples: int,
        precomputed_size: Optional[int],  # Avoid TF init in __len__
        episode_queue_prefetch_buffer: int,  # Phase 3: overlap episode fetch/decode
        tfds_read_cycle_length: int,
        tfds_read_block_length: int,
        tfds_read_decode_parallelism: int,
        tfds_read_interleave_parallelism: int,
        pipeline_episode_concurrency: int,  # Phase 4: parallel episode processing via interleave
        pipeline_transform_parallelism: int,
        pipeline_interleave_parallelism: int,
        private_threadpool_size: int,  # 0 = use shared global threadpool
        pair_frames_mode: str = "endpoints",
        pair_frames_stride: int = 1,
        pair_frames_n: int = 2,
        tfds_source: str = "gcs",
        tfds_local_root: Optional[str] = None,
    ):
        super().__init__()

        if dataset_name not in OXE_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(OXE_DATASETS.keys())}"
            )

        self.config = OXE_DATASETS.get(dataset_name)

        self.split = split
        self.offset = int(offset)
        if self.offset <= 0:
            raise ValueError("offset must be a positive integer (>= 1)")
        self.image_size = image_size
        self.final_stream_prefetch_buffer = final_stream_prefetch_buffer
        self.is_train = bool(is_train)
        self.output_batch_size = int(output_batch_size) if output_batch_size is not None else None
        if self.output_batch_size is not None and self.output_batch_size <= 0:
            raise ValueError("output_batch_size must be a positive integer or None")
        self.output_action_dim = int(output_action_dim) if output_action_dim is not None else None
        self.output_state_dim = int(output_state_dim) if output_state_dim is not None else None
        if self.output_action_dim is not None and self.output_action_dim <= 0:
            raise ValueError("output_action_dim must be a positive integer or None")
        if self.output_state_dim is not None and self.output_state_dim <= 0:
            raise ValueError("output_state_dim must be a positive integer or None")
        
        self.tfds_read_cycle_length = tfds_read_cycle_length
        self.tfds_read_block_length = tfds_read_block_length
        self.tfds_read_decode_parallelism = tfds_read_decode_parallelism
        self.tfds_read_interleave_parallelism = tfds_read_interleave_parallelism
        
        self.pipeline_episode_concurrency = pipeline_episode_concurrency
        self.pipeline_transform_parallelism = pipeline_transform_parallelism
        self.pipeline_interleave_parallelism = pipeline_interleave_parallelism
        
        self.return_metadata = return_metadata
        self.persistent_iterator = persistent_iterator
        self.samples_per_episode = samples_per_episode
        self.seed = seed
        self._rng = random.Random(seed)
        self._tf_seed: Optional[int] = None
        self.debug_use_synthetic_data = bool(debug_use_synthetic_data)
        self.debug_synthetic_num_samples = int(debug_synthetic_num_samples)
        self._precomputed_size = precomputed_size
        self.episode_queue_prefetch_buffer = episode_queue_prefetch_buffer
        self.private_threadpool_size = private_threadpool_size

        self.episode_queue_shuffle_buffer = int(episode_queue_shuffle_buffer)
        self.intra_episode_sample_shuffle_buffer = int(intra_episode_sample_shuffle_buffer)

        self.tfds_source = str(tfds_source)
        self.tfds_local_root = tfds_local_root

        self.pair_frames_mode = str(pair_frames_mode)
        if self.pair_frames_mode not in {"endpoints", "all", "stride", "fixed_n"}:
            raise ValueError(
                "pair_frames_mode must be one of: 'endpoints', 'all', 'stride', 'fixed_n'"
            )
        self.pair_frames_stride = int(pair_frames_stride)
        if self.pair_frames_stride <= 0:
            raise ValueError("pair_frames_stride must be a positive integer (>= 1)")
        self.pair_frames_n = int(pair_frames_n)
        if self.pair_frames_mode == "fixed_n" and self.pair_frames_n < 2:
            raise ValueError("pair_frames_n must be >= 2 when pair_frames_mode='fixed_n'")

        # Lazy initialization of tf.data pipeline
        self._builder = None
        self._num_episodes = None
        # Persistent pipeline - created once, reused across epochs
        self._persistent_pipeline = None
        # Persistent iterator - avoids shuffle buffer refill on each epoch
        self._pipeline_iterator = None
        # Idempotency flag for cleanup
        self._cleaned_up = False

    def _compute_pair_frame_indices(self, offset: int) -> list[int]:
        """
        Compute (Python-side) step indices in [0, offset] to include in the output.

        Contract:
        - indices[0] == 0
        - indices[-1] == offset
        - len(indices) >= 2 when offset > 0
        - non-decreasing order (duplicates allowed)
        """
        offset = int(offset)
        if offset <= 0:
            raise ValueError("offset must be >= 1 to form frame pairs")

        mode = str(self.pair_frames_mode)
        if mode == "endpoints":
            return [0, offset]
        if mode == "all":
            return list(range(0, offset + 1))
        if mode == "stride":
            stride = int(self.pair_frames_stride)
            if stride <= 0:
                raise ValueError("pair_frames_stride must be >= 1")
            idxs = list(range(0, offset + 1, stride))
            if not idxs or idxs[0] != 0:
                idxs = [0] + idxs
            if idxs[-1] != offset:
                idxs.append(offset)
            return idxs
        if mode == "fixed_n":
            n = int(self.pair_frames_n)
            if n < 2:
                raise ValueError("pair_frames_n must be >= 2 when pair_frames_mode='fixed_n'")
            idxs = np.rint(np.linspace(0, offset, num=n)).astype(np.int32).tolist()
            if not idxs:
                idxs = [0, offset]
            idxs[0] = 0
            idxs[-1] = offset
            idxs = [int(min(offset, max(0, x))) for x in idxs]
            idxs.sort()
            return idxs

        raise ValueError(f"Unknown pair_frames_mode: {mode}")

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
        if self.debug_use_synthetic_data:
            return
        if self._builder is not None:
            return

        # Import TF only when needed (CPU-only, avoid TF GPU init/JIT).
        tf = _import_tensorflow_cpu_only()
        import tensorflow_datasets as tfds

        builder_dir = _resolve_tfds_builder_dir(
            gcs_builder_dir=self.config.gcs_path,
            source=self.tfds_source,
            local_root=self.tfds_local_root,
        )
        logger.info(
            f"OXE TFDS source: {self.tfds_source} | dataset={self.config.name} | builder_dir={builder_dir}"
        )
        self._builder = tfds.builder_from_directory(builder_dir)
        self._num_episodes = self._builder.info.splits[
            self.split.split("[")[0]
        ].num_examples

    def __len__(self):
        """
        Approximate length in yielded items.

        When `output_batch_size` is set, this dataset yields batches (created in
        tf.data) and the length is in batches. When `output_batch_size` is None,
        this dataset yields individual samples and the length is in samples.

        IMPORTANT: If precomputed_size is set, returns that value directly to avoid
        TensorFlow initialization. This prevents startup hangs and OOM issues when
        DataLoader workers fork before TF is initialized.
        """
        if self._precomputed_size is not None:
            if self.output_batch_size is None:
                return int(self._precomputed_size)
            if self.is_train:
                return max(
                    1, int(self._precomputed_size) // int(self.output_batch_size)
                )
            return int(
                math.ceil(float(self._precomputed_size) / float(self.output_batch_size))
            )
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
        tf = _import_tensorflow_cpu_only()

        self._init_rng_for_worker()

        if self.debug_use_synthetic_data:
            dataset_name = tf.constant(self.config.name, dtype=tf.string)
            offset = tf.constant(int(self.offset), dtype=tf.int32)
            pair_frame_indices = self._compute_pair_frame_indices(int(self.offset))
            num_frames = int(len(pair_frame_indices))
            action_dim = int(self.config.action_dim)
            state_dim = int(self.config.state_dim)
            output_action_dim = int(self.output_action_dim) if self.output_action_dim is not None else action_dim
            output_state_dim = int(self.output_state_dim) if self.output_state_dim is not None else state_dim
            h = int(self.image_size)
            w = int(self.image_size)

            robot_val = tf.constant(
                "widowx" if self.config.name == "robonet" else "", dtype=tf.string
            )

            def _make(idx: tf.Tensor):
                pair = tf.zeros((num_frames, h, w, 3), dtype=tf.uint8)
                if not self.return_metadata:
                    return pair

                episode_id = tf.strings.join(
                    [dataset_name, tf.constant(":"), tf.strings.as_string(idx)]
                )
                meta = {
                    "dataset_name": dataset_name,
                    "dataset_type": dataset_name,
                    "episode_id": episode_id,
                    "frame_idx": tf.cast(idx, tf.int32),
                    "offset": offset,
                    "language": tf.constant("dummy instruction", dtype=tf.string),
                    "action": tf.zeros((output_action_dim,), dtype=tf.float32),
                    "initial_state": tf.zeros((output_state_dim,), dtype=tf.float32),
                    "robot": robot_val,
                }
                return pair, meta

            return tf.data.Dataset.range(int(self.debug_synthetic_num_samples)).map(
                _make, num_parallel_calls=tf.data.AUTOTUNE
            )

        self._init_tfds()

        import tensorflow_datasets as tfds

        tfds_read_cycle_length = int(self.tfds_read_cycle_length)
        tfds_read_block_length = int(self.tfds_read_block_length)
        
        tfds_read_decode_parallelism = int(self.tfds_read_decode_parallelism)
        if tfds_read_decode_parallelism == -1:
            tfds_read_decode_parallelism = tf.data.AUTOTUNE
            
        tfds_read_interleave_parallelism = int(self.tfds_read_interleave_parallelism)
        if tfds_read_interleave_parallelism == -1:
            tfds_read_interleave_parallelism = tf.data.AUTOTUNE
        elif tfds_read_interleave_parallelism > 0:
            # TF/TFDS constraint: when `interleave_cycle_length` is fixed, the
            # interleave `num_parallel_calls` must be <= cycle_length (or AUTOTUNE).
            # When we scale `cycle_length` down for multi-dataset mixing, clamp to
            # avoid InvalidArgumentError.
            tfds_read_interleave_parallelism = min(
                tfds_read_interleave_parallelism, tfds_read_cycle_length
            )

        pipeline_transform_parallelism = int(self.pipeline_transform_parallelism)
        if pipeline_transform_parallelism == -1:
            pipeline_transform_parallelism = tf.data.AUTOTUNE

        pipeline_interleave_parallelism = int(self.pipeline_interleave_parallelism)
        if pipeline_interleave_parallelism == -1:
            pipeline_interleave_parallelism = tf.data.AUTOTUNE
            
        pipeline_episode_concurrency = int(self.pipeline_episode_concurrency)

        read_config = tfds.ReadConfig(
            try_autocache=False,                                    # our dataset is too big for that
            add_tfds_id=True,                                       # adds a unique id to each episode like tfrecord...
            shuffle_seed=self.seed,
            interleave_cycle_length=tfds_read_cycle_length,
            interleave_block_length=tfds_read_block_length,
            # input_context=
            # experimental_interleave_sort_fn=                      # this can be used to overwrite shuffle_files=True
            skip_prefetch=True, # done at the end of our pipeline
            num_parallel_calls_for_decode=tfds_read_decode_parallelism,
            num_parallel_calls_for_interleave_files=tfds_read_interleave_parallelism,
            # enable_ordering_guard=                                # True by default, throws exception if ordered ds is shuffled
            assert_cardinality=False,   # this ensures that the read length matches the metadata
                                        # len but if some files may be missing turn this to False
        )
        ds = self._builder.as_dataset(split=self.split, read_config=read_config)

        # Shard episodes across DataLoader workers to avoid duplicates when num_workers > 0.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        # Shuffle the queue of episodes ("tickets") before processing.
        if self.episode_queue_shuffle_buffer > 0:
            ds = ds.shuffle(self.episode_queue_shuffle_buffer, seed=self._tf_seed)

        # Prefetch episodes (keep small; each episode can be large)
        if int(self.episode_queue_prefetch_buffer) == -1:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        elif int(self.episode_queue_prefetch_buffer) > 0:
            ds = ds.prefetch(self.episode_queue_prefetch_buffer)

        image_key = self.config.image_key
        instruction_key = self.config.instruction_key
        state_key = self.config.state_key
        offset = int(self.offset)
        image_size = int(self.image_size)
        return_metadata = bool(self.return_metadata)
        dataset_name = self.config.name
        pair_frames_mode = str(self.pair_frames_mode)
        pair_frame_indices = self._compute_pair_frame_indices(offset)
        pair_num_frames = int(len(pair_frame_indices))
        action_dim = int(self.config.action_dim)
        state_dim = int(self.config.state_dim)
        output_action_dim = int(self.output_action_dim) if self.output_action_dim is not None else action_dim
        output_state_dim = int(self.output_state_dim) if self.output_state_dim is not None else state_dim
        action_key = self.config.action_key
        action_is_dict = bool(self.config.action_is_dict)
        instruction_in_step = bool(self.config.instruction_in_step)
        robot_key = self.config.robot_key
        samples_per_episode = int(self.samples_per_episode) if self.samples_per_episode else 0

        offset_tf = tf.constant(offset, dtype=tf.int32)
        dataset_name_tf = tf.constant(dataset_name, dtype=tf.string)
        pair_frame_indices_tf = tf.constant(pair_frame_indices, dtype=tf.int32)
        pair_frame_prefix_indices_tf = tf.constant(pair_frame_indices[:-1], dtype=tf.int32)
        # When `intra_episode_sample_shuffle_buffer` is 0, users expect "no shuffling"
        # rather than a hidden, large per-episode shuffle buffer (which can be very
        # expensive when `samples_per_episode` is small).
        #
        # Also treat a buffer of 1 as effectively "no shuffle" (it cannot randomize).
        per_episode_sample_shuffle = (
            int(self.intra_episode_sample_shuffle_buffer)
            if self.intra_episode_sample_shuffle_buffer > 1
            else 0
        )

        def _get_keypath(container: dict, keypath: str):
            """
            Resolve `keypath` from a nested dict structure.

            Important: some OXE TFDS datasets use literal keys that contain `/`
            (e.g., `clip_function_input/base_pose_tool_reached`). Prefer a direct
            lookup first, and only treat `/` as a nesting delimiter if the full
            key is not present.
            """
            if not keypath:
                raise ValueError("keypath must be a non-empty string")
            if keypath in container:
                return container[keypath]
            if "/" not in keypath:
                return container[keypath]
            cur = container
            for part in keypath.split("/"):
                cur = cur[part]
            return cur

        def _pad_or_truncate_1d(vec: tf.Tensor, target_dim: int) -> tf.Tensor:
            vec = tf.convert_to_tensor(vec)
            vec = vec[:target_dim]
            pad = tf.maximum(0, target_dim - tf.shape(vec)[0])
            vec = tf.pad(vec, paddings=[[0, pad]])
            return tf.ensure_shape(vec, [target_dim])

        def _strip_null_bytes(s: tf.Tensor) -> tf.Tensor:
            s = tf.convert_to_tensor(s, dtype=tf.string)
            return tf.strings.regex_replace(s, "\x00+$", "")

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

        def _resize_frames_batch(frames: tf.Tensor) -> tf.Tensor:
            frames = tf.cast(frames, tf.uint8)
            shape = tf.shape(frames)
            h = shape[1]
            w = shape[2]
            needs_resize = tf.logical_or(tf.not_equal(h, image_size), tf.not_equal(w, image_size))

            def _do_resize():
                resized = tf.image.resize(frames, [image_size, image_size])
                return tf.cast(resized, tf.uint8)

            frames = tf.cond(needs_resize, _do_resize, lambda: tf.cast(frames, tf.uint8))
            return tf.ensure_shape(frames, [None, image_size, image_size, 3])

        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = worker_info.num_workers if worker_info is not None else 1
        worker_id_tf = tf.constant(int(worker_id), dtype=tf.int32)
        num_shards_tf = tf.constant(int(num_shards), dtype=tf.int32)
        colon_tf = tf.constant(":", dtype=tf.string)

        # Enumerate episodes so we can construct a stable episode_id even when the
        # underlying RLDS dataset does not provide an "episode_id" field.
        ds = ds.enumerate()

        def process_episode_to_pairs(ep_idx, episode):
            steps_ds = episode["steps"]

            episode_id_tf = tf.strings.join(
                [
                    dataset_name_tf,
                    colon_tf,
                    tf.strings.as_string(num_shards_tf),
                    colon_tf,
                    tf.strings.as_string(worker_id_tf),
                    colon_tf,
                    tf.strings.as_string(ep_idx),
                ]
            )

            if not return_metadata:
                frames_ds = steps_ds.map(
                    lambda s: _get_keypath(s["observation"], image_key),
                    num_parallel_calls=pipeline_transform_parallelism,
                )

                # Fast path: if we only take a single sample per episode and do not
                # need per-episode sampling shuffle, avoid the scan + batch(8)/unbatch
                # pipeline. Taking just `offset+1` frames and forming the first pair
                # is much cheaper and avoids decoding/resizing unnecessary frames.
                if samples_per_episode == 1 and per_episode_sample_shuffle <= 0:
                    # We still have to iterate through `offset` intermediate frames to
                    # reach the target frame, but we can avoid resizing unnecessary
                    # frames by selecting and resizing only the requested indices.
                    # Batch the prefix once so we don't read/decode frames twice.
                    frames_prefix = frames_ds.take(offset + 1).batch(
                        offset + 1, drop_remainder=True
                    )

                    def _make_one_pair(frames):
                        selected = tf.gather(frames, pair_frame_indices_tf, axis=0)
                        selected = _resize_frames_batch(selected)
                        return tf.ensure_shape(
                            selected, [pair_num_frames, image_size, image_size, 3]
                        )

                    return frames_prefix.map(_make_one_pair, num_parallel_calls=1)

                frames_ds = frames_ds.batch(8).map(
                    _resize_frames_batch, num_parallel_calls=pipeline_transform_parallelism
                ).unbatch()

                dummy_pair = tf.zeros(
                    [pair_num_frames, image_size, image_size, 3], dtype=tf.uint8
                )

                def _init_state():
                    frames_ta = tf.TensorArray(
                        tf.uint8,
                        size=offset,
                        element_shape=(image_size, image_size, 3),
                        clear_after_read=False,
                    )
                    step_idx = tf.constant(0, dtype=tf.int32)
                    return frames_ta, step_idx

                if pair_frames_mode == "endpoints":

                    def _scan_fn(state, frame):
                        frames_ta, step_idx = state
                        pos = tf.math.mod(step_idx, offset_tf)
                        ready = tf.greater_equal(step_idx, offset_tf)

                        def _emit():
                            start_frame = frames_ta.read(pos)
                            pair = tf.stack([start_frame, frame], axis=0)
                            pair = tf.ensure_shape(
                                pair, [2, image_size, image_size, 3]
                            )
                            return pair

                        out_pair = tf.cond(ready, _emit, lambda: dummy_pair)
                        frames_ta = frames_ta.write(pos, frame)
                        return (frames_ta, step_idx + 1), out_pair

                else:

                    def _scan_fn(state, frame):
                        frames_ta, step_idx = state
                        pos = tf.math.mod(step_idx, offset_tf)
                        ready = tf.greater_equal(step_idx, offset_tf)

                        def _emit():
                            # `pair_frame_prefix_indices_tf` excludes the last endpoint
                            # (offset), which is the current `frame`.
                            ta_positions = tf.math.mod(
                                pos + pair_frame_prefix_indices_tf, offset_tf
                            )
                            prefix_frames = frames_ta.gather(ta_positions)
                            prefix_frames = tf.ensure_shape(
                                prefix_frames,
                                [pair_num_frames - 1, image_size, image_size, 3],
                            )
                            selected = tf.concat(
                                [prefix_frames, tf.expand_dims(frame, axis=0)], axis=0
                            )
                            return tf.ensure_shape(
                                selected,
                                [pair_num_frames, image_size, image_size, 3],
                            )

                        out_pair = tf.cond(ready, _emit, lambda: dummy_pair)
                        frames_ta = frames_ta.write(pos, frame)
                        return (frames_ta, step_idx + 1), out_pair

                pairs_ds = frames_ds.scan(_init_state(), _scan_fn).skip(offset)
                if samples_per_episode > 0:
                    if per_episode_sample_shuffle > 0:
                        pairs_ds = pairs_ds.shuffle(
                            per_episode_sample_shuffle, seed=self._tf_seed
                        )
                    pairs_ds = pairs_ds.take(samples_per_episode)
                return pairs_ds

            if robot_key:
                robot_raw = episode["episode_metadata"][robot_key]
                robot_tf = robot_raw if robot_raw.dtype == tf.string else tf.strings.as_string(robot_raw)
            else:
                robot_tf = tf.constant("", dtype=tf.string)

            def _extract_language(step):
                if instruction_in_step:
                    instr = _get_keypath(step, instruction_key)
                else:
                    instr = _get_keypath(step["observation"], instruction_key)
                # String tensor (Bridge/RT-1) vs encoded ints (language_table)
                if instr.dtype != tf.string:
                    instr = tf.strings.unicode_encode(tf.cast(instr, tf.int32), "UTF-8")
                return _strip_null_bytes(instr)

            def _extract_action(step):
                if action_is_dict:
                    if not action_key:
                        raise ValueError("Config error: action_is_dict=True requires action_key")
                    a = _get_keypath(step["action"], action_key)
                else:
                    a = step["action"]
                a = tf.cast(tf.reshape(a, [-1])[:action_dim], tf.float32)
                return tf.ensure_shape(a, [action_dim])

            allow_missing_state = bool(getattr(self.config, "allow_missing_state", False))

            def _extract_state(step):
                if state_dim <= 0:
                    return tf.zeros((0,), dtype=tf.float32)
                if state_key is None:
                    if allow_missing_state:
                        return tf.zeros((state_dim,), dtype=tf.float32)
                    raise ValueError(
                        f"Config error: state_key is None for dataset={dataset_name}"
                    )
                s = _get_keypath(step["observation"], state_key)
                s = tf.cast(tf.reshape(s, [-1])[:state_dim], tf.float32)
                return tf.ensure_shape(s, [state_dim])

            # Single-pass episode processing:
            # Use a ring buffer via `Dataset.scan()` to emit endpoint pairs [t, t+offset]
            # without building sliding windows of full image tensors (which can be very slow
            # and memory-heavy for larger offsets).
            def _step_to_features(step):
                frame = _get_keypath(step["observation"], image_key)
                action = _extract_action(step)
                state = _extract_state(step)
                language = _extract_language(step)
                return frame, action, state, language

            # Some datasets do not have a usable state (`state_dim <= 0`). We observed
            # segfaults for Roboturk when using `Dataset.scan()` in the metadata path.
            # Work around this by avoiding `scan()` entirely and forming pairs using
            # `Dataset.zip()` + `skip(offset)` and action windows.
            if state_dim <= 0 and return_metadata:
                if pair_frames_mode != "endpoints":
                    raise NotImplementedError(
                        "state_dim<=0 metadata pairing currently supports only pair_frames_mode='endpoints'"
                    )

                frames_ds = steps_ds.map(
                    lambda s: _get_keypath(s["observation"], image_key),
                    num_parallel_calls=pipeline_transform_parallelism,
                )
                frames_ds = frames_ds.batch(8).map(
                    _resize_frames_batch, num_parallel_calls=pipeline_transform_parallelism
                ).unbatch()

                actions_ds = steps_ds.map(
                    _extract_action, num_parallel_calls=pipeline_transform_parallelism
                )
                langs_ds = steps_ds.map(
                    _extract_language, num_parallel_calls=pipeline_transform_parallelism
                )
                states_ds = steps_ds.map(
                    _extract_state, num_parallel_calls=pipeline_transform_parallelism
                )

                # Pair frames: (t, t+offset)
                pairs_ds = tf.data.Dataset.zip((frames_ds, frames_ds.skip(offset))).map(
                    lambda a, b: tf.ensure_shape(tf.stack([a, b], axis=0), [2, image_size, image_size, 3]),
                    num_parallel_calls=pipeline_transform_parallelism,
                )

                # Cumulative action over [t, t+offset) using windowed reductions.
                action_sums_ds = actions_ds.window(
                    offset, shift=1, drop_remainder=True
                ).flat_map(
                    lambda w: w.batch(offset, drop_remainder=True).map(
                        lambda a: tf.reduce_sum(a, axis=0),
                        num_parallel_calls=1,
                    )
                )

                # Frame indices correspond to the start step index t.
                frame_idxs_ds = tf.data.Dataset.range(2**31 - 1, output_type=tf.int32)

                def _to_pair_and_meta(pair, cumulative_action, initial_state, language, frame_idx):
                    if output_action_dim != action_dim:
                        cumulative_action = _pad_or_truncate_1d(
                            cumulative_action, output_action_dim
                        )
                    if output_state_dim != state_dim:
                        initial_state = _pad_or_truncate_1d(initial_state, output_state_dim)
                    meta = {
                        "episode_id": episode_id_tf,
                        "frame_idx": frame_idx,
                        "offset": offset_tf,
                        "language": language,
                        "dataset_type": dataset_name_tf,
                        "dataset_name": dataset_name_tf,
                        "action": cumulative_action,
                        "initial_state": initial_state,
                        "robot": robot_tf,
                    }
                    return pair, meta

                out_ds = tf.data.Dataset.zip(
                    (pairs_ds, action_sums_ds, states_ds, langs_ds, frame_idxs_ds)
                ).map(_to_pair_and_meta, num_parallel_calls=pipeline_transform_parallelism)

                if samples_per_episode > 0:
                    if per_episode_sample_shuffle > 0:
                        out_ds = out_ds.shuffle(
                            per_episode_sample_shuffle, seed=self._tf_seed
                        )
                    out_ds = out_ds.take(samples_per_episode)
                return out_ds

            # Fast path: if we only take a single sample per episode and do not need
            # per-episode sampling shuffle, avoid decoding/resizing the full episode
            # and avoid the scan pipeline altogether.
            if samples_per_episode == 1 and per_episode_sample_shuffle <= 0:
                prefix_steps = steps_ds.take(offset + 1)
                prefix_feats = prefix_steps.map(
                    _step_to_features, num_parallel_calls=pipeline_transform_parallelism
                )
                prefix_batch = prefix_feats.batch(offset + 1, drop_remainder=True)

                frame_idx0 = tf.constant(0, dtype=tf.int32)

                def _make_one_sample(frames, actions, states, languages):
                    selected = tf.gather(frames, pair_frame_indices_tf, axis=0)
                    selected = _resize_frames_batch(selected)
                    selected = tf.ensure_shape(
                        selected, [pair_num_frames, image_size, image_size, 3]
                    )

                    cumulative_action = tf.reduce_sum(actions[:offset], axis=0)
                    initial_state = states[0]
                    language = languages[0]

                    if output_action_dim != action_dim:
                        cumulative_action = _pad_or_truncate_1d(
                            cumulative_action, output_action_dim
                        )
                    if output_state_dim != state_dim:
                        initial_state = _pad_or_truncate_1d(
                            initial_state, output_state_dim
                        )

                    meta = {
                        "episode_id": episode_id_tf,
                        "frame_idx": frame_idx0,
                        "offset": offset_tf,
                        "language": language,
                        "dataset_type": dataset_name_tf,
                        "dataset_name": dataset_name_tf,
                        "action": cumulative_action,
                        "initial_state": initial_state,
                        "robot": robot_tf,
                    }
                    return selected, meta

                return prefix_batch.map(_make_one_sample, num_parallel_calls=1)

            features_ds = steps_ds.map(_step_to_features, num_parallel_calls=pipeline_transform_parallelism)

            def _resize_features_batch(frames, actions, states, languages):
                return _resize_frames_batch(frames), actions, states, languages

            features_ds = features_ds.batch(8).map(
                _resize_features_batch, num_parallel_calls=pipeline_transform_parallelism
            ).unbatch()

            dummy_pair = tf.zeros([pair_num_frames, image_size, image_size, 3], dtype=tf.uint8)
            dummy_action = tf.zeros([action_dim], dtype=tf.float32)
            dummy_state = tf.zeros([state_dim], dtype=tf.float32)
            dummy_lang = tf.constant("", dtype=tf.string)
            dummy_idx = tf.constant(-1, dtype=tf.int32)

            # NOTE: Some datasets intentionally have `state_dim=0` (no usable state).
            # Creating a `tf.TensorArray` with `element_shape=(0,)` has been observed to
            # segfault for some TFDS builders (Roboturk) when used inside `Dataset.scan()`.
            # We special-case that case and do not track states via a TensorArray.

            if state_dim > 0:

                def _init_state():
                    frames_ta = tf.TensorArray(
                        tf.uint8,
                        size=offset,
                        element_shape=(image_size, image_size, 3),
                        clear_after_read=False,
                    )
                    actions_ta = tf.TensorArray(
                        tf.float32,
                        size=offset,
                        element_shape=(action_dim,),
                        clear_after_read=False,
                    )
                    states_ta = tf.TensorArray(
                        tf.float32,
                        size=offset,
                        element_shape=(state_dim,),
                        clear_after_read=False,
                    )
                    langs_ta = tf.TensorArray(
                        tf.string,
                        size=offset,
                        element_shape=(),
                        clear_after_read=False,
                    )
                    action_sum = tf.zeros([action_dim], dtype=tf.float32)
                    step_idx = tf.constant(0, dtype=tf.int32)
                    return frames_ta, actions_ta, states_ta, langs_ta, action_sum, step_idx

                def _scan_fn(state, features):
                    frames_ta, actions_ta, states_ta, langs_ta, action_sum, step_idx = state
                    frame, action, obs_state, language = features

                    pos = tf.math.mod(step_idx, offset_tf)
                    ready = tf.greater_equal(step_idx, offset_tf)

                    def _emit():
                        start_state = states_ta.read(pos)
                        start_lang = langs_ta.read(pos)
                        frame_idx = tf.cast(step_idx - offset_tf, tf.int32)

                        if pair_frames_mode == "endpoints":
                            start_frame = frames_ta.read(pos)
                            pair = tf.stack([start_frame, frame], axis=0)
                            pair = tf.ensure_shape(pair, [2, image_size, image_size, 3])
                            return pair, frame_idx, action_sum, start_state, start_lang

                        ta_positions = tf.math.mod(
                            pos + pair_frame_prefix_indices_tf, offset_tf
                        )
                        prefix_frames = frames_ta.gather(ta_positions)
                        prefix_frames = tf.ensure_shape(
                            prefix_frames,
                            [pair_num_frames - 1, image_size, image_size, 3],
                        )
                        selected = tf.concat(
                            [prefix_frames, tf.expand_dims(frame, axis=0)], axis=0
                        )
                        selected = tf.ensure_shape(
                            selected, [pair_num_frames, image_size, image_size, 3]
                        )
                        return selected, frame_idx, action_sum, start_state, start_lang

                    out_pair, out_frame_idx, out_action, out_init_state, out_lang = tf.cond(
                        ready,
                        _emit,
                        lambda: (
                            dummy_pair,
                            dummy_idx,
                            dummy_action,
                            dummy_state,
                            dummy_lang,
                        ),
                    )

                    old_action = tf.cond(
                        ready, lambda: actions_ta.read(pos), lambda: dummy_action
                    )
                    action_sum = action_sum + action - old_action

                    frames_ta = frames_ta.write(pos, frame)
                    actions_ta = actions_ta.write(pos, action)
                    states_ta = states_ta.write(pos, obs_state)
                    langs_ta = langs_ta.write(pos, language)

                    return (
                        frames_ta,
                        actions_ta,
                        states_ta,
                        langs_ta,
                        action_sum,
                        step_idx + 1,
                    ), (out_pair, out_frame_idx, out_action, out_init_state, out_lang)

            else:

                def _init_state():
                    frames_ta = tf.TensorArray(
                        tf.uint8,
                        size=offset,
                        element_shape=(image_size, image_size, 3),
                        clear_after_read=False,
                    )
                    actions_ta = tf.TensorArray(
                        tf.float32,
                        size=offset,
                        element_shape=(action_dim,),
                        clear_after_read=False,
                    )
                    action_sum = tf.zeros([action_dim], dtype=tf.float32)
                    step_idx = tf.constant(0, dtype=tf.int32)
                    return frames_ta, actions_ta, action_sum, step_idx

                def _scan_fn(state, features):
                    frames_ta, actions_ta, action_sum, step_idx = state
                    frame, action, _obs_state, language = features

                    pos = tf.math.mod(step_idx, offset_tf)
                    ready = tf.greater_equal(step_idx, offset_tf)

                    def _emit():
                        # For state_dim=0 datasets, we do not keep a string TensorArray.
                        # Using a tf.string TensorArray inside `Dataset.scan()` has been
                        # observed to segfault for some builders (Roboturk). We instead
                        # emit the per-step language at the *end* of the window. For
                        # most datasets the instruction is constant across the episode.
                        start_lang = language
                        frame_idx = tf.cast(step_idx - offset_tf, tf.int32)
                        start_state = dummy_state

                        if pair_frames_mode == "endpoints":
                            start_frame = frames_ta.read(pos)
                            pair = tf.stack([start_frame, frame], axis=0)
                            pair = tf.ensure_shape(pair, [2, image_size, image_size, 3])
                            return pair, frame_idx, action_sum, start_state, start_lang

                        ta_positions = tf.math.mod(
                            pos + pair_frame_prefix_indices_tf, offset_tf
                        )
                        prefix_frames = frames_ta.gather(ta_positions)
                        prefix_frames = tf.ensure_shape(
                            prefix_frames,
                            [pair_num_frames - 1, image_size, image_size, 3],
                        )
                        selected = tf.concat(
                            [prefix_frames, tf.expand_dims(frame, axis=0)], axis=0
                        )
                        selected = tf.ensure_shape(
                            selected, [pair_num_frames, image_size, image_size, 3]
                        )
                        return selected, frame_idx, action_sum, start_state, start_lang

                    out_pair, out_frame_idx, out_action, out_init_state, out_lang = tf.cond(
                        ready,
                        _emit,
                        lambda: (
                            dummy_pair,
                            dummy_idx,
                            dummy_action,
                            dummy_state,
                            dummy_lang,
                        ),
                    )

                    old_action = tf.cond(
                        ready, lambda: actions_ta.read(pos), lambda: dummy_action
                    )
                    action_sum = action_sum + action - old_action

                    frames_ta = frames_ta.write(pos, frame)
                    actions_ta = actions_ta.write(pos, action)

                    return (
                        frames_ta,
                        actions_ta,
                        action_sum,
                        step_idx + 1,
                    ), (out_pair, out_frame_idx, out_action, out_init_state, out_lang)

            scanned_ds = features_ds.scan(_init_state(), _scan_fn).skip(offset)

            def _to_pair_and_meta(pair, frame_idx, cumulative_action, initial_state, language):
                if output_action_dim != action_dim:
                    cumulative_action = _pad_or_truncate_1d(cumulative_action, output_action_dim)
                if output_state_dim != state_dim:
                    initial_state = _pad_or_truncate_1d(initial_state, output_state_dim)
                meta = {
                    "episode_id": episode_id_tf,
                    "frame_idx": frame_idx,
                    "offset": offset_tf,
                    "language": language,
                    "dataset_type": dataset_name_tf,
                    "dataset_name": dataset_name_tf,
                    "action": cumulative_action,
                    "initial_state": initial_state,
                    "robot": robot_tf,
                }
                return pair, meta

            out_ds = scanned_ds.map(_to_pair_and_meta, num_parallel_calls=pipeline_transform_parallelism)
            if samples_per_episode > 0:
                if per_episode_sample_shuffle > 0:
                    out_ds = out_ds.shuffle(per_episode_sample_shuffle, seed=self._tf_seed)
                out_ds = out_ds.take(samples_per_episode)
            return out_ds

        if pipeline_episode_concurrency == -1:
            cycle_length = tf.data.AUTOTUNE
        else:
            cycle_length = max(1, pipeline_episode_concurrency)

        tf_ds = ds.interleave(
            process_episode_to_pairs,
            cycle_length=cycle_length,
            block_length=1,
            num_parallel_calls=pipeline_interleave_parallelism,
            deterministic=False,
        )

        # Apply tf.data options (private thread pool, non-determinism for speed)
        options = tf.data.Options()
        options.experimental_deterministic = False
        options.experimental_slack = True
        if self.private_threadpool_size > 0:
            options.threading.private_threadpool_size = self.private_threadpool_size

        tf_ds = tf_ds.with_options(options)

        return tf_ds

    def _get_or_create_pipeline(self):
        """Get existing pipeline or create new one (lazy, persistent)."""
        if self._persistent_pipeline is None:
            tf = _import_tensorflow_cpu_only()
            tf_ds = self._create_tf_pipeline()

            if self.is_train:
                tf_ds = tf_ds.repeat()

            if (
                self.return_metadata
                and (self.output_action_dim is not None or self.output_state_dim is not None)
            ):
                out_action_dim = self.output_action_dim
                out_state_dim = self.output_state_dim

                def _pad_or_truncate_1d(vec, target_dim: int):
                    vec = tf.convert_to_tensor(vec)
                    vec = vec[:target_dim]
                    pad = tf.maximum(0, target_dim - tf.shape(vec)[0])
                    vec = tf.pad(vec, paddings=[[0, pad]])
                    return tf.ensure_shape(vec, [target_dim])

                def _pad_meta(pair, meta):
                    meta = dict(meta)
                    if out_action_dim is not None:
                        meta["action"] = _pad_or_truncate_1d(meta["action"], int(out_action_dim))
                    if out_state_dim is not None:
                        meta["initial_state"] = _pad_or_truncate_1d(meta["initial_state"], int(out_state_dim))
                    return pair, meta

                tf_ds = tf_ds.map(_pad_meta, num_parallel_calls=tf.data.AUTOTUNE)

            if self.output_batch_size is not None:
                tf_ds = tf_ds.batch(
                    int(self.output_batch_size),
                    drop_remainder=bool(self.is_train),
                )

            prefetch_buffer = int(self.final_stream_prefetch_buffer)
            if prefetch_buffer == -1:
                tf_ds = tf_ds.prefetch(tf.data.AUTOTUNE)
            elif prefetch_buffer > 0:
                tf_ds = tf_ds.prefetch(prefetch_buffer)

            self._persistent_pipeline = tf_ds
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
        default_dataset_name = self.config.name
        offset = self.offset

        for item in tf_iter:
            if self.return_metadata:
                pair_tf, meta_tf = item
                if self.output_batch_size is None:
                    episode_id = _decode_str(meta_tf["episode_id"].numpy())
                    language = _decode_str(meta_tf["language"].numpy())
                    robot = _decode_str(meta_tf["robot"].numpy())
                    dataset_name = (
                        _decode_str(meta_tf["dataset_name"].numpy())
                        if "dataset_name" in meta_tf
                        else default_dataset_name
                    ) or default_dataset_name

                    # NOTE: For metadata-enabled pipelines we intentionally avoid TF->Torch DLPack
                    # conversion. We have observed dataset-specific segfaults when mixing DLPack
                    # conversion with subsequent `.numpy()` calls on other tensors in the same
                    # nested tf.data element (e.g., Roboturk). `.numpy()` is slower but stable.
                    pair_np = pair_tf.numpy()
                    pair_pt = torch.from_numpy(pair_np).permute(3, 0, 1, 2)

                    meta = {
                        "episode_id": episode_id,
                        "frame_idx": int(meta_tf["frame_idx"].numpy()),
                        "offset": offset,
                        "language": language,
                        "dataset_name": dataset_name,
                        "dataset_type": dataset_name,
                        "action": meta_tf["action"].numpy(),
                        "initial_state": meta_tf["initial_state"].numpy(),
                        "robot": robot,
                    }
                    yield {"frames": pair_pt, **meta}
                    continue

                # Batched output: convert one batched tensor instead of stacking N samples in PyTorch.
                episode_ids = [_decode_str(x) for x in meta_tf["episode_id"].numpy()]
                frame_idxs = [int(x) for x in meta_tf["frame_idx"].numpy()]
                languages = [_decode_str(x) for x in meta_tf["language"].numpy()]
                robots = [_decode_str(x) for x in meta_tf["robot"].numpy()]

                if "dataset_name" in meta_tf:
                    dataset_names = [
                        _decode_str(x) or default_dataset_name
                        for x in meta_tf["dataset_name"].numpy()
                    ]
                else:
                    dataset_names = [default_dataset_name for _ in episode_ids]

                pair_np = pair_tf.numpy()
                pair_pt = torch.from_numpy(pair_np).permute(0, 4, 1, 2, 3)

                actions_np = meta_tf["action"].numpy()
                initial_states_np = meta_tf["initial_state"].numpy()
                actions = [actions_np[i] for i in range(actions_np.shape[0])]
                initial_states = [initial_states_np[i] for i in range(initial_states_np.shape[0])]

                yield {
                    "frames": pair_pt,
                    "episode_id": episode_ids,
                    "frame_idx": frame_idxs,
                    "offset": offset,
                    "language": languages,
                    "dataset_name": dataset_names,
                    "dataset_type": dataset_names,
                    "action": actions,
                    "initial_state": initial_states,
                    "robot": robots,
                }
            else:
                if self.output_batch_size is None:
                    try:
                        yield torch.utils.dlpack.from_dlpack(
                            tf.experimental.dlpack.to_dlpack(item)
                        ).permute(3, 0, 1, 2)
                    except Exception:
                        pair_np = item.numpy()
                        yield torch.from_numpy(pair_np).permute(3, 0, 1, 2)
                else:
                    try:
                        yield torch.utils.dlpack.from_dlpack(
                            tf.experimental.dlpack.to_dlpack(item)
                        ).permute(0, 4, 1, 2, 3)
                    except Exception:
                        pair_np = item.numpy()
                        yield torch.from_numpy(pair_np).permute(0, 4, 1, 2, 3)


class MultiOXEFramePairDataset(IterableDataset):
    """
    PyTorch IterableDataset that interleaves frame pairs from multiple OXE datasets.

    Alternates between datasets based on weights to ensure good mixing.
    Each dataset can have different configs (action format, image size, etc.)

    Args:
        datasets: List of dataset configs, each with:
            - name: Dataset name (e.g., "bridge", "language_table")
            - train_split: TFDS split string for training
            - val_split: TFDS split string for validation
            - pair_offset_steps: Frame offset (in steps) for pairs
            - approx_num_pairs: Estimated pair count (used for weighting + __len__)
            - weight: Optional sampling weight (numeric or "proportionate")
        image_size: Target image size (shared)
        final_stream_prefetch_buffer: tf.data prefetch buffer size (after mixing)
        episode_queue_shuffle_buffer: Shuffle buffer for incoming episodes (0 to disable)
        intra_episode_sample_shuffle_buffer: Shuffle buffer for per-episode samples (0 to disable)
        global_stream_shuffle_buffer: Shuffle buffer for the final mixed stream (0 to disable)
        return_metadata: If True, return dict with metadata
        is_train: If True, use train_split; else use val_split
        num_parallel_episodes: Number of episodes to process in parallel
    """

    def __init__(
        self,
        datasets: list,
        final_stream_prefetch_buffer: int,
        episode_queue_prefetch_buffer: int,
        episode_queue_shuffle_buffer: int,
        intra_episode_sample_shuffle_buffer: int,
        global_stream_shuffle_buffer: int,
        image_size: int,
        return_metadata: bool,
        is_train: bool,
        output_batch_size: int,
        persistent_iterator: bool,  # Keep iterators alive to avoid shuffle buffer refill
        samples_per_episode: int,
        seed: Optional[int],
        debug_use_synthetic_data: bool,
        debug_synthetic_num_samples: int,
        pipeline_episode_concurrency_total: int,
        pipeline_transform_parallelism: int,
        pipeline_interleave_parallelism: int,
        mix_block_length: int,
        mix_selector_run_length: int,
        parallelism_mode: str,
        per_dataset_stream_prefetch_buffer: int,
        mixing_strategy: str,
        per_dataset_private_threadpool_size: int,  # 0 = use shared global threadpool
        tfds_read_cycle_length: int,
        tfds_read_block_length: int,
        tfds_read_decode_parallelism: int,
        tfds_read_interleave_parallelism: int,
        pair_frames_mode: str = "endpoints",
        pair_frames_stride: int = 1,
        pair_frames_n: int = 2,
        tfds_source: str = "gcs",
        tfds_local_root: Optional[str] = None,
    ):
        super().__init__()

        self.image_size = image_size
        self.dataset_configs = datasets
        self.final_stream_prefetch_buffer = final_stream_prefetch_buffer
        self.per_dataset_stream_prefetch_buffer = int(per_dataset_stream_prefetch_buffer)
        # Allow -1 to delegate to tf.data.AUTOTUNE inside each per-dataset pipeline.
        if self.per_dataset_stream_prefetch_buffer < -1:
            raise ValueError("per_dataset_stream_prefetch_buffer must be >= -1")
        self.return_metadata = return_metadata
        self.is_train = is_train
        self.output_batch_size = int(output_batch_size)
        if self.output_batch_size <= 0:
            raise ValueError("output_batch_size must be a positive integer")
        self.persistent_iterator = persistent_iterator
        self.samples_per_episode = samples_per_episode
        self.seed = seed
        self.debug_use_synthetic_data = bool(debug_use_synthetic_data)
        self.debug_synthetic_num_samples = int(debug_synthetic_num_samples)
        
        self.pipeline_episode_concurrency_total = pipeline_episode_concurrency_total
        self.pipeline_transform_parallelism = pipeline_transform_parallelism
        self.pipeline_interleave_parallelism = pipeline_interleave_parallelism
        
        self.episode_queue_prefetch_buffer = episode_queue_prefetch_buffer
        self.episode_queue_shuffle_buffer = int(episode_queue_shuffle_buffer)
        self.intra_episode_sample_shuffle_buffer = int(intra_episode_sample_shuffle_buffer)
        self.global_stream_shuffle_buffer = int(global_stream_shuffle_buffer)
        self.mix_block_length = int(mix_block_length)
        if self.mix_block_length <= 0:
            raise ValueError("mix_block_length must be a positive integer")
        self.mix_selector_run_length = int(mix_selector_run_length)
        if self.mix_selector_run_length <= 0:
            raise ValueError("mix_selector_run_length must be a positive integer")
        self.parallelism_mode = str(parallelism_mode)
        if self.parallelism_mode not in {"divide", "sqrt", "full"}:
            raise ValueError(
                "parallelism_mode must be one of: 'divide', 'sqrt', 'full'"
            )
        self.mixing_strategy = str(mixing_strategy)
        if self.mixing_strategy not in {"sample", "choose", "python"}:
            raise ValueError("mixing_strategy must be one of: 'sample', 'choose', 'python'")
        if self.mixing_strategy == "python":
            raise ValueError(
                "mixing_strategy='python' is not supported with batched OXE output. "
                "Use `data.adapter.tf.mixing.strategy=sample|choose`."
            )
        self.per_dataset_private_threadpool_size = int(per_dataset_private_threadpool_size)
        
        self.tfds_read_cycle_length = tfds_read_cycle_length
        self.tfds_read_block_length = tfds_read_block_length
        self.tfds_read_decode_parallelism = tfds_read_decode_parallelism
        self.tfds_read_interleave_parallelism = tfds_read_interleave_parallelism

        self.tfds_source = str(tfds_source)
        self.tfds_local_root = tfds_local_root

        self.pair_frames_mode = str(pair_frames_mode)
        self.pair_frames_stride = int(pair_frames_stride)
        self.pair_frames_n = int(pair_frames_n)

        # Will be populated lazily
        self._datasets = None
        self._weights = None
        self._persistent_pipeline = None
        self._pipeline_iterator = None

    def _init_datasets(self):
        """Initialize individual datasets lazily."""
        if self._datasets is not None:
            return

        # Compute weights first (from config-provided sizes) so we can drop datasets
        # with zero weight early. This makes weight=0 act as a true disable switch
        # and avoids unnecessary TFDS pipelines.
        sizes = [int(cfg["approx_num_pairs"]) for cfg in self.dataset_configs]
        weights_all = self._compute_weights(self.dataset_configs, sizes)

        kept: list[tuple[dict, float]] = []
        for cfg, w in zip(self.dataset_configs, weights_all):
            if float(w) <= 0.0:
                name = cfg.get("name", "<unknown>")
                logger.info(f"Dropping dataset '{name}' with non-positive weight={w:.6f}")
                continue
            kept.append((cfg, float(w)))

        if not kept:
            raise ValueError("All datasets have non-positive weight; nothing to sample.")

        self._datasets = []
        self._weights = [w for _, w in kept]

        # Compute per-dataset shuffle buffers for the kept datasets.
        n_datasets = len(kept)
        episode_prefetch_buffer = (
            self.episode_queue_prefetch_buffer // n_datasets if self.is_train else 0
        )
        if self.is_train and self.episode_queue_shuffle_buffer > 0:
            episode_buffer_per_ds = max(
                1, int(self.episode_queue_shuffle_buffer) // n_datasets
            )
        else:
            episode_buffer_per_ds = 0

        # Avoid per-dataset global stream shuffles when mixing datasets. Apply one
        # global blender after mixing (see `_create_tf_pipeline()`).
        global_stream_shuffle_buffer_per_ds = 0

        # Avoid oversubscribing CPU/threadpools: each underlying dataset has its own
        # interleave/map parallelism. If we pass the full parallelism to every dataset,
        # total concurrency multiplies by number of datasets and can slow down.
        if self.parallelism_mode == "full":
            divisor = 1.0
        elif self.parallelism_mode == "sqrt":
            divisor = float(n_datasets) ** 0.5
        else:
            divisor = float(n_datasets)

        tfds_read_cycle_length = max(
            1, int(float(self.tfds_read_cycle_length) / divisor)
        )
        # Preserve tf.data.AUTOTUNE (-1) for per-dataset pipelines when explicitly requested.
        # Otherwise, scale down to avoid oversubscribing CPU/threadpools.
        if int(self.pipeline_episode_concurrency_total) == -1:
            pipeline_episode_concurrency = -1
        else:
            pipeline_episode_concurrency = max(
                1, int(float(self.pipeline_episode_concurrency_total) / divisor)
            )
        # Preserve tf.data.AUTOTUNE (-1) for per-dataset pipelines when explicitly requested.
        # Otherwise, scale down to avoid oversubscribing CPU/threadpools.
        if int(self.pipeline_transform_parallelism) == -1:
            pipeline_transform_parallelism = -1
        else:
            pipeline_transform_parallelism = max(
                1, int(float(self.pipeline_transform_parallelism) / divisor)
            )
        if int(self.pipeline_interleave_parallelism) == -1:
            pipeline_interleave_parallelism = -1
        else:
            pipeline_interleave_parallelism = max(
                1, int(float(self.pipeline_interleave_parallelism) / divisor)
            )

        # Prefetching and mixing interaction:
        #
        # - When mixing per-sample (`mix_block_length=1`), a small *per-dataset* prefetch
        #   can help keep `sample_from_datasets()` from blocking on dataset switches.
        # - When mixing in blocks (`mix_block_length>1`), per-dataset prefetch is only
        #   effective if it buffers *blocks*, not individual samples. Otherwise, the
        #   mixer may still block while waiting for the next block to be assembled.
        #
        # To reflect that, we interpret `per_dataset_stream_prefetch_buffer` as:
        # - samples (when mix_block_length <= 1)
        # - blocks  (when mix_block_length > 1 and n_datasets > 1)
        mix_block_length = int(self.mix_block_length)
        use_block_prefetch = mix_block_length > 1 and len(kept) > 1

        prefetch_buffer_per_ds_samples = (
            0 if use_block_prefetch else int(self.per_dataset_stream_prefetch_buffer)
        )
        self._per_dataset_block_prefetch_buffer = (
            int(self.per_dataset_stream_prefetch_buffer) if use_block_prefetch else 0
        )

        output_action_dim = None
        output_state_dim = None
        if self.return_metadata:
            max_action_dim = 0
            max_state_dim = 0
            for cfg, _w in kept:
                name = str(cfg.get("name"))
                if name not in OXE_DATASETS:
                    raise ValueError(f"Unknown dataset in config: {name}")
                max_action_dim = max(max_action_dim, int(OXE_DATASETS[name].action_dim))
                max_state_dim = max(max_state_dim, int(OXE_DATASETS[name].state_dim))
            output_action_dim = max_action_dim
            output_state_dim = max_state_dim

        for cfg, _w in kept:

            # Get split based on train/val mode
            split = cfg["train_split"] if self.is_train else cfg["val_split"]

            precomputed_size = int(cfg["approx_num_pairs"])

            ds = OXEFramePairDataset(
                image_size=self.image_size,
                dataset_name=cfg["name"],
                split=split,
                offset=int(cfg["pair_offset_steps"]),
                final_stream_prefetch_buffer=prefetch_buffer_per_ds_samples,
                return_metadata=self.return_metadata,
                is_train=bool(self.is_train),
                output_batch_size=None,  # batch after mixing (one place)
                output_action_dim=output_action_dim,
                output_state_dim=output_state_dim,
                persistent_iterator=self.persistent_iterator,
                samples_per_episode=self.samples_per_episode,
                seed=self._get_dataset_seed(cfg),
                debug_use_synthetic_data=self.debug_use_synthetic_data,
                debug_synthetic_num_samples=self.debug_synthetic_num_samples,
                precomputed_size=precomputed_size,
                episode_queue_prefetch_buffer=episode_prefetch_buffer,
                episode_queue_shuffle_buffer=episode_buffer_per_ds,
                intra_episode_sample_shuffle_buffer=self.intra_episode_sample_shuffle_buffer,
                private_threadpool_size=self.per_dataset_private_threadpool_size,
                tfds_read_cycle_length=tfds_read_cycle_length,
                tfds_read_block_length=self.tfds_read_block_length,
                tfds_read_decode_parallelism=self.tfds_read_decode_parallelism,
                tfds_read_interleave_parallelism=self.tfds_read_interleave_parallelism,
                pipeline_episode_concurrency=pipeline_episode_concurrency,
                pipeline_transform_parallelism=pipeline_transform_parallelism,
                pipeline_interleave_parallelism=pipeline_interleave_parallelism,
                pair_frames_mode=self.pair_frames_mode,
                pair_frames_stride=self.pair_frames_stride,
                pair_frames_n=self.pair_frames_n,
                tfds_source=self.tfds_source,
                tfds_local_root=self.tfds_local_root,
            )
            self._datasets.append(ds)

    def _get_dataset_seed(self, cfg: dict) -> Optional[int]:
        # If an explicit per-dataset seed is provided, use it as-is.
        if "seed" in cfg and cfg["seed"] is not None:
            return int(cfg["seed"])
        if self.seed is None:
            return None
        # Derive deterministic per-dataset seed from global seed + dataset name.
        name_hash = zlib.crc32(str(cfg.get("name", "")).encode("utf-8")) & 0x7FFFFFFF
        return (int(self.seed) + int(name_hash)) & 0x7FFFFFFF

    def reset_iterator(self):
        """Force reset the underlying iterator (useful for cyclic sampling)."""
        self._pipeline_iterator = None
        if self._persistent_pipeline is not None:
            self._pipeline_iterator = iter(self._persistent_pipeline)

    def _create_tf_pipeline(self):
        """
        Create a single tf.data pipeline that mixes datasets with weights.

        This avoids Python-side per-sample selection (which blocks parallelism)
        and allows tf.data to overlap I/O and preprocessing across datasets.
        """
        tf = _import_tensorflow_cpu_only()

        self._init_datasets()

        tf_datasets = [ds._get_or_create_pipeline() for ds in self._datasets]
        weights = [float(w) for w in self._weights]

        # Reduce per-element cross-dataset switching overhead by sampling blocks from the
        # same dataset and unbatching afterwards. This keeps the external element type
        # identical while amortizing `sample_from_datasets()` bookkeeping.
        mix_block_length = int(self.mix_block_length)
        direct_batch_mixing = (
            mix_block_length > 1
            and len(tf_datasets) > 1
            and int(self.output_batch_size) == mix_block_length
        )
        if mix_block_length > 1 and len(tf_datasets) > 1:
            tf_datasets = [
                ds.batch(mix_block_length, drop_remainder=True) for ds in tf_datasets
            ]
            # Optional: prefetch *blocks* per dataset, so switches do not have to
            # synchronously assemble the next block on demand.
            block_prefetch = int(getattr(self, "_per_dataset_block_prefetch_buffer", 0))
            if block_prefetch == -1:
                tf_datasets = [ds.prefetch(tf.data.AUTOTUNE) for ds in tf_datasets]
            elif block_prefetch > 0:
                tf_datasets = [ds.prefetch(block_prefetch) for ds in tf_datasets]

        seed = int(self.seed) if self.seed is not None else None
        if len(tf_datasets) == 1:
            mixed = tf_datasets[0]
        elif self.mixing_strategy == "choose":
            # Approximate weighted sampling by repeating indices proportional to weights
            # and shuffling the selector. This can be faster than
            # `sample_from_datasets()` in some settings.
            scale = 100
            counts = [max(1, int(round(w * scale))) for w in weights]
            selector_indices = []
            for i, c in enumerate(counts):
                selector_indices.extend([i] * int(c))
            selector_ds = tf.data.Dataset.from_tensor_slices(
                tf.constant(selector_indices, dtype=tf.int64)
            ).repeat()
            if self.is_train:
                if seed is None:
                    selector_ds = selector_ds.shuffle(len(selector_indices))
                else:
                    selector_ds = selector_ds.shuffle(len(selector_indices), seed=seed)
            # Reduce cross-dataset switching by repeating each selector choice for a
            # fixed run length. This keeps memory constant (unlike increasing
            # `mix_block_length`) while reducing "cold switch" stalls on GCS.
            selector_run = int(self.mix_selector_run_length)
            if selector_run > 1:
                selector_ds = selector_ds.flat_map(
                    lambda idx: tf.data.Dataset.from_tensors(idx).repeat(selector_run)
                )
            mixed = tf.data.Dataset.choose_from_datasets(tf_datasets, selector_ds)
        else:
            if seed is None:
                mixed = tf.data.Dataset.sample_from_datasets(
                    tf_datasets,
                    weights=weights,
                    stop_on_empty_dataset=not self.is_train,
                )
            else:
                mixed = tf.data.Dataset.sample_from_datasets(
                    tf_datasets,
                    weights=weights,
                    seed=seed,
                    stop_on_empty_dataset=not self.is_train,
                )

        # If we mixed batches of exactly the training batch size, keep them as-is
        # to avoid an unbatch->batch roundtrip.
        if (mix_block_length > 1 and len(tf_datasets) > 1) and (not direct_batch_mixing):
            mixed = mixed.unbatch()

        options = tf.data.Options()
        options.experimental_deterministic = False
        options.experimental_slack = True

        mixed = mixed.with_options(options)

        if self.is_train:
            mixed = mixed.repeat()

            if self.global_stream_shuffle_buffer > 0:
                if seed is None:
                    mixed = mixed.shuffle(self.global_stream_shuffle_buffer)
                else:
                    mixed = mixed.shuffle(self.global_stream_shuffle_buffer, seed=seed)

        if not direct_batch_mixing:
            mixed = mixed.batch(self.output_batch_size, drop_remainder=bool(self.is_train))

        prefetch_buffer = int(self.final_stream_prefetch_buffer)
        if prefetch_buffer == -1:
            mixed = mixed.prefetch(tf.data.AUTOTUNE)
        elif prefetch_buffer > 0:
            mixed = mixed.prefetch(prefetch_buffer)

        return mixed

    def _get_or_create_pipeline(self):
        if self._persistent_pipeline is None:
            self._persistent_pipeline = self._create_tf_pipeline()
        return self._persistent_pipeline

    def _get_or_create_iterator(self):
        tf_ds = self._get_or_create_pipeline()
        if self.persistent_iterator:
            if self._pipeline_iterator is None:
                self._pipeline_iterator = iter(tf_ds)
            return self._pipeline_iterator
        return iter(tf_ds)

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
        """Approximate total length in yielded batches across all datasets."""
        self._init_datasets()
        total_pairs = sum(len(ds) for ds in self._datasets)
        if self.is_train:
            return max(1, int(total_pairs) // int(self.output_batch_size))
        return int(math.ceil(float(total_pairs) / float(self.output_batch_size)))

    def cleanup(self, *, ignore_errors: bool = False) -> None:
        """
        Explicitly release TensorFlow resources from all underlying datasets.

        Call this when you are done with the dataset and want to free memory.
        This method is idempotent - safe to call multiple times.

        """
        datasets_to_cleanup = self._datasets
        self._datasets = None  # Clear reference first to prevent re-entry
        self._weights = None
        self._persistent_pipeline = None
        self._pipeline_iterator = None

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

    def _iter_python_mixing(self, tf, _decode_str):
        """
        Python-level mixing: iterate individual dataset tf.data pipelines and mix
        samples using Python random selection instead of tf.data.sample_from_datasets().

        This can avoid tf.data coordination overhead that occurs when sample_from_datasets()
        switches between pipelines, at the cost of some Python overhead.
        """
        import random as _random

        self._init_datasets()

        # Create iterators for each dataset's tf.data pipeline
        iterators = [iter(ds._get_or_create_pipeline()) for ds in self._datasets]
        weights = list(self._weights)
        rng = _random.Random(self.seed) if self.seed is not None else _random.Random()

        # Loop for __len__ steps to define epoch length
        steps_total = len(self)

        for _ in range(steps_total):
            # Choose dataset based on weights using cumulative probability
            r = rng.random()
            cumsum = 0.0
            dataset_idx = 0
            for i, w in enumerate(weights):
                cumsum += w
                if r <= cumsum:
                    dataset_idx = i
                    break

            # Fetch item from chosen dataset, cycling if exhausted
            try:
                item = next(iterators[dataset_idx])
            except StopIteration:
                # Dataset exhausted: reset its iterator and retry
                self._datasets[dataset_idx].reset_iterator()
                iterators[dataset_idx] = iter(self._datasets[dataset_idx]._get_or_create_pipeline())
                try:
                    item = next(iterators[dataset_idx])
                except StopIteration:
                    continue  # Failsafe if dataset is truly empty

            # Convert item to PyTorch format (same as main __iter__)
            if self.return_metadata:
                pair_tf, meta_tf = item
                pair_np = pair_tf.numpy()
                pair_pt = torch.from_numpy(pair_np).permute(3, 0, 1, 2)

                meta = {
                    "episode_id": _decode_str(meta_tf["episode_id"].numpy()),
                    "frame_idx": int(meta_tf["frame_idx"].numpy()),
                    "offset": int(meta_tf["offset"].numpy()),
                    "language": _decode_str(meta_tf["language"].numpy()),
                    "dataset_type": _decode_str(meta_tf["dataset_type"].numpy()),
                    "dataset_name": _decode_str(meta_tf["dataset_name"].numpy()),
                    "action": meta_tf["action"].numpy(),
                    "initial_state": meta_tf["initial_state"].numpy(),
                    "robot": _decode_str(meta_tf["robot"].numpy()),
                }
                yield {"frames": pair_pt, **meta}
            else:
                try:
                    yield torch.utils.dlpack.from_dlpack(
                        tf.experimental.dlpack.to_dlpack(item)
                    ).permute(3, 0, 1, 2)
                except Exception:
                    pair_np = item.numpy()
                    yield torch.from_numpy(pair_np).permute(3, 0, 1, 2)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup(ignore_errors=True)
        except Exception:
            pass

    def __iter__(self):
        tf = _import_tensorflow_cpu_only()

        def _decode_str(val) -> str:
            if isinstance(val, bytes):
                return val.decode("utf-8").rstrip("\x00")
            return str(val) if val else ""

        tf_iter = self._get_or_create_iterator()

        for item in tf_iter:
            if self.return_metadata:
                pair_tf, meta_tf = item
                pair_np = pair_tf.numpy()
                pair_pt = torch.from_numpy(pair_np).permute(0, 4, 1, 2, 3)

                episode_ids = [_decode_str(x) for x in meta_tf["episode_id"].numpy()]
                frame_idxs = [int(x) for x in meta_tf["frame_idx"].numpy()]

                offsets = meta_tf["offset"].numpy()
                offset = int(offsets[0]) if len(offsets) else 0

                languages = [_decode_str(x) for x in meta_tf["language"].numpy()]
                dataset_types = [_decode_str(x) for x in meta_tf["dataset_type"].numpy()]
                dataset_names = [_decode_str(x) for x in meta_tf["dataset_name"].numpy()]
                robots = [_decode_str(x) for x in meta_tf["robot"].numpy()]

                actions_np = meta_tf["action"].numpy()
                initial_states_np = meta_tf["initial_state"].numpy()
                actions = [actions_np[i] for i in range(actions_np.shape[0])]
                initial_states = [initial_states_np[i] for i in range(initial_states_np.shape[0])]

                yield {
                    "frames": pair_pt,
                    "episode_id": episode_ids,
                    "frame_idx": frame_idxs,
                    "offset": offset,
                    "language": languages,
                    "dataset_type": dataset_types,
                    "dataset_name": dataset_names,
                    "action": actions,
                    "initial_state": initial_states,
                    "robot": robots,
                }
            else:
                try:
                    yield torch.utils.dlpack.from_dlpack(
                        tf.experimental.dlpack.to_dlpack(item)
                    ).permute(0, 4, 1, 2, 3)
                except Exception:
                    pair_np = item.numpy()
                    yield torch.from_numpy(pair_np).permute(0, 4, 1, 2, 3)


def get_oxe_dataset_info(
    dataset_name: str = "language_table",
    *,
    tfds_source: str = "gcs",
    tfds_local_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Get information about an OXE dataset."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    if dataset_name not in OXE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = OXE_DATASETS[dataset_name]
    builder_dir = _resolve_tfds_builder_dir(
        gcs_builder_dir=config.gcs_path, source=tfds_source, local_root=tfds_local_root
    )
    builder = tfds.builder_from_directory(builder_dir)

    return {
        "name": config.name,
        "gcs_path": config.gcs_path,
        "builder_dir": builder_dir,
        "splits": {
            name: split.num_examples for name, split in builder.info.splits.items()
        },
        "image_shape": config.image_shape,
        "control_frequency_hz": config.control_frequency_hz,
        "features": str(builder.info.features),
    }
