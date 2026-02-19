"""
Shared OXE infrastructure: dataset registry, TFDS resolution, and helpers.

Used by both oxe.py (v1) and oxe_v2.py (v2) adapters.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TensorFlow CPU-only import guard
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset config & registry
# ---------------------------------------------------------------------------


@dataclass
class OXEDatasetConfig:
    """Configuration for an OXE dataset."""

    name: str
    gcs_path: str
    image_key: str = "rgb"
    instruction_key: str = "instruction"
    state_key: Optional[str] = "effector_translation"
    image_shape: Tuple[int, int, int] = (360, 640, 3)
    control_frequency_hz: float = 10.0
    action_dim: int = 2
    state_dim: int = 2
    action_key: Optional[str] = None
    action_is_dict: bool = False
    instruction_in_step: bool = False
    robot_key: Optional[str] = None
    avg_episode_length: int = 30
    allow_missing_state: bool = False


OXE_DATASETS = {
    "language_table": OXEDatasetConfig(
        name="language_table",
        gcs_path="gs://gresearch/robotics/language_table/0.1.0",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
        avg_episode_length=40,
    ),
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
    "bridge": OXEDatasetConfig(
        name="bridge",
        gcs_path="gs://gresearch/robotics/bridge/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        state_key="state",
        image_shape=(480, 640, 3),
        control_frequency_hz=5.0,
        action_dim=3,
        state_dim=2,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=50,
    ),
    "rt1": OXEDatasetConfig(
        name="rt1",
        gcs_path="gs://gresearch/robotics/fractal20220817_data/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        state_key="base_pose_tool_reached",
        image_shape=(256, 320, 3),
        control_frequency_hz=3.0,
        action_dim=3,
        state_dim=3,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=30,
    ),
    "robonet": OXEDatasetConfig(
        name="robonet",
        gcs_path="gs://gresearch/robotics/robo_net/0.1.0",
        image_key="image",
        instruction_key="language_instruction",
        state_key="state",
        image_shape=(240, 320, 3),
        control_frequency_hz=10.0,
        action_dim=3,
        state_dim=3,
        action_key=None,
        action_is_dict=False,
        instruction_in_step=True,
        robot_key="robot",
        avg_episode_length=30,
    ),
    "aloha_mobile": OXEDatasetConfig(
        name="aloha_mobile",
        gcs_path="gs://gresearch/robotics/aloha_mobile/0.0.1",
        image_key="cam_high",
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        image_shape=(480, 640, 3),
        control_frequency_hz=10.0,
        action_dim=16,
        state_dim=14,
        action_is_dict=False,
        avg_episode_length=200,
    ),
    "droid": OXEDatasetConfig(
        name="droid",
        gcs_path="gs://gresearch/robotics/droid/1.0.1",
        image_key="exterior_image_1_left",
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="cartesian_position",
        image_shape=(180, 320, 3),
        control_frequency_hz=10.0,
        action_dim=7,
        state_dim=6,
        action_is_dict=False,
        avg_episode_length=300,
    ),
    "berkeley_autolab_ur5": OXEDatasetConfig(
        name="berkeley_autolab_ur5",
        gcs_path="gs://gresearch/robotics/berkeley_autolab_ur5/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="robot_state",
        image_shape=(480, 640, 3),
        control_frequency_hz=10.0,
        action_dim=3,
        state_dim=15,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=90,
    ),
    "jaco_play": OXEDatasetConfig(
        name="jaco_play",
        gcs_path="gs://gresearch/robotics/jaco_play/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="end_effector_cartesian_pos",
        image_shape=(224, 224, 3),
        control_frequency_hz=10.0,
        action_dim=3,
        state_dim=7,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=70,
    ),
    "kuka": OXEDatasetConfig(
        name="kuka",
        gcs_path="gs://gresearch/robotics/kuka/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key=None,
        image_shape=(512, 640, 3),
        control_frequency_hz=10.0,
        action_dim=3,
        state_dim=0,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=15,
        allow_missing_state=True,
    ),
    "taco_play": OXEDatasetConfig(
        name="taco_play",
        gcs_path="gs://gresearch/robotics/taco_play/0.1.0",
        image_key="rgb_static",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="robot_obs",
        image_shape=(150, 200, 3),
        control_frequency_hz=10.0,
        action_dim=7,
        state_dim=15,
        action_key="actions",
        action_is_dict=True,
        avg_episode_length=60,
    ),
    "roboturk": OXEDatasetConfig(
        name="roboturk",
        gcs_path="gs://gresearch/robotics/roboturk/0.1.0",
        image_key="front_rgb",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key=None,
        image_shape=(480, 640, 3),
        control_frequency_hz=10.0,
        action_dim=3,
        state_dim=0,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=90,
        allow_missing_state=True,
    ),
    "bc_z": OXEDatasetConfig(
        name="bc_z",
        gcs_path="gs://gresearch/robotics/bc_z/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="present/xyz",
        state_dim=3,
        image_shape=(171, 213, 3),
        action_key="future/xyz_residual",
        action_is_dict=True,
        action_dim=30,
        avg_episode_length=30,
    ),
    "berkeley_cable_routing": OXEDatasetConfig(
        name="berkeley_cable_routing",
        gcs_path="gs://gresearch/robotics/berkeley_cable_routing/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="robot_state",
        state_dim=2,
        image_shape=(128, 128, 3),
        action_key="world_vector",
        action_is_dict=True,
        action_dim=3,
        avg_episode_length=30,
    ),
    "columbia_cairlab_pusht_real": OXEDatasetConfig(
        name="columbia_cairlab_pusht_real",
        gcs_path="gs://gresearch/robotics/columbia_cairlab_pusht_real/0.1.0",
        image_key="image",
        image_shape=(240, 320, 3),
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="robot_state",
        state_dim=2,
        action_key="world_vector",
        action_is_dict=True,
        action_dim=3,
        avg_episode_length=30,
    ),
    "mimic_play": OXEDatasetConfig(
        name="mimic_play",
        gcs_path="gs://gresearch/robotics/mimic_play/0.0.1",
        image_key="image/front_image_1",
        image_shape=(120, 120, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state/ee_pose",
        state_dim=7,
        action_dim=7,
        avg_episode_length=200,
    ),
    "berkeley_fanuc_manipulation": OXEDatasetConfig(
        name="berkeley_fanuc_manipulation",
        gcs_path="gs://gresearch/robotics/berkeley_fanuc_manipulation/0.1.0",
        image_key="image",
        image_shape=(224, 224, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=6,
        avg_episode_length=60,
    ),
    "dobbe": OXEDatasetConfig(
        name="dobbe",
        gcs_path="gs://gresearch/robotics/dobbe/0.0.1",
        image_key="wrist_image",
        image_shape=(256, 256, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
    "uiuc_d3field": OXEDatasetConfig(
        name="uiuc_d3field",
        gcs_path="gs://gresearch/robotics/uiuc_d3field/0.1.0",
        image_key="image_1",
        image_shape=(360, 640, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=3,
        avg_episode_length=60,
    ),
    "ucsd_kitchen_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="ucsd_kitchen_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(480, 640, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=8,
        avg_episode_length=60,
    ),
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="ucsd_pick_and_place_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/ucsd_pick_and_place_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(224, 224, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=4,
        avg_episode_length=60,
    ),
    "furniture_bench_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="furniture_bench_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/furniture_bench_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(224, 224, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=8,
        avg_episode_length=60,
    ),
    "maniskill_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="maniskill_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(256, 256, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
    "robo_set": OXEDatasetConfig(
        name="robo_set",
        gcs_path="gs://gresearch/robotics/robo_set/0.0.1",
        image_key="image_left",
        image_shape=(240, 424, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=8,
        avg_episode_length=60,
    ),
    "stanford_hydra_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="stanford_hydra_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/stanford_hydra_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(240, 320, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
    "stanford_robocook_converted_externally_to_rlds": OXEDatasetConfig(
        name="stanford_robocook_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/stanford_robocook_converted_externally_to_rlds/0.1.0",
        image_key="image_1",
        image_shape=(256, 256, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
    "spoc": OXEDatasetConfig(
        name="spoc",
        gcs_path="gs://gresearch/robotics/spoc/0.0.1",
        image_key="image",
        image_shape=(224, 384, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key=None,
        state_dim=0,
        allow_missing_state=True,
        action_dim=9,
        avg_episode_length=60,
    ),
    "tidybot": OXEDatasetConfig(
        name="tidybot",
        gcs_path="gs://gresearch/robotics/tidybot/0.0.1",
        image_key="image",
        image_shape=(360, 640, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        action_dim=0,
        state_key=None,
        state_dim=0,
        allow_missing_state=True,
        avg_episode_length=60,
    ),
    "toto": OXEDatasetConfig(
        name="toto",
        gcs_path="gs://gresearch/robotics/toto/0.1.0",
        image_key="image",
        image_shape=(480, 640, 3),
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="state",
        state_dim=2,
        action_key="world_vector",
        action_is_dict=True,
        action_dim=3,
        avg_episode_length=60,
    ),
    "viola": OXEDatasetConfig(
        name="viola",
        gcs_path="gs://gresearch/robotics/viola/0.1.0",
        image_key="agentview_rgb",
        image_shape=(224, 224, 3),
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key=None,
        state_dim=0,
        allow_missing_state=True,
        action_key="world_vector",
        action_is_dict=True,
        action_dim=3,
        avg_episode_length=60,
    ),
    "vima_converted_externally_to_rlds": OXEDatasetConfig(
        name="vima_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/vima_converted_externally_to_rlds/0.0.1",
        image_key="image",
        image_shape=(128, 256, 3),
        instruction_key="multimodal_instruction",
        instruction_in_step=True,
        state_key=None,
        state_dim=0,
        allow_missing_state=True,
        action_key="pose0_position",
        action_is_dict=True,
        action_dim=3,
        avg_episode_length=60,
    ),
    "utaustin_mutex": OXEDatasetConfig(
        name="utaustin_mutex",
        gcs_path="gs://gresearch/robotics/utaustin_mutex/0.1.0",
        image_key="image",
        image_shape=(128, 128, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
    "fmb": OXEDatasetConfig(
        name="fmb",
        gcs_path="gs://gresearch/robotics/fmb/0.0.1",
        image_key="image_side_1",
        image_shape=(256, 256, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state_gripper_pose",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
}


# ---------------------------------------------------------------------------
# TFDS builder resolution
# ---------------------------------------------------------------------------


def _parse_tfds_prepared_dataset_dir(builder_dir: str) -> tuple[str, str]:
    parts = [p for p in str(builder_dir).rstrip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"Invalid TFDS builder dir (expected .../<dataset>/<version>): {builder_dir!r}"
        )
    return parts[-2], parts[-1]


def _resolve_tfds_builder_dir(
    *, gcs_builder_dir: str, source: str, local_root: Optional[str]
) -> str:
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

    return local_builder_dir if Path(local_builder_dir).exists() else str(gcs_builder_dir)


# ---------------------------------------------------------------------------
# Frame-pair index computation
# ---------------------------------------------------------------------------


def compute_pair_frame_indices(
    offset: int, mode: str = "endpoints", stride: int = 1, n: int = 2
) -> list[int]:
    """
    Compute step indices in [0, offset] to include in frame-pair output.

    Contract:
    - indices[0] == 0, indices[-1] == offset
    - len(indices) >= 2 when offset > 0
    - non-decreasing order
    """
    offset = int(offset)
    if offset <= 0:
        raise ValueError("offset must be >= 1 to form frame pairs")

    if mode == "endpoints":
        return [0, offset]
    if mode == "all":
        return list(range(0, offset + 1))
    if mode == "stride":
        stride = int(stride)
        if stride <= 0:
            raise ValueError("pair_frames_stride must be >= 1")
        idxs = list(range(0, offset + 1, stride))
        if not idxs or idxs[0] != 0:
            idxs = [0] + idxs
        if idxs[-1] != offset:
            idxs.append(offset)
        return idxs
    if mode == "fixed_n":
        n = int(n)
        if n < 2:
            raise ValueError("pair_frames_n must be >= 2 when mode='fixed_n'")
        idxs = np.rint(np.linspace(0, offset, num=n)).astype(np.int32).tolist()
        if not idxs:
            idxs = [0, offset]
        idxs[0] = 0
        idxs[-1] = offset
        idxs = [int(min(offset, max(0, x))) for x in idxs]
        idxs.sort()
        return idxs

    raise ValueError(f"Unknown pair_frames_mode: {mode}")


# ---------------------------------------------------------------------------
# TF helper functions (used inside tf.data pipelines)
# ---------------------------------------------------------------------------


def pad_or_truncate_1d(vec, target_dim: int):
    """Pad/truncate a 1D tensor to exactly target_dim."""
    import tensorflow as tf

    vec = tf.convert_to_tensor(vec)
    vec = vec[:target_dim]
    pad = tf.maximum(0, target_dim - tf.shape(vec)[0])
    vec = tf.pad(vec, paddings=[[0, pad]])
    return tf.ensure_shape(vec, [target_dim])


def strip_null_bytes(s):
    """Remove trailing null bytes from a tf.string tensor."""
    import tensorflow as tf

    s = tf.convert_to_tensor(s, dtype=tf.string)
    return tf.strings.regex_replace(s, "\x00+$", "")


def resolve_nested_key(container, keypath: str):
    """Navigate a nested dict/tensor structure via slash-separated key path.

    Some OXE TFDS datasets use literal keys containing '/' (e.g.
    'future/xyz_residual'). Try a direct lookup first; only treat '/' as a
    nesting delimiter if the full key is not present.
    """
    if "/" not in keypath:
        return container[keypath]
    if keypath in container:
        return container[keypath]
    cur = container
    for part in keypath.split("/"):
        cur = cur[part]
    return cur


# ---------------------------------------------------------------------------
# Dataset info query
# ---------------------------------------------------------------------------


def get_oxe_dataset_info(
    dataset_name: str = "language_table",
    *,
    tfds_source: str = "gcs",
    tfds_local_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Get information about an OXE dataset."""
    import tensorflow_datasets as tfds

    tf = _import_tensorflow_cpu_only()

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
