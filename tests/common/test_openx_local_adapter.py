import io
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from common.adapters.openx_local import OpenXLocalIndexedPairIterable
from common.data_factory import create_datamodule


def _jpeg_bytes(seed: int, h: int = 24, w: int = 24) -> bytes:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = (seed * 17) % 255
    arr[..., 1] = (seed * 31) % 255
    arr[..., 2] = (seed * 47) % 255
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_episode(ep_seed: int, n_steps: int) -> dict:
    steps = []
    for i in range(n_steps):
        steps.append(
            {
                "action": {
                    "world_vector": np.asarray([1.0, 0.0, -1.0], dtype=np.float32),
                    "rotation_delta": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                    "gripper_closedness_action": np.asarray([0.5], dtype=np.float32),
                },
                "is_first": i == 0,
                "is_last": i == n_steps - 1,
                "is_terminal": i == n_steps - 1,
                "observation": {
                    "image": _jpeg_bytes(ep_seed * 100 + i),
                    "natural_language_instruction": b"push the object",
                    "state": np.asarray([float(i), float(i + 1)], dtype=np.float32),
                },
                "reward": 0.0,
            }
        )
    return {"steps": steps}


def _make_episode_with_image_key(ep_seed: int, n_steps: int, image_key: str) -> dict:
    steps = []
    for i in range(n_steps):
        steps.append(
            {
                "action": {
                    "world_vector": np.asarray([1.0, 0.0, -1.0], dtype=np.float32),
                    "rotation_delta": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                    "gripper_closedness_action": np.asarray([0.5], dtype=np.float32),
                },
                "is_first": i == 0,
                "is_last": i == n_steps - 1,
                "is_terminal": i == n_steps - 1,
                "observation": {
                    image_key: _jpeg_bytes(ep_seed * 100 + i),
                    "natural_language_instruction": b"push the object",
                    "state": np.asarray([float(i), float(i + 1)], dtype=np.float32),
                },
                "reward": 0.0,
            }
        )
    return {"steps": steps}


def _write_shard(
    root: Path, dataset_name: str, shard_name: str, episodes: list[dict]
) -> Path:
    dataset_dir = root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shard_path = dataset_dir / shard_name
    with tarfile.open(shard_path, "w") as tf:
        for i, episode in enumerate(episodes):
            member_name = f"sample_{i:012d}.data.pickle"
            payload = pickle.dumps(episode, protocol=pickle.HIGHEST_PROTOCOL)
            info = tarfile.TarInfo(name=member_name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return shard_path


def test_openx_local_indexed_iterable_emits_standard_sample(tmp_path: Path):
    root = tmp_path / "openx"
    _write_shard(
        root=root,
        dataset_name="bridge",
        shard_name="bridge_00000.tar",
        episodes=[_make_episode(ep_seed=0, n_steps=5), _make_episode(ep_seed=1, n_steps=5)],
    )

    ds = OpenXLocalIndexedPairIterable(
        root=str(root),
        dataset_entries=[
            {
                "name": "bridge",
                "train_split": "train[:100%]",
                "val_split": "train[50%:]",
                "pair_offset_steps": 1,
                "weight": 1.0,
                "approx_num_pairs": None,
            }
        ],
        split_key="train_split",
        image_size=32,
        return_metadata=True,
        max_shards_per_dataset=None,
        pairs_per_episode=None,
        index_workers=1,
        index_cache_dir=None,
        seed=42,
        resample_each_epoch=True,
        stopping_strategy="all_exhausted",
    )

    assert len(ds) == 8  # 2 episodes * (5 - 1)

    sample = next(iter(ds))
    assert sample["frames"].shape == (3, 2, 32, 32)
    assert sample["dataset_name"] == "bridge"
    assert isinstance(sample["language"], str)
    assert sample["action"].shape == (3,)
    assert sample["initial_state"].shape == (2,)
    assert isinstance(sample["episode_id"], str)
    assert isinstance(sample["frame_idx"], int)


def test_openx_local_indexed_supports_unknown_dataset_name(tmp_path: Path):
    root = tmp_path / "openx"
    dataset_name = "nyu_door_opening_surprising_effectiveness"
    _write_shard(
        root=root,
        dataset_name=dataset_name,
        shard_name=f"{dataset_name}_00000.tar",
        episodes=[_make_episode(ep_seed=2, n_steps=4)],
    )

    ds = OpenXLocalIndexedPairIterable(
        root=str(root),
        dataset_entries=[
            {
                "name": dataset_name,
                "train_split": "train",
                "val_split": "train",
                "pair_offset_steps": 1,
                "weight": 1.0,
                "approx_num_pairs": None,
            }
        ],
        split_key="train_split",
        image_size=24,
        return_metadata=True,
        max_shards_per_dataset=None,
        pairs_per_episode=None,
        index_workers=1,
        index_cache_dir=None,
        seed=1,
        resample_each_epoch=True,
        stopping_strategy="all_exhausted",
    )

    sample = next(iter(ds))
    assert sample["dataset_name"] == dataset_name
    assert sample["frames"].shape == (3, 2, 24, 24)


def test_openx_local_datamodule_via_data_factory(tmp_path: Path):
    root = tmp_path / "openx"
    _write_shard(
        root=root,
        dataset_name="bridge",
        shard_name="bridge_00000.tar",
        episodes=[_make_episode(ep_seed=3, n_steps=6), _make_episode(ep_seed=4, n_steps=6)],
    )

    cfg_data = {
        "backend": "oxe_local_indexed",
        "preprocess": {"image_size": 32, "return_metadata": True},
        "loader": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": None,
        },
        "dataset": {
            "oxe": {
                "datasets": [
                    {
                        "name": "bridge",
                        "train_split": "train[:100%]",
                        "val_split": "train[50%:]",
                        "pair_offset_steps": 1,
                        "weight": 1.0,
                        "approx_num_pairs": None,
                    }
                ]
            }
        },
        "adapter": {
            "openx_local": {
                "root": str(root),
                "mode": "indexed",
                "max_shards_per_dataset": 0,
                "pairs_per_episode": 0,
                "index_workers": 1,
                "index_cache_dir": str(tmp_path / "index_cache"),
                "index_rebuild": False,
                "index_max_open_shards": 8,
                "weights_by_size": False,
                "max_pairs": None,
                "seed": 123,
                "resample_each_epoch": True,
                "stopping_strategy": "all_exhausted",
            }
        },
    }

    dm = create_datamodule(cfg_data)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    assert batch["frames"].shape == (2, 3, 2, 32, 32)
    assert len(batch["episode_id"]) == 2
    assert len(batch["frame_idx"]) == 2
    assert len(batch["dataset_name"]) == 2
    assert len(batch["language"]) == 2


def test_openx_local_index_cache_written(tmp_path: Path):
    root = tmp_path / "openx"
    cache_dir = tmp_path / "index_cache"
    _write_shard(
        root=root,
        dataset_name="bridge",
        shard_name="bridge_00000.tar",
        episodes=[_make_episode(ep_seed=5, n_steps=5)],
    )

    ds = OpenXLocalIndexedPairIterable(
        root=str(root),
        dataset_entries=[
            {
                "name": "bridge",
                "train_split": "train[:100%]",
                "val_split": "train[:100%]",
                "pair_offset_steps": 1,
                "weight": 1.0,
                "approx_num_pairs": None,
            }
        ],
        split_key="train_split",
        image_size=24,
        return_metadata=True,
        max_shards_per_dataset=None,
        pairs_per_episode=None,
        index_workers=1,
        index_cache_dir=str(cache_dir),
        seed=7,
        resample_each_epoch=True,
        stopping_strategy="all_exhausted",
    )
    assert len(ds) > 0
    cache_files = list(cache_dir.glob("*.pkl"))
    assert len(cache_files) >= 1


def test_openx_local_indexed_full_datamodule_builds_and_samples(tmp_path: Path):
    root = tmp_path / "openx"
    _write_shard(
        root=root,
        dataset_name="bridge",
        shard_name="bridge_00000.tar",
        episodes=[_make_episode(ep_seed=6, n_steps=6), _make_episode(ep_seed=7, n_steps=6)],
    )

    cfg_data = {
        "backend": "oxe_local_indexed",
        "preprocess": {"image_size": 28, "return_metadata": True},
        "loader": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": None,
        },
        "dataset": {
            "oxe": {
                "datasets": [
                    {
                        "name": "bridge",
                        "train_split": "train[:100%]",
                        "val_split": "train[:100%]",
                        "pair_offset_steps": 1,
                        "weight": 1.0,
                        "approx_num_pairs": None,
                    }
                ]
            }
        },
        "adapter": {
            "openx_local": {
                "root": str(root),
                "mode": "indexed_full",
                "max_shards_per_dataset": 0,
                "pairs_per_episode": 0,
                "index_workers": 1,
                "index_cache_dir": str(tmp_path / "index_cache"),
                "index_rebuild": False,
                "index_max_open_shards": 4,
                "weights_by_size": False,
                "max_pairs": None,
                "seed": 42,
                "resample_each_epoch": True,
                "stopping_strategy": "all_exhausted",
            }
        },
    }

    dm = create_datamodule(cfg_data)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    assert batch["frames"].shape == (2, 3, 2, 28, 28)
    assert len(batch["episode_id"]) == 2
    assert len(batch["frame_idx"]) == 2
    assert len(batch["dataset_name"]) == 2

    index_meta_files = list(
        (tmp_path / "index_cache" / "episode_index").glob("*/meta.json")
    )
    assert len(index_meta_files) >= 2  # train split + val split


def test_openx_local_autodiscover_uses_rgb_fallback_key(tmp_path: Path):
    root = tmp_path / "openx"
    _write_shard(
        root=root,
        dataset_name="language_table",
        shard_name="language_table_00000.tar",
        episodes=[_make_episode_with_image_key(ep_seed=8, n_steps=5, image_key="rgb")],
    )

    cfg_data = {
        "backend": "oxe_local_indexed",
        "preprocess": {"image_size": 24, "return_metadata": True},
        "loader": {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": None,
        },
        "dataset": {"oxe": {"datasets": []}},
        "adapter": {
            "openx_local": {
                "root": str(root),
                "mode": "indexed_full",
                "max_shards_per_dataset": 0,
                "pairs_per_episode": 0,
                "index_workers": 1,
                "index_cache_dir": str(tmp_path / "index_cache"),
                "index_rebuild": False,
                "index_max_open_shards": 4,
                "weights_by_size": False,
                "auto_discover": True,
                "auto_train_split": "train[:100%]",
                "auto_val_split": "train[:100%]",
                "auto_pair_offset_steps": 1,
                "auto_weight": 1.0,
                "max_pairs": None,
                "seed": 42,
                "resample_each_epoch": True,
                "stopping_strategy": "all_exhausted",
            }
        },
    }

    dm = create_datamodule(cfg_data)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch["frames"].shape == (1, 3, 2, 24, 24)
    assert batch["dataset_name"][0] == "language_table"
