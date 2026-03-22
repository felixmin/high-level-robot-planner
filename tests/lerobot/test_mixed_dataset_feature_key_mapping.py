from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets import mixed_dataset as md


@dataclass
class _FakeMeta:
    features: dict
    fps: int = 10
    root: str = "/tmp/fake"

    def __post_init__(self) -> None:
        self.episodes = [
            {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 4}
        ]
        self.total_episodes = 1


def _make_stats(feature_key: str) -> dict[str, dict[str, np.ndarray]]:
    return {
        feature_key: {
            "mean": np.asarray([0.0], dtype=np.float32),
            "std": np.asarray([1.0], dtype=np.float32),
            "min": np.asarray([0.0], dtype=np.float32),
            "max": np.asarray([1.0], dtype=np.float32),
            "count": np.asarray([4], dtype=np.int64),
        }
    }


def _make_stats_for_features(features: dict[str, dict]) -> dict[str, dict[str, np.ndarray]]:
    return {feature_key: _make_stats(feature_key)[feature_key] for feature_key in features}


def test_load_dataset_mix_config_parses_feature_key_mapping(tmp_path):
    mix_path = tmp_path / "mix.yaml"
    mix_path.write_text(
        """
sources:
  - name: source_a
    repo_id: repo/a
    weight: 1.0
    supervision: multitask
    feature_key_mapping:
      observation.images.cam_high: observation.images.image
"""
    )

    cfg = md.load_dataset_mix_config(mix_path)

    assert cfg.sources[0].feature_key_mapping == {
        "observation.images.cam_high": "observation.images.image"
    }


def test_load_dataset_mix_config_parses_compatibility_options(tmp_path):
    mix_path = tmp_path / "mix.yaml"
    mix_path.write_text(
        """
compatibility:
  retained_features:
    - observation.images.image
  enforce_matching_fps: false
  enforce_matching_delta_timestamps: false
  allow_visual_shape_mismatch: true
sources:
  - name: source_a
    repo_id: repo/a
    weight: 1.0
    supervision: multitask
"""
    )

    cfg = md.load_dataset_mix_config(mix_path)

    assert cfg.retained_features == ("observation.images.image",)
    assert cfg.enforce_matching_fps is False
    assert cfg.enforce_matching_delta_timestamps is False
    assert cfg.allow_visual_shape_mismatch is True


def test_validate_mixed_sources_uses_remapped_feature_schema(monkeypatch):
    feature_a = "observation.images.cam_high"
    feature_b = "observation.images.rgb_front"
    canonical_feature = "observation.images.image"
    monkeypatch.setattr(
        md,
        "_aggregate_selected_stats",
        lambda meta, selected_episodes: _make_stats(next(iter(meta.features))),
    )

    source_a = md.LogicalSource(
        source_index=0,
        config=md.DatasetMixSourceConfig(
            name="source_a",
            repo_id="repo/a",
            supervision="multitask",
            feature_key_mapping={feature_a: canonical_feature},
        ),
        meta=_FakeMeta(
            features={feature_a: {"dtype": "video", "shape": [8, 3, 224, 224]}}
        ),
        delta_timestamps={feature_a: [0.0, 0.1]},
    )
    source_b = md.LogicalSource(
        source_index=1,
        config=md.DatasetMixSourceConfig(
            name="source_b",
            repo_id="repo/b",
            supervision="multitask",
            feature_key_mapping={feature_b: canonical_feature},
        ),
        meta=_FakeMeta(
            features={feature_b: {"dtype": "video", "shape": [8, 3, 224, 224]}}
        ),
        delta_timestamps={feature_b: [0.0, 0.1]},
    )

    md.validate_mixed_sources([source_a, source_b])

    assert list(source_a.features) == [canonical_feature]
    assert list(source_b.features) == [canonical_feature]
    assert source_a.camera_keys == [canonical_feature]
    assert source_b.camera_keys == [canonical_feature]


def test_logical_source_remaps_item_keys_and_pad_mask(monkeypatch):
    raw_feature = "observation.images.cam_high"
    canonical_feature = "observation.images.image"
    monkeypatch.setattr(
        md,
        "_aggregate_selected_stats",
        lambda meta, selected_episodes: _make_stats(next(iter(meta.features))),
    )

    source = md.LogicalSource(
        source_index=0,
        config=md.DatasetMixSourceConfig(
            name="source_a",
            repo_id="repo/a",
            supervision="multitask",
            feature_key_mapping={raw_feature: canonical_feature},
        ),
        meta=_FakeMeta(
            features={raw_feature: {"dtype": "video", "shape": [8, 3, 224, 224]}}
        ),
        delta_timestamps={raw_feature: [0.0, 0.1]},
    )

    class _FakeDataset:
        _absolute_to_relative_idx = None

        def __getitem__(self, index: int):
            return {
                raw_feature: "frames",
                f"{raw_feature}_is_pad": "pad-mask",
                "task": "keep-me",
            }

    monkeypatch.setattr(source, "_get_dataset", lambda: _FakeDataset())

    item = source.get_item(0)

    assert item[canonical_feature] == "frames"
    assert item[f"{canonical_feature}_is_pad"] == "pad-mask"
    assert raw_feature not in item
    assert f"{raw_feature}_is_pad" not in item


def test_validate_mixed_sources_allows_retained_subset_with_mixed_fps(monkeypatch):
    monkeypatch.setattr(
        md,
        "_aggregate_selected_stats",
        lambda meta, selected_episodes: _make_stats_for_features(meta.features),
    )

    source_a = md.LogicalSource(
        source_index=0,
        config=md.DatasetMixSourceConfig(
            name="libero_like",
            repo_id="repo/a",
            supervision="multitask",
        ),
        meta=_FakeMeta(
            features={
                "observation.images.image": {"dtype": "image", "shape": [256, 256, 3]},
                "observation.images.image2": {"dtype": "image", "shape": [256, 256, 3]},
            },
            fps=10,
        ),
        delta_timestamps={
            "observation.images.image": [0.0, 0.5],
            "observation.images.image2": [0.0, 0.5],
        },
        retained_features=("observation.images.image",),
    )
    source_b = md.LogicalSource(
        source_index=1,
        config=md.DatasetMixSourceConfig(
            name="bridge_like",
            repo_id="repo/b",
            supervision="multitask",
            feature_key_mapping={
                "observation.images.primary": "observation.images.image",
            },
        ),
        meta=_FakeMeta(
            features={
                "observation.images.primary": {"dtype": "video", "shape": [480, 640, 3]},
                "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
            },
            fps=5,
        ),
        delta_timestamps={
            "observation.images.primary": [0.0, 0.4],
            "observation.images.wrist": [0.0, 0.4],
        },
        retained_features=("observation.images.image",),
    )

    md.validate_mixed_sources(
        [source_a, source_b],
        enforce_matching_fps=False,
        enforce_matching_delta_timestamps=False,
        allow_visual_shape_mismatch=True,
    )

    assert list(source_a.features) == ["observation.images.image"]
    assert list(source_b.features) == ["observation.images.image"]


def test_logical_source_filters_non_retained_features(monkeypatch):
    monkeypatch.setattr(
        md,
        "_aggregate_selected_stats",
        lambda meta, selected_episodes: _make_stats_for_features(meta.features),
    )

    source = md.LogicalSource(
        source_index=0,
        config=md.DatasetMixSourceConfig(
            name="bridge_like",
            repo_id="repo/a",
            supervision="multitask",
            feature_key_mapping={
                "observation.images.primary": "observation.images.image",
            },
        ),
        meta=_FakeMeta(
            features={
                "observation.images.primary": {"dtype": "video", "shape": [480, 640, 3]},
                "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
            }
        ),
        delta_timestamps={
            "observation.images.primary": [0.0, 0.1],
            "observation.images.wrist": [0.0, 0.1],
        },
        retained_features=("observation.images.image",),
    )

    class _FakeDataset:
        _absolute_to_relative_idx = None

        def __getitem__(self, index: int):
            return {
                "observation.images.primary": "frames",
                "observation.images.primary_is_pad": "pad-mask",
                "observation.images.wrist": "drop-me",
                "observation.images.wrist_is_pad": "drop-mask",
            }

    monkeypatch.setattr(source, "_get_dataset", lambda: _FakeDataset())

    item = source.get_item(0)

    assert item["observation.images.image"] == "frames"
    assert item["observation.images.image_is_pad"] == "pad-mask"
    assert "observation.images.wrist" not in item
    assert "observation.images.wrist_is_pad" not in item


def test_logical_source_uses_raw_delta_timestamps_for_underlying_dataset(monkeypatch):
    monkeypatch.setattr(
        md,
        "_aggregate_selected_stats",
        lambda meta, selected_episodes: _make_stats_for_features(meta.features),
    )

    source = md.LogicalSource(
        source_index=0,
        config=md.DatasetMixSourceConfig(
            name="bridge_like",
            repo_id="repo/a",
            supervision="multitask",
            feature_key_mapping={
                "observation.images.primary": "observation.images.image",
            },
        ),
        meta=_FakeMeta(
            features={
                "observation.images.primary": {"dtype": "video", "shape": [480, 640, 3]},
                "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
            }
        ),
        delta_timestamps={
            "observation.images.primary": [0.0, 0.5],
            "observation.images.wrist": [0.0, 0.5],
        },
        retained_features=("observation.images.image",),
    )

    created = {}

    class _FakeDataset:
        _absolute_to_relative_idx = None

        def __getitem__(self, index: int):
            return {"observation.images.primary": "frames"}

    def _fake_lerobot_dataset(*args, **kwargs):
        created["delta_timestamps"] = kwargs["delta_timestamps"]
        return _FakeDataset()

    monkeypatch.setattr(md, "LeRobotDataset", _fake_lerobot_dataset)

    source.get_item(0)

    assert created["delta_timestamps"] == {
        "observation.images.primary": [0.0, 0.5]
    }
