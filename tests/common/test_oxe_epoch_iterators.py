import pytest


def test_oxe_frame_pair_dataset_repeats_only_for_train(monkeypatch):
    import importlib.util
    from pathlib import Path

    oxe_path = Path(__file__).resolve().parents[2] / "packages" / "common" / "adapters" / "oxe.py"
    spec = importlib.util.spec_from_file_location("_oxe_mod_for_test", oxe_path)
    assert spec is not None and spec.loader is not None
    oxe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oxe_mod)

    calls = {"repeat": 0}

    class DummyDS:
        def repeat(self):
            calls["repeat"] += 1
            return self

        def map(self, *args, **kwargs):
            return self

        def batch(self, *args, **kwargs):
            return self

        def prefetch(self, *args, **kwargs):
            return self

    monkeypatch.setattr(oxe_mod, "_import_tensorflow_cpu_only", lambda: None, raising=True)

    def _fake_create_tf_pipeline(self):
        return DummyDS()

    monkeypatch.setattr(
        oxe_mod.OXEFramePairDataset, "_create_tf_pipeline", _fake_create_tf_pipeline, raising=True
    )

    train_ds = oxe_mod.OXEFramePairDataset(
        dataset_name="language_table",
        split="train[:10]",
        offset=1,
        final_stream_prefetch_buffer=0,
        episode_queue_shuffle_buffer=0,
        intra_episode_sample_shuffle_buffer=0,
        image_size=64,
        return_metadata=False,
        is_train=True,
        output_batch_size=4,
        output_action_dim=None,
        output_state_dim=None,
        persistent_iterator=True,
        samples_per_episode=0,
        seed=None,
        debug_use_synthetic_data=False,
        debug_synthetic_num_samples=10,
        precomputed_size=100,
        episode_queue_prefetch_buffer=0,
        tfds_read_cycle_length=1,
        tfds_read_block_length=1,
        tfds_read_decode_parallelism=-1,
        tfds_read_interleave_parallelism=-1,
        tfds_read_skip_steps_decoding=False,
        pipeline_episode_concurrency=1,
        pipeline_transform_parallelism=1,
        pipeline_interleave_parallelism=1,
        private_threadpool_size=0,
    )
    _ = train_ds._get_or_create_pipeline()

    val_ds = oxe_mod.OXEFramePairDataset(
        dataset_name="language_table",
        split="train[:10]",
        offset=1,
        final_stream_prefetch_buffer=0,
        episode_queue_shuffle_buffer=0,
        intra_episode_sample_shuffle_buffer=0,
        image_size=64,
        return_metadata=False,
        is_train=False,
        output_batch_size=4,
        output_action_dim=None,
        output_state_dim=None,
        persistent_iterator=True,
        samples_per_episode=0,
        seed=None,
        debug_use_synthetic_data=False,
        debug_synthetic_num_samples=10,
        precomputed_size=100,
        episode_queue_prefetch_buffer=0,
        tfds_read_cycle_length=1,
        tfds_read_block_length=1,
        tfds_read_decode_parallelism=-1,
        tfds_read_interleave_parallelism=-1,
        tfds_read_skip_steps_decoding=False,
        pipeline_episode_concurrency=1,
        pipeline_transform_parallelism=1,
        pipeline_interleave_parallelism=1,
        private_threadpool_size=0,
    )
    _ = val_ds._get_or_create_pipeline()

    assert calls["repeat"] == 1


def test_oxe_datamodule_forces_nonpersistent_val_iterator():
    pytest.importorskip("lightning")
    from common.data import OXEDataModule

    dm = OXEDataModule(
        datasets=[
            {
                "name": "language_table",
                "train_split": "train[:90%]",
                "val_split": "train[90%:]",
                "pair_offset_steps": 10,
                "weight": 1.0,
                "approx_num_pairs": 1000,
            }
        ],
        preprocess={"image_size": 64, "return_metadata": True},
        loader={"batch_size": 4, "num_workers": 0, "pin_memory": False},
        adapter={
            "tf": {
                "debug": {"use_synthetic_data": False, "synthetic_num_samples": 10},
                "iterator": {"persistent": True},
                "sampling": {"samples_per_episode": 0, "seed": 123},
                "train": {
                    "episode_queue_shuffle_buffer": 1,
                    "intra_episode_sample_shuffle_buffer": 0,
                    "global_stream_shuffle_buffer": 1,
                },
                "val": {
                    "episode_queue_shuffle_buffer": 0,
                    "intra_episode_sample_shuffle_buffer": 0,
                    "global_stream_shuffle_buffer": 0,
                },
                "prefetch": {
                    "final_stream_buffer": 0,
                    "per_dataset_stream_buffer": 0,
                    "episode_queue_buffer": 0,
                },
                "tfds_read": {
                    "source": "auto",
                    "local_root": None,
                    "cycle_length": 1,
                    "block_length": 1,
                    "decode_parallelism": -1,
                    "interleave_parallelism": -1,
                    "skip_steps_decoding": False,
                },
                "pipeline": {
                    "episode_concurrency": 1,
                    "transform_parallelism": 1,
                    "interleave_parallelism": 1,
                    "emit_encoded_pairs": False,
                    "post_mix_decode_resize": False,
                },
                "mixing": {
                    "mix_block_length": 1,
                    "parallelism_mode": "divide",
                    "strategy": "sample",
                    "selector_run_length": 1,
                    "python_prefetch_queue_size": 2,
                    "python_prefetch_min_ready_datasets": 1,
                    "python_prefetch_wait_timeout_s": 600,
                    "per_dataset_private_threadpool_size": 0,
                },
                "pair_frames": {"mode": "endpoints", "stride": 1, "n": 2},
            }
        },
    )

    dm.setup()

    assert dm.train_dataset.persistent_iterator is True
    assert dm.val_dataset.persistent_iterator is False
