#!/usr/bin/env python3
"""
Benchmark OXE datasets using tfds.benchmark and PyTorch throughput measurement.

Usage:
    python experiments/benchmark_oxe.py [config_name]

Example:
    python experiments/benchmark_oxe.py laq_oxe_multi
"""

import sys
import time
from pathlib import Path
import argparse
import logging

# Add packages to path
workspace_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(workspace_root / "packages"))

import torch
import tensorflow_datasets as tfds
import tensorflow as tf
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Disable TF GPU
tf.config.set_visible_devices([], "GPU")

def load_config(config_name: str):
    """Load data config using Hydra."""
    GlobalHydra.instance().clear()
    config_dir = str(workspace_root / "config")
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        # We load the main config but override data with our target
        cfg = compose(config_name="config", overrides=[f"data={config_name}"])
        return OmegaConf.to_container(cfg.data, resolve=True)

def benchmark_tf_dataset(name: str, ds_config: dict, batch_size: int = 32):
    """Run tfds.benchmark on a single OXE dataset component."""
    from common.adapters.oxe import OXEFramePairDataset

    print(f"\n{'='*80}")
    print(f"Benchmarking TF Pipeline: {name}")
    print(f"{'='*80}")

    # Create the dataset adapter
    adapter = OXEFramePairDataset(
        dataset_name=ds_config["name"],
        gcs_path=ds_config.get("gcs_path"),
        split=ds_config.get("train_split", "train[:90%]"),
        offset=ds_config["offset"],
        final_stream_prefetch_buffer=ds_config["final_stream_prefetch_buffer"],
        episode_queue_shuffle_buffer=ds_config["episode_queue_shuffle_buffer"],
        intra_episode_sample_shuffle_buffer=ds_config["intra_episode_sample_shuffle_buffer"],
        image_size=ds_config["image_size"],
        return_metadata=True,  # Usually we want metadata
        persistent_iterator=bool(ds_config.get("persistent_iterator", False)),
        samples_per_episode=int(ds_config.get("samples_per_episode", 0)),
        seed=ds_config.get("sampling_seed"),
        precomputed_size=ds_config.get("size"),
        episode_queue_prefetch_buffer=int(ds_config.get("episode_queue_prefetch_buffer", 0)),
        private_threadpool_size=int(ds_config.get("per_dataset_private_threadpool_size", 0)),
        tfds_read_cycle_length=int(ds_config.get("tfds_read_cycle_length", 1)),
        tfds_read_block_length=int(ds_config.get("tfds_read_block_length", 1)),
        tfds_read_decode_parallelism=int(ds_config.get("tfds_read_decode_parallelism", -1)),
        tfds_read_interleave_parallelism=int(ds_config.get("tfds_read_interleave_parallelism", -1)),
        pipeline_episode_concurrency=int(ds_config.get("pipeline_episode_concurrency", 1)),
        pipeline_transform_parallelism=int(ds_config.get("pipeline_transform_parallelism", 1)),
        pipeline_interleave_parallelism=int(ds_config.get("pipeline_interleave_parallelism", 1)),
    )

    # Get the underlying tf.data.Dataset
    tf_ds = adapter._get_or_create_pipeline()
    
    # Batch it for the benchmark (tfds.benchmark expects batches usually, or handles them)
    # tfds.benchmark iterates the dataset. If we want items/sec, we can pass it directly.
    # But usually we want to see batch throughput.
    tf_ds_batched = tf_ds.batch(batch_size)

    # Run benchmark
    # num_iter=None means auto-detect
    tfds.benchmark(tf_ds_batched, batch_size=batch_size, num_iter=100)
    
    # Cleanup
    adapter.cleanup()

def benchmark_pytorch_throughput(config: dict, batch_size: int = 32):
    """Measure throughput of the final PyTorch iterable (Multi or Single)."""
    from common.adapters.oxe import MultiOXEFramePairDataset, OXEFramePairDataset
    from torch.utils.data import DataLoader

    print(f"\n{'='*80}")
    print(f"Benchmarking PyTorch DataLoader Throughput (Combined)")
    print(f"{'='*80}")

    # Instantiate dataset
    if "datasets" in config:
        # Multi-dataset
        ds = MultiOXEFramePairDataset(
            datasets=config["datasets"],
            final_stream_prefetch_buffer=config["final_stream_prefetch_buffer"],
            episode_queue_prefetch_buffer=int(config.get("episode_queue_prefetch_buffer", 0)),
            episode_queue_shuffle_buffer=config["episode_queue_shuffle_buffer"],
            intra_episode_sample_shuffle_buffer=config["intra_episode_sample_shuffle_buffer"],
            global_stream_shuffle_buffer=config["global_stream_shuffle_buffer"],
            image_size=config["image_size"],
            return_metadata=bool(config.get("return_metadata", True)),
            is_train=True,
            output_batch_size=batch_size,
            persistent_iterator=bool(config.get("persistent_iterator", True)),
            samples_per_episode=int(config.get("samples_per_episode", 0)),
            seed=config.get("sampling_seed"),
            debug_use_synthetic_data=bool(config.get("debug_use_synthetic_data", False)),
            debug_synthetic_num_samples=int(config.get("debug_synthetic_num_samples", 1000)),
            pipeline_episode_concurrency_total=int(config.get("pipeline_episode_concurrency", 1)),
            pipeline_transform_parallelism=int(config.get("pipeline_transform_parallelism", 1)),
            pipeline_interleave_parallelism=int(config.get("pipeline_interleave_parallelism", 1)),
            mix_block_length=int(config.get("multi_dataset_mix_block_length", 1)),
            parallelism_mode=str(config.get("multi_dataset_parallelism_mode", "divide")),
            per_dataset_stream_prefetch_buffer=int(config.get("per_dataset_stream_prefetch_buffer", 0)),
            mixing_strategy=str(config.get("multi_dataset_mixing_strategy", "sample")),
            python_prefetch_queue_size=int(config.get("python_prefetch_queue_size", 2)),
            python_prefetch_min_ready_datasets=int(config.get("python_prefetch_min_ready_datasets", 1)),
            python_prefetch_wait_timeout_s=float(config.get("python_prefetch_wait_timeout_s", 600)),
            per_dataset_private_threadpool_size=int(
                config.get("per_dataset_private_threadpool_size", 0)
            ),
            tfds_read_cycle_length=int(config.get("tfds_read_cycle_length", 1)),
            tfds_read_block_length=int(config.get("tfds_read_block_length", 1)),
            tfds_read_decode_parallelism=int(config.get("tfds_read_decode_parallelism", -1)),
            tfds_read_interleave_parallelism=int(config.get("tfds_read_interleave_parallelism", -1)),
            mix_selector_run_length=int(config.get("mix_selector_run_length", 1)),
            tfds_read_skip_steps_decoding=bool(config.get("tfds_read_skip_steps_decoding", False)),
            emit_encoded_pairs=bool(config.get("emit_encoded_pairs", False)),
            post_mix_decode_resize=bool(config.get("post_mix_decode_resize", False)),
            pair_frames_mode=str(config.get("pair_frames_mode", "endpoints")),
            pair_frames_stride=int(config.get("pair_frames_stride", 1)),
            pair_frames_n=int(config.get("pair_frames_n", 2)),
            tfds_source=str(config.get("tfds_source", "gcs")),
            tfds_local_root=config.get("tfds_local_root"),
        )
    else:
        # Single dataset (legacy/single config)
        ds = OXEFramePairDataset(
            dataset_name=config.get("dataset_name", config.get("name")),
            gcs_path=config.get("gcs_path"),
            split=config.get("train_split", "train"),
            offset=config.get("offset", 1),
            final_stream_prefetch_buffer=config["final_stream_prefetch_buffer"],
            episode_queue_shuffle_buffer=config["episode_queue_shuffle_buffer"],
            intra_episode_sample_shuffle_buffer=config["intra_episode_sample_shuffle_buffer"],
            image_size=config.get("image_size", 256),
            return_metadata=bool(config.get("return_metadata", True)),
            persistent_iterator=bool(config.get("persistent_iterator", True)),
            samples_per_episode=int(config.get("samples_per_episode", 0)),
            seed=config.get("sampling_seed"),
            precomputed_size=config.get("size"),
            episode_queue_prefetch_buffer=int(config.get("episode_queue_prefetch_buffer", 0)),
            private_threadpool_size=int(config.get("per_dataset_private_threadpool_size", 0)),
            tfds_read_cycle_length=int(config.get("tfds_read_cycle_length", 1)),
            tfds_read_block_length=int(config.get("tfds_read_block_length", 1)),
            tfds_read_decode_parallelism=int(config.get("tfds_read_decode_parallelism", -1)),
            tfds_read_interleave_parallelism=int(config.get("tfds_read_interleave_parallelism", -1)),
            pipeline_episode_concurrency=int(config.get("pipeline_episode_concurrency", 1)),
            pipeline_transform_parallelism=int(config.get("pipeline_transform_parallelism", 1)),
            pipeline_interleave_parallelism=int(config.get("pipeline_interleave_parallelism", 1)),
        )

    # Create DataLoader
    # num_workers=0 is standard for IterableDataset wrapping TF (threading issues)
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        num_workers=config.get("num_workers", 0),
        pin_memory=True
    )

    print("Warming up...")
    iterator = iter(loader)
    
    # Warmup
    for _ in range(5):
        next(iterator)

    print("Measuring...")
    start_time = time.time()
    num_batches = 50
    
    for i in range(num_batches):
        try:
            batch = next(iterator)
            if i % 10 == 0:
                sys.stdout.write(f".")
                sys.stdout.flush()
        except StopIteration:
            break
    
    end_time = time.time()
    duration = end_time - start_time
    total_samples = num_batches * batch_size
    
    print(f"\nProcessed {num_batches} batches ({total_samples} samples) in {duration:.2f}s")
    print(f"Throughput: {num_batches/duration:.2f} batches/sec")
    print(f"Throughput: {total_samples/duration:.2f} samples/sec")
    
    if hasattr(ds, "cleanup"):
        ds.cleanup()

def main():
    parser = argparse.ArgumentParser(description="Benchmark OXE datasets")
    parser.add_argument("config", nargs="?", default="laq_oxe_multi", help="Config name in config/data/")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--skip-tfds", action="store_true", help="Skip individual TFDS benchmarks")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch combined benchmark")
    
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # 1. Benchmark individual components using tfds.benchmark
    if not args.skip_tfds:
        if "datasets" in config:
            for i, ds_cfg in enumerate(config["datasets"]):
                ds_cfg_merged = dict(config)
                ds_cfg_merged.update(ds_cfg)
                benchmark_tf_dataset(
                    ds_cfg_merged.get("name", f"dataset_{i}"), ds_cfg_merged, args.batch_size
                )
        else:
            # Single dataset config
            benchmark_tf_dataset(config.get("dataset_name", config.get("name", "unknown")), config, args.batch_size)

    # 2. Benchmark combined PyTorch throughput
    if not args.skip_pytorch:
        benchmark_pytorch_throughput(config, args.batch_size)

if __name__ == "__main__":
    main()
