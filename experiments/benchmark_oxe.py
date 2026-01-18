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
        split=ds_config.get("train_split", "train[:90%]"),
        offset=ds_config.get("offset", 1),
        image_size=ds_config.get("image_size", 256),
        shuffle_buffer=ds_config.get("shuffle_buffer", 100),
        prefetch_buffer=ds_config.get("prefetch_buffer", 2),
        return_metadata=True, # Usually we want metadata
        # We don't need persistent iterator for benchmark, but we need the pipeline
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
            image_size=config.get("image_size", 256),
            shuffle_buffer=config.get("shuffle_buffer", 200),
            prefetch_buffer=config.get("prefetch_buffer", 2),
            return_metadata=config.get("return_metadata", True),
            is_train=True
        )
    else:
        # Single dataset (legacy/single config)
        ds = OXEFramePairDataset(
            dataset_name=config.get("dataset_name", config.get("name")),
            split=config.get("train_split", "train"),
            offset=config.get("offset", 1),
            image_size=config.get("image_size", 256),
            shuffle_buffer=config.get("shuffle_buffer", 100),
            prefetch_buffer=config.get("prefetch_buffer", 2),
            return_metadata=config.get("return_metadata", True),
            precomputed_size=config.get("size")
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
                # Merge global settings if not present
                if "offset" not in ds_cfg:
                    ds_cfg["offset"] = config.get("offset", 1)
                if "image_size" not in ds_cfg:
                    ds_cfg["image_size"] = config.get("image_size", 256)
                
                benchmark_tf_dataset(ds_cfg.get("name", f"dataset_{i}"), ds_cfg, args.batch_size)
        else:
            # Single dataset config
            benchmark_tf_dataset(config.get("dataset_name", config.get("name", "unknown")), config, args.batch_size)

    # 2. Benchmark combined PyTorch throughput
    if not args.skip_pytorch:
        benchmark_pytorch_throughput(config, args.batch_size)

if __name__ == "__main__":
    main()
