#!/usr/bin/env python3
"""
Benchmark HuggingFace-based OXE dataloader throughput.

This script compares the HuggingFace dataloader with the TensorFlow-based one
to measure performance differences.

Example:
  conda run -n hlrp python scripts/bench_hf_dataloader.py \
      --dataset bridge \
      --batch_size 32 \
      --num_batches 100 \
      --num_workers 4
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add packages to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

from common.adapters.huggingface_oxe import (
    HFOXEFramePairDataset,
    hf_collate_fn,
)


def benchmark_hf_dataloader(
    dataset_name: str,
    batch_size: int,
    num_batches: int,
    num_workers: int,
    image_size: int = 256,
    warmup_batches: int = 10,
):
    """Benchmark HuggingFace dataloader throughput."""
    print(f"\n{'='*60}")
    print(f"HuggingFace Dataloader Benchmark")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Image size: {image_size}")
    print(f"Warmup batches: {warmup_batches}")
    print(f"Measured batches: {num_batches}")

    # Create dataset
    print("\nCreating dataset...")
    ds = HFOXEFramePairDataset(
        dataset_name=dataset_name,
        split="train",
        offset=5,
        image_size=image_size,
        shuffle_buffer=100,
        return_metadata=True,
        samples_per_episode=0,
    )

    # Create dataloader
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=hf_collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Warmup
    print(f"Warming up ({warmup_batches} batches)...")
    it = iter(loader)
    for _ in range(warmup_batches):
        batch = next(it)
        # Touch data to ensure it's loaded
        _ = batch["frames"].shape

    # Benchmark
    print(f"Benchmarking ({num_batches} batches)...")
    batch_times = []
    samples_count = 0

    start_time = time.perf_counter()
    for i in range(num_batches):
        t0 = time.perf_counter()
        batch = next(it)
        _ = batch["frames"].shape  # Touch data
        t1 = time.perf_counter()

        batch_times.append(t1 - t0)
        samples_count += batch["frames"].shape[0]

        if (i + 1) % 20 == 0:
            print(f"  Batch {i+1}/{num_batches}...")

    total_time = time.perf_counter() - start_time

    # Results
    batch_times_arr = np.array(batch_times)
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total samples: {samples_count}")
    print(f"Samples/sec: {samples_count / total_time:.1f}")
    print(f"Batches/sec: {num_batches / total_time:.2f}")
    print(f"Mean batch time: {batch_times_arr.mean():.4f}s")
    print(f"Std batch time: {batch_times_arr.std():.4f}s")
    print(f"P50 batch time: {np.percentile(batch_times_arr, 50):.4f}s")
    print(f"P90 batch time: {np.percentile(batch_times_arr, 90):.4f}s")
    print(f"P99 batch time: {np.percentile(batch_times_arr, 99):.4f}s")

    return {
        "samples_per_sec": samples_count / total_time,
        "batches_per_sec": num_batches / total_time,
        "mean_batch_time": batch_times_arr.mean(),
        "p50_batch_time": np.percentile(batch_times_arr, 50),
        "p90_batch_time": np.percentile(batch_times_arr, 90),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HuggingFace OXE dataloader"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bridge",
        choices=["bridge", "language_table", "rt1"],
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--num_batches", type=int, default=100, help="Number of batches to measure"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Image size"
    )
    parser.add_argument(
        "--warmup_batches", type=int, default=10, help="Warmup batches"
    )

    args = parser.parse_args()

    benchmark_hf_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_workers=args.num_workers,
        image_size=args.image_size,
        warmup_batches=args.warmup_batches,
    )


if __name__ == "__main__":
    main()
