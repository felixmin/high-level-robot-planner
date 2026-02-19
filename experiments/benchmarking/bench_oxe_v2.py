#!/usr/bin/env python3
"""
Smoke test / benchmark for OXE v2 dataloader.

Usage:
    # Synthetic (no data needed):
    python experiments/benchmarking/bench_oxe_v2.py --synthetic

    # Real data (1 dataset, local or GCS):
    python experiments/benchmarking/bench_oxe_v2.py --dataset language_table

    # Full 29-dataset mix:
    python experiments/benchmarking/bench_oxe_v2.py --config oxe_local_spe0_v2

    # Custom:
    python experiments/benchmarking/bench_oxe_v2.py --dataset language_table --batch-size 8 --steps 50
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add packages to path
workspace_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(workspace_root / "packages"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_rss_mb():
    """Current process RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0.0


def run_synthetic(batch_size, steps, image_size):
    from common.adapters.oxe_v2 import OXEFramePairDatasetV2

    logger.info(f"Synthetic: bs={batch_size}, steps={steps}, image_size={image_size}")

    ds = OXEFramePairDatasetV2(
        dataset_entries=[
            {"name": "language_table", "pair_offset_steps": 5, "weight": 1.0,
             "train_split": "train[:90%]", "val_split": "train[90%:]"},
            {"name": "bridge", "pair_offset_steps": 5, "weight": 1.0,
             "train_split": "train[:90%]", "val_split": "train[90%:]"},
        ],
        image_size=image_size,
        batch_size=batch_size,
        use_synthetic_data=True,
        synthetic_num_samples=max(1000, steps * batch_size * 2),
        total_threads=4,
        ram_budget_gb=1,
        shuffle_sample_buffer=0,
        seed=42,
        train=True,
    )

    return _run_iteration(ds, steps)


def run_single_dataset(dataset_name, batch_size, steps, image_size, tfds_source, tfds_local_root):
    from common.adapters.oxe_v2 import OXEFramePairDatasetV2

    logger.info(f"Single dataset: {dataset_name}, bs={batch_size}, steps={steps}")

    ds = OXEFramePairDatasetV2(
        dataset_entries=[
            {"name": dataset_name, "pair_offset_steps": 5, "weight": 1.0,
             "train_split": "train[:90%]", "val_split": "train[90%:]"},
        ],
        image_size=image_size,
        batch_size=batch_size,
        total_threads=8,
        ram_budget_gb=1,
        shuffle_sample_buffer=500,
        tfds_source=tfds_source,
        tfds_local_root=tfds_local_root,
        seed=42,
        train=True,
    )

    return _run_iteration(ds, steps)


def run_config(config_name, batch_size, steps):
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from common.data_factory import create_datamodule

    logger.info(f"Config: {config_name}, bs={batch_size}, steps={steps}")

    GlobalHydra.instance().clear()
    config_dir = str(workspace_root / "config" / "data")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
        data_cfg = OmegaConf.to_container(cfg, resolve=True)

    # @package data nests everything under "data" key
    if "data" in data_cfg:
        data_cfg = data_cfg["data"]
    if batch_size:
        data_cfg["loader"]["batch_size"] = batch_size

    dm = create_datamodule(data_cfg)
    dm.setup()

    logger.info("Train dataloader created, starting iteration...")
    dl = dm.train_dataloader()

    rss_start = get_rss_mb()
    max_rss = rss_start
    t0 = time.time()

    for i, batch in enumerate(dl):
        if i >= steps:
            break
        rss = get_rss_mb()
        max_rss = max(max_rss, rss)
        if i % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  step {i:4d} | {rate:.2f} step/s | RSS {rss:.0f} MB | max RSS {max_rss:.0f} MB")

    elapsed = time.time() - t0
    rate = steps / elapsed if elapsed > 0 else 0

    logger.info(f"\nDone: {steps} steps in {elapsed:.1f}s ({rate:.2f} step/s)")
    logger.info(f"RSS: start={rss_start:.0f} MB, max={max_rss:.0f} MB, delta={max_rss - rss_start:.0f} MB")

    dm.teardown()
    return {"steps": steps, "elapsed": elapsed, "rate": rate, "max_rss_mb": max_rss}


def _run_iteration(ds, steps):
    rss_start = get_rss_mb()
    max_rss = rss_start
    t0 = time.time()

    it = iter(ds)
    for i in range(steps):
        batch = next(it)
        rss = get_rss_mb()
        max_rss = max(max_rss, rss)
        if i % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            frames = batch["frames"]
            logger.info(
                f"  step {i:4d} | {rate:.2f} step/s | RSS {rss:.0f} MB | max RSS {max_rss:.0f} MB | "
                f"frames {tuple(frames.shape)} | datasets: {set(batch['dataset_name'])}"
            )

    elapsed = time.time() - t0
    rate = steps / elapsed if elapsed > 0 else 0

    logger.info(f"\nDone: {steps} steps in {elapsed:.1f}s ({rate:.2f} step/s)")
    logger.info(f"RSS: start={rss_start:.0f} MB, max={max_rss:.0f} MB, delta={max_rss - rss_start:.0f} MB")

    ds.cleanup()
    return {"steps": steps, "elapsed": elapsed, "rate": rate, "max_rss_mb": max_rss}


def main():
    parser = argparse.ArgumentParser(description="OXE v2 dataloader benchmark")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--dataset", type=str, help="Single dataset name (e.g. language_table)")
    parser.add_argument("--config", type=str, help="Hydra data config name (e.g. oxe_local_spe0_v2)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--tfds-source", type=str, default="auto")
    parser.add_argument("--tfds-local-root", type=str, default=None)
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic(args.batch_size, args.steps, args.image_size)
    elif args.dataset:
        run_single_dataset(
            args.dataset, args.batch_size, args.steps, args.image_size,
            args.tfds_source, args.tfds_local_root,
        )
    elif args.config:
        run_config(args.config, args.batch_size, args.steps)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python experiments/benchmarking/bench_oxe_v2.py --synthetic")
        print("  python experiments/benchmarking/bench_oxe_v2.py --dataset language_table")
        print("  python experiments/benchmarking/bench_oxe_v2.py --config oxe_local_spe0_v2 --batch-size 8 --steps 100")


if __name__ == "__main__":
    main()
