#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

workspace_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(workspace_root / "packages"))
sys.path.insert(0, str(workspace_root / "lerobot" / "src"))

from common.lerobot_v3_data import LeRobotV3DataModule
from common.lerobot_v3_types import BatchedDatasetSample, Stage1Batch
from foundation.backends.interfaces import FoundationBatch


@dataclass(frozen=True)
class BenchmarkResult:
    batch_size: int
    num_workers: int
    warmup_steps: int
    measured_steps: int
    setup_s: float
    first_batch_s: float
    mean_batch_s: float
    p50_batch_s: float
    p90_batch_s: float
    p99_batch_s: float
    batches_per_s: float
    samples_per_s: float
    approx_payload_mb: float
    rss_before_mb: float
    rss_after_setup_mb: float
    rss_after_run_mb: float


def _rss_mb() -> float:
    process = psutil.Process()
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _estimate_sample_payload_bytes(batch: Any) -> int:
    if torch.is_tensor(batch):
        return _tensor_bytes(batch)
    if isinstance(batch, (str, bytes)) or batch is None:
        return 0
    if isinstance(batch, dict):
        return sum(_estimate_sample_payload_bytes(value) for value in batch.values())
    if isinstance(batch, (list, tuple)):
        return sum(_estimate_sample_payload_bytes(value) for value in batch)
    if isinstance(batch, (BatchedDatasetSample, Stage1Batch, FoundationBatch)):
        return sum(_estimate_sample_payload_bytes(value) for value in vars(batch).values())
    return 0


def _shutdown_loader_iterator(iterator: Any) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _measure_loader(*, loader, warmup_steps: int, measured_steps: int) -> dict[str, float]:
    iterator = iter(loader)
    batch_times: list[float] = []
    try:
        t0 = time.perf_counter()
        first_batch = next(iterator)
        first_batch_s = time.perf_counter() - t0
        payload_mb = float(_estimate_sample_payload_bytes(first_batch)) / (1024.0 * 1024.0)

        for _ in range(max(0, warmup_steps - 1)):
            _ = next(iterator)

        for _ in range(measured_steps):
            t0 = time.perf_counter()
            _ = next(iterator)
            batch_times.append(time.perf_counter() - t0)
    finally:
        _shutdown_loader_iterator(iterator)

    total_measured_s = float(sum(batch_times))
    return {
        "first_batch_s": first_batch_s,
        "mean_batch_s": float(statistics.mean(batch_times)),
        "p50_batch_s": _percentile(batch_times, 50.0),
        "p90_batch_s": _percentile(batch_times, 90.0),
        "p99_batch_s": _percentile(batch_times, 99.0),
        "batches_per_s": float(measured_steps) / total_measured_s,
        "samples_per_s": float(measured_steps * int(loader.batch_size)) / total_measured_s,
        "approx_payload_mb": payload_mb,
    }


def _load_config(config_path: Path) -> dict[str, Any]:
    raw_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(type(cfg))
    if "loader" in cfg and "dataset" in cfg:
        return cfg
    if "data" in cfg and isinstance(cfg["data"], dict):
        return cfg["data"]
    if "defaults" in cfg:
        config_dir = workspace_root / "config"
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            composed = compose(config_name="config", overrides=[f"data={config_path.stem}"])
        composed_data = OmegaConf.to_container(composed.data, resolve=True)
        if not isinstance(composed_data, dict):
            raise TypeError(type(composed_data))
        return composed_data
    raise KeyError(f"Unsupported data config shape in {config_path}")


def _make_datamodule(
    *,
    cfg: dict[str, Any],
    batch_size: int,
    num_workers: int,
    output_format: str,
    steps_per_epoch: int,
) -> LeRobotV3DataModule:
    loader_cfg = dict(cfg["loader"])
    loader_cfg["batch_size"] = int(batch_size)
    loader_cfg["num_workers"] = int(num_workers)
    loader_cfg["prefetch_factor"] = None if int(num_workers) == 0 else int(loader_cfg.get("prefetch_factor", 2))

    adapter_cfg = dict(cfg["adapter"]["lerobot_v3"])
    adapter_cfg["steps_per_epoch"] = int(steps_per_epoch)

    return LeRobotV3DataModule(
        sources=list(cfg["dataset"]["lerobot"]["sources"]),
        request=dict(cfg["request"]),
        loader=loader_cfg,
        adapter=adapter_cfg,
        output_format=str(output_format),
    )


def summarize_config_counts(*, cfg: dict[str, Any]) -> dict[str, Any]:
    datamodule = _make_datamodule(
        cfg=cfg,
        batch_size=1,
        num_workers=0,
        output_format="stage1",
        steps_per_epoch=8,
    )
    datamodule.setup()

    total_episodes = sum(int(source.meta.total_episodes) for source in datamodule.sources)
    train_episodes = sum(int(len(source.compiled_train_index.episodes.episode_index)) for source in datamodule.sources)
    val_episodes = sum(int(len(source.compiled_val_index.episodes.episode_index)) for source in datamodule.sources)
    train_samples = int(len(datamodule.train_dataset))
    val_samples = int(len(datamodule.val_dataset))

    by_source = []
    for source in datamodule.sources:
        by_source.append(
            {
                "repo_id": source.repo_id,
                "weight": float(source.weight),
                "total_episodes": int(source.meta.total_episodes),
                "train_episodes": int(len(source.compiled_train_index.episodes.episode_index)),
                "val_episodes": int(len(source.compiled_val_index.episodes.episode_index)),
                "train_samples": int(source.compiled_train_index.episodes.valid_anchor_count.sum()),
                "val_samples": int(source.compiled_val_index.episodes.valid_anchor_count.sum()),
            }
        )

    return {
        "num_sources": len(datamodule.sources),
        "total_episodes": total_episodes,
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "sources": by_source,
    }


def run_single_benchmark(
    *,
    cfg: dict[str, Any],
    batch_size: int,
    num_workers: int,
    output_format: str,
    warmup_steps: int,
    measured_steps: int,
) -> BenchmarkResult:
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    datamodule = _make_datamodule(
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        output_format=output_format,
        steps_per_epoch=max(8, warmup_steps + measured_steps + 2),
    )
    datamodule.setup()
    setup_s = time.perf_counter() - t0
    rss_after_setup = _rss_mb()

    loader = datamodule.train_dataloader()
    metrics = _measure_loader(loader=loader, warmup_steps=warmup_steps, measured_steps=measured_steps)
    rss_after_run = _rss_mb()

    del loader
    del datamodule
    gc.collect()

    return BenchmarkResult(
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        warmup_steps=int(warmup_steps),
        measured_steps=int(measured_steps),
        setup_s=float(setup_s),
        first_batch_s=float(metrics["first_batch_s"]),
        mean_batch_s=float(metrics["mean_batch_s"]),
        p50_batch_s=float(metrics["p50_batch_s"]),
        p90_batch_s=float(metrics["p90_batch_s"]),
        p99_batch_s=float(metrics["p99_batch_s"]),
        batches_per_s=float(metrics["batches_per_s"]),
        samples_per_s=float(metrics["samples_per_s"]),
        approx_payload_mb=float(metrics["approx_payload_mb"]),
        rss_before_mb=float(rss_before),
        rss_after_setup_mb=float(rss_after_setup),
        rss_after_run_mb=float(rss_after_run),
    )


def write_results(
    *,
    output_dir: Path,
    config_path: Path,
    output_format: str,
    counts: dict[str, Any],
    results: list[BenchmarkResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_path.txt").write_text(str(config_path) + "\n")
    (output_dir / "counts.json").write_text(json.dumps(counts, indent=2))
    (output_dir / "results.json").write_text(json.dumps([asdict(result) for result in results], indent=2))

    if not results:
        return

    with (output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    best_by_batch: dict[int, BenchmarkResult] = {}
    for result in results:
        current = best_by_batch.get(result.batch_size)
        if current is None or result.samples_per_s > current.samples_per_s:
            best_by_batch[result.batch_size] = result

    lines = [
        "# LeRobot v3 Config Benchmark",
        "",
        f"- config: `{config_path}`",
        f"- output_format: `{output_format}`",
        f"- num_sources: `{counts['num_sources']}`",
        f"- total_episodes: `{counts['total_episodes']}`",
        f"- train_episodes: `{counts['train_episodes']}`",
        f"- val_episodes: `{counts['val_episodes']}`",
        f"- train_samples: `{counts['train_samples']}`",
        f"- val_samples: `{counts['val_samples']}`",
        "",
        "## Best by batch size",
        "",
        "| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for batch_size in sorted(best_by_batch):
        result = best_by_batch[batch_size]
        lines.append(
            f"| {result.batch_size} | {result.num_workers} | {result.samples_per_s:.1f} | "
            f"{result.mean_batch_s:.4f} | {result.first_batch_s:.4f} | {result.approx_payload_mb:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Full results",
            "",
            "| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb | rss_after_run_mb |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        lines.append(
            f"| {result.batch_size} | {result.num_workers} | {result.samples_per_s:.1f} | "
            f"{result.mean_batch_s:.4f} | {result.first_batch_s:.4f} | {result.approx_payload_mb:.2f} | {result.rss_after_run_mb:.1f} |"
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a LeRobot-v3 data config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-format", default="stage1")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--num-workers", nargs="+", type=int, default=[0, 4, 8, 12, 16, 20, 24])
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root
        / "experiments"
        / "benchmarking"
        / "results"
        / f"lerobot_v3_config_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    counts = summarize_config_counts(cfg=cfg)
    print(json.dumps({k: v for k, v in counts.items() if k != "sources"}, indent=2), flush=True)
    write_results(
        output_dir=args.output_dir,
        config_path=args.config,
        output_format=str(args.output_format),
        counts=counts,
        results=[],
    )

    results: list[BenchmarkResult] = []
    for batch_size in args.batch_sizes:
        for num_workers in args.num_workers:
            print(f"[bench] batch_size={batch_size} num_workers={num_workers}", flush=True)
            result = run_single_benchmark(
                cfg=cfg,
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                output_format=str(args.output_format),
                warmup_steps=int(args.warmup_steps),
                measured_steps=int(args.steps),
            )
            results.append(result)
            write_results(
                output_dir=args.output_dir,
                config_path=args.config,
                output_format=str(args.output_format),
                counts=counts,
                results=results,
            )
            print(
                f"  -> {result.samples_per_s:.1f} samples/s | mean={result.mean_batch_s:.4f}s | first={result.first_batch_s:.4f}s",
                flush=True,
            )
    print(f"[bench] wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
