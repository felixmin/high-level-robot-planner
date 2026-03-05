#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

workspace_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(workspace_root / "packages"))
sys.path.insert(0, str(workspace_root / "lerobot" / "src"))

from common.lerobot_v3_data import LeRobotV3DataModule
from common.lerobot_v3_types import BatchedDatasetSample, Stage1Batch
from stage2.backends.interfaces import Stage2Batch


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    description: str
    output_format: str
    sources: list[dict[str, Any]]
    request: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkResult:
    scenario: str
    description: str
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


def _source_cfg(
    *,
    repo_id: str,
    camera_map: dict[str, str],
    weight: float = 1.0,
    val_episode_count: int = 2,
) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "weight": weight,
        "camera_map": camera_map,
        "state_key": "observation.state",
        "action_key": "action",
        "video_backend": "pyav",
        "val_episode_count": val_episode_count,
    }


def _request_cfg(
    *,
    image_requests: dict[str, list[int]],
    image_size: int = 96,
    include_actions: bool,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "image_requests": {role: {"deltas_steps": deltas} for role, deltas in image_requests.items()},
        "include_task_text": True,
        "include_metadata": True,
        "pad_missing_future": True,
        "image_size": [image_size, image_size],
        "image_dtype": "uint8",
    }
    if include_actions:
        cfg["state_request"] = {"deltas_steps": [0]}
        cfg["action_request"] = {"deltas_steps": list(range(16))}
    return cfg


def _adapter_cfg(*, steps_per_epoch: int) -> dict[str, Any]:
    return {
        "seed": 7,
        "steps_per_epoch": steps_per_epoch,
        "resample_each_epoch": True,
        "weights_mode": "explicit",
    }


def _scenarios() -> dict[str, BenchmarkScenario]:
    return {
        "stage1_single_nyu": BenchmarkScenario(
            name="stage1_single_nyu",
            description="single-source Stage 1 batch on lerobot/nyu_rot_dataset",
            output_format="stage1",
            sources=[
                _source_cfg(
                    repo_id="lerobot/nyu_rot_dataset",
                    camera_map={"primary": "observation.images.image"},
                    val_episode_count=2,
                )
            ],
            request=_request_cfg(
                image_requests={"primary": [0, 1]},
                include_actions=False,
            ),
        ),
        "stage1_mixed_nyu_asu": BenchmarkScenario(
            name="stage1_mixed_nyu_asu",
            description="mixed-source Stage 1 batch on lerobot/nyu_rot_dataset + lerobot/asu_table_top",
            output_format="stage1",
            sources=[
                _source_cfg(
                    repo_id="lerobot/nyu_rot_dataset",
                    camera_map={"primary": "observation.images.image"},
                    weight=0.5,
                    val_episode_count=2,
                ),
                _source_cfg(
                    repo_id="lerobot/asu_table_top",
                    camera_map={"primary": "observation.images.image"},
                    weight=0.5,
                    val_episode_count=5,
                ),
            ],
            request=_request_cfg(
                image_requests={"primary": [0, 1]},
                include_actions=False,
            ),
        ),
        "stage1_multicamera_cmu": BenchmarkScenario(
            name="stage1_multicamera_cmu",
            description="single-source Stage 1 multicamera pair batch on lerobot/cmu_franka_exploration_dataset",
            output_format="stage1",
            sources=[
                _source_cfg(
                    repo_id="lerobot/cmu_franka_exploration_dataset",
                    camera_map={
                        "primary": "observation.images.image",
                        "secondary": "observation.images.highres_image",
                    },
                    val_episode_count=10,
                )
            ],
            request=_request_cfg(
                image_requests={
                    "primary": [0, 1],
                    "secondary": [0, 1],
                },
                include_actions=False,
            ),
        ),
        "stage2_nyu": BenchmarkScenario(
            name="stage2_nyu",
            description="single-source Stage 2-style batch on lerobot/nyu_rot_dataset",
            output_format="stage2",
            sources=[
                _source_cfg(
                    repo_id="lerobot/nyu_rot_dataset",
                    camera_map={"primary": "observation.images.image"},
                    val_episode_count=2,
                )
            ],
            request=_request_cfg(
                image_requests={"primary": [0, 1]},
                include_actions=True,
            ),
        ),
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _rss_mb() -> float:
    process = psutil.Process()
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _estimate_sample_payload_bytes(batch: Any) -> int:
    total = 0
    if torch.is_tensor(batch):
        return _tensor_bytes(batch)
    if isinstance(batch, (str, bytes)) or batch is None:
        return 0
    if isinstance(batch, dict):
        return sum(_estimate_sample_payload_bytes(value) for value in batch.values())
    if isinstance(batch, (list, tuple)):
        return sum(_estimate_sample_payload_bytes(value) for value in batch)
    if isinstance(batch, (BatchedDatasetSample, Stage1Batch, Stage2Batch)):
        return sum(_estimate_sample_payload_bytes(value) for value in vars(batch).values())
    return 0


def _shutdown_loader_iterator(iterator: Any) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _measure_loader(*, loader, warmup_steps: int, measured_steps: int) -> dict[str, float]:
    iterator = iter(loader)
    batch_times: list[float] = []
    first_batch_s = float("nan")
    payload_mb = float("nan")
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
        "mean_batch_s": float(statistics.mean(batch_times)) if batch_times else float("nan"),
        "p50_batch_s": _percentile(batch_times, 50.0),
        "p90_batch_s": _percentile(batch_times, 90.0),
        "p99_batch_s": _percentile(batch_times, 99.0),
        "batches_per_s": (float(measured_steps) / total_measured_s) if total_measured_s > 0.0 else float("nan"),
        "samples_per_s": (float(measured_steps) * float(loader.batch_size) / total_measured_s)
        if total_measured_s > 0.0
        else float("nan"),
        "approx_payload_mb": payload_mb,
    }


def run_single_benchmark(
    *,
    scenario: BenchmarkScenario,
    batch_size: int,
    num_workers: int,
    warmup_steps: int,
    measured_steps: int,
) -> BenchmarkResult:
    loader_cfg = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": True,
        "prefetch_factor": 2 if int(num_workers) > 0 else None,
    }

    rss_before = _rss_mb()
    t0 = time.perf_counter()
    datamodule = LeRobotV3DataModule(
        sources=scenario.sources,
        request=scenario.request,
        loader=loader_cfg,
        adapter=_adapter_cfg(steps_per_epoch=max(8, warmup_steps + measured_steps + 2)),
        output_format=scenario.output_format,
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
        scenario=scenario.name,
        description=scenario.description,
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


def run_benchmark_grid(
    *,
    scenario_names: list[str],
    batch_sizes: list[int],
    num_workers_values: list[int],
    warmup_steps: int,
    measured_steps: int,
) -> list[BenchmarkResult]:
    scenarios = _scenarios()
    results: list[BenchmarkResult] = []
    for scenario_name in scenario_names:
        scenario = scenarios[scenario_name]
        for batch_size in batch_sizes:
            for num_workers in num_workers_values:
                print(
                    f"[bench] scenario={scenario_name} batch_size={batch_size} num_workers={num_workers}",
                    flush=True,
                )
                result = run_single_benchmark(
                    scenario=scenario,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    warmup_steps=warmup_steps,
                    measured_steps=measured_steps,
                )
                results.append(result)
                print(
                    f"  -> {result.samples_per_s:.1f} samples/s | {result.mean_batch_s:.4f}s mean batch | first={result.first_batch_s:.4f}s",
                    flush=True,
                )
    return results


def write_benchmark_results(*, output_dir: Path, results: list[BenchmarkResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.md"

    json_path.write_text(json.dumps([asdict(result) for result in results], indent=2))

    fieldnames = list(asdict(results[0]).keys()) if results else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    lines = [
        "# LeRobot v3 Dataloader Benchmark",
        "",
        "| scenario | batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.scenario} | {result.batch_size} | {result.num_workers} | {result.samples_per_s:.1f} | "
            f"{result.mean_batch_s:.4f} | {result.first_batch_s:.4f} | {result.approx_payload_mb:.2f} |"
        )
    summary_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    scenario_names = list(_scenarios().keys())
    parser = argparse.ArgumentParser(description="Benchmark LeRobot v3 dataloader throughput")
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        choices=scenario_names,
        help="Scenario name to benchmark. Repeat to select multiple scenarios. Defaults to all.",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--num-workers", nargs="+", type=int, default=[0, 2, 4, 8])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root
        / "experiments"
        / "benchmarking"
        / "results"
        / f"lerobot_v3_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = args.scenarios if args.scenarios else list(_scenarios().keys())
    results = run_benchmark_grid(
        scenario_names=scenarios,
        batch_sizes=[int(x) for x in args.batch_sizes],
        num_workers_values=[int(x) for x in args.num_workers],
        warmup_steps=int(args.warmup_steps),
        measured_steps=int(args.steps),
    )
    write_benchmark_results(output_dir=args.output_dir, results=results)
    print(f"[bench] wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
