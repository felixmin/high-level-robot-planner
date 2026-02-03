#!/usr/bin/env python3
"""
Experimental benchmark: Python-level mixer with per-dataset background prefetch.

Motivation
----------
tf.data mixing ops (`choose_from_datasets` / `sample_from_datasets`) typically only
advance the currently-selected dataset pipeline. When mixing many GCS-streamed
datasets this can produce rare but very large stalls on dataset switches.

This script tests an alternative: keep *all* dataset pipelines "hot" by running
one lightweight background thread per dataset, prefetching batches into a small
queue, and then mixing batches in Python by weighted sampling from those queues.

This is meant for experimentation only (not a production training path yet).
"""

import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Optional

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from common.cache_env import configure_cache_env, resolve_cache_dir
from common.data_factory import create_datamodule
from common.unified_logging import resolve_runs_dir, setup_unified_logging


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


@dataclass
class BatchRec:
    idx: int
    dt_s: float
    dataset_name: str


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    runs_dir = None
    try:
        if HydraConfig.initialized():
            runs_dir = Path(str(HydraConfig.get().runtime.output_dir))
    except Exception:
        runs_dir = None
    if runs_dir is None:
        runs_dir = resolve_runs_dir(
            logging_root_dir=cfg.logging.get("root_dir"),
            logging_runs_dir=cfg.logging.get("runs_dir"),
            workspace_root=workspace_root,
            experiment_name=f"{OmegaConf.select(cfg, 'experiment.name')}_bench_oxe_python_prefetch",
        )

    logger, output_dir = setup_unified_logging(
        runs_dir=runs_dir,
        job_id=cfg.logging.get("job_id"),
        log_level=cfg.logging.get("level", "INFO"),
        logger_name="bench.oxe_python_prefetch",
    )

    cache_dir = resolve_cache_dir(cfg=cfg, workspace_root=workspace_root)
    if cache_dir is not None:
        configure_cache_env(cache_dir=cache_dir, logger=logger)

    bench_cfg = cfg["benchmark"]
    warmup_steps = int(bench_cfg["warmup_steps"])
    steps = int(bench_cfg["steps"])
    queue_size = int(bench_cfg["python_prefetch_queue_size"])
    min_ready = int(bench_cfg["python_prefetch_min_ready_datasets"])
    if queue_size <= 0:
        raise ValueError("benchmark.python_prefetch_queue_size must be >= 1")
    if min_ready <= 0:
        raise ValueError("benchmark.python_prefetch_min_ready_datasets must be >= 1")

    bs = int(cfg.data.loader.batch_size)

    if cfg.data.backend != "oxe_tf":
        raise ValueError(
            f"bench_oxe_python_prefetch_mixer expects data.backend='oxe_tf', got {cfg.data.backend!r}"
        )

    if bool(OmegaConf.select(cfg, "data.preprocess.return_metadata", default=False)):
        raise ValueError("This benchmark expects data.preprocess.return_metadata=false")

    logger.info("OXE python-prefetch mixer benchmark")
    logger.info(f"  - batch_size: {bs}")
    logger.info(f"  - warmup_steps: {warmup_steps}")
    logger.info(f"  - measured_steps: {steps}")
    logger.info(f"  - per-dataset queue_size: {queue_size}")

    datamodule = create_datamodule(cfg.data)
    datamodule.setup()
    train_ds = getattr(datamodule, "train_dataset", None)
    if train_ds is None or not hasattr(train_ds, "_init_datasets"):
        raise TypeError("Expected an OXE TF Multi dataset with `_init_datasets()`")

    train_ds._init_datasets()
    children = list(getattr(train_ds, "_datasets", []) or [])
    weights = list(getattr(train_ds, "_weights", []) or [])
    if not children or not weights or len(children) != len(weights):
        raise RuntimeError("Failed to access child datasets/weights for python mixer benchmark")

    # Import TF in the same way as the adapter (CPU-only).
    from common.adapters.oxe import _import_tensorflow_cpu_only  # noqa: PLC0415

    tf = _import_tensorflow_cpu_only()

    dataset_names: list[str] = []
    queues: list[Queue] = []
    threads: list[Thread] = []
    stop = Event()

    for child in children:
        name = getattr(getattr(child, "config", None), "name", None) or "dataset"
        dataset_names.append(str(name))
        queues.append(Queue(maxsize=queue_size))

    # Normalize weights for sampling.
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    rng = np.random.default_rng(int(cfg.seed))

    def _worker(i: int) -> None:
        name = dataset_names[i]
        q = queues[i]
        try:
            pipe = children[i]._get_or_create_pipeline()
            pipe = pipe.batch(bs, drop_remainder=True).prefetch(1)
            it = iter(pipe)
            while not stop.is_set():
                # Backpressure when the queue is full.
                item = next(it)
                q.put(item)
        except Exception as e:
            logger.error(f"worker failed for {name}: {e}")
            stop.set()

    logger.info(f"Starting {len(children)} prefetch threads")
    for i in range(len(children)):
        t = Thread(target=_worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    # Wait until at least `min_ready` datasets have produced one batch.
    start_wait = time.perf_counter()
    while not stop.is_set():
        ready = sum(1 for q in queues if q.qsize() > 0)
        if ready >= min_ready:
            break
        if time.perf_counter() - start_wait > 600:
            raise TimeoutError("Timed out waiting for dataset queues to fill")
        time.sleep(0.05)
    logger.info(
        f"Initial queue fill: {time.perf_counter() - start_wait:.1f}s (ready={ready}/{len(queues)})"
    )

    total = warmup_steps + steps
    dts: list[float] = []
    recs: list[BatchRec] = []
    per_ds: dict[str, list[float]] = defaultdict(list)

    # Consume batches with weighted sampling over dataset queues.
    # IMPORTANT: we always sample from the full distribution and block on `get()`
    # so the measured time includes any starvation (queue empty) events.
    for idx in range(total):
        if stop.is_set():
            raise RuntimeError("Stopping early due to background worker failure")
        ds_idx = int(rng.choice(len(children), p=w))
        name = dataset_names[ds_idx]
        t0 = time.perf_counter()
        item = queues[ds_idx].get()

        # Convert to torch (similar to the adapter’s DLPack path).
        try:
            pt = torch.utils.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(item))
        except Exception:
            pt = torch.from_numpy(item.numpy())
        _ = pt.permute(0, 4, 1, 2, 3)

        dt = float(time.perf_counter() - t0)
        if idx >= warmup_steps:
            dts.append(dt)
            per_ds[name].append(dt)
            recs.append(BatchRec(idx=idx, dt_s=dt, dataset_name=name))

    stop.set()
    for t in threads:
        t.join(timeout=1.0)

    mean_s = float(np.mean(np.asarray(dts, dtype=np.float64))) if dts else float("nan")
    p50_s = _percentile(dts, 50.0)
    p90_s = _percentile(dts, 90.0)
    batches_per_s = float(len(dts)) / float(sum(dts)) if dts else float("nan")
    samples_per_s = float(bs) * batches_per_s if dts else float("nan")

    logger.info("Results (python-prefetch mixer)")
    logger.info(f"  - mean batch time: {mean_s:.4f}s")
    logger.info(f"  - p50 batch time:  {p50_s:.4f}s")
    logger.info(f"  - p90 batch time:  {p90_s:.4f}s")
    logger.info(f"  - batches/s:       {batches_per_s:.3f}")
    logger.info(f"  - samples/s:       {samples_per_s:.3f}")

    # Small per-dataset summary (mean).
    slowest = sorted(per_ds.items(), key=lambda kv: np.mean(kv[1]) if kv[1] else 0.0, reverse=True)[:8]
    logger.info("Slowest datasets (mean batch time, top 8)")
    for name, xs in slowest:
        logger.info(f"  - {name}: n={len(xs)} mean={float(np.mean(xs)):.4f}s p90={_percentile(xs,90.0):.4f}s")

    out_path = Path(output_dir) / "python_prefetch_batch_times.txt"
    out_path.write_text(
        "\n".join([f"{r.idx}\t{r.dataset_name}\t{r.dt_s:.6f}" for r in recs]),
        encoding="utf-8",
    )
    logger.info(f"✓ Wrote batch records: {out_path}")


if __name__ == "__main__":
    main()
