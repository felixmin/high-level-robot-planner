#!/usr/bin/env python3
"""
Benchmark OXE dataloader throughput without training.

This isolates tf.data + TF->Torch conversion + PyTorch DataLoader collation from
model/optimizer overhead.

Example:
  conda run -n hlrp python scripts/bench_oxe_dataloader.py \\
    experiment=laq_oxe_local \\
    benchmark.steps=300 benchmark.warmup_steps=20 \\
    data.loader.batch_size=128 \\
    'data.dataset.oxe.datasets=[{name:language_table,train_split:\"train[:10000]\",val_split:\"train[10000:10020]\",pair_offset_steps:10,weight:1.0,approx_num_pairs:1000000}]'
"""

import sys
import time
from pathlib import Path

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

import hydra
import lightning.pytorch as pl
import numpy as np
from omegaconf import DictConfig, OmegaConf

from common.data_factory import create_datamodule
from common.unified_logging import resolve_runs_dir, setup_unified_logging


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    runs_dir = resolve_runs_dir(
        logging_root_dir=cfg.logging.get("root_dir"),
        logging_runs_dir=cfg.logging.get("runs_dir"),
        workspace_root=workspace_root,
        experiment_name=f"{OmegaConf.select(cfg, 'experiment.name')}_bench_oxe_dataloader",
    )
    logger, output_dir = setup_unified_logging(
        runs_dir=runs_dir,
        job_id=cfg.logging.get("job_id"),
        log_level=cfg.logging.get("level", "INFO"),
        logger_name="bench.oxe_dataloader",
    )

    bench_cfg = cfg.get("benchmark") or {}
    warmup_steps = int(bench_cfg.get("warmup_steps", 20))
    steps = int(bench_cfg.get("steps", 200))
    compute_sleep_s = float(bench_cfg.get("compute_sleep_s", 0.0))
    tf_profile = bool(bench_cfg.get("tf_profile", False))
    torch_profile = bool(bench_cfg.get("torch_profile", False))

    if cfg.data.backend != "oxe_tf":
        raise ValueError(
            f"bench_oxe_dataloader expects data.backend='oxe_tf', got {cfg.data.backend!r}"
        )

    datamodule = create_datamodule(cfg.data)
    datamodule.setup()

    dataset_names = [d.name for d in cfg.data.dataset.oxe.datasets]
    logger.info("OXE dataloader benchmark")
    logger.info(f"  - datasets: {dataset_names} (n={len(dataset_names)})")
    logger.info(f"  - batch_size: {cfg.data.loader.batch_size}")
    logger.info(f"  - warmup_steps: {warmup_steps}")
    logger.info(f"  - measured_steps: {steps}")
    logger.info(f"  - compute_sleep_s: {compute_sleep_s}")
    logger.info(f"  - tf_profile: {tf_profile}")
    logger.info(f"  - torch_profile: {torch_profile}")

    dl = datamodule.train_dataloader()

    tf_profiler_ctx = None
    if tf_profile:
        try:
            from common.adapters.oxe import _import_tensorflow_cpu_only

            tf = _import_tensorflow_cpu_only()
            tf_profile_dir = Path(output_dir) / "tf_profile"
            tf_profile_dir.mkdir(parents=True, exist_ok=True)
            tf.profiler.experimental.start(str(tf_profile_dir))
            tf_profiler_ctx = tf_profile_dir
            logger.info(f"✓ TensorFlow profiler started: {tf_profile_dir}")
        except Exception as e:
            logger.warning(f"Failed to start TensorFlow profiler: {e}")

    torch_prof = None
    if torch_profile:
        try:
            import torch

            torch_profile_dir = Path(output_dir) / "torch_profile"
            torch_profile_dir.mkdir(parents=True, exist_ok=True)
            torch_prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(torch_profile_dir)
                ),
            )
            torch_prof.__enter__()
            logger.info(f"✓ Torch profiler started: {torch_profile_dir}")
        except Exception as e:
            logger.warning(f"Failed to start Torch profiler: {e}")
            torch_prof = None

    batch_times_s: list[float] = []
    measured = 0
    total = warmup_steps + steps
    overall_start = time.perf_counter()

    it = iter(dl)
    for i in range(total):
        t0 = time.perf_counter()
        batch = next(it)
        _ = getattr(batch, "shape", None)
        if torch_prof is not None:
            torch_prof.step()
        dt = time.perf_counter() - t0
        if i >= warmup_steps:
            batch_times_s.append(float(dt))
            measured += 1
        if compute_sleep_s > 0:
            time.sleep(compute_sleep_s)

    overall_dt = time.perf_counter() - overall_start

    if torch_prof is not None:
        try:
            torch_prof.__exit__(None, None, None)
        except Exception:
            pass

    if tf_profiler_ctx is not None:
        try:
            from common.adapters.oxe import _import_tensorflow_cpu_only

            tf = _import_tensorflow_cpu_only()
            tf.profiler.experimental.stop()
            logger.info("✓ TensorFlow profiler stopped")
        except Exception:
            pass

    mean_s = float(np.mean(np.asarray(batch_times_s, dtype=np.float64))) if batch_times_s else float("nan")
    p50_s = _percentile(batch_times_s, 50.0)
    p90_s = _percentile(batch_times_s, 90.0)

    bs = int(cfg.data.loader.batch_size)
    samples_per_sec = (float(bs) * float(measured)) / float(sum(batch_times_s)) if batch_times_s else float("nan")
    batches_per_sec = float(measured) / float(sum(batch_times_s)) if batch_times_s else float("nan")

    logger.info("Results")
    logger.info(f"  - mean batch time: {mean_s:.4f}s")
    logger.info(f"  - p50 batch time:  {p50_s:.4f}s")
    logger.info(f"  - p90 batch time:  {p90_s:.4f}s")
    logger.info(f"  - batches/s:       {batches_per_sec:.2f}")
    logger.info(f"  - samples/s:       {samples_per_sec:.1f}")
    logger.info(f"  - total wall time: {overall_dt:.1f}s")

    # Keep Lightning from warning about missing trainer state when datamodule cleans up.
    _ = pl.Trainer(enable_checkpointing=False, logger=False)


if __name__ == "__main__":
    main()
