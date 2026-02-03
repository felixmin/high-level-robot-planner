#!/usr/bin/env python3
"""
Detailed benchmark for OXE TF streaming throughput.

This script is meant to diagnose *where* latency spikes come from when mixing
many datasets (especially when streaming from GCS).

It reports:
- per-batch fetch times
- dominant dataset per batch (requires return_metadata=true)
- switch vs stay latency (dominant dataset changes)
- first-seen batch index per dataset (to separate startup vs steady-state)
- slowest batches with dataset composition

Optionally, enable tf.data per-op stats:
  HLRP_OXE_TF_DATA_STATS=1 conda run -n hlrp python scripts/bench_oxe_dataloader_detailed.py ...

Example:
  CUDA_VISIBLE_DEVICES="" conda run -n hlrp python scripts/bench_oxe_dataloader_detailed.py \\
    logging.use_wandb=false data=laq_oxe_cluster_mirror_extended_smoke \\
    data.loader.batch_size=2 benchmark.warmup_steps=20 benchmark.steps=200 \\
    data.adapter.tf.tfds_read.source=gcs
"""

import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

import hydra
import numpy as np
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
class BatchInfo:
    idx: int
    dt_s: float
    fetch_dt_s: float
    materialize_dt_s: float
    is_warmup: bool
    dominant_dataset: Optional[str]
    dataset_counts: Dict[str, int]


def _normalize_dataset_names(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple)):
        out: list[str] = []
        for x in val:
            if x is None:
                out.append("")
            elif isinstance(x, bytes):
                out.append(x.decode("utf-8", errors="replace"))
            else:
                out.append(str(x))
        return out
    # Fallback: numpy array / torch tensor of strings/bytes
    try:
        if hasattr(val, "tolist"):
            return _normalize_dataset_names(val.tolist())
    except Exception:
        pass
    return [str(val)]

def _best_effort_materialize(obj: Any, leaf_budget: int = 32) -> None:
    """
    Best-effort eager materialization to force tf.data work to execute.

    We intentionally keep this lightweight (budgeted) because some batches can be large.
    """

    budget = int(leaf_budget)

    def _walk(x: Any) -> None:
        nonlocal budget
        if budget <= 0:
            return
        if x is None:
            return
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
            return
        if isinstance(x, (list, tuple)):
            for v in x:
                _walk(v)
            return
        # Eager tensors / numpy arrays
        try:
            if hasattr(x, "numpy"):
                _ = x.numpy()
                budget -= 1
        except Exception:
            return


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
            experiment_name=f"{OmegaConf.select(cfg, 'experiment.name')}_bench_oxe_dataloader_detailed",
        )

    logger, output_dir = setup_unified_logging(
        runs_dir=runs_dir,
        job_id=cfg.logging.get("job_id"),
        log_level=cfg.logging.get("level", "INFO"),
        logger_name="bench.oxe_dataloader_detailed",
    )

    cache_dir = resolve_cache_dir(cfg=cfg, workspace_root=workspace_root)
    if cache_dir is not None:
        configure_cache_env(cache_dir=cache_dir, logger=logger)

    bench_cfg = cfg.get("benchmark") or {}
    warmup_steps = int(bench_cfg.get("warmup_steps", 20))
    steps = int(bench_cfg.get("steps", 200))
    compute_sleep_s = float(bench_cfg.get("compute_sleep_s", 0.0))
    detailed_mode = str(bench_cfg.get("detailed_mode", "torch")).lower().strip()
    tf_materialize = bool(bench_cfg.get("tf_materialize", detailed_mode == "tf"))
    prime_each_dataset_samples = int(bench_cfg.get("prime_each_dataset_samples", 0))
    prime_materialize = bool(bench_cfg.get("prime_materialize", False))
    if detailed_mode not in {"torch", "tf"}:
        raise ValueError("benchmark.detailed_mode must be one of: 'torch', 'tf'")

    if cfg.data.backend != "oxe_tf":
        raise ValueError(
            f"bench_oxe_dataloader_detailed expects data.backend='oxe_tf', got {cfg.data.backend!r}"
        )

    datamodule = create_datamodule(cfg.data)
    datamodule.setup()

    # Optional: prime each child dataset once so the measured run reflects steady-state
    # throughput rather than "first-seen dataset" cold-start stalls.
    #
    # This is especially important for multi-dataset GCS streaming, where the first element
    # from a dataset can pay a large one-time cost (open first shard, initialize readers,
    # compile tf.function graph, etc.).
    if prime_each_dataset_samples > 0:
        train_ds = getattr(datamodule, "train_dataset", None)
        children = []
        try:
            if train_ds is not None and hasattr(train_ds, "_init_datasets"):
                train_ds._init_datasets()
            children = list(getattr(train_ds, "_datasets", []) or [])
        except Exception:
            children = []

        if children:
            logger.info(
                f"Priming per-dataset pipelines: n_datasets={len(children)} samples_each={prime_each_dataset_samples} materialize={prime_materialize}"
            )
            prime_start = time.perf_counter()
            for child in children:
                child_name = getattr(getattr(child, "config", None), "name", None)
                if child_name is None:
                    child_name = getattr(child, "dataset_name", None) or "dataset"
                t0 = time.perf_counter()
                try:
                    child_pipe = child._get_or_create_pipeline()
                    child_it = iter(child_pipe)
                    for _ in range(prime_each_dataset_samples):
                        elem = next(child_it)
                        if prime_materialize:
                            _best_effort_materialize(elem)
                    logger.info(
                        f"  - primed {child_name}: {time.perf_counter() - t0:.2f}s"
                    )
                except Exception as e:
                    logger.warning(f"  - priming failed for {child_name}: {e}")
            logger.info(
                f"✓ Priming done in {time.perf_counter() - prime_start:.1f}s"
            )
        else:
            logger.info("Priming requested but no child datasets found; skipping.")

    dl = datamodule.train_dataloader()

    dataset_names = [d.name for d in cfg.data.dataset.oxe.datasets]
    logger.info("OXE detailed dataloader benchmark")
    logger.info(f"  - datasets: {dataset_names} (n={len(dataset_names)})")
    logger.info(f"  - batch_size: {cfg.data.loader.batch_size}")
    logger.info(f"  - warmup_steps: {warmup_steps}")
    logger.info(f"  - measured_steps: {steps}")
    logger.info(f"  - compute_sleep_s: {compute_sleep_s}")
    return_metadata = bool(OmegaConf.select(cfg, "data.preprocess.return_metadata", default=False))
    logger.info(f"  - detailed_mode: {detailed_mode}")
    if detailed_mode == "tf":
        logger.info(f"  - tf_materialize: {tf_materialize}")
    logger.info(f"  - return_metadata: {return_metadata}")
    if prime_each_dataset_samples > 0:
        logger.info(
            f"  - prime_each_dataset_samples: {prime_each_dataset_samples} (materialize={prime_materialize})"
        )

    total = warmup_steps + steps

    # Two measurement modes:
    # - torch: time `next(iter(torch_dataloader))` (includes TF pipeline + TF->PyTorch conversion)
    # - tf: time `next(iter(tf_dataset))` and optionally materialize tensors, to isolate TF-side stalls
    it = iter(dl)
    tf_iter = None
    if detailed_mode == "tf":
        train_ds = getattr(datamodule, "train_dataset", None)
        if train_ds is None or not hasattr(train_ds, "_get_or_create_pipeline"):
            raise TypeError(
                "TF mode requires an OXE streaming dataset with `_get_or_create_pipeline()`"
            )
        tf_ds = train_ds._get_or_create_pipeline()
        tf_iter = iter(tf_ds)

    batch_infos: list[BatchInfo] = []
    first_seen: dict[str, int] = {}
    per_dominant: dict[str, list[float]] = defaultdict(list)
    switch_times: list[float] = []
    stay_times: list[float] = []
    prev_dom: Optional[str] = None

    slowest: list[BatchInfo] = []
    slowest_k = int(bench_cfg.get("detailed_top_k", 10))

    overall_start = time.perf_counter()
    measured_dts: list[float] = []
    measured_fetch_dts: list[float] = []
    measured_materialize_dts: list[float] = []

    for idx in range(total):
        t0 = time.perf_counter()
        if detailed_mode == "tf":
            batch = next(tf_iter)  # type: ignore[arg-type]
        else:
            batch = next(it)
        fetch_dt = float(time.perf_counter() - t0)

        materialize_dt = 0.0
        if detailed_mode == "tf" and tf_materialize:
            t1 = time.perf_counter()
            try:
                if return_metadata:
                    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                        raise TypeError(
                            "TF mode with return_metadata=true expects elements as (pair_tf, meta_tf)"
                        )
                    pair_tf, _meta_tf = batch
                    _ = pair_tf.numpy()
                else:
                    _ = batch.numpy()  # type: ignore[union-attr]
            except Exception:
                # Best-effort materialization. If this fails, keep going to still surface
                # iterator stalls (fetch_dt).
                pass
            materialize_dt = float(time.perf_counter() - t1)

        dt = fetch_dt + materialize_dt

        if compute_sleep_s > 0:
            time.sleep(compute_sleep_s)

        dominant: Optional[str] = None
        counts: Dict[str, int] = {}

        if return_metadata:
            if detailed_mode == "tf":
                # TF mode returns either:
                # - (pair_tf, meta_tf) for return_metadata=true
                # - pair_tf for return_metadata=false
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    raise TypeError(
                        "TF mode with return_metadata=true expects elements as (pair_tf, meta_tf)"
                    )
                _pair_tf, meta_tf = batch
                try:
                    # meta_tf["dataset_name"] is a vector of strings (batched).
                    raw = meta_tf["dataset_name"].numpy()
                    names = _normalize_dataset_names(raw)
                except Exception:
                    names = []
            else:
                if isinstance(batch, dict) and "dataset_name" in batch:
                    names = _normalize_dataset_names(batch.get("dataset_name"))
                else:
                    names = []

            if names:
                c = Counter(names)
                counts = dict(c)
                dominant = c.most_common(1)[0][0] if c else None
                for name in c.keys():
                    if name not in first_seen:
                        first_seen[name] = idx

        is_warmup = idx < warmup_steps
        info = BatchInfo(
            idx=idx,
            dt_s=dt,
            fetch_dt_s=fetch_dt,
            materialize_dt_s=materialize_dt,
            is_warmup=is_warmup,
            dominant_dataset=dominant,
            dataset_counts=counts,
        )
        batch_infos.append(info)

        if not is_warmup:
            measured_dts.append(dt)
            measured_fetch_dts.append(fetch_dt)
            measured_materialize_dts.append(materialize_dt)

        if (dominant is not None) and (not is_warmup):
            per_dominant[dominant].append(dt)
            if prev_dom is None:
                prev_dom = dominant
            else:
                if dominant != prev_dom:
                    switch_times.append(dt)
                else:
                    stay_times.append(dt)
                prev_dom = dominant

        # Maintain top-k slowest
        if slowest_k > 0:
            slowest.append(info)
            slowest.sort(key=lambda x: x.dt_s, reverse=True)
            if len(slowest) > slowest_k:
                slowest.pop()

    overall_dt = time.perf_counter() - overall_start

    mean_s = float(np.mean(np.asarray(measured_dts, dtype=np.float64))) if measured_dts else float("nan")
    p50_s = _percentile(measured_dts, 50.0)
    p90_s = _percentile(measured_dts, 90.0)

    bs = int(cfg.data.loader.batch_size)
    samples_per_sec = (float(bs) * float(len(measured_dts))) / float(sum(measured_dts)) if measured_dts else float("nan")
    batches_per_sec = float(len(measured_dts)) / float(sum(measured_dts)) if measured_dts else float("nan")

    logger.info("Results (overall)")
    logger.info(f"  - mean batch time: {mean_s:.4f}s")
    logger.info(f"  - p50 batch time:  {p50_s:.4f}s")
    logger.info(f"  - p90 batch time:  {p90_s:.4f}s")
    if detailed_mode == "tf":
        logger.info(
            f"  - mean fetch:      {float(np.mean(measured_fetch_dts)) if measured_fetch_dts else float('nan'):.4f}s"
        )
        logger.info(
            f"  - mean materialize:{float(np.mean(measured_materialize_dts)) if measured_materialize_dts else float('nan'):.4f}s"
        )
    logger.info(f"  - batches/s:       {batches_per_sec:.3f}")
    logger.info(f"  - samples/s:       {samples_per_sec:.3f}")
    logger.info(f"  - total wall time: {overall_dt:.1f}s")

    if per_dominant:
        logger.info("Results (by dominant dataset)")
        for name, dts in sorted(per_dominant.items(), key=lambda kv: np.mean(kv[1]) if kv[1] else 0.0, reverse=True):
            logger.info(
                f"  - {name}: n={len(dts)} mean={float(np.mean(dts)):.4f}s p50={_percentile(dts,50.0):.4f}s p90={_percentile(dts,90.0):.4f}s first_seen_batch={first_seen.get(name)}"
            )

    if switch_times or stay_times:
        logger.info("Switching analysis (dominant dataset)")
        if switch_times:
            logger.info(
                f"  - switch batches: n={len(switch_times)} mean={float(np.mean(switch_times)):.4f}s p90={_percentile(switch_times,90.0):.4f}s"
            )
        if stay_times:
            logger.info(
                f"  - stay batches:   n={len(stay_times)} mean={float(np.mean(stay_times)):.4f}s p90={_percentile(stay_times,90.0):.4f}s"
            )

    if slowest:
        logger.info(f"Top {len(slowest)} slowest batches")
        for b in slowest:
            if b.dominant_dataset is None:
                logger.info(
                    f"  - idx={b.idx} dt={b.dt_s:.4f}s warmup={b.is_warmup} (no dataset_name)"
                )
            else:
                logger.info(
                    f"  - idx={b.idx} dt={b.dt_s:.4f}s warmup={b.is_warmup} dom={b.dominant_dataset} counts={b.dataset_counts}"
                )

    # Persist raw batch times for offline analysis.
    out_jsonl = Path(output_dir) / "batch_times.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for b in batch_infos:
            f.write(
                json.dumps(
                    {
                        "idx": b.idx,
                        "dt_s": b.dt_s,
                        "fetch_dt_s": b.fetch_dt_s,
                        "materialize_dt_s": b.materialize_dt_s,
                        "is_warmup": b.is_warmup,
                        "dominant_dataset": b.dominant_dataset,
                        "dataset_counts": b.dataset_counts,
                    }
                )
                + "\n"
            )
    logger.info(f"✓ Wrote {len(batch_infos)} batch records: {out_jsonl}")

    # If tf.data stats are enabled, attempt to dump summaries.
    try:
        tf_stats_path = Path(output_dir) / "tf_data_stats.txt"
        summaries: list[str] = []
        train_ds = getattr(datamodule, "train_dataset", None)
        if train_ds is not None and hasattr(train_ds, "tf_data_stats_summary"):
            s = train_ds.tf_data_stats_summary()
            if s:
                summaries.append("=== mixed ===\n" + s)
        # Multi dataset: include per-dataset summaries if available.
        for child in getattr(train_ds, "_datasets", []) or []:
            if hasattr(child, "tf_data_stats_summary"):
                s = child.tf_data_stats_summary()
                if s:
                    summaries.append(f"=== {getattr(child, 'config', None).name if getattr(child,'config',None) else 'dataset'} ===\n{s}")
        if summaries:
            tf_stats_path.write_text("\n\n".join(summaries), encoding="utf-8")
            logger.info(f"✓ Wrote tf.data stats: {tf_stats_path}")
    except Exception as e:
        logger.warning(f"Failed to write tf.data stats summaries: {e}")


if __name__ == "__main__":
    main()
