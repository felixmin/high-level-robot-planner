#!/usr/bin/env python3
"""
Inspect how OXE TFDS datasets behave with TFDS `SkipDecoding()` on RLDS `steps`.

In this mode, TFDS may return:
- `tf.string` encoded image bytes (common)
- `tf.uint8` tensors for datasets that store raw images

This script checks, for a configured multi-dataset list:
- image keypath exists under steps/observation
- the dtype and shape we observe for that stream
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

import tensorflow as tf
import tensorflow_datasets as tfds

from common.adapters.oxe import OXE_DATASETS


def _get_keypath(container: Any, keypath: str):
    if keypath in container:
        return container[keypath]
    if "/" not in keypath:
        return container[keypath]
    cur = container
    for part in keypath.split("/"):
        cur = cur[part]
    return cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-yaml",
        type=str,
        default=str(workspace_root / "config/data/dataset/oxe_cluster_mirror_large.yaml"),
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--split",
        type=str,
        default="",
        help="Optional TFDS split override (e.g. 'train[:1]'). If empty, uses the YAML train_split.",
    )
    args = ap.parse_args()

    dataset_yaml = Path(args.dataset_yaml)
    cfg = yaml.safe_load(dataset_yaml.read_text())
    datasets = cfg["oxe"]["datasets"]
    if args.limit and args.limit > 0:
        datasets = datasets[: args.limit]

    # Keep TF on CPU for this inspection.
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    ok = 0
    bad = 0

    for entry in datasets:
        name = entry["name"]
        split = str(args.split) if args.split else entry.get("train_split", "train[:1]")
        if name not in OXE_DATASETS:
            print(f"[MISSING] {name}: not in OXE_DATASETS", flush=True)
            bad += 1
            continue

        builder_dir = OXE_DATASETS[name].gcs_path
        image_key = OXE_DATASETS[name].image_key
        print(f"[CHECK] {name}: split={split} builder_dir={builder_dir}", flush=True)
        try:
            b = tfds.builder_from_directory(builder_dir)
            ds = b.as_dataset(
                split=split,
                shuffle_files=False,
                decoders={"steps": tfds.decode.SkipDecoding()},
            )
            ep = next(iter(ds))
            obs = ep["steps"]["observation"]
            seq = _get_keypath(obs, image_key)
            dtype = getattr(seq, "dtype", None)
            shape = getattr(seq, "shape", None)
            dtype_name = getattr(dtype, "name", str(dtype))
            if dtype not in {tf.string, tf.uint8}:
                print(
                    f"[BAD] {name}: image_key={image_key} dtype={dtype_name} shape={shape}",
                    flush=True,
                )
                bad += 1
                continue
            print(
                f"[OK]  {name}: image_key={image_key} dtype={dtype_name} shape={shape}",
                flush=True,
            )
            ok += 1
        except Exception as e:
            print(f"[ERR] {name}: image_key={image_key} split={split} err={type(e).__name__}: {e}", flush=True)
            bad += 1

    print(f"\nSummary: ok={ok} bad={bad}", flush=True)


if __name__ == "__main__":
    main()
