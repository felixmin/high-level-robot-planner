#!/usr/bin/env python3
"""
Inspect a TFDS-prepared OXE (RLDS) dataset directory and print candidate keys.

This is a small helper for expanding `packages/common/adapters/oxe.py`:
- finds candidate image tensors under `steps[*].observation`
- prints the observed dict structure (shallow) and candidate keypaths

Example:
  conda run -n hlrp python scripts/inspect_oxe_dataset_sample.py \\
    gs://gresearch/robotics/mimic_play/0.0.1 \\
    gs://gresearch/robotics/berkeley_cable_routing/0.1.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class Candidate:
    keypath: str
    dtype: str
    shape: tuple[int, ...]


def _iter_candidates(obs: Any, prefix: str = "") -> Iterable[Candidate]:
    if isinstance(obs, dict):
        for k, v in obs.items():
            yield from _iter_candidates(v, prefix=f"{prefix}{k}/")
        return

    try:
        dtype = getattr(getattr(obs, "dtype", None), "name", None) or str(obs.dtype)
        shape = tuple(int(x) for x in obs.shape)
    except Exception:
        return

    keypath = prefix[:-1] if prefix.endswith("/") else prefix
    yield Candidate(keypath=keypath, dtype=dtype, shape=shape)


def _pick_split(builder) -> str:
    # Prefer train, but fall back to whatever exists.
    for name in ("train", "test", "val", "validation"):
        if name in builder.info.splits:
            return f"{name}[:1]"
    # Any split
    first = next(iter(builder.info.splits.keys()))
    return f"{first}[:1]"


def _get_first_step(builder_dir: str):
    import tensorflow_datasets as tfds

    b = tfds.builder_from_directory(builder_dir)
    split = _pick_split(b)
    ds = b.as_dataset(split=split)
    ep = next(iter(ds))
    step0 = next(iter(ep["steps"]))
    return split, step0


def _summarize_obs(obs: Any, max_depth: int = 2, prefix: str = "", depth: int = 0) -> None:
    if depth > max_depth:
        return
    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, dict):
                keys = list(v.keys())
                print(f"{prefix}{k}: dict keys={keys[:10]}")
                _summarize_obs(v, max_depth=max_depth, prefix=f"{prefix}{k}/", depth=depth + 1)
            else:
                try:
                    print(f"{prefix}{k}: shape={tuple(v.shape)} dtype={v.dtype}")
                except Exception:
                    print(f"{prefix}{k}: {type(v)}")
    else:
        print(f"{prefix}{type(obs)}")


def _best_image_key(cands: list[Candidate]) -> Optional[str]:
    # Heuristic: uint8 HxWx3
    imgs = [
        c
        for c in cands
        if c.dtype == "uint8"
        and len(c.shape) == 3
        and c.shape[-1] == 3
        and c.shape[0] > 0
        and c.shape[1] > 0
    ]
    if not imgs:
        return None
    # Prefer a conventional key name if present.
    preferred = ("image", "rgb", "front_rgb", "cam_high", "wrist_image")
    for p in preferred:
        for c in imgs:
            if c.keypath == p or c.keypath.endswith("/" + p):
                return c.keypath
    return imgs[0].keypath


def inspect(builder_dir: str) -> None:
    split, step0 = _get_first_step(builder_dir)
    obs = step0["observation"]

    print("\n===", builder_dir, "===")
    print("split:", split)
    print("observation structure:")
    _summarize_obs(obs, max_depth=3)

    cands = list(_iter_candidates(obs))
    image_key = _best_image_key(cands)

    image_cands = [
        c
        for c in cands
        if c.dtype == "uint8" and len(c.shape) == 3 and c.shape[-1] == 3
    ]
    str_cands = [c for c in cands if c.dtype == "string"]

    print("candidate image keypaths:")
    for c in image_cands[:30]:
        print(f"  - {c.keypath} (shape={c.shape})")
    if len(image_cands) > 30:
        print(f"  ... ({len(image_cands) - 30} more)")

    if str_cands:
        print("candidate string keypaths:")
        for c in str_cands[:20]:
            print(f"  - {c.keypath} (shape={c.shape})")
        if len(str_cands) > 20:
            print(f"  ... ({len(str_cands) - 20} more)")

    print("suggested:")
    if image_key is None:
        print("  image_key: <none found>")
    else:
        print(f"  image_key: {image_key}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("builder_dirs", nargs="+", help="TFDS prepared dataset dirs (local or gs://...)")
    args = ap.parse_args()

    for d in args.builder_dirs:
        inspect(str(d))


if __name__ == "__main__":
    main()
