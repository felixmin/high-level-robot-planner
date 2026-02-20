"""
Create Lightning DataModules from the Hydra `cfg.data` schema.

This module is intentionally strict:
- backend selection is explicit (`data.backend`)
- required keys must be present (no silent fallbacks)
"""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


def _to_dict(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, DictConfig):
        out = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(out, dict):
            raise TypeError(f"Expected DictConfig -> dict, got {type(out)}")
        return out
    if isinstance(cfg, dict):
        return cfg
    raise TypeError(f"Expected DictConfig or dict, got {type(cfg)}")


def create_datamodule(cfg_data: Any):
    """
    Create a Lightning DataModule from `cfg.data`.

    Expected schema:
      data:
        backend: local_files | oxe_tf | oxe_tf_v2 | oxe_hf | oxe_local_indexed
        preprocess: {image_size: int, return_metadata: bool}
        loader: {batch_size: int, num_workers: int, pin_memory: bool, prefetch_factor: int|null}
        dataset: {...}   # backend-specific
        adapter: {...}   # backend-specific
        split: {...}     # local_files only
        subset: {...}    # local_files only
    """
    data = _to_dict(cfg_data)

    backend = data["backend"]
    preprocess = data["preprocess"]
    loader = data["loader"]
    dataset = data["dataset"]

    if backend == "local_files":
        from common.data import LAQDataModule

        split = data["split"]
        subset = data["subset"]
        local = dataset["local_files"]

        return LAQDataModule(
            sources=local["sources"],
            image_size=int(preprocess["image_size"]),
            batch_size=int(loader["batch_size"]),
            num_workers=int(loader["num_workers"]),
            pin_memory=bool(loader["pin_memory"]),
            prefetch_factor=loader["prefetch_factor"],
            min_frames=int(local["min_frames_per_scene"]),
            pair_offsets_frames=list(local["pair_offsets_frames"]),
            filters=local["filters"],
            return_metadata=bool(preprocess["return_metadata"]),
            split_mode=str(split["mode"]),
            split_seed=int(split["seed"]),
            val_ratio=float(split["val_ratio"]),
            val_scene_filters=split["val_scene_filters"],
            val_counts_per_dataset=split["val_counts_per_dataset"],
            subset_max_pairs=subset["max_pairs"],
            subset_strategy=str(subset["strategy"]),
            subset_seed=int(subset["seed"]),
        )

    if backend == "oxe_tf":
        from common.data import OXEDataModule

        adapter = data["adapter"]
        oxe = dataset["oxe"]
        return OXEDataModule(
            datasets=list(oxe["datasets"]),
            preprocess=preprocess,
            loader=loader,
            adapter=adapter,
        )

    if backend == "oxe_tf_v2":
        from common.data import OXEDataModuleV2

        adapter = data["adapter"]
        oxe = dataset["oxe"]
        return OXEDataModuleV2(
            datasets=list(oxe["datasets"]),
            preprocess=preprocess,
            loader=loader,
            adapter=adapter,
        )

    if backend == "oxe_hf":
        from common.adapters.huggingface_oxe import HFOXEDataModule

        adapter = data["adapter"]
        hf_oxe = dataset["hf_oxe"]
        return HFOXEDataModule(
            datasets=list(hf_oxe["datasets"]),
            preprocess=preprocess,
            loader=loader,
            adapter=adapter,
        )

    if backend == "oxe_local_indexed":
        from common.data import OpenXLocalDataModule

        adapter = data["adapter"]
        oxe = dataset["oxe"]
        return OpenXLocalDataModule(
            datasets=list(oxe["datasets"]),
            preprocess=preprocess,
            loader=loader,
            adapter=adapter,
        )

    raise ValueError(f"Unknown data.backend: {backend!r}")
