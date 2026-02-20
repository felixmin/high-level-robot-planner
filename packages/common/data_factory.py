"""Create Lightning DataModules from the Hydra `cfg.data` schema."""

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
        backend: oxe_local_indexed
        preprocess: {image_size: int, return_metadata: bool}
        loader: {batch_size: int, num_workers: int, pin_memory: bool, prefetch_factor: int|null}
        dataset: {...}
        adapter: {...}
    """
    data = _to_dict(cfg_data)

    backend = data["backend"]
    preprocess = data["preprocess"]
    loader = data["loader"]
    dataset = data["dataset"]

    if backend != "oxe_local_indexed":
        raise ValueError(
            f"Only data.backend='oxe_local_indexed' is supported, got {backend!r}"
        )

    from common.data import OpenXLocalDataModule

    adapter = data["adapter"]
    oxe = dataset["oxe"]
    return OpenXLocalDataModule(
        datasets=list(oxe["datasets"]),
        preprocess=preprocess,
        loader=loader,
        adapter=adapter,
    )
