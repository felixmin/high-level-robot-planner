"""Shared LAQ checkpoint loading utilities."""

from __future__ import annotations

import inspect
from typing import Union

import torch
from omegaconf import OmegaConf

from laq.inference import LAQEncoderVQInference
from laq.task import LAQTask


def load_laq_task_from_checkpoint(
    checkpoint_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> LAQTask:
    """
    Load a full LAQTask from a Lightning checkpoint.

    Tries LAQTask.load_from_checkpoint first; falls back to manual
    hparam extraction + state_dict load for cross-version compatibility.
    """
    ckpt_path = str(checkpoint_path)

    try:
        load_kwargs: dict = {"map_location": map_location, "weights_only": False}
        if "strict" in inspect.signature(LAQTask.load_from_checkpoint).parameters:
            load_kwargs["strict"] = strict
        return LAQTask.load_from_checkpoint(ckpt_path, **load_kwargs)
    except TypeError:
        pass
    except RuntimeError as exc:
        if "weights_only" not in str(exc).lower():
            raise

    load_kwargs: dict = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    checkpoint = torch.load(ckpt_path, **load_kwargs)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")

    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise KeyError("Checkpoint missing 'hyper_parameters' dict")

    model_config = hparams.get("model_config")
    training_config = hparams.get("training_config")
    if isinstance(model_config, dict):
        model_config = OmegaConf.create(model_config)
    if isinstance(training_config, dict):
        training_config = OmegaConf.create(training_config)
    if model_config is None or training_config is None:
        raise KeyError("Checkpoint hyper_parameters missing model_config or training_config")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint missing 'state_dict'")

    task = LAQTask(model_config=model_config, training_config=training_config)
    task.load_state_dict(state_dict, strict=strict)
    return task


def load_laq_model_weights_only(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    strip_model_prefix: bool = True,
    drop_optimizer_keys: bool = True,
) -> tuple[list[str], list[str], int]:
    """
    Load model weights from a checkpoint, ignoring optimizer/scheduler state.

    Returns (missing_keys, unexpected_keys, loaded_tensor_count).
    """
    load_kwargs: dict = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    checkpoint = torch.load(str(checkpoint_path), **load_kwargs)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict dict, got {type(state_dict)}")

    if drop_optimizer_keys:
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith(("optimizer", "lr_scheduler"))
        }
    if strip_model_prefix:
        state_dict = {
            k[len("model."):] if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected), len(state_dict)


def load_laq_encoder_vq_inference_from_checkpoint(
    checkpoint_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    prune_decoders: bool = True,
) -> LAQEncoderVQInference:
    """
    Load a LAQEncoderVQInference from a checkpoint.

    Loads the full task then prunes decoder/teacher modules to reclaim VRAM.
    Caller is responsible for setting eval mode and moving to the target device.
    """
    task = load_laq_task_from_checkpoint(checkpoint_path, map_location=map_location, strict=strict)
    return LAQEncoderVQInference(task.model, prune_decoders=prune_decoders)
