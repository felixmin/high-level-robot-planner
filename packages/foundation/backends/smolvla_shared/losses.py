from __future__ import annotations

import torch
import torch.nn.functional as F

from foundation.backends.interfaces import FoundationBatch
from foundation.backends.smolvla_shared.model import SmolVLASharedCore


def latent_flow_loss(
    *,
    core: SmolVLASharedCore,
    batch: FoundationBatch,
    target_vectors: torch.Tensor,
) -> torch.Tensor:
    return core.latent_flow_loss(batch=batch, target_vectors=target_vectors)


def action_regression_loss(
    *,
    core: SmolVLASharedCore,
    batch: FoundationBatch,
    target_actions: torch.Tensor,
) -> torch.Tensor:
    pred_actions = core.predict_actions(batch=batch)
    if target_actions.ndim != 2:
        raise ValueError(f"Expected target_actions [B,A], got {tuple(target_actions.shape)}")
    target = target_actions.to(device=pred_actions.device, dtype=pred_actions.dtype)
    if target.shape != pred_actions.shape:
        raise ValueError(
            f"target_actions shape mismatch: expected {tuple(pred_actions.shape)}, got {tuple(target.shape)}"
        )
    return F.mse_loss(pred_actions, target)
