"""
Optical Flow Supervision for LAQ.

Provides motion-enriched latent representations by training the VAE's latent
to predict optical flow between frames. Uses online knowledge distillation
from a frozen RAFT teacher model.

Components:
- RAFTTeacher: Frozen optical flow model for ground-truth generation
- FlowDecoder: Transformer that predicts flow from latent + context image
"""

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)

# Supported RAFT variants
FlowModelType = Literal["raft_small", "raft_large"]


@dataclass
class FlowConfig:
    """Configuration for flow supervision.

    All fields are required when flow supervision is enabled.
    No defaults - caller must explicitly configure.
    """
    model: FlowModelType
    loss_weight: float
    decoder_depth: int
    warmup_steps: int = 0  # Steps to linearly ramp up flow loss (0 = no warmup)

    def __post_init__(self):
        if self.model not in ("raft_small", "raft_large"):
            raise ValueError(f"flow.model must be 'raft_small' or 'raft_large', got '{self.model}'")
        if self.loss_weight <= 0:
            raise ValueError(f"flow.loss_weight must be positive, got {self.loss_weight}")
        if self.decoder_depth <= 0:
            raise ValueError(f"flow.decoder_depth must be positive, got {self.decoder_depth}")
        if self.warmup_steps < 0:
            raise ValueError(f"flow.warmup_steps must be non-negative, got {self.warmup_steps}")

    def get_weight(self, step: int) -> float:
        """Get effective loss weight at given training step.

        Linearly ramps from 0 to loss_weight over warmup_steps.
        """
        if self.warmup_steps == 0:
            return self.loss_weight

        warmup_factor = min(1.0, step / self.warmup_steps)
        return self.loss_weight * warmup_factor


class RAFTTeacher(nn.Module):
    """
    Frozen RAFT optical flow model for ground-truth generation.

    Not registered as a submodule to avoid checkpoint pollution.
    Weights are always loaded fresh from torchvision on first use.

    Uses half precision (float16) for efficiency and official transforms
    for proper input normalization.
    """

    def __init__(self, model_name: FlowModelType):
        super().__init__()
        self._model_name = model_name
        self._model: Optional[nn.Module] = None
        self._transforms = None

    def _load_model(self, device: torch.device) -> nn.Module:
        """Lazy-load RAFT model on first use."""
        if self._model is not None:
            return self._model

        from torchvision.models.optical_flow import (
            raft_small, raft_large,
            Raft_Small_Weights, Raft_Large_Weights
        )

        if self._model_name == "raft_small":
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights)
            logger.info("Loaded RAFT-Small optical flow teacher")
        else:
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights)
            logger.info("Loaded RAFT-Large optical flow teacher")

        # Store official transforms for proper normalization
        self._transforms = weights.transforms()

        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self._model = model
        return model

    @torch.no_grad()
    def compute_flow(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute optical flow between two frames.

        Uses autocast for mixed precision when on CUDA for efficiency.

        Args:
            frame1: First frame [B, C, 1, H, W] in [0, 1] range
            frame2: Second frame [B, C, 1, H, W] in [0, 1] range

        Returns:
            flow: Optical flow field [B, 2, H, W] (dx, dy per pixel)
        """
        model = self._load_model(frame1.device)

        # Remove time dimension: [B, C, 1, H, W] -> [B, C, H, W]
        img1 = frame1.squeeze(2)
        img2 = frame2.squeeze(2)

        # Scale to [0, 255] range and convert to uint8 for transforms
        img1 = (img1 * 255.0).clamp(0, 255).to(torch.uint8)
        img2 = (img2 * 255.0).clamp(0, 255).to(torch.uint8)

        # Apply official RAFT transforms (handles normalization)
        img1_t, img2_t = self._transforms(img1, img2)
        img1_t = img1_t.contiguous()
        img2_t = img2_t.contiguous()

        # Use autocast for mixed precision on CUDA (faster, less VRAM)
        device_type = "cuda" if frame1.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=frame1.is_cuda):
            # RAFT returns list of flow predictions at different refinement levels
            # Take the last (most refined) prediction
            flow_predictions = model(img1_t, img2_t)

        # Return flow in full precision for loss computation
        return flow_predictions[-1].float()

    def state_dict(self, *args, **kwargs):
        """Override to prevent RAFT weights from being saved."""
        return {}

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Override to skip loading (RAFT is always loaded fresh)."""
        pass


class FlowDecoder(nn.Module):
    """
    Transformer decoder that predicts optical flow from latent action + context image.

    Architecture mirrors the auxiliary pixel decoder:
    - Input: Pixel context tokens from first frame (spatial layout)
    - Cross-attention context: Quantized latent action (motion encoding)
    - Output: Dense optical flow field [B, 2, H, W]

    The latent encodes "what motion happened" while the image provides
    "where objects are" - together they predict "which pixels moved where".
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        image_size: tuple[int, int],
        effective_grid_size: tuple[int, int],
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        """
        Args:
            dim: Transformer embedding dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            dim_head: Dimension per attention head
            image_size: Output image size (H, W)
            effective_grid_size: Spatial grid size after encoding (h, w)
            attn_dropout: Attention dropout rate
            ff_dropout: Feed-forward dropout rate
        """
        super().__init__()

        from laq.models.attention import Transformer

        self.dim = dim
        self.effective_grid_size = effective_grid_size

        # Transformer with cross-attention for action conditioning
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
            has_cross_attn=True,
            dim_context=dim,
        )

        # Output projection: dim -> 2 channels (dx, dy) per patch
        image_height, image_width = image_size
        eff_h, eff_w = effective_grid_size
        patch_h = image_height // eff_h
        patch_w = image_width // eff_w

        self.to_flow = nn.Sequential(
            nn.Linear(dim, 2 * patch_h * patch_w),
            Rearrange(
                'b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)',
                p1=patch_h, p2=patch_w, c=2
            )
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict optical flow from context + action.

        Args:
            context_tokens: Pixel context from first frame [B, 1, h, w, d]
            action_tokens: Quantized latent action [B, 1, h', w', d]
            attn_bias: Spatial attention bias

        Returns:
            pred_flow: Predicted optical flow [B, 2, H, W]
        """
        b = context_tokens.shape[0]
        h, w = self.effective_grid_size

        video_shape = tuple(context_tokens.shape[:-1])

        # Flatten for transformer
        context_flat = rearrange(context_tokens, 'b t h w d -> (b t) (h w) d')
        action_flat = rearrange(action_tokens, 'b t h w d -> (b t) (h w) d')

        # Run transformer with cross-attention
        out = self.transformer(
            context_flat,
            attn_bias=attn_bias,
            video_shape=video_shape,
            context=action_flat,
        )

        # Reshape and project to flow
        out = rearrange(out, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)
        pred_flow = self.to_flow(out).squeeze(2)  # [B, 2, H, W]

        return pred_flow


def compute_flow_loss(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute MSE loss between predicted and ground-truth flow.

    Flow is normalized by image dimensions to put values in ~[-1, 1] range,
    making the loss scale comparable to other losses (DINO, pixel).

    Args:
        pred_flow: Predicted flow [B, 2, H, W]
        gt_flow: Ground-truth flow from RAFT [B, 2, H, W]
        normalize: If True, normalize flow by image size before loss

    Returns:
        loss: Scalar MSE loss
    """
    if normalize:
        # Normalize flow by image dimensions to get ~[-1, 1] range
        # Flow channel 0 = dx (horizontal), normalize by W
        # Flow channel 1 = dy (vertical), normalize by H
        _, _, H, W = gt_flow.shape

        # Create normalization tensor [1, 2, 1, 1] for broadcasting
        norm = torch.tensor([W, H], device=gt_flow.device, dtype=gt_flow.dtype)
        norm = norm.view(1, 2, 1, 1)

        pred_flow = pred_flow / norm
        gt_flow = gt_flow / norm

    return F.mse_loss(pred_flow, gt_flow)
