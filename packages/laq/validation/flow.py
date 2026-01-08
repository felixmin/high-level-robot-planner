"""
Flow visualization strategy for LAQ validation.

Visualizes predicted vs ground-truth optical flow to monitor flow decoder learning.
"""

from typing import Any, Dict

import torch
from torchvision.utils import make_grid, flow_to_image
from einops import rearrange
import lightning.pytorch as pl

from .core import ValidationStrategy, ValidationCache


class FlowVisualizationStrategy(ValidationStrategy):
    """
    Visualize optical flow predictions vs RAFT ground truth.

    Creates side-by-side comparisons:
    - Frame t (source)
    - Frame t+k (target)
    - Ground-truth flow (from RAFT)
    - Predicted flow (from flow decoder)

    Only runs if the model has flow supervision enabled.
    """

    def __init__(
        self,
        name: str = "flow_visualization",
        enabled: bool = True,
        num_samples: int = 8,
        every_n_validations: int = 1,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            **kwargs,
        )
        self.num_samples = num_samples

    def needs_caching(self) -> bool:
        return True  # Need frames for visualization

    def needs_codes(self) -> bool:
        return False

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate flow visualization comparing predicted vs ground-truth."""
        metrics = {}
        wandb_logger = self._get_wandb_logger(trainer)

        if wandb_logger is None:
            return metrics

        # Check if model has flow decoder
        model = pl_module.model
        if model.flow_decoder is None or model.flow_teacher is None:
            return metrics

        # Get frames from cache
        all_frames = cache.get_all_frames()
        if all_frames is None or len(all_frames) == 0:
            return metrics

        # Sample frames
        n_samples = min(self.num_samples, len(all_frames))
        indices = torch.randperm(len(all_frames))[:n_samples]
        frames = all_frames[indices]

        # Generate flow visualization
        flow_grid = self._create_flow_grid(frames, pl_module)

        if flow_grid is not None:
            bucket_name = cache.bucket_name or ""
            prefix = f"val/{bucket_name}" if bucket_name else "val"
            wandb_logger.log_image(
                key=f"{prefix}/flow_comparison{metric_suffix}",
                images=[flow_grid],
                caption=[f"Step {trainer.global_step} (GT flow | Pred flow)"],
            )

        return metrics

    def _create_flow_grid(
        self,
        frames: torch.Tensor,
        pl_module: pl.LightningModule,
    ) -> torch.Tensor:
        """Create visualization grid comparing GT and predicted flow."""
        if len(frames) == 0:
            return None

        model = pl_module.model
        device = pl_module.device

        model.eval()
        with torch.no_grad():
            frames = frames.to(device)

            # Extract frame pairs
            first_frame = frames[:, :, :1]  # [B, C, 1, H, W]
            rest_frames = frames[:, :, 1:]  # [B, C, 1, H, W]

            # Get ground-truth flow from RAFT teacher
            gt_flow = model.flow_teacher.compute_flow(first_frame, rest_frames)

            # Get predicted flow from model
            # We need to run encoding and get the latent action
            enc_first_tokens, enc_rest_tokens, first_tokens, last_tokens = (
                model._encode_frames(first_frame, rest_frames)
            )
            tokens, _, _, _ = model.vq(first_tokens, last_tokens, codebook_training_only=False)

            action_h, action_w = model.action_shape
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=action_h, w=action_w)

            # Get pixel context
            dec_first_frame_tokens = model.decoder_context_projection(first_frame)

            # Compute attention bias
            h, w = model.patch_height_width
            attn_bias = model.spatial_rel_pos_bias(h, w, device=device)

            # Predict flow
            pred_flow = model.flow_decoder(
                dec_first_frame_tokens,
                tokens.detach(),
                attn_bias,
            )

        model.train()

        # Convert flow to RGB images using color wheel
        # flow_to_image expects [B, 2, H, W] and returns [B, 3, H, W] in [0, 255]
        gt_flow_rgb = flow_to_image(gt_flow).float() / 255.0
        pred_flow_rgb = flow_to_image(pred_flow).float() / 255.0

        # Get source and target frames
        frame_t = frames[:, :, 0].cpu()  # [B, C, H, W]
        frame_t_plus = frames[:, :, 1].cpu()

        gt_flow_rgb = gt_flow_rgb.cpu()
        pred_flow_rgb = pred_flow_rgb.cpu()

        # Stack: [frame_t, frame_t+k, gt_flow, pred_flow]
        imgs = torch.stack([frame_t, frame_t_plus, gt_flow_rgb, pred_flow_rgb], dim=0)
        imgs = rearrange(imgs, 'r b c h w -> (b r) c h w')
        imgs = imgs.clamp(0.0, 1.0)

        return make_grid(imgs, nrow=4, normalize=False)
