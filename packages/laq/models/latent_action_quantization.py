"""
Latent Action Quantization (LAQ) Model.

A VQ-VAE that encodes frame-to-frame transitions into discrete latent action codes.
Supports modular decoder objectives: DINO, Pixel, Flow (training) and Aux (interpretability).

Decoder Types:
- DINO decoder: Predicts next frame's DINO tokens (renamed from dec_spatial_transformer)
- Pixel decoder: Predicts next frame's pixels with gradients flowing to encoder
- Flow decoder: Predicts optical flow via RAFT knowledge distillation
- Aux decoder: Predicts pixels for visualization only (gradients detached from encoder)
"""

import logging
import math
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, pack
from einops.layers.torch import Rearrange
from torch import nn

from laq.models.attention import ContinuousPositionBias, Transformer
from laq.models.dino import DINOEncoder, DINOFeatureExtractor, DINOWrapper
from laq.models.nsvq import NSVQ

logger = logging.getLogger(__name__)


def exists(val):
    return val is not None


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    if len(ret) != 2:
        raise ValueError(f"Expected pair, got {ret}")
    return ret


class LatentActionQuantization(nn.Module):
    """
    Latent Action Quantization (LAQ) model.

    Encodes frame pairs into discrete latent action codes using VQ-VAE style
    quantization. Uses transformer-based encoder/decoder with modular objectives.

    Architecture:
    - Encoder: Processes frame pairs through spatial/temporal transformers
    - VQ (NSVQ): Quantizes latent representations to discrete codes
    - Training Decoders (at least one required):
      - DINO decoder: Predicts next frame's DINO tokens
      - Pixel decoder: Predicts next frame's pixels (gradients to encoder)
      - Flow decoder: Predicts optical flow (gradients to encoder)
    - Interpretability Decoder (optional):
      - Aux decoder: Predicts pixels for visualization (gradients detached)

    Returns:
        (loss, metrics_dict) where metrics_dict contains diagnostic values
    """
    # Default codebook replacement schedule: Replace unused codebook entries at diminishing frequency
    # This helps codebook utilization early in training without overhead later
    # Format: (interval, until_step) - replace every `interval` steps until `until_step`
    DEFAULT_CODEBOOK_REPLACE_SCHEDULE = [
        (10, 100),    # Every 10 steps for first 100 steps
        (100, 1000),  # Every 100 steps for steps 100-1000
        (500, 5000),  # Every 500 steps for steps 1000-5000
    ]

    def __init__(
        self,
        *,
        dim,
        quant_dim,
        codebook_size,
        image_size,
        patch_size,
        spatial_depth,
        temporal_depth,
        dim_head = 64,
        heads = 8,
        channels = 3,
        attn_dropout = 0.,
        ff_dropout = 0.,
        code_seq_len = 1,
        use_dinov3_encoder = False,
        dinov3_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m",
        dinov3_pool_to_grid = None,  # Pool DINO features to this grid size (e.g., 8 for 8x8)
        # Training decoder flags (at least one must be True, or flow_config must be set)
        use_dino_decoder = True,
        use_pixel_decoder = False,
        # Interpretability decoder flag (optional, for visualization only)
        use_aux_decoder = True,
        # Flow supervision config (optional - set to enable flow loss)
        flow_config: Optional["FlowConfig"] = None,
        # Codebook replacement schedule (optional - uses default if not provided)
        codebook_replace_schedule: Optional[list] = None,
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        # Store decoder flags
        self.use_dino_decoder = use_dino_decoder
        self.use_pixel_decoder = use_pixel_decoder
        self.use_aux_decoder = use_aux_decoder
        self.flow_config = flow_config

        # Validate at least one training decoder is enabled
        training_decoders = self._get_enabled_training_decoders()
        if not training_decoders:
            raise ValueError(
                "At least one training decoder must be enabled. "
                "Set use_dino_decoder=True, use_pixel_decoder=True, or provide flow_config."
            )
        logger.info(f"Enabled training decoders: {training_decoders}")
        if use_aux_decoder:
            logger.info("Aux decoder enabled for interpretability")

        self.codebook_replace_schedule = (
            codebook_replace_schedule
            if codebook_replace_schedule is not None
            else self.DEFAULT_CODEBOOK_REPLACE_SCHEDULE
        )
        self.code_seq_len = code_seq_len
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        if use_dinov3_encoder:
            logger.info(f"Using DINOv3 Encoder: {dinov3_model_name}")
            self.dino_feature_extractor = DINOFeatureExtractor(
                model_name=dinov3_model_name,
                target_size=self.image_size[0],  # Assuming square
            )
            # DINOEncoder returns [B, 1, h, w, d]
            # Pool to match original grid size if specified (reduces memory 16x for attention)
            self.dino_encoder = DINOEncoder(
                self.dino_feature_extractor,
                dim,
                pool_to_grid=dinov3_pool_to_grid,
            )

            self.encoder_projection = DINOWrapper(self.dino_encoder)

            # Use encoder's output grid size (accounts for pooling)
            output_grid = self.dino_encoder.output_grid_size
            self._effective_grid_size = (output_grid, output_grid)
            logger.info(f"  - Effective grid size: {self._effective_grid_size}")
        else:
            self._effective_grid_size = (image_height // patch_height, image_width // patch_width)
            self.encoder_projection = None

        # Decoder Pixel Projection (Also used for Encoder if DINO is disabled)
        # Ensure patch size matches the effective grid size (critical if DINO pooling is used)
        eff_h, eff_w = self._effective_grid_size
        pixel_p1 = image_height // eff_h
        pixel_p2 = image_width // eff_w
        
        self.pixel_projection = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = pixel_p1, p2 = pixel_p2),
            nn.LayerNorm(channels * pixel_p1 * pixel_p2),
            nn.Linear(channels * pixel_p1 * pixel_p2, dim),
            nn.LayerNorm(dim)
        )
        
        self.decoder_context_projection = self.pixel_projection

        if self.encoder_projection is None:
            self.encoder_projection = self.pixel_projection


        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
        )
        
        transformer_with_action_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
            has_cross_attn = True,
            dim_context = dim,
        )

        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)


        self.vq = NSVQ(
            dim=dim,
            num_embeddings=codebook_size,
            embedding_dim=quant_dim,
            code_seq_len=code_seq_len,
            patch_size=patch_size,
            image_size=image_size,
            grid_size=self._effective_grid_size
        )

        # Compute pixel projection parameters (shared by pixel decoders)
        eff_h, eff_w = self._effective_grid_size
        eff_patch_h = image_height // eff_h
        eff_patch_w = image_width // eff_w

        # --- DINO Decoder (Training) ---
        # Predicts next frame's DINO embeddings
        if use_dino_decoder:
            self.dino_decoder = Transformer(depth=spatial_depth, **transformer_with_action_kwargs)
        else:
            self.dino_decoder = None

        # --- Pixel Decoder (Training) ---
        # Predicts next frame's pixels with gradients flowing to encoder
        if use_pixel_decoder:
            self.pixel_decoder = Transformer(depth=spatial_depth, **transformer_with_action_kwargs)
            self.pixel_to_pixels = nn.Sequential(
                nn.Linear(dim, channels * eff_patch_h * eff_patch_w),
                Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1=eff_patch_h, p2=eff_patch_w)
            )
        else:
            self.pixel_decoder = None
            self.pixel_to_pixels = None

        # --- Aux Decoder (Interpretability) ---
        # Predicts next frame's pixels for visualization (gradients detached from encoder)
        if use_aux_decoder:
            self.aux_decoder = Transformer(depth=spatial_depth, **transformer_with_action_kwargs)
            self.aux_to_pixels = nn.Sequential(
                nn.Linear(dim, channels * eff_patch_h * eff_patch_w),
                Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1=eff_patch_h, p2=eff_patch_w)
            )
        else:
            self.aux_decoder = None
            self.aux_to_pixels = None

        # --- Flow Decoder (Optional) ---
        # Predicts optical flow from pixel context + latent action
        if flow_config is not None:
            from laq.models.flow import FlowDecoder, RAFTTeacher

            logger.info(
                f"Initializing flow supervision (model={flow_config.model}, "
                f"depth={flow_config.decoder_depth}, weight={flow_config.loss_weight})"
            )

            self.flow_decoder = FlowDecoder(
                dim=dim,
                depth=flow_config.decoder_depth,
                heads=heads,
                dim_head=dim_head,
                image_size=self.image_size,
                effective_grid_size=self._effective_grid_size,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
            self.flow_teacher = RAFTTeacher(flow_config.model)
        else:
            self.flow_decoder = None
            self.flow_teacher = None

        # Pre-compute action shape from code_seq_len (used in forward/inference)
        if math.sqrt(code_seq_len) % 1 == 0:
            self._action_shape = (int(math.sqrt(code_seq_len)), int(math.sqrt(code_seq_len)))
        elif code_seq_len == 2:
            self._action_shape = (2, 1)
        else:
            raise ValueError(
                f"code_seq_len must be a square number or 2, got {code_seq_len}"
            )

    def _get_enabled_training_decoders(self) -> List[str]:
        """
        Get list of enabled training decoders.

        Training decoders contribute gradients to the encoder/VQ.
        At least one must be enabled for meaningful training.

        Returns:
            List of enabled decoder names (e.g., ["dino", "flow"])
        """
        decoders = []
        if self.use_dino_decoder:
            decoders.append("dino")
        if self.use_pixel_decoder:
            decoders.append("pixel")
        if self.flow_config is not None:
            decoders.append("flow")
        return decoders

    def _should_replace_codebook(self, step: int) -> bool:
        """
        Check if unused codebook entries should be replaced at this step.

        Uses codebook_replace_schedule for diminishing frequency replacement:
        - More frequent early in training to ensure good codebook utilization
        - Less frequent later to reduce overhead
        """
        for interval, until_step in self.codebook_replace_schedule:
            if step < until_step and step % interval == 0:
                return True
        return False

    def load(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        pt = torch.load(str(path), weights_only=False)
        pt = {k.replace('module.', ''): v for k, v in pt.items()}
        self.load_state_dict(pt, strict=False)

    @property
    def patch_height_width(self):
        return self._effective_grid_size

    @property
    def action_shape(self):
        """Returns (action_h, action_w) for reshaping latent codes."""
        return self._action_shape

    def _encode_frames(self, first_frame, rest_frames):
        """
        Encode frame pair through encoder projection and temporal transformer.

        Args:
            first_frame: First frame [B, C, 1, H, W]
            rest_frames: Second frame [B, C, 1, H, W]

        Returns:
            enc_first_frame_tokens: Encoded first frame tokens [B, 1, h, w, d]
            enc_rest_frames_tokens: Encoded second frame tokens [B, 1, h, w, d]
            first_tokens_packed: Packed first frame latents [B, h*w, d]
            last_tokens_packed: Packed second frame latents [B, h*w, d]
        """
        enc_first_frame_tokens = self.encoder_projection(first_frame)
        enc_rest_frames_tokens = self.encoder_projection(rest_frames)
        enc_tokens = torch.cat((enc_first_frame_tokens, enc_rest_frames_tokens), dim=1)

        first_tokens, last_tokens = self.encode(enc_tokens)

        first_tokens_packed, _ = pack([first_tokens], 'b * d')
        last_tokens_packed, _ = pack([last_tokens], 'b * d')

        return enc_first_frame_tokens, enc_rest_frames_tokens, first_tokens_packed, last_tokens_packed

    def encode(
        self,
        tokens
    ):
        """
        Encodes continuous video tokens into latent representations.

        Args:
            tokens: Continuous feature vectors (embeddings) of shape [B, T, h, w, d].
                    These are NOT discrete indices.
                    h, w = patch_height_width (e.g., 8x8)
                    d = dim (e.g., 1024)

        Returns:
            first_tokens: Latent representation of the first frame [B, 1, h, w, d]
            last_tokens: Latent representation of the last frame [B, 1, h, w, d]
        """
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        
        first_tokens = tokens[:, :1]
        last_tokens = tokens[:, 1:]
        
        return first_tokens, last_tokens

        

    def decode(
        self,
        tokens,
        actions,
    ):
        """
        Decodes latent actions + context frame into reconstructed video.

        This uses the AUXILIARY PIXEL DECODER path for visualization/inference.
        Returns None if aux_decoder is disabled.

        Args:
            tokens: Continuous embeddings of the first frame (PIXEL CONTEXT) [B, 1, h, w, d]
            actions: Continuous embeddings of the latent action [B, 1, h', w', d].

        Returns:
            recon_video: Reconstructed pixel values [B, C, 1, H, W], or None if aux_decoder disabled
        """
        if self.aux_decoder is None:
            return None

        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=h, w=w)

        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        actions = rearrange(actions, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)

        # Use AUX decoder for pixel reconstruction
        tokens = self.aux_decoder(tokens, attn_bias=attn_bias, video_shape=video_shape, context=actions)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        # Use AUX projector
        recon_video = self.aux_to_pixels(tokens)

        return recon_video
    

    def forward(
        self,
        video,
        step=0,
        return_recons_only=False,
        return_only_codebook_ids=False,
    ):
        """
        Forward pass for training.

        Args:
            video: Input frame pairs [B, C, 2, H, W] or [B, C, H, W] for single frame
            step: Training step (for codebook replacement scheduling)
            return_recons_only: If True, return only reconstructed frames
            return_only_codebook_ids: If True, return only codebook indices

        Returns:
            If return_recons_only: reconstructed frames [B, C, H, W]
            If return_only_codebook_ids: codebook indices [B, code_seq_len]
            Otherwise: (loss, metrics_dict)
        """
        if video.ndim not in {4, 5}:
            raise ValueError(f"Expected 4D or 5D input, got {video.ndim}D")

        if video.ndim == 4:
            video = rearrange(video, 'b c h w -> b c 1 h w')

        b, c, f, *image_dims = video.shape
        device = video.device

        if tuple(image_dims) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}, got {tuple(image_dims)}")

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        # Encode both frames
        enc_first_frame_tokens, enc_rest_frames_tokens, first_tokens, last_tokens = (
            self._encode_frames(first_frame, rest_frames)
        )

        tokens, _, _, indices = self.vq(first_tokens, last_tokens, codebook_training_only=False)

        num_unique_indices = indices.unique().size(0)

        if step != 0 and self._should_replace_codebook(step):
            logger.debug(f"Replacing unused codebook entries at step {step}")
            self.vq.replace_unused_codebooks(tokens.shape[0])

        if return_only_codebook_ids:
            return indices

        action_h, action_w = self.action_shape
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=action_h, w=action_w)

        # Initialize loss and metrics
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        metrics = {"num_unique_codes": num_unique_indices}

        # Precompute attention bias (shared by all decoders)
        h_dec, w_dec = self.patch_height_width
        attn_bias = self.spatial_rel_pos_bias(h_dec, w_dec, device=device)

        # --- 1. DINO Decoder Path (Training) ---
        # Predict DINO/encoder tokens of next frame from encoder context + action
        if self.dino_decoder is not None:
            dino_context = enc_first_frame_tokens  # [B, 1, h, w, d]
            video_shape = tuple(dino_context.shape[:-1])

            # Flatten for transformer
            dino_context_flat = rearrange(dino_context, 'b t h w d -> (b t) (h w) d')
            dino_action_flat = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

            # Run DINO decoder
            pred_dino_tokens = self.dino_decoder(
                dino_context_flat,
                attn_bias=attn_bias,
                video_shape=video_shape,
                context=dino_action_flat
            )

            # Reshape back
            pred_dino_tokens = rearrange(
                pred_dino_tokens, '(b t) (h w) d -> b t h w d',
                b=b, h=h_dec, w=w_dec
            )

            # DINO loss: MSE to target encoder tokens
            target_dino_tokens = enc_rest_frames_tokens.detach()
            dino_loss = F.mse_loss(pred_dino_tokens, target_dino_tokens)
            total_loss = total_loss + dino_loss
            metrics["dino_loss"] = dino_loss.detach()

        # --- 2. Pixel Decoder Path (Training) ---
        # Predict pixels of next frame with gradients flowing to encoder
        if self.pixel_decoder is not None:
            # Use pixel-based projection for decoder context
            pixel_context = self.decoder_context_projection(first_frame)
            video_shape = tuple(pixel_context.shape[:-1])

            # Flatten for transformer
            pixel_context_flat = rearrange(pixel_context, 'b t h w d -> (b t) (h w) d')
            pixel_action_flat = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

            # Run pixel decoder (gradients flow to encoder)
            pred_pixel_tokens = self.pixel_decoder(
                pixel_context_flat,
                attn_bias=attn_bias,
                video_shape=video_shape,
                context=pixel_action_flat
            )

            # Reshape and project to pixels
            pred_pixel_tokens = rearrange(
                pred_pixel_tokens, '(b t) (h w) d -> b t h w d',
                b=b, h=h_dec, w=w_dec
            )
            pred_pixels = self.pixel_to_pixels(pred_pixel_tokens)

            # Pixel loss with gradients to encoder
            pixel_loss = F.mse_loss(rest_frames, pred_pixels)
            total_loss = total_loss + pixel_loss
            metrics["pixel_loss"] = pixel_loss.detach()

        # --- 3. Aux Decoder Path (Interpretability) ---
        # Predict pixels for visualization (gradients detached from encoder)
        if self.aux_decoder is not None or return_recons_only:
            if self.aux_decoder is None:
                # Aux decoder disabled but reconstructions requested - return None
                if return_recons_only:
                    return None

            # Use pixel-based projection for decoder context
            aux_context = self.decoder_context_projection(first_frame)

            # Detach action so aux loss doesn't affect encoder/VQ
            aux_actions = tokens.detach()

            # Run aux decoder (via self.decode)
            recon_video = self.decode(aux_context, aux_actions)
            recon_frames = rearrange(recon_video, 'b c 1 h w -> b c h w')

            if return_recons_only:
                return recon_frames

            # Aux loss: pixel reconstruction (only trains aux decoder, not encoder)
            aux_loss = F.mse_loss(rest_frames, recon_video)
            total_loss = total_loss + aux_loss
            metrics["aux_loss"] = aux_loss.detach()

        # --- 4. Flow Decoder Path (Training) ---
        # Predict optical flow with gradients flowing to encoder
        if self.flow_decoder is not None:
            from laq.models.flow import compute_flow_loss

            # Use pixel-based projection for decoder context
            flow_context = self.decoder_context_projection(first_frame)

            # NO detach - gradients flow to encoder for motion-aware representations
            flow_actions = tokens

            # Compute ground-truth flow from RAFT teacher
            gt_flow = self.flow_teacher.compute_flow(first_frame, rest_frames)

            # Predict flow from context + action
            pred_flow = self.flow_decoder(
                flow_context,
                flow_actions,
                attn_bias,
            )

            # Flow loss with gradients to encoder
            flow_loss = compute_flow_loss(pred_flow, gt_flow)
            flow_weight = self.flow_config.get_weight(step)
            total_loss = total_loss + flow_weight * flow_loss
            metrics["flow_loss"] = flow_loss.detach()
            metrics["flow_weight"] = flow_weight

        return total_loss, metrics
        

    def inference(
        self,
        video,
        return_only_codebook_ids=False,
        user_action_token_num=None,
    ):
        """
        Inference pass (no loss computation).

        Args:
            video: Input frame pairs [B, C, 2, H, W] or [B, C, H, W]
            return_only_codebook_ids: If True, return only codebook indices
            user_action_token_num: Optional override for action token selection

        Returns:
            If return_only_codebook_ids: codebook indices [B, code_seq_len]
            Otherwise: reconstructed frames [B, C, H, W], or None if aux_decoder disabled
        """
        if video.ndim not in {4, 5}:
            raise ValueError(f"Expected 4D or 5D input, got {video.ndim}D")

        if video.ndim == 4:
            video = rearrange(video, 'b c h w -> b c 1 h w')

        if tuple(video.shape[3:]) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}, got {tuple(video.shape[3:])}")

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        # Encode both frames (enc tokens not needed for inference)
        _, _, first_tokens, last_tokens = self._encode_frames(first_frame, rest_frames)

        if user_action_token_num is not None:
            tokens, indices = self.vq.inference(
                first_tokens, last_tokens, user_action_token_num=user_action_token_num
            )
        else:
            tokens, indices = self.vq.inference(first_tokens, last_tokens)

        if return_only_codebook_ids:
            return indices

        # Aux decoder required for pixel reconstruction
        if self.aux_decoder is None:
            return None

        action_h, action_w = self.action_shape
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=action_h, w=action_w)

        # Decoder uses pixel projection context
        dec_first_frame_tokens = self.decoder_context_projection(first_frame)

        recon_video = self.decode(dec_first_frame_tokens, actions=tokens)
        recon_frames = rearrange(recon_video, 'b c 1 h w -> b c h w')

        return recon_frames

