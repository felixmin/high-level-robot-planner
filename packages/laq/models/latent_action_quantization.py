import logging
import math
from pathlib import Path

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
    quantization. Uses transformer-based encoder/decoder with optional DINO
    features for improved representations.

    Training architecture:
    - Encoder: Processes frame pairs through spatial/temporal transformers
    - VQ (NSVQ): Quantizes latent representations to discrete codes
    - Main Decoder: Predicts next frame's DINO tokens (main learning objective)
    - Aux Decoder: Predicts next frame's pixels (for visualization, detached gradients)

    Returns:
        (loss, metrics_dict) where metrics_dict contains diagnostic values
    """
    # Codebook replacement schedule: Replace unused codebook entries at diminishing frequency
    # This helps codebook utilization early in training without overhead later
    # Format: (interval, until_step) - replace every `interval` steps until `until_step`
    CODEBOOK_REPLACE_SCHEDULE = [
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
            
        # --- Main Decoder (DINO-to-DINO) ---
        # Predicts next frame's DINO embeddings
        self.dec_spatial_transformer = Transformer(depth = spatial_depth, **transformer_with_action_kwargs)
        
        # --- Auxiliary Decoder (Pixels-to-Pixels) ---
        # Predicts next frame's Pixels for visualization (gradients stopped at latent z)
        self.aux_decoder = Transformer(depth = spatial_depth, **transformer_with_action_kwargs)
        
        # Decoder pixel projection - must match effective grid size
        eff_h, eff_w = self._effective_grid_size
        eff_patch_h = image_height // eff_h
        eff_patch_w = image_width // eff_w
        
        # Projector for the Aux Decoder output
        self.aux_to_pixels = nn.Sequential(
            nn.Linear(dim, channels * eff_patch_h * eff_patch_w),
            Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1 = eff_patch_h, p2 = eff_patch_w)
        )

        # Pre-compute action shape from code_seq_len (used in forward/inference)
        if math.sqrt(code_seq_len) % 1 == 0:
            self._action_shape = (int(math.sqrt(code_seq_len)), int(math.sqrt(code_seq_len)))
        elif code_seq_len == 2:
            self._action_shape = (2, 1)
        else:
            raise ValueError(
                f"code_seq_len must be a square number or 2, got {code_seq_len}"
            )

    def _should_replace_codebook(self, step: int) -> bool:
        """
        Check if unused codebook entries should be replaced at this step.

        Uses CODEBOOK_REPLACE_SCHEDULE for diminishing frequency replacement:
        - More frequent early in training to ensure good codebook utilization
        - Less frequent later to reduce overhead
        """
        for interval, until_step in self.CODEBOOK_REPLACE_SCHEDULE:
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
        
        This uses the AUXILIARY PIXEL DECODER path.
        Use this for visualization and inference.

        Args:
            tokens: Continuous embeddings of the first frame (PIXEL CONTEXT) [B, 1, h, w, d]
            actions: Continuous embeddings of the latent action [B, 1, h', w', d].

        Returns:
            recon_video: Reconstructed pixel values [B, C, 1, H, W]
        """
        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        video_shape = tuple(tokens.shape[:-1])


        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        actions = rearrange(actions, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        # Use AUX decoder for pixel reconstruction
        tokens = self.aux_decoder(tokens, attn_bias = attn_bias, video_shape = video_shape, context=actions)
        

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        rest_frames_tokens = tokens

        # Use AUX projector
        recon_video = self.aux_to_pixels(rest_frames_tokens)

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
        
        # --- 1. Main Path: DINO Decoding (Learning) ---
        # Predict DINO tokens of next frame from DINO context of first frame + Action
        
        # Prepare inputs for Main Decoder
        main_context = enc_first_frame_tokens # DINO context [B, 1, h, w, d]
        main_query = tokens # Action [B, 1, h, w, d]
        
        # Helper for main decoder pass (similar to decode() but uses dec_spatial_transformer)
        b_main = main_context.shape[0]
        h_main, w_main = self.patch_height_width
        
        # Flatten for transformer
        main_context_flat = rearrange(main_context, 'b t h w d -> (b t) (h w) d')
        main_query_flat = rearrange(main_query, 'b t h w d -> (b t) (h w) d')
        
        video_shape = tuple(main_context.shape[:-1])
        attn_bias = self.spatial_rel_pos_bias(h_main, w_main, device=main_context.device)
        
        # Run Main Decoder
        pred_dino_tokens = self.dec_spatial_transformer(
            main_context_flat, 
            attn_bias=attn_bias, 
            video_shape=video_shape, 
            context=main_query_flat
        )
        
        # Reshape back
        pred_dino_tokens = rearrange(pred_dino_tokens, '(b t) (h w) d -> b t h w d', b=b_main, h=h_main, w=w_main)
        
        # Main Loss: MSE between Predicted DINO tokens and True DINO tokens (enc_rest_frames_tokens)
        # Target must be detached to act as ground truth
        target_dino_tokens = enc_rest_frames_tokens.detach()
        main_loss = F.mse_loss(pred_dino_tokens, target_dino_tokens)


        # --- 2. Aux Path: Pixel Decoding (Visualization) ---
        # Predict Pixels of next frame from Pixel context + Detached Action
        
        # Use pixel-based projection for decoder context
        dec_first_frame_tokens = self.decoder_context_projection(first_frame)
        
        # Detach action so Aux loss doesn't affect Encoder/VQ
        aux_actions = tokens.detach()
        
        # Run Aux Decoder (via self.decode which is now mapped to Aux)
        recon_video = self.decode(dec_first_frame_tokens, aux_actions)

        recon_frames = rearrange(recon_video, 'b c 1 h w -> b c h w')

        if return_recons_only:
            return recon_frames

        # Aux loss: pixel reconstruction (gradients don't flow to encoder/VQ)
        aux_loss = F.mse_loss(rest_frames, recon_video)

        total_loss = main_loss + aux_loss

        metrics = {
            "num_unique_codes": num_unique_indices,
            "main_dino_loss": main_loss.detach(),
            "aux_pixel_loss": aux_loss.detach(),
        }

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
            Otherwise: reconstructed frames [B, C, H, W]
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

        action_h, action_w = self.action_shape
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=action_h, w=action_w)

        # Decoder uses pixel projection context
        dec_first_frame_tokens = self.decoder_context_projection(first_frame)

        recon_video = self.decode(dec_first_frame_tokens, actions=tokens)
        recon_frames = rearrange(recon_video, 'b c 1 h w -> b c h w')

        return recon_frames

