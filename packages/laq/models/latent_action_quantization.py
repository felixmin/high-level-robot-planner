from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, pack, repeat
from einops.layers.torch import Rearrange

from laq.models.attention import Transformer, ContinuousPositionBias
from laq.models.nsvq import NSVQ
from laq.models.dino import DINOFeatureExtractor, DINOEncoder, DINOWrapper

def exists(val):
    return val is not None


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
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
        perceptual_loss_config = None,  # Dict with enabled, weight, layers, model_name
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
            print(f"Using DINOv3 Encoder: {dinov3_model_name}")
            self.dino_feature_extractor = DINOFeatureExtractor(
                model_name=dinov3_model_name,
                target_size=self.image_size[0] # Assuming square
            )
            # DINOEncoder returns [B, 1, h, w, d]
            # Pool to match original grid size if specified (reduces memory 16x for attention)
            self.dino_encoder = DINOEncoder(
                self.dino_feature_extractor, 
                dim,
                pool_to_grid=dinov3_pool_to_grid
            )
            
            self.encoder_projection = DINOWrapper(self.dino_encoder)
            
            # Use encoder's output grid size (accounts for pooling)
            output_grid = self.dino_encoder.output_grid_size
            self._effective_grid_size = (output_grid, output_grid)
            print(f"  - Effective grid size: {self._effective_grid_size}")
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

        # Perceptual Loss Setup (optional DINO-based loss on reconstructions)
        self.perceptual_loss_config = perceptual_loss_config
        self.use_perceptual_loss = (
            perceptual_loss_config is not None
            and perceptual_loss_config.get("enabled", False)
        )

        if self.use_perceptual_loss:
            # Strict validation - all required params must be present
            required_keys = ["model_name", "weight", "layers"]
            missing = [k for k in required_keys if k not in perceptual_loss_config]
            if missing:
                raise ValueError(f"perceptual_loss_config missing required keys: {missing}")

            print(f"Initializing DINO Perceptual Loss with {perceptual_loss_config['model_name']}")
            self.perceptual_loss_net = DINOFeatureExtractor(
                model_name=perceptual_loss_config["model_name"],
                freeze=True,
                target_size=image_height
            )
            self.perceptual_loss_net.eval()
            for p in self.perceptual_loss_net.parameters():
                p.requires_grad = False

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
        assert path.exists()
        pt = torch.load(str(path))
        pt = {k.replace('module.', '') if 'module.' in k else k: v for k, v in pt.items()}
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebooks[indices]

        return self.decode(codes)

    @property
    def patch_height_width(self):
        return self._effective_grid_size

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
        step = 0,
        mask = None,
        return_recons_only = False,
        return_only_codebook_ids = False,
    ):
        """
        Forward pass for training.

        Flow:
        1. Preprocess: Image -> Patch Embeddings (Continuous 'tokens')
        2. Encode: Patches -> Latent Features
        3. VQ (NSVQ): Latent Features -> Quantized Vectors (with noise injection for grad flow)
        4. Decode: Quantized Vectors + First Frame -> Reconstructed Next Frame
        """
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]


        enc_first_frame_tokens = self.encoder_projection(first_frame)
        enc_rest_frames_tokens = self.encoder_projection(rest_frames)
        enc_tokens = torch.cat((enc_first_frame_tokens, enc_rest_frames_tokens), dim = 1)

        shape = enc_tokens.shape
        *_, h, w, _ = shape

        first_tokens, last_tokens = self.encode(enc_tokens)

        first_tokens, first_packed_fhw_shape = pack([first_tokens], 'b * d')
        last_tokens, last_packed_fhw_shape = pack([last_tokens], 'b * d')
        

        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)
        self.lookup_free_quantization = False
        vq_kwargs = dict(mask = vq_mask) if not self.lookup_free_quantization else dict()

        
        tokens, perplexity, codebook_usage, indices = self.vq(first_tokens, last_tokens, codebook_training_only = False)
        
        num_unique_indices = indices.unique().size(0)
        

        
        if step != 0 and self._should_replace_codebook(step):
            print(f"update codebook {step}")
            self.vq.replace_unused_codebooks(tokens.shape[0])

        if return_only_codebook_ids:
            return indices
        
        if math.sqrt(self.code_seq_len) % 1 == 0: # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            ## error
            print("code_seq_len should be square number or defined as 2")
            return
        
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = action_h, w = action_w)
        
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

        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w')
        video = rest_frames 

        if return_recons_only:
            return returned_recon

        if exists(mask):
            # variable lengthed video / images training
            aux_loss = F.mse_loss(video, recon_video, reduction='none')
            aux_loss = aux_loss[repeat(mask, 'b t -> b c t', c=c)]
            aux_loss = aux_loss.mean()
        else:
            aux_loss = F.mse_loss(video, recon_video)

        # Perceptual loss (optional)
        perceptual_loss = torch.tensor(0.0, device=video.device)
        if self.use_perceptual_loss:
            target_frame = rest_frames.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
            layers = self.perceptual_loss_config["layers"]

            with torch.no_grad():
                target_feats = self.perceptual_loss_net(
                    target_frame, output_hidden_states=True, layer_indices=layers
                )

            # Clamp reconstruction to [0, 1] - DINO normalization assumes valid images
            recon_clamped = torch.clamp(returned_recon, 0.0, 1.0)
            recon_feats = self.perceptual_loss_net(
                recon_clamped, output_hidden_states=True, layer_indices=layers
            )

            perceptual_loss = sum(
                F.l1_loss(r_f, t_f) for r_f, t_f in zip(recon_feats, target_feats)
            )

        # Combine losses
        perceptual_weight = (
            self.perceptual_loss_config["weight"] if self.use_perceptual_loss else 0.0
        )
        total_loss = main_loss + aux_loss + perceptual_weight * perceptual_loss

        metrics = {
            "num_unique_codes": num_unique_indices,
            "main_dino_loss": main_loss.detach(),
            "aux_pixel_loss": aux_loss.detach(),
            "perceptual_loss": perceptual_loss.detach(),
        }

        return total_loss, metrics
        

    def inference(
        self,
        video,
        step = 0,
        mask = None,
        return_only_codebook_ids=False,
        user_action_token_num=None
    ):
        
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        enc_first_frame_tokens = self.encoder_projection(first_frame)
        enc_rest_frames_tokens = self.encoder_projection(rest_frames)
        enc_tokens = torch.cat((enc_first_frame_tokens, enc_rest_frames_tokens), dim = 1)


        shape = enc_tokens.shape
        *_, h, w, _ = shape

        first_tokens, last_tokens = self.encode(enc_tokens)

        # quantize
        first_tokens, first_packed_fhw_shape = pack([first_tokens], 'b * d')
        last_tokens, last_packed_fhw_shape = pack([last_tokens], 'b * d')

        if user_action_token_num is not None:
            tokens, indices = self.vq.inference(first_tokens, last_tokens, user_action_token_num=user_action_token_num)
        else:
            tokens, indices = self.vq.inference(first_tokens, last_tokens)

        
    
        if return_only_codebook_ids:
            return indices

        if math.sqrt(self.code_seq_len) % 1 == 0: # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            print("code_seq_len should be square number or defined as 2")
            return
        

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = action_h, w = action_w)
        
        # Decoder uses pixel projection context
        dec_first_frame_tokens = self.decoder_context_projection(first_frame)
        
        recon_video = self.decode(dec_first_frame_tokens, actions=tokens)
        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w')
        video = rest_frames 
        
        return returned_recon

