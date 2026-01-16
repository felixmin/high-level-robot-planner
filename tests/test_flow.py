"""
Tests for optical flow supervision module.

Tests FlowConfig validation, FlowDecoder architecture, and integration with LAQ.
"""

import pytest
import torch

from laq.models.flow import FlowConfig, FlowDecoder, RAFTTeacher, compute_flow_loss


class TestFlowConfig:
    """Test FlowConfig validation."""

    def test_valid_config_raft_small(self):
        """Test valid config with raft_small."""
        config = FlowConfig(
            model="raft_small",
            loss_weight=0.1,
            decoder_depth=4,
        )
        assert config.model == "raft_small"
        assert config.loss_weight == 0.1
        assert config.decoder_depth == 4

    def test_valid_config_raft_large(self):
        """Test valid config with raft_large."""
        config = FlowConfig(
            model="raft_large",
            loss_weight=0.5,
            decoder_depth=8,
        )
        assert config.model == "raft_large"

    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="must be 'raft_small' or 'raft_large'"):
            FlowConfig(
                model="invalid_model",
                loss_weight=0.1,
                decoder_depth=4,
            )

    def test_invalid_loss_weight_zero(self):
        """Test that zero loss weight raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            FlowConfig(
                model="raft_small",
                loss_weight=0.0,
                decoder_depth=4,
            )

    def test_invalid_loss_weight_negative(self):
        """Test that negative loss weight raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            FlowConfig(
                model="raft_small",
                loss_weight=-0.1,
                decoder_depth=4,
            )

    def test_invalid_decoder_depth(self):
        """Test that invalid decoder depth raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            FlowConfig(
                model="raft_small",
                loss_weight=0.1,
                decoder_depth=0,
            )

    def test_warmup_weight_at_zero(self):
        """Test that weight is 0 at step 0 with warmup."""
        config = FlowConfig(
            model="raft_small",
            loss_weight=1.0,
            decoder_depth=4,
            warmup_steps=1000,
        )
        assert config.get_weight(0) == 0.0

    def test_warmup_weight_at_half(self):
        """Test that weight is 0.5 at halfway through warmup."""
        config = FlowConfig(
            model="raft_small",
            loss_weight=1.0,
            decoder_depth=4,
            warmup_steps=1000,
        )
        assert config.get_weight(500) == 0.5

    def test_warmup_weight_at_end(self):
        """Test that weight is full after warmup completes."""
        config = FlowConfig(
            model="raft_small",
            loss_weight=1.0,
            decoder_depth=4,
            warmup_steps=1000,
        )
        assert config.get_weight(1000) == 1.0
        assert config.get_weight(2000) == 1.0  # Stays at max

    def test_no_warmup(self):
        """Test that weight is full immediately with no warmup."""
        config = FlowConfig(
            model="raft_small",
            loss_weight=0.5,
            decoder_depth=4,
            warmup_steps=0,
        )
        assert config.get_weight(0) == 0.5
        assert config.get_weight(100) == 0.5


class TestFlowDecoder:
    """Test FlowDecoder architecture."""

    @pytest.fixture
    def decoder(self):
        """Create a small flow decoder for testing."""
        return FlowDecoder(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            image_size=(64, 64),
            effective_grid_size=(4, 4),
        )

    def test_output_shape(self, decoder):
        """Test that output has correct shape [B, 2, H, W]."""
        batch_size = 2
        h, w = 4, 4
        dim = 64

        context_tokens = torch.randn(batch_size, 1, h, w, dim)
        action_tokens = torch.randn(batch_size, 1, 1, 1, dim)  # code_seq_len=1
        attn_bias = torch.zeros(4, h * w, h * w)

        output = decoder(context_tokens, action_tokens, attn_bias)

        assert output.shape == (batch_size, 2, 64, 64)

    def test_gradient_flow(self, decoder):
        """Test that gradients flow through decoder."""
        context_tokens = torch.randn(2, 1, 4, 4, 64, requires_grad=True)
        action_tokens = torch.randn(2, 1, 1, 1, 64, requires_grad=True)
        attn_bias = torch.zeros(4, 16, 16)

        output = decoder(context_tokens, action_tokens, attn_bias)
        loss = output.sum()
        loss.backward()

        assert context_tokens.grad is not None
        assert action_tokens.grad is not None


class TestRAFTTeacher:
    """Test RAFTTeacher lazy loading and inference."""

    def test_lazy_loading(self):
        """Test that RAFT is not loaded until first use."""
        teacher = RAFTTeacher("raft_small")
        assert teacher._model is None

    def test_state_dict_empty(self):
        """Test that state_dict returns empty to avoid checkpoint pollution."""
        teacher = RAFTTeacher("raft_small")
        state_dict = teacher.state_dict()
        assert state_dict == {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RAFT")
    def test_compute_flow_shape(self):
        """Test that compute_flow returns correct shape."""
        teacher = RAFTTeacher("raft_small")

        # RAFT requires minimum image size
        frame1 = torch.randn(1, 3, 1, 128, 128).cuda()
        frame2 = torch.randn(1, 3, 1, 128, 128).cuda()

        flow = teacher.compute_flow(frame1, frame2)

        assert flow.shape == (1, 2, 128, 128)
        assert flow.device.type == "cuda"


class TestComputeFlowLoss:
    """Test flow loss computation."""

    def test_mse_loss_normalized(self):
        """Test that flow loss is normalized MSE."""
        pred_flow = torch.randn(2, 2, 64, 64)
        gt_flow = torch.randn(2, 2, 64, 64)

        loss = compute_flow_loss(pred_flow, gt_flow, normalize=True)

        # Manually compute normalized MSE
        H, W = 64, 64
        norm = torch.tensor([W, H]).view(1, 2, 1, 1)
        pred_norm = pred_flow / norm
        gt_norm = gt_flow / norm
        expected = torch.nn.functional.mse_loss(pred_norm, gt_norm)
        assert torch.allclose(loss, expected)

    def test_mse_loss_unnormalized(self):
        """Test that unnormalized flow loss is raw MSE."""
        pred_flow = torch.randn(2, 2, 64, 64)
        gt_flow = torch.randn(2, 2, 64, 64)

        loss = compute_flow_loss(pred_flow, gt_flow, normalize=False)

        expected = torch.nn.functional.mse_loss(pred_flow, gt_flow)
        assert torch.allclose(loss, expected)

    def test_zero_loss_identical(self):
        """Test that identical flows give zero loss."""
        flow = torch.randn(2, 2, 64, 64)
        loss = compute_flow_loss(flow, flow)
        assert loss.item() == 0.0

    def test_normalization_reduces_scale(self):
        """Test that normalization brings flow to smaller scale."""
        # Large pixel displacements
        pred_flow = torch.randn(2, 2, 256, 256) * 100  # ~[-100, 100] pixels
        gt_flow = torch.randn(2, 2, 256, 256) * 100

        loss_unnorm = compute_flow_loss(pred_flow, gt_flow, normalize=False)
        loss_norm = compute_flow_loss(pred_flow, gt_flow, normalize=True)

        # Normalized loss should be much smaller (by factor of ~256^2)
        assert loss_norm < loss_unnorm / 1000


class TestLAQWithFlow:
    """Integration tests for LAQ with flow supervision."""

    @pytest.fixture
    def training_config(self):
        """Create training config for testing."""
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "optimizer": {
                "type": "AdamW",
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
            "scheduler": {"type": "none"},
        })

    @pytest.fixture
    def laq_config_with_flow(self):
        """Create minimal LAQ config with flow for testing."""
        from omegaconf import OmegaConf
        # Use image_size=128, patch_size=16 for 8x8 grid (supported by NSVQ)
        return OmegaConf.create({
            "dim": 64,
            "quant_dim": 16,
            "codebook_size": 8,
            "image_size": 128,
            "patch_size": 16,
            "spatial_depth": 2,
            "temporal_depth": 2,
            "dim_head": 16,
            "heads": 4,
            "code_seq_len": 1,
            "use_aux_loss": False,
            "flow": {
                "model": "raft_small",
                "loss_weight": 0.1,
                "decoder_depth": 2,
            }
        })

    @pytest.fixture
    def laq_config_without_flow(self):
        """Create minimal LAQ config without flow for testing."""
        from omegaconf import OmegaConf
        # Use image_size=128, patch_size=16 for 8x8 grid (supported by NSVQ)
        return OmegaConf.create({
            "dim": 64,
            "quant_dim": 16,
            "codebook_size": 8,
            "image_size": 128,
            "patch_size": 16,
            "spatial_depth": 2,
            "temporal_depth": 2,
            "dim_head": 16,
            "heads": 4,
            "code_seq_len": 1,
            "use_aux_loss": False,
            # No flow config
        })

    def test_flow_config_parsed(self, laq_config_with_flow, training_config):
        """Test that flow config is correctly parsed in task."""
        from laq.task import LAQTask

        task = LAQTask(
            model_config=laq_config_with_flow,
            training_config=training_config,
        )

        assert task.model.flow_config is not None
        assert task.model.flow_config.model == "raft_small"
        assert task.model.flow_decoder is not None
        assert task.model.flow_teacher is not None

    def test_no_flow_when_not_configured(self, laq_config_without_flow, training_config):
        """Test that flow is disabled when not configured."""
        from laq.task import LAQTask

        task = LAQTask(
            model_config=laq_config_without_flow,
            training_config=training_config,
        )

        assert task.model.flow_config is None
        assert task.model.flow_decoder is None
        assert task.model.flow_teacher is None


class TestHydraConfigWithFlow:
    """Test Hydra configuration with flow supervision."""

    @pytest.fixture
    def config_dir(self):
        """Get path to config directory."""
        from pathlib import Path
        return str(Path(__file__).parent.parent / "config")

    @pytest.fixture
    def ensure_user_config(self, config_dir):
        """Create empty user_config/local.yaml if it doesn't exist (for optional include)."""
        from pathlib import Path
        user_config_dir = Path(config_dir) / "user_config"
        user_config_path = user_config_dir / "local.yaml"
        created = False
        if not user_config_path.exists():
            user_config_dir.mkdir(parents=True, exist_ok=True)
            user_config_path.write_text("# Temporary user config for testing\n")
            created = True
        yield user_config_path
        if created:
            user_config_path.unlink()

    def test_laq_oxe_all_val_3_loads(self, config_dir, ensure_user_config):
        """Test that laq_oxe_all_val_3 config loads correctly."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=laq_oxe_all_val_3"])

            # Validate experiment
            assert cfg.experiment.name == "laq_oxe_all_val_3"
            assert "flow" in cfg.experiment.description.lower()

            # Validate flow config
            assert cfg.model.flow.model == "raft_large"
            assert cfg.model.flow.loss_weight == 1.0
            assert cfg.model.flow.decoder_depth == 4
            assert cfg.model.flow.warmup_steps == 10000

            # Validate decoder flags
            assert cfg.model.use_dino_decoder is True
            assert cfg.model.use_aux_decoder is True

            # Validate flow visualization strategy
            assert cfg.training.validation.strategies.flow_visualization.enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
