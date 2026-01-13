"""
Tests for Hydra configuration composition.
"""

import pytest
from hydra import compose, initialize_config_dir
from pathlib import Path
from omegaconf import OmegaConf


@pytest.fixture
def config_dir():
    """Get path to config directory."""
    return str(Path(__file__).parent.parent / "config")


class TestExperimentConfigs:
    """Test experiment configuration composition."""

    def test_laq_debug_config(self, config_dir):
        """Test LAQ debug configuration loads correctly."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=laq_debug"])

            # Validate experiment metadata
            assert cfg.experiment.name == "laq_debug"
            assert "debug" in cfg.experiment.description.lower()

            # Validate model config (LAPA ViT-based LAQ)
            assert cfg.model.name == "laq_vit"
            assert cfg.model.dim == 1024
            assert cfg.model.codebook_size == 8
            assert cfg.model.patch_size == 32

            # Validate data config (uses multi-dataset)
            assert hasattr(cfg.data, "sources")
            assert cfg.data.batch_size == 4

            # Validate training config
            assert cfg.training.epochs == 3
            assert hasattr(cfg.training, "optimizer")
            assert cfg.training.optimizer.type == "AdamW"

            # Validate cluster config
            assert cfg.cluster.name == "local"
            assert cfg.cluster.compute.num_nodes == 1
            assert cfg.cluster.compute.gpus_per_node == 1

    def test_laq_full_config(self, config_dir):
        """Test LAQ full configuration loads correctly."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=laq_full"])

            # Validate experiment (uses openx_v1 not laq_full)
            assert cfg.experiment.name == "laq_openx_v1"
            assert "openx" in cfg.experiment.description.lower()

            # Validate model (same LAPA ViT LAQ model)
            assert cfg.model.name == "laq_vit"
            assert cfg.model.codebook_size == 8

            # Validate training (full training with 100 epochs)
            assert cfg.training.epochs == 100
    
            # Validate cluster config (H100 single node)
            assert cfg.cluster.name == "lrz_h100"
            assert cfg.cluster.compute.num_nodes == 1
            assert cfg.cluster.compute.gpus_per_node == 1
    def test_vla_7b_config(self, config_dir):
        """Test VLA 7B configuration loads correctly."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=vla_7b"])

            # Validate experiment
            assert cfg.experiment.name == "vla_7b_foundation"
            assert "foundation" in cfg.experiment.description.lower()

            # Validate VLA model
            assert cfg.model.name == "vla_7b"
            assert hasattr(cfg.model, "vision")
            assert hasattr(cfg.model, "llm")
            assert cfg.model.llm.model_name == "meta-llama/Llama-2-7b-hf"

            # Validate latent-labeled data
            assert cfg.data.dataset_name == "bridge"
            assert cfg.data.return_metadata is True

            # Validate FSDP training
            assert hasattr(cfg.training, "fsdp")
            assert cfg.training.fsdp.sharding_strategy == "FULL_SHARD"

            # Validate multi-node cluster
            assert cfg.cluster.name == "lrz_h100_multinode"
            assert cfg.cluster.compute.num_nodes == 4
            assert cfg.cluster.compute.gpus_per_node == 4

    def test_vla_cosmos2_tokens_debug_config(self, config_dir):
        """Test Cosmos-Reason2 token-based VLA debug configuration loads correctly."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config", overrides=["experiment=vla_cosmos2_tokens_debug"]
            )

            assert cfg.experiment.name == "vla_cosmos2_tokens_debug"
            assert "cosmos" in cfg.experiment.description.lower()
            assert cfg.model.name == "vla_cosmos2_tokens"
            assert cfg.model.vla.model_name == "nvidia/Cosmos-Reason2-2B"
            assert cfg.model.action_tokens.codebook_size == 8
            assert cfg.model.action_tokens.code_seq_len == 4
            assert cfg.cluster.name == "local"


class TestConfigComposition:
    """Test configuration composition and overrides."""

    def test_cli_overrides(self, config_dir):
        """Test CLI parameter overrides work correctly."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=laq_debug",
                    "data.batch_size=16",
                    "training.optimizer.lr=5e-5",
                    "seed=123"
                ]
            )

            # Verify overrides
            assert cfg.data.batch_size == 16
            assert cfg.training.optimizer.lr == 5e-5
            assert cfg.seed == 123

            # Verify base config still loaded
            assert cfg.experiment.name == "laq_debug"

    def test_nested_overrides(self, config_dir):
        """Test deeply nested configuration overrides."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=laq_debug",
                    "model.dim=512",
                    "training.optimizer.betas=[0.95,0.999]"
                ]
            )

            assert cfg.model.dim == 512
            assert cfg.training.optimizer.betas == [0.95, 0.999]

    def test_config_is_valid_omegaconf(self, config_dir):
        """Test that loaded config is a valid OmegaConf DictConfig."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=laq_debug"])

            # Should be DictConfig, not plain dict
            assert OmegaConf.is_config(cfg)
            assert OmegaConf.is_dict(cfg)


class TestExperimentConsistency:
    """Test that all experiments load without errors."""

    @pytest.mark.parametrize(
        "experiment",
        ["laq_debug", "laq_full", "laq_normal", "vla_7b", "vla_cosmos2_tokens_debug"],
    )
    def test_all_experiments_load(self, config_dir, experiment):
        """Test that all available experiments load successfully."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

            # All configs should have these top-level keys
            assert hasattr(cfg, "experiment")
            assert hasattr(cfg, "model")
            assert hasattr(cfg, "data")
            assert hasattr(cfg, "training")
            assert hasattr(cfg, "cluster")

            # Experiment should have name and description
            assert cfg.experiment.name is not None
            assert cfg.experiment.description is not None

    def test_laq_normal_config(self, config_dir):
        """Test LAQ normal training configuration loads correctly."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=laq_normal"])

            # Validate experiment
            assert cfg.experiment.name == "laq_normal"

            # Validate model (same as debug)
            assert cfg.model.name == "laq_vit"
            assert cfg.model.dim == 1024

            # Validate data config (uses multi-dataset)
            assert hasattr(cfg.data, "sources")

            # Validate training (5000 epochs)
            assert cfg.training.epochs == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
