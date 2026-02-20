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

    def test_laq_hf_local_config(self, config_dir):
        """Test HuggingFace LAQ configuration loads correctly."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=laq_hf_local"])

            # Validate experiment metadata
            assert cfg.experiment.name == "laq_hf_local"
            assert "huggingface" in cfg.experiment.description.lower()

            # Validate model config (LAPA ViT-based LAQ)
            assert cfg.model.name == "laq_vit"
            assert cfg.model.dim == 1024
            assert cfg.model.codebook_size == 8
            assert cfg.model.patch_size == 32

            # Validate data config
            assert cfg.data.backend == "oxe_hf"
            assert hasattr(cfg.data, "dataset")
            assert len(cfg.data.dataset.hf_oxe.datasets) >= 1

            # Validate training config
            assert cfg.training.epochs == 100
            assert hasattr(cfg.training, "optimizer")
            assert cfg.training.optimizer.type == "AdamW"
            assert bool(cfg.training.dataset_usage_logger.enabled) is False
            assert bool(cfg.training.dataset_usage_logger.log_on_validation_end) is True

            # Validate cluster config
            assert cfg.cluster.name == "local_dev"
            assert bool(cfg.cluster.slurm.enabled) is False

    def test_laq_oxe_local_config(self, config_dir):
        """Test OXE local configuration loads correctly."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=laq_oxe_local"])

            # Validate experiment
            assert cfg.experiment.name == "laq_oxe_local"
            assert "oxe" in cfg.experiment.description.lower()

            # Validate model (same LAPA ViT LAQ model)
            assert cfg.model.name == "laq_vit"
            assert cfg.model.codebook_size == 8

            # Validate data (OXE-style)
            assert cfg.data.backend == "oxe_tf"
            assert len(cfg.data.dataset.oxe.datasets) == 4
            assert hasattr(cfg.data.adapter.tf.train, "episode_queue_shuffle_buffer")
            assert hasattr(cfg.data.adapter.tf.train, "global_stream_shuffle_buffer")

            # Validate cluster config (H100 single node)
            assert cfg.cluster.name == "local_dev"

    def test_data_oxe_local_indexed_config(self, config_dir):
        """Test local indexed OpenX data preset loads correctly."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=["experiment=laq_oxe_local", "data=oxe_local_indexed"],
            )

            assert cfg.data.backend == "oxe_local_indexed"
            assert hasattr(cfg.data.adapter, "openx_local")
            assert cfg.data.adapter.openx_local.mode == "indexed"
            assert bool(cfg.data.adapter.openx_local.index_rebuild) is False
            assert int(cfg.data.adapter.openx_local.index_max_open_shards) >= 1
            assert bool(cfg.data.adapter.openx_local.weights_by_size) is False
            assert len(cfg.data.dataset.oxe.datasets) >= 1

    def test_data_oxe_local_indexed_full_override(self, config_dir):
        """Test indexed_full mode override composes."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=laq_oxe_local",
                    "data=oxe_local_indexed",
                    "data.adapter.openx_local.mode=indexed_full",
                ],
            )

            assert cfg.data.backend == "oxe_local_indexed"
            assert cfg.data.adapter.openx_local.mode == "indexed_full"

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
            assert cfg.data.backend == "oxe_tf"
            assert cfg.data.dataset.oxe.datasets[0].name == "bridge"
            assert bool(cfg.data.preprocess.return_metadata) is True

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
            assert cfg.model.action_tokens.codebook_size == 4096
            assert cfg.model.action_tokens.code_seq_len == 1
            assert cfg.data.adapter.tf.val.episode_queue_shuffle_buffer == 200
            assert cfg.training.validation.check_interval == 100
            assert cfg.training.validation.limit_batches == 4
            assert bool(cfg.training.validation.visualization.enabled) is True
            assert bool(cfg.training.dataset_usage_logger.enabled) is True
            assert bool(cfg.training.dataset_usage_logger.log_on_validation_end) is True
            assert cfg.training.checkpoint.every_n_train_steps == 100
            assert cfg.cluster.name == "local_dev"

    def test_vla_cosmos2_tokens_config(self, config_dir):
        """Test Cosmos-Reason2 token-based VLA (non-debug) configuration loads correctly."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=vla_cosmos2_tokens"])

            assert cfg.experiment.name == "vla_cosmos2_tokens"
            assert cfg.model.name == "vla_cosmos2_tokens"
            assert cfg.model.vla.model_name == "nvidia/Cosmos-Reason2-2B"
            assert cfg.model.action_tokens.codebook_size == 8
            assert cfg.model.action_tokens.code_seq_len == 4
            assert cfg.data.backend == "oxe_tf"
            assert len(cfg.data.dataset.oxe.datasets) == 4
            assert cfg.data.loader.batch_size == 64
            assert cfg.training.validation.check_interval == 1000
            assert cfg.training.checkpoint.every_n_train_steps == 1000
            assert bool(cfg.training.dataset_usage_logger.enabled) is True
            assert cfg.cluster.name == "mcml_h100"


class TestConfigComposition:
    """Test configuration composition and overrides."""

    def test_cli_overrides(self, config_dir):
        """Test CLI parameter overrides work correctly."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=laq_hf_local",
                    "data.loader.batch_size=16",
                    "training.optimizer.lr=5e-5",
                    "seed=123",
                ],
            )

            # Verify overrides
            assert cfg.data.loader.batch_size == 16
            assert cfg.training.optimizer.lr == 5e-5
            assert cfg.seed == 123

            # Verify base config still loaded
            assert cfg.experiment.name == "laq_hf_local"

    def test_nested_overrides(self, config_dir):
        """Test deeply nested configuration overrides."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=laq_hf_local",
                    "model.dim=512",
                    "training.optimizer.betas=[0.95,0.999]",
                ],
            )

            assert cfg.model.dim == 512
            assert cfg.training.optimizer.betas == [0.95, 0.999]

    def test_config_is_valid_omegaconf(self, config_dir):
        """Test that loaded config is a valid OmegaConf DictConfig."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=laq_hf_local"])

            # Should be DictConfig, not plain dict
            assert OmegaConf.is_config(cfg)
            assert OmegaConf.is_dict(cfg)


class TestExperimentConsistency:
    """Test that all experiments load without errors."""

    @pytest.mark.parametrize(
        "experiment",
        [
            "laq_compare_sweep",
            "laq_hf_local",
            "laq_lr_sweep",
            "laq_oxe_all_val",
            "laq_oxe_all_val_3",
            "laq_oxe_cluster",
            "laq_oxe_eval",
            "laq_oxe_local",
            "laq_oxe_local_debug",
            "vla_7b",
            "vla_cosmos2_tokens_debug",
            "vla_cosmos2_tokens",
        ],
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

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
