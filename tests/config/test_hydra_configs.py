"""
Tests for Hydra configuration composition.
"""

import pytest
from hydra import compose, initialize_config_dir
from pathlib import Path


@pytest.fixture
def config_dir():
    """Get path to config directory."""
    return str(Path(__file__).parent.parent.parent / "config")


def test_laq_debug_config(config_dir):
    """Test LAQ debug configuration loads correctly."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=laq_debug"])
        
        assert cfg.experiment.name == "laq_debug"
        # New LAPA config structure: model config is merged at top level via @_here_
        # Check that model config exists (channels should be at top level)
        assert hasattr(cfg, 'channels') or hasattr(cfg, 'model')
        # If model is merged, channels should be accessible directly
        if hasattr(cfg, 'channels'):
            assert cfg.channels == 3
        elif hasattr(cfg, 'model'):
            assert cfg.model.channels == 3
        assert cfg.data.batch_size == 8
        assert cfg.training.epochs == 5


def test_laq_full_config(config_dir):
    """Test LAQ full configuration loads correctly."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=laq_full"])
        
        assert cfg.experiment.name == "laq_openx_v1"
        assert cfg.data.batch_size == 256
        assert cfg.training.epochs == 100


def test_vla_config(config_dir):
    """Test VLA 7B configuration loads correctly."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=vla_7b"])
        
        assert cfg.llm.model_name == "meta-llama/Llama-2-7b-hf"  # Check LLM config directly
        assert cfg.cluster.compute.num_nodes == 4


def test_config_override(config_dir):
    """Test CLI overrides work correctly."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=laq_debug",
                "data.batch_size=16",
                "optimizer.lr=5e-5"  # Optimizer is at top level
            ]
        )
        
        assert cfg.data.batch_size == 16
        assert cfg.optimizer.lr == 5e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

