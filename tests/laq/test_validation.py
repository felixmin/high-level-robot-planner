"""
Tests for LAQ validation strategies.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from laq.validation import (
    ValidationCache,
    ValidationStrategy,
    BasicVisualizationStrategy,
    LatentTransferStrategy,
    ClusteringStrategy,
    create_validation_strategies,
)


class TestValidationCache:
    """Test ValidationCache functionality."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        cache = ValidationCache()
        assert len(cache.frames) == 0
        assert len(cache.latents) == 0
        assert len(cache.codes) == 0
        assert cache.fixed_frames is None
        assert len(cache.metadata) == 0

    def test_cache_append_and_get(self):
        """Test appending and retrieving cached data."""
        cache = ValidationCache()

        # Add some fake data
        frames1 = torch.randn(4, 3, 2, 64, 64)
        frames2 = torch.randn(4, 3, 2, 64, 64)
        cache.frames.append(frames1)
        cache.frames.append(frames2)

        all_frames = cache.get_all_frames()
        assert all_frames.shape == (8, 3, 2, 64, 64)

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = ValidationCache()
        cache.frames.append(torch.randn(4, 3, 2, 64, 64))
        cache.latents.append(torch.randn(4, 32))

        cache.clear()
        assert len(cache.frames) == 0
        assert len(cache.latents) == 0
    
    def test_cache_metadata(self):
        """Test metadata storage and retrieval."""
        cache = ValidationCache()
        
        # Add metadata batches
        meta1 = [{"dataset_type": "youtube"}, {"dataset_type": "youtube"}]
        meta2 = [{"dataset_type": "bridge"}, {"dataset_type": "bridge"}]
        cache.metadata.append(meta1)
        cache.metadata.append(meta2)
        
        all_meta = cache.get_all_metadata()
        assert len(all_meta) == 4
        assert all_meta[0]["dataset_type"] == "youtube"
        assert all_meta[2]["dataset_type"] == "bridge"
    
    def test_get_frames_by_dataset_type(self):
        """Test filtering frames by dataset type."""
        cache = ValidationCache()
        
        # Add frames with metadata
        frames = torch.randn(4, 3, 2, 64, 64)
        cache.frames.append(frames)
        cache.metadata.append([
            {"dataset_type": "youtube"},
            {"dataset_type": "bridge"},
            {"dataset_type": "youtube"},
            {"dataset_type": "bridge"},
        ])
        
        youtube_frames = cache.get_frames_by_dataset_type("youtube")
        bridge_frames = cache.get_frames_by_dataset_type("bridge")
        
        assert youtube_frames.shape[0] == 2
        assert bridge_frames.shape[0] == 2


class TestValidationStrategy:
    """Test ValidationStrategy base class."""

    def test_should_run_every_n(self):
        """Test should_run logic with every_n_validations."""
        strategy = BasicVisualizationStrategy(
            enabled=True,
        )

        # Should run on first validation
        assert strategy.should_run()

        # Increment counter
        strategy.increment_count()

        # Should still run (every_n_validations=1)
        assert strategy.should_run()

    def test_should_run_disabled(self):
        """Test disabled strategy never runs."""
        strategy = BasicVisualizationStrategy(enabled=False)
        assert not strategy.should_run()

    def test_should_run_periodic(self):
        """Test periodic strategy runs at correct intervals."""
        strategy = LatentTransferStrategy(
            enabled=True,
            every_n_validations=5,
        )

        # Should run on first validation (count=0)
        assert strategy.should_run()

        # Skip next 4
        for i in range(4):
            strategy.increment_count()
            assert not strategy.should_run()

        # Should run again on 5th
        strategy.increment_count()  # count=5
        assert strategy.should_run()


class TestCreateValidationStrategies:
    """Test strategy creation from config."""

    def test_create_all_strategies(self):
        """Test creating all strategy types."""
        config = {
            "basic": {"enabled": True},
            "latent_transfer": {"enabled": True, "every_n_validations": 5},
            "clustering": {"enabled": True, "every_n_validations": 10},
        }
        strategies = create_validation_strategies(config)

        assert len(strategies) == 3
        assert any(s.name == "basic_visualization" for s in strategies)
        assert any(s.name == "latent_transfer" for s in strategies)
        assert any(s.name == "clustering" for s in strategies)

    def test_create_only_basic(self):
        """Test creating only basic strategy."""
        config = {
            "basic": {"enabled": True},
            "latent_transfer": {"enabled": False},
            "clustering": {"enabled": False},
        }
        strategies = create_validation_strategies(config)

        assert len(strategies) == 1
        assert strategies[0].name == "basic_visualization"

    def test_empty_config(self):
        """Test with empty config."""
        strategies = create_validation_strategies({})
        assert len(strategies) == 0


class TestBasicVisualizationStrategy:
    """Test BasicVisualizationStrategy."""

    def test_needs_caching(self):
        """Test that basic strategy needs caching."""
        strategy = BasicVisualizationStrategy()
        assert strategy.needs_caching()

    def test_run_with_empty_cache(self):
        """Test run with empty cache returns empty metrics."""
        strategy = BasicVisualizationStrategy()
        cache = ValidationCache()

        # Create mock pl_module and trainer
        pl_module = MagicMock()
        trainer = MagicMock()
        trainer.loggers = []

        metrics = strategy.run(cache, pl_module, trainer)
        assert metrics == {}


class TestLatentTransferStrategy:
    """Test LatentTransferStrategy."""

    def test_needs_caching(self):
        """Test that latent transfer strategy needs caching."""
        strategy = LatentTransferStrategy()
        assert strategy.needs_caching()

    def test_run_with_insufficient_data(self):
        """Test run with insufficient data returns empty metrics."""
        strategy = LatentTransferStrategy(num_pairs=10)
        cache = ValidationCache()

        # Only add 2 frames (need at least 4)
        cache.frames.append(torch.randn(2, 3, 2, 64, 64))

        pl_module = MagicMock()
        trainer = MagicMock()
        trainer.loggers = []

        metrics = strategy.run(cache, pl_module, trainer)
        assert metrics == {}


class TestClusteringStrategy:
    """Test ClusteringStrategy."""

    def test_needs_caching(self):
        """Test that clustering strategy needs caching."""
        strategy = ClusteringStrategy()
        assert strategy.needs_caching()

    def test_run_with_insufficient_data(self):
        """Test run with insufficient data returns empty metrics."""
        strategy = ClusteringStrategy(num_clusters=16)
        cache = ValidationCache()

        # Only add 5 codes (need at least num_clusters)
        cache.codes.append(torch.randint(0, 8, (5, 4)))

        pl_module = MagicMock()
        trainer = MagicMock()
        trainer.loggers = []

        metrics = strategy.run(cache, pl_module, trainer)
        assert metrics == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
