"""
Tests for OXE dataset weighting computation.

Tests the mixed absolute mode weighting algorithm:
- All proportionate: Weights by dataset size
- All explicit: Fixed numeric weights
- Mixed mode: Combine explicit and proportionate
- Edge cases: Empty datasets, weights >= 1.0
"""

import pytest
from unittest.mock import MagicMock


class TestComputeWeights:
    """Test _compute_weights() method in isolation."""

    @pytest.fixture
    def weight_calculator(self):
        """Create a minimal mock that has the _compute_weights method."""
        # Import the actual class to get the real method
        from common.adapters.oxe import MultiOXEFramePairDataset

        # Create instance with minimal config (won't be initialized)
        instance = MultiOXEFramePairDataset.__new__(MultiOXEFramePairDataset)
        return instance._compute_weights

    def test_all_proportionate_equal_sizes(self, weight_calculator):
        """All datasets use proportionate weighting with equal sizes."""
        configs = [
            {"name": "ds_a"},  # weight omitted = proportionate
            {"name": "ds_b"},
            {"name": "ds_c"},
        ]
        sizes = [1000, 1000, 1000]

        weights = weight_calculator(configs, sizes)

        # Should be equal distribution
        assert len(weights) == 3
        assert pytest.approx(weights[0], abs=0.001) == 1 / 3
        assert pytest.approx(weights[1], abs=0.001) == 1 / 3
        assert pytest.approx(weights[2], abs=0.001) == 1 / 3
        assert pytest.approx(sum(weights), abs=0.001) == 1.0

    def test_all_proportionate_different_sizes(self, weight_calculator):
        """All datasets use proportionate weighting with different sizes."""
        configs = [
            {"name": "ds_a"},
            {"name": "ds_b"},
            {"name": "ds_c"},
        ]
        sizes = [1000, 2000, 1000]  # Total: 4000

        weights = weight_calculator(configs, sizes)

        # Should be proportional to size
        assert pytest.approx(weights[0], abs=0.001) == 0.25  # 1000/4000
        assert pytest.approx(weights[1], abs=0.001) == 0.50  # 2000/4000
        assert pytest.approx(weights[2], abs=0.001) == 0.25  # 1000/4000
        assert pytest.approx(sum(weights), abs=0.001) == 1.0

    def test_explicit_proportionate_keyword(self, weight_calculator):
        """Explicit 'proportionate' keyword works same as omitting weight."""
        configs = [
            {"name": "ds_a", "weight": "proportionate"},
            {"name": "ds_b"},  # omitted
        ]
        sizes = [1000, 1000]

        weights = weight_calculator(configs, sizes)

        assert pytest.approx(weights[0], abs=0.001) == 0.5
        assert pytest.approx(weights[1], abs=0.001) == 0.5

    def test_all_explicit_weights(self, weight_calculator):
        """All datasets have explicit numeric weights."""
        configs = [
            {"name": "ds_a", "weight": 0.3},
            {"name": "ds_b", "weight": 0.5},
            {"name": "ds_c", "weight": 0.2},
        ]
        sizes = [1000, 2000, 3000]  # Sizes are ignored

        weights = weight_calculator(configs, sizes)

        # Should use explicit weights directly
        assert pytest.approx(weights[0], abs=0.001) == 0.3
        assert pytest.approx(weights[1], abs=0.001) == 0.5
        assert pytest.approx(weights[2], abs=0.001) == 0.2
        assert pytest.approx(sum(weights), abs=0.001) == 1.0

    def test_mixed_mode_basic(self, weight_calculator):
        """Mix of explicit and proportionate weights."""
        configs = [
            {"name": "ds_a", "weight": 0.3},  # Explicit: 30%
            {"name": "ds_b"},  # Proportionate
            {"name": "ds_c"},  # Proportionate
        ]
        sizes = [500, 1000, 2000]  # Proportionate total: 3000

        weights = weight_calculator(configs, sizes)

        # ds_a: 0.3 (explicit)
        # Remaining: 0.7
        # ds_b: 0.7 * (1000/3000) = 0.233
        # ds_c: 0.7 * (2000/3000) = 0.467
        assert pytest.approx(weights[0], abs=0.001) == 0.3
        assert pytest.approx(weights[1], abs=0.001) == 0.7 * (1000 / 3000)
        assert pytest.approx(weights[2], abs=0.001) == 0.7 * (2000 / 3000)
        assert pytest.approx(sum(weights), abs=0.001) == 1.0

    def test_mixed_mode_multiple_explicit(self, weight_calculator):
        """Multiple explicit weights with some proportionate."""
        configs = [
            {"name": "ds_a", "weight": 0.2},  # Explicit: 20%
            {"name": "ds_b", "weight": 0.3},  # Explicit: 30%
            {"name": "ds_c"},  # Proportionate: gets 50%
        ]
        sizes = [1000, 1000, 1000]

        weights = weight_calculator(configs, sizes)

        # ds_a: 0.2, ds_b: 0.3, ds_c: 0.5 (remaining)
        assert pytest.approx(weights[0], abs=0.001) == 0.2
        assert pytest.approx(weights[1], abs=0.001) == 0.3
        assert pytest.approx(weights[2], abs=0.001) == 0.5
        assert pytest.approx(sum(weights), abs=0.001) == 1.0

    def test_explicit_weights_exceed_one(self, weight_calculator, caplog):
        """Explicit weights >= 1.0 should warn and normalize for sampling correctness."""
        import logging

        configs = [
            {"name": "ds_a", "weight": 0.6},
            {"name": "ds_b", "weight": 0.5},  # Total: 1.1
            {"name": "ds_c"},  # Proportionate: gets 0
        ]
        sizes = [1000, 1000, 1000]

        with caplog.at_level(logging.WARNING):
            weights = weight_calculator(configs, sizes)

        # Should warn about exceeding 1.0
        assert "1.1" in caplog.text or "1.100" in caplog.text
        assert ">= 1.0" in caplog.text

        # Weights are normalized to sum to 1.0 for sampling correctness
        # Original: [0.6, 0.5, 0.0] -> normalized by 1.1
        assert pytest.approx(weights[0], abs=0.001) == 0.6 / 1.1
        assert pytest.approx(weights[1], abs=0.001) == 0.5 / 1.1
        assert pytest.approx(weights[2], abs=0.001) == 0.0
        assert pytest.approx(sum(weights), abs=0.001) == 1.0

    def test_explicit_weights_equal_one(self, weight_calculator, caplog):
        """Explicit weights exactly 1.0 should warn."""
        import logging

        configs = [
            {"name": "ds_a", "weight": 0.5},
            {"name": "ds_b", "weight": 0.5},
            {"name": "ds_c"},  # Proportionate: gets 0
        ]
        sizes = [1000, 1000, 1000]

        with caplog.at_level(logging.WARNING):
            weights = weight_calculator(configs, sizes)

        # Should warn
        assert ">= 1.0" in caplog.text

        assert pytest.approx(weights[0], abs=0.001) == 0.5
        assert pytest.approx(weights[1], abs=0.001) == 0.5
        assert pytest.approx(weights[2], abs=0.001) == 0.0

    def test_zero_size_proportionate(self, weight_calculator):
        """Proportionate datasets with zero size get equal split."""
        configs = [
            {"name": "ds_a"},
            {"name": "ds_b"},
        ]
        sizes = [0, 0]  # Both have zero size

        weights = weight_calculator(configs, sizes)

        # Equal split when all have zero size
        assert pytest.approx(weights[0], abs=0.001) == 0.5
        assert pytest.approx(weights[1], abs=0.001) == 0.5

    def test_partial_zero_size_proportionate(self, weight_calculator):
        """Mix of zero and non-zero sizes in proportionate datasets."""
        configs = [
            {"name": "ds_a"},  # Proportionate
            {"name": "ds_b"},  # Proportionate
        ]
        sizes = [0, 1000]  # One has zero size

        weights = weight_calculator(configs, sizes)

        # All weight goes to non-zero size dataset
        assert pytest.approx(weights[0], abs=0.001) == 0.0
        assert pytest.approx(weights[1], abs=0.001) == 1.0

    def test_integer_weights(self, weight_calculator):
        """Integer weights are supported and normalized."""
        configs = [
            {"name": "ds_a", "weight": 1},  # Integer
            {"name": "ds_b", "weight": 2},  # Integer
        ]
        sizes = [1000, 1000]

        weights = weight_calculator(configs, sizes)

        # Integer weights 1 and 2 sum to 3.0, so they get normalized
        # This ensures the sampling loop works correctly
        assert pytest.approx(weights[0], abs=0.001) == 1 / 3
        assert pytest.approx(weights[1], abs=0.001) == 2 / 3
        assert pytest.approx(sum(weights), abs=0.001) == 1.0

    def test_invalid_weight_type(self, weight_calculator):
        """Invalid weight type raises ValueError."""
        configs = [
            {"name": "ds_a", "weight": "invalid"},
        ]
        sizes = [1000]

        with pytest.raises(ValueError, match="Invalid weight"):
            weight_calculator(configs, sizes)

    def test_invalid_weight_list(self, weight_calculator):
        """List as weight raises ValueError."""
        configs = [
            {"name": "ds_a", "weight": [0.5]},
        ]
        sizes = [1000]

        with pytest.raises(ValueError, match="Invalid weight"):
            weight_calculator(configs, sizes)

    def test_single_dataset_proportionate(self, weight_calculator):
        """Single proportionate dataset gets 100%."""
        configs = [{"name": "ds_a"}]
        sizes = [1000]

        weights = weight_calculator(configs, sizes)

        assert pytest.approx(weights[0], abs=0.001) == 1.0

    def test_single_dataset_explicit(self, weight_calculator):
        """Single explicit dataset gets normalized to 1.0."""
        configs = [{"name": "ds_a", "weight": 0.5}]
        sizes = [1000]

        weights = weight_calculator(configs, sizes)

        # Single dataset is normalized to 1.0 for sampling correctness
        assert pytest.approx(weights[0], abs=0.001) == 1.0

    def test_real_world_example(self, weight_calculator):
        """Test with realistic dataset sizes (from plan example)."""
        configs = [
            {"name": "my_specific_task", "weight": 0.3},  # Explicit: 30%
            {"name": "language_table"},  # Proportionate
            {"name": "robonet"},  # Proportionate
        ]
        # Realistic sizes: specific task is small, others are large
        sizes = [10_000, 1_000_000, 2_000_000]

        weights = weight_calculator(configs, sizes)

        # my_specific_task: 0.3 (explicit)
        # Remaining: 0.7, split by size ratio
        # language_table: 0.7 * (1M / 3M) = 0.233
        # robonet: 0.7 * (2M / 3M) = 0.467
        assert pytest.approx(weights[0], abs=0.001) == 0.3
        assert pytest.approx(weights[1], abs=0.001) == 0.7 * (1_000_000 / 3_000_000)
        assert pytest.approx(weights[2], abs=0.001) == 0.7 * (2_000_000 / 3_000_000)
        assert pytest.approx(sum(weights), abs=0.001) == 1.0


class TestWeightingIntegration:
    """Integration tests that verify weighting works with real dataset configs."""

    @pytest.fixture
    def mock_datasets_factory(self):
        """Factory to create mock datasets with specific sizes."""
        def create_mock_dataset(size: int):
            mock = MagicMock()
            mock.__len__ = MagicMock(return_value=size)
            return mock
        return create_mock_dataset

    def test_weights_used_in_init(self, mock_datasets_factory, monkeypatch):
        """Verify _compute_weights is called during _init_datasets."""
        from common.adapters.oxe import MultiOXEFramePairDataset

        # Track calls to _compute_weights
        compute_weights_calls = []
        original_compute_weights = MultiOXEFramePairDataset._compute_weights

        def tracking_compute_weights(self, configs, sizes):
            compute_weights_calls.append((configs, sizes))
            return original_compute_weights(self, configs, sizes)

        monkeypatch.setattr(
            MultiOXEFramePairDataset,
            "_compute_weights",
            tracking_compute_weights
        )

        # Mock OXEFramePairDataset to avoid TF initialization
        from common.adapters import oxe

        def mock_oxe_dataset(*args, **kwargs):
            mock = MagicMock()
            mock.__len__ = MagicMock(return_value=1000)
            mock.seed = kwargs.get("seed")
            return mock

        monkeypatch.setattr(oxe, "OXEFramePairDataset", mock_oxe_dataset)

        # Create and initialize
        ds = MultiOXEFramePairDataset(
            datasets=[
                {"name": "language_table", "offset": 5},
                {"name": "bridge", "offset": 5, "weight": 0.4},
            ],
            prefetch_buffer=0,
            image_size=64,
        )
        ds._init_datasets()

        # Verify _compute_weights was called
        assert len(compute_weights_calls) == 1
        configs, sizes = compute_weights_calls[0]
        assert len(configs) == 2
        assert sizes == [1000, 1000]

        # Verify weights were set
        assert ds._weights is not None
        assert len(ds._weights) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
