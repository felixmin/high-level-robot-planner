# LAPA Testing Guide

This directory contains the test suite for the LAPA (Latent Action Pretraining) project.

## Test Organization

```
tests/
├── conftest.py                          # Shared fixtures and pytest configuration
├── unit/                                # Unit tests for individual components
│   ├── test_lapa_attention.py          # Transformer, PEG, position bias tests
│   ├── test_lapa_encoder.py            # Encoder and patch embedding tests
│   ├── test_lapa_nsvq.py               # NSVQ quantization tests
│   └── test_lapa_decoder.py            # Cross-attention decoder tests
├── integration/                         # Integration tests
│   ├── test_lapa_full_model.py         # Full LAPA pipeline tests
│   └── test_laq_lightning_module.py    # Lightning module integration tests
├── config/                              # Configuration tests
│   └── test_hydra_configs.py           # Hydra config validation
└── test_common.py                       # Common test utilities
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Slow tests excluded
pytest -m "not slow"

# GPU tests only (requires CUDA)
pytest -m gpu
```

### Run Specific Test Files

```bash
# Test encoder
pytest tests/unit/test_lapa_encoder.py

# Test full model
pytest tests/integration/test_lapa_full_model.py

# Test with verbose output
pytest tests/unit/test_lapa_attention.py -v
```

### Run Specific Test Functions

```bash
# Run a single test
pytest tests/unit/test_lapa_encoder.py::TestLAPAEncoder::test_encoder_forward

# Run all tests in a class
pytest tests/unit/test_lapa_nsvq.py::TestNSVQ
```

## Coverage

Run tests with coverage reporting:

```bash
# Generate coverage report
pytest --cov=packages.laq --cov-report=html

# View HTML report
open htmlcov/index.html
```

## Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests for component interactions
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.gpu` - Tests that require GPU (CUDA or MPS)

## Fixtures

Common fixtures are defined in `conftest.py`:

- `device` - Auto-selects best available device (CUDA/MPS/CPU)
- `batch_size` - Default batch size (2)
- `image_size` - LAPA image size (256)
- `patch_size` - LAPA patch size (32)
- `lapa_config` - Complete LAPA model configuration
- `sample_frame_pair` - Sample input frames `[B, 3, 2, 256, 256]`
- `sample_frame_tokens` - Sample encoder outputs
- `sample_action_tokens` - Sample NSVQ outputs

## Writing New Tests

### Unit Test Example

```python
import pytest
import torch
from packages.laq.models.encoder import LAPAEncoder

@pytest.mark.unit
class TestMyComponent:
    def test_forward_pass(self, device, lapa_config):
        model = LAPAEncoder(**lapa_config).to(device)
        x = torch.randn(2, 3, 2, 256, 256, device=device)
        output = model(x)
        assert output is not None
```

### Integration Test Example

```python
import pytest
import torch
from packages.laq.models.lapa import LAPA

@pytest.mark.integration
class TestMyIntegration:
    def test_end_to_end(self, device, lapa_config):
        model = LAPA(**lapa_config).to(device)
        frames = torch.randn(2, 3, 2, 256, 256, device=device)
        reconstructed, actions, indices = model(frames)
        assert reconstructed.shape == (2, 3, 256, 256)
```

## CI/CD Integration

Tests are automatically run in CI/CD pipelines. Local testing ensures:

1. **Code Quality**: All tests pass before committing
2. **No Regressions**: Changes don't break existing functionality
3. **Documentation**: Tests serve as usage examples

## Debugging Failed Tests

```bash
# Run with detailed output
pytest -vv --tb=long

# Stop at first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s
```

## Performance Testing

For benchmarking:

```bash
# Run slow tests (includes performance tests)
pytest -m slow

# Profile test execution time
pytest --durations=10
```

## Best Practices

1. **Fast Tests**: Unit tests should run quickly (<1s each)
2. **Isolation**: Each test should be independent
3. **Fixtures**: Use fixtures for common setup
4. **Markers**: Tag tests appropriately
5. **Assertions**: Use clear, specific assertions
6. **Cleanup**: Tests should not leave side effects

## Requirements

All testing dependencies are in `environment.yml`:

```yaml
- pytest>=7.4.0
- pytest-cov>=4.1.0
```

Install with:

```bash
conda env update -f environment.yml
```


