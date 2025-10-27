#!/usr/bin/env python3
"""Quick validation that environment is ready."""

from pathlib import Path
from hydra import compose, initialize_config_dir
import torch

print("Testing setup...")

# Test 1: Core imports
print("✓ PyTorch imported")
print("✓ Hydra imported")

# Test 2: GPU/MPS check
if torch.backends.mps.is_available():
    print("✓ MPS (Apple Silicon GPU) available")
elif torch.cuda.is_available():
    print(f"✓ CUDA available ({torch.cuda.device_count()} GPUs)")
else:
    print("⚠ CPU only (no GPU acceleration)")

# Test 3: Config loading
config_dir = Path(__file__).parent / "config"
with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(config_name="config", overrides=["experiment=laq_debug"])
    print(f"✓ Loaded experiment: {cfg.experiment.name}")
    print(f"✓ Batch size: {cfg.data.batch_size}")
    print(f"✓ Training epochs: {cfg.training.epochs}")

print("\n🎉 All tests passed! Ready to start coding!")
print("\nNext step: Implement LAQ Encoder")
print("  File: packages/laq/models/encoder.py")
print("  Reference: PLAN.md Section 2.1.1")

