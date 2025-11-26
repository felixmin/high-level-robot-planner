#!/usr/bin/env python
"""
Run all LAPA component tests.

This script tests all LAPA components to ensure they work correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("Running All LAPA Component Tests")
print("=" * 70)

# Test 1: Attention components
print("\n" + "=" * 70)
print("Test 1/6: Transformer Components (Transformer, PEG, ContinuousPositionBias)")
print("=" * 70)
try:
    from models import attention
    attention.test_peg()
    attention.test_continuous_position_bias()
    attention.test_transformer()
    print("✅ Attention components test PASSED")
except Exception as e:
    print(f"❌ Attention components test FAILED: {e}")
    sys.exit(1)

# Test 2: NSVQ
print("\n" + "=" * 70)
print("Test 2/6: NSVQ (Delta Quantization)")
print("=" * 70)
try:
    from models import nsvq
    nsvq.test_nsvq()
    print("✅ NSVQ test PASSED")
except Exception as e:
    print(f"❌ NSVQ test FAILED: {e}")
    sys.exit(1)

# Test 3: Encoder
print("\n" + "=" * 70)
print("Test 3/6: LAPA Encoder (Spatial-Temporal Transformer)")
print("=" * 70)
try:
    from models import encoder
    encoder.test_encoder()
    print("✅ Encoder test PASSED")
except Exception as e:
    print(f"❌ Encoder test FAILED: {e}")
    sys.exit(1)

# Test 4: Decoder
print("\n" + "=" * 70)
print("Test 4/6: LAPA Decoder (Cross-Attention)")
print("=" * 70)
try:
    from models import decoder
    decoder.test_decoder()
    print("✅ Decoder test PASSED")
except Exception as e:
    print(f"❌ Decoder test FAILED: {e}")
    sys.exit(1)

# Test 5: Complete LAPA model
print("\n" + "=" * 70)
print("Test 5/6: Complete LAPA Model")
print("=" * 70)
try:
    from models import lapa
    lapa.test_lapa()
    print("✅ LAPA model test PASSED")
except Exception as e:
    print(f"❌ LAPA model test FAILED: {e}")
    sys.exit(1)

# Test 6: Dataset
print("\n" + "=" * 70)
print("Test 6/6: Dataset Preprocessing")
print("=" * 70)
try:
    import data
    data.test_dataset()
    print("✅ Dataset test PASSED")
except Exception as e:
    print(f"❌ Dataset test FAILED: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("🎉 ALL TESTS PASSED! 🎉")
print("=" * 70)
print("\nLAPA implementation is ready for training!")
print("All components are working correctly:")
print("  ✅ Transformer components (Transformer, PEG, ContinuousPositionBias)")
print("  ✅ NSVQ (delta quantization with single codebook)")
print("  ✅ Encoder (spatial-temporal transformer)")
print("  ✅ Decoder (cross-attention)")
print("  ✅ Complete LAPA model")
print("  ✅ Dataset preprocessing")
print("\nNext steps:")
print("  1. Prepare your video dataset")
print("  2. Run preprocessing: python scripts/1_videos_to_webdataset.py")
print("  3. Start training: python scripts/2_train_laq.py")

