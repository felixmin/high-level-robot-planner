# Open Issues

**Created**: 2025-12-21
**Updated**: 2025-12-21
**Status**: Partially Fixed

## Critical Issues

*None*

## Medium Priority Issues

### 1. Hardcoded `language_table` filtering in validation strategies (UNCOMMITTED CHANGES)

**Location**: `packages/laq/validation.py` (lines 1538, 1701, 1871, 2062)

**Analysis**: The filter `if meta.get("dataset_name") != "language_table": continue` breaks generic scatter strategies. Bridge has actions too but would be silently skipped.

**Decision**: Need bucket/dataset binding for strategies. Two options:
1. **Per-strategy dataset filter** (simple): Add `dataset_filter: List[str]` param to each scatter strategy
2. **Bucket-strategy binding** (already partially exists): Use existing `strategy_bucket_bindings` to route strategies to specific buckets

Option 2 is cleaner since bucket infrastructure exists. The binding would specify which buckets each strategy operates on.

**Status**: OPEN - Needs design decision on whether to use per-strategy filter or leverage bucket bindings.

---

### 2. Duplicate code in validation scatter strategies

**Location**: `packages/laq/validation.py`

**Analysis**: 4 scatter strategies share ~70% code: sample filtering, scatter plot creation, sample limiting.

**Fix**: Extract base class `MetadataScatterStrategy` with shared methods:
- `_filter_samples_with_metadata(cache, required_keys) -> (values, codes, metadata)`
- `_create_scatter_figure(x, y, colors, title, xlabel, ylabel) -> PIL.Image`
- `_limit_samples(data, n) -> data`

**Status**: OPEN - Larger refactor, low urgency since code works.

---

### 3. Magic numbers in codebook replacement schedule

**Location**: `packages/laq/models/latent_action_quantization.py` (line 344)

**Analysis**: Hardcoded schedule `(step % 10 < 100) or (step % 100 < 1000) or (step % 500 < 5000)` is unclear.

**Options**:
1. Move to model config with structured schedule
2. Define as class constants with docstring
3. Use exponential backoff formula

Option 1 (config) is cleanest but requires config schema changes. Option 2 is minimal change.

**Status**: OPEN - Needs config schema design.

---

## Fixes Applied

1. ✅ Removed duplicate `DINOWrapper` class (`packages/laq/models/dino.py`)
2. ✅ Added `state_key="effector_translation"` to 3 OXE dataset configs (`packages/common/adapters/oxe.py`)
3. ✅ Changed print to `logger.warning()` in OXE iterator (`packages/common/adapters/oxe.py`)
4. ✅ Fixed EMACallback to handle OmegaConf DictConfig (`packages/laq/callbacks.py`)
5. ✅ Fixed deprecated `typing.Tuple` to builtin `tuple` (`packages/laq/models/attention.py`)
6. ✅ Fixed test files for 4-value model return (multiple test files)

---

## Code Quality Suggestions

### 1. Add type hints to validation strategies
Low priority. Methods work, just missing return type annotations.

### 2. Add unit tests for scatter strategies
**Status**: OPEN - Should add tests for:
- `ActionTokenScatterStrategy.run()` with mock cache
- `_create_scatter()` output validation
- Edge cases (empty cache, missing metadata)
