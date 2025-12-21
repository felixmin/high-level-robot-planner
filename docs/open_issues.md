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

### 3. Duplicate class definition for `DINOWrapper`

**Location**: `packages/laq/models/dino.py`

**Analysis**: Copy-paste error. Second definition (with `ndim == 5` check) is correct.

**Fix**: Remove first definition, keep second.

**Status**: ✅ FIXED

---

### 4. Missing `state_key` in some OXE dataset configs

**Location**: `packages/common/adapters/oxe.py` (lines 80-103)

**Analysis**: 3 language_table variants missing `state_key="effector_translation"`. Would cause state scatter to fail silently.

**Fix**: Add `state_key="effector_translation"` to all 3 configs.

**Status**: ✅ FIXED

---

## Low Priority Issues

### 5. Inconsistent error handling in OXE iterator

**Location**: `packages/common/adapters/oxe.py` (line 452)

**Analysis**: Using `print()` instead of `logging`. Minor but inconsistent.

**Fix**: Use `logger.warning()` with module-level logger.

**Status**: ✅ FIXED

---

### 6. Magic numbers in codebook replacement schedule

**Location**: `packages/laq/models/latent_action_quantization.py` (line 344)

**Analysis**: Hardcoded schedule `(step % 10 < 100) or (step % 100 < 1000) or (step % 500 < 5000)` is unclear.

**Options**:
1. Move to model config with structured schedule
2. Define as class constants with docstring
3. Use exponential backoff formula

Option 1 (config) is cleanest but requires config schema changes. Option 2 is minimal change.

**Status**: OPEN - Needs config schema design.

---

### 7. EMACallback assumes model has specific constructor

**Location**: `packages/laq/callbacks.py` (lines 391-394)

**Analysis**: `**pl_module.model_config` fails if config is OmegaConf DictConfig.

**Fix**: Convert to dict: `dict(pl_module.model_config)`.

**Status**: ✅ FIXED

---

### 8. ValidationCache metadata structure inconsistency

**Location**: `packages/laq/validation.py` (lines 178-186)

**Analysis**: Comment clarified - both `add_sample` and `add_batch` now store metadata as lists consistently. `get_all_metadata` handles flattening.

**Fix**: Updated comment to clarify intentional design.

**Status**: ✅ FIXED (was not a bug, just unclear comment)

---

### 9. Deprecated type hints in attention.py

**Location**: `packages/laq/models/attention.py` (lines 62, 309)

**Analysis**: `typing.Tuple` deprecated since Python 3.9, removed in 3.13.

**Fix**: Replace with builtin `tuple[int, int, int, int]`.

**Status**: ✅ FIXED

---

## Code Quality Suggestions

### 1. Add type hints to validation strategies
Low priority. Methods work, just missing return type annotations.

### 2. Consider using `Protocol` for strategy interface
Not worth changing - ABC works fine and is more explicit.

### 3. Add unit tests for scatter strategies
**Status**: OPEN - Should add tests for:
- `ActionTokenScatterStrategy.run()` with mock cache
- `_create_scatter()` output validation
- Edge cases (empty cache, missing metadata)