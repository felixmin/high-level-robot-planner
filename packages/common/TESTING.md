# Data Loading Testing Strategy

## Current State

**Tests:** `tests/common/test_data.py` (36 tests, all passing)

**Coverage:**
- ✅ `ImageVideoDataset`, `MetadataAwareDataset`, `LAQDataModule`
- ✅ `SceneMetadata`, `SceneFilter`, CSV parsing
- ✅ Integration with LAQ model (forward pass)

**Problems:**
- All tests depend on real dataset at `/mnt/data/datasets/youtube_new/JNBtHDVoNQc_stabilized`
- Will skip on CI or other machines (`pytest.skip` if path missing)
- Brittle to dataset changes (CSV schema, number of scenes, folder structure)
- Hard-coded assertions: `assert len(dm.train_dataset) == 90`

## Recommended Refactor

### 1. Add Synthetic Fixtures

**Location:** `tests/conftest.py`

**Create:**
```
tiny_dataset fixture:
  - tmp_path/dataset/scene_000/ with 3 dummy JPGs (64x64)
  - tmp_path/dataset/scenes.csv with 1 row
  - Minimal columns: scene_idx, scene_folder, start_frame, end_frame, stabilized_label, max_trans
```

**Benefits:**
- Fast (no I/O on real data)
- Always available (CI, any machine)
- Immune to WIP dataset changes

### 2. Split Test Files

**`tests/common/test_data_unit.py`** (new):
- Use `tiny_dataset` fixture
- Test logic: filtering, metadata parsing, dataset indexing, collate functions
- Should run in <1s
- No `pytest.skip` conditions

**`tests/common/test_data_integration.py`** (refactor existing):
- Use real dataset at `/mnt/data/datasets/youtube_new/...`
- Mark all tests: `@pytest.mark.integration`
- Relax assertions (compute expected values, don't hard-code)
- Keep model integration test here

**Run strategy:**
```bash
pytest -m "not integration"  # Fast unit tests (CI default)
pytest -m integration         # Heavy real-data tests (manual)
```

### 3. Relax Integration Test Assertions

**Bad:**
```python
assert len(dm.train_dataset) == 90  # Breaks if dataset changes
```

**Good:**
```python
total = len(dm.train_dataset) + len(dm.val_dataset)
assert total == min(max_samples, len(full_dataset))
assert len(dm.val_dataset) == int(total * dm.val_split)
```

### 4. Remove / Minimize Prints

Current tests have many `print(f"✓ ...")` statements.

**Options:**
- Remove entirely (rely on pytest assertion messages)
- Keep only in integration tests for debugging
- Use logging instead of print

### 5. Use Parametrization

**Current:**
```python
for offset in [1, 10, 30, 60]:
    dataset = ImageVideoDataset(..., offset=offset)
    sample = dataset[0]
    assert sample.shape == (3, 2, 256, 256)
```

**Better:**
```python
@pytest.mark.parametrize("offset", [1, 10, 30, 60])
def test_dataset_different_offsets(tiny_dataset, offset):
    dataset = ImageVideoDataset(tiny_dataset, offset=offset)
    sample = dataset[0]
    assert sample.shape == (3, 2, 256, 256)
```

Gives clearer test output (4 separate test cases).

## Action Items

**Priority 1 (blocking for CI/portability):**
- [ ] Add `tiny_dataset` fixture to `tests/conftest.py`
- [ ] Create `tests/common/test_data_unit.py` with core logic tests
- [ ] Mark existing tests in `test_data.py` with `@pytest.mark.integration`

**Priority 2 (code quality):**
- [ ] Relax hard-coded assertions in integration tests
- [ ] Parametrize repeated test loops
- [ ] Remove/reduce print statements

**Priority 3 (nice-to-have):**
- [ ] Add docstrings explaining what each test validates
- [ ] Add test for CSV with extra/missing columns (extensibility)
- [ ] Mock `Image.open` for pure unit tests (no file I/O)

## When to Revisit

- Before setting up CI pipeline
- When dataset schema changes (CSV columns)
- When dataset location changes
- When onboarding new developers (synthetic tests = instant setup)

## Notes

Integration tests are valuable for validating real pipeline, but should be:
1. Clearly marked (`@pytest.mark.integration`)
2. Robust to missing data (`pytest.skip` is fine)
3. Relaxed on exact counts (test logic, not data size)
4. Optional for CI (can run manually with `-m integration`)
