Here’s an updated version of your **Data Loading Testing Strategy** that’s aligned with the **future pair-level dataloader** (`MetadataAwarePairDataset`, overfitting-on-one-pair, offsets, etc.).

---

# Data Loading Testing Strategy (Updated for Pair-Level Dataloader)

## Current State

**Tests:** `tests/common/test_data.py` (36 tests, all passing)

**Coverage:**

* ✅ `ImageVideoDataset`, `MetadataAwareDataset`, `LAQDataModule`
* ✅ `SceneMetadata`, `SceneFilter`, CSV parsing
* ✅ Integration with LAQ model (forward pass)

**Problems:**

* All tests depend on real dataset at
  `/mnt/data/datasets/youtube_new/JNBtHDVoNQc_stabilized`
* Will skip on CI or other machines (`pytest.skip` if path missing)
* Brittle to dataset changes (CSV schema, number of scenes, folder structure)
* Hard-coded assertions like: `assert len(dm.train_dataset) == 90`
* Tests only cover **scene-level indexing**, while the future design is **pair-level indexing**

---

## Target Architecture (Relevance for Testing)

We are moving towards:

* **Pair-level dataset**: `MetadataAwarePairDataset`

  * Each index corresponds to a *concrete* pair: `(scene_idx, first_frame_idx, second_frame_idx, offset)`.
  * Pairs are **pre-enumerated** in `__init__` (no randomness in `__getitem__`).
* `LAQDataModule`:

  * When `pair_level=True`, `full_dataset` is pair-level, and:

    * `len(full_dataset)` = number of **pairs**, not scenes.
    * `max_samples` and `val_split` operate on **pairs**.
* Overfitting debug mode:

  * `pair_level=True`, `max_samples=1`, `val_split=0.0` → train on **one fixed image pair**.

Tests must reflect these semantics.

---

## Recommended Refactor

### 1. Add Synthetic Fixtures

**Location:** `tests/conftest.py`

**Create a `tiny_dataset` fixture:**

* Directory structure under `tmp_path`:

  ```
  tmp_path / "dataset" /
      scenes.csv
      scene_000/ frame_000.jpg, frame_001.jpg, frame_002.jpg
      scene_001/ frame_000.jpg, frame_001.jpg
  ```
* `scenes.csv` with minimal columns:

  * `scene_idx, scene_folder, start_frame, end_frame, stabilized_label, max_trans`
  * Example:

    * `scene_000`: `stabilized_label="stabilized"`, `start_frame=0`, `end_frame=3`
    * `scene_001`: `stabilized_label="unstable"`, `start_frame=0`, `end_frame=2`

**Properties:**

* Tiny, deterministic, fast to create.
* Always available on CI / any machine.
* Supports:

  * Scene-level filtering,
  * Pair-level enumeration,
  * Different offsets (e.g. `offset=1` → known pair counts).

**Benefits:**

* No dependency on real `/mnt/data/...`.
* Stable against dataset schema and content changes.
* Enables precise expectations (e.g. exact pair counts).

---

### 2. Split Tests into Unit vs Integration

#### `tests/common/test_data_unit.py` (new)

Pure logic tests on **synthetic data** via `tiny_dataset`:

* `SceneMetadata` / `SceneFilter`

  * CSV parsing, extras behavior, boolean and comparison filters.
* `ImageVideoDataset` / `MetadataAwareDataset` (legacy behavior)

  * Length, single-sample shape, basic sanity checks.
* **`MetadataAwarePairDataset` (pair-level)**:

  * Correct **pair counts** for known frame counts and offsets.
  * Correct mapping index → `(scene_idx, first_frame_idx, second_frame_idx, offset)`.
* `LAQDataModule` (with `pair_level=True`):

  * `max_samples` and `val_split` work on **pairs**.
  * Overfitting config: `max_samples=1`, `val_split=0.0` → `len(train_dataset)==1`, `len(val_dataset)==0`.

**Properties:**

* No `pytest.skip` needed.
* Should run in <1s.
* Use parametrization where possible.

#### `tests/common/test_data_integration.py` (refactor existing)

Keep tests that use the **real** dataset at
`/mnt/data/datasets/youtube_new/JNBtHDVoNQc_stabilized`.

* Mark all tests:

  ```python
  @pytest.mark.integration
  ```
* Still exercise:

  * Real CSV schema,
  * Real folder structure,
  * Model forward pass with LAQ and real batches.
* Should be robust to data size changes (see next section).

**Run strategy:**

```bash
# CI / default: fast, synthetic, deterministic
pytest -m "not integration"

# Optional heavy tests
pytest -m integration
```

---

### 3. Relax Integration Test Assertions

**Avoid:**

```python
assert len(dm.train_dataset) == 90
```

This breaks as soon as:

* Dataset changes size,
* New scenes are added/removed,
* Filters or offsets change.

**Use formula-based checks instead (pair-level aware):**

Assuming:

* `full_dataset` is the underlying dataset *before* train/val split.
* `max_samples` and `val_split` are known from config.

Example:

```python
total_pairs = len(dm.train_dataset) + len(dm.val_dataset)

# Total should be min(max_samples, len(full_dataset))
assert total_pairs == min(max_samples, len(full_dataset))

# Validation fraction should respect val_split (up to flooring effects)
assert len(dm.val_dataset) == int(total_pairs * dm.val_split)
```

If dataset uses `pair_level=True`, `len(full_dataset)` is the **number of pairs**, not scenes.

This keeps tests about **logic**, not initial data sizes.

---

### 4. Add Pair-Level Specific Tests

These are new unit tests in `test_data_unit.py` using `tiny_dataset`.

#### a) Pair count vs frame count

Given `scene_000` has 3 frames and `offset=1`:

* Valid pairs: (0,1), (1,2) → 2 pairs.

Example test:

```python
def test_pair_dataset_pair_count_tiny_dataset(tiny_dataset):
    ds = MetadataAwarePairDataset(folder=tiny_dataset, offsets=[1])
    assert len(ds) == 2  # from known fixture definition
    sample = ds[0]
    assert sample["frames"].shape == (3, 2, 256, 256)
```

Add parametrization for different offsets (e.g. `[1, 2]`).

#### b) Scene-level filtering reflected at pair-level

If only `scene_000` is `stabilized`:

```python
def test_pair_dataset_respects_scene_filters(tiny_dataset):
    ds_all = MetadataAwarePairDataset(folder=tiny_dataset, filters=None, offsets=[1])
    ds_stab = MetadataAwarePairDataset(
        folder=tiny_dataset,
        filters={"stabilized_label": "stabilized"},
        offsets=[1],
    )

    assert len(ds_stab) > 0
    assert len(ds_stab) <= len(ds_all)
```

This ensures “filter scenes first, then enumerate pairs” is working.

#### c) Overfitting-on-one-pair via `LAQDataModule`

```python
def test_overfit_single_pair_pairlevel(tiny_dataset):
    dm = LAQDataModule(
        folder=tiny_dataset,
        use_metadata=True,
        pair_level=True,
        offsets=[1],
        max_samples=1,
        val_split=0.0,
        batch_size=1,
        num_workers=0,
        return_metadata=True,
    )
    dm.setup("fit")

    assert len(dm.train_dataset) == 1
    assert len(dm.val_dataset) == 0

    batch = next(iter(dm.train_dataloader()))
    assert batch["frames"].shape == (1, 3, 2, 256, 256)
```

This directly validates the **overfit-on-one-pair** debug path.

---

### 5. Prints, Logging & Parametrization

**Prints:**

* Remove or reduce `print("✓ ...")` in unit tests.
* Optional in integration tests for debugging.
* Prefer pytest assertion messages or logging if needed.

**Parametrization:**

Replace manual loops with `@pytest.mark.parametrize`, especially for offsets.

Example:

```python
@pytest.mark.parametrize("offset", [1, 10, 30, 60])
def test_image_video_dataset_offsets(tiny_dataset, offset):
    ds = ImageVideoDataset(folder=tiny_dataset, offset=offset)
    sample = ds[0]
    assert sample.shape == (3, 2, 256, 256)
```

Makes failing cases clearer and isolates failures.

---

### 6. Action Items

**Priority 1 (CI-ready / portability):**

* [ ] Add `tiny_dataset` fixture in `tests/conftest.py`.
* [ ] Create `tests/common/test_data_unit.py` with:

  * Scene metadata/filter tests,
  * `ImageVideoDataset` / `MetadataAwareDataset` basics,
  * **`MetadataAwarePairDataset` tests** (pair count, offsets, filtering),
  * `LAQDataModule` tests with `pair_level=True` (including overfit-on-one-pair).
* [ ] Mark existing real-data tests as `@pytest.mark.integration`.

**Priority 2 (code quality):**

* [ ] Relax hard-coded size assertions in integration tests (use formulas).
* [ ] Parametrize repeated loops (offsets, configs).
* [ ] Remove or minimize print statements.

**Priority 3 (nice-to-have):**

* [ ] Add docstrings to tests explaining what each validates (especially pair-level ones).
* [ ] Add tests for CSV with extra/missing columns (extensibility).
* [ ] Optionally mock `Image.open` / `np.load` for pure logic tests (no I/O), if needed later.

---

## When to Revisit

* When pair-level metadata (e.g. pair-level flow quality) is added → add tests for **pair-level filtering**.
* When global parquet index or multiple sources (YouTube + others) are introduced → add tests for:

  * Source tags,
  * Multiple roots / scene folders.
* Before setting up or tightening CI → ensure `pytest -m "not integration"` passes fast and deterministically.

---

This document is now consistent with the **future pair-level dataloader** and your immediate need to **overfit on a single image pair** while keeping tests robust, CI-friendly, and easy to extend.
