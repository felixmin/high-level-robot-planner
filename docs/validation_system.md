# Validation System Architecture

## Overview

The LAQ validation system uses **bucket-strategy binding** to enable flexible, multi-dataset validation with automatic applicability checks. This architecture addresses the challenge of running different validation strategies on heterogeneous datasets with varying metadata availability.

## Core Concepts

### 1. Validation Buckets

Buckets are named data subsets defined by metadata filters. Samples are routed to matching buckets during validation.

```yaml
validation:
  buckets:
    youtube_iid:
      filters: {dataset_type: "youtube"}
      max_samples: 100
    bridge_holdout:
      filters: {dataset_type: "bridge", environment: "toykitchen7"}
      max_samples: 100
      is_holdout: true  # Mark as OOD for analysis
```

**Key Properties**:
- `filters`: Metadata conditions for bucket membership (supports operators like `!=`, `>`, `<`)
- `max_samples`: Maximum samples cached per bucket
- `is_holdout`: Flag for distribution shift data

### 2. Validation Strategies

Strategies define validation logic (e.g., reconstruction visualization, clustering). Each strategy:
- Declares metadata requirements via `required_metadata()`
- Declares minimum sample count via `min_samples`
- Checks data availability via `can_run(cache)`
- Runs validation via `run(cache, pl_module, trainer, metric_suffix)`

**Example**:
```python
class ActionTokenScatterStrategy(ValidationStrategy):
    def required_metadata(self) -> List[str]:
        return ["action"]  # Requires action metadata

    def needs_codes(self) -> bool:
        return True  # Requires codebook indices

    def can_run(self, cache: ValidationCache) -> Tuple[bool, str]:
        # Checks if cache has enough samples with valid action metadata
        count = cache.count_samples_with_metadata(["action"])
        if count < self.min_samples:
            return False, f"Only {count} samples with action (need {self.min_samples})"
        return True, ""
```

### 3. Strategy-Bucket Binding

Strategies are bound to specific buckets via configuration:

```yaml
validation:
  strategy_bucket_bindings:
    action_token_scatter:
      buckets: [language_table]  # Only runs on language_table bucket
    latent_transfer:
      buckets: [bridge_iid, bridge_holdout]
      compare_buckets: true  # Run separately per bucket with metric suffix
```

**Binding Modes**:
- **Merged mode** (`compare_buckets: false`): Combine bucket data and run once
- **Compare mode** (`compare_buckets: true`): Run separately on each bucket with metric suffix (e.g., `val/latent_transfer_mse_bridge_iid` vs `val/latent_transfer_mse_bridge_holdout`)

## Data Flow

```
Validation Batch
       │
       ▼
Extract metadata per sample
       │
       ▼
Route to matching buckets ──┐
       │                     │
       ├──> youtube_iid      │
       ├──> bridge_iid       │  Per-bucket caches
       ├──> bridge_holdout   │
       └──> language_table   │
                             │
       ┌─────────────────────┘
       │
       ▼
At validation end:
  For each strategy:
    ├─> Check should_run() (frequency)
    ├─> Get assigned buckets
    ├─> Check can_run() per bucket (data availability)
    └─> Run strategy
        ├─> Compare mode: run separately per bucket
        └─> Merged mode: combine bucket data, run once
```

## Available Validation Strategies

| Strategy | Needs Codes | Required Metadata | Description |
|----------|-------------|-------------------|-------------|
| `basic_visualization` | No | - | Reconstruction grids for train/val |
| `latent_transfer` | No | - | Test action transfer between scenes |
| `clustering` | Yes | - | K-means clustering of latent codes |
| `codebook_histogram` | Yes | - | Codebook usage distribution |
| `sequence_histogram` | Yes | - | Top latent sequence frequencies |
| `all_sequences_histogram` | Yes | - | Full sequence distribution (long tail) |
| `action_token_scatter` | Yes | `action` | 2D action space colored by token |
| `action_sequence_scatter` | Yes | `action` | 2D action space colored by sequence |
| `top_sequences_scatter` | Yes | `action` | Highlight top N sequences in action space |
| `state_sequence_scatter` | Yes | `initial_state` | State space colored by sequence |

## Dataset Compatibility

### language_table (OXE)
- **Action**: 2D (x, y)
- **State**: 2D effector_translation
- **Instruction**: Encoded tensor
- **Compatible strategies**: All

### bridge (OXE)
- **Action**: 3D world_vector (uses first 2 dims for 2D plots)
- **State**: 7D robot state (uses first 2 dims for plots)
- **Instruction**: String tensor
- **Compatible strategies**: All (action strategies use first 2 action dims)

### YouTube (Local)
- **Action**: Not available
- **State**: Not available
- **Compatible strategies**: Basic visualization, latent transfer, clustering, histograms

### Bridge (Local)
- **Action**: Not available (currently not extracted)
- **State**: Not available (currently not extracted)
- **Compatible strategies**: Basic visualization, latent transfer, clustering, histograms

## Configuration Example

```yaml
validation:
  check_interval: 0.01  # Validate every 1% of epoch
  num_fixed_samples: 8
  num_random_samples: 8
  max_cached_samples: 1024

  # Define buckets
  buckets:
    youtube_iid:
      filters: {dataset_type: "youtube"}
      max_samples: 100
    bridge_iid:
      filters: {dataset_type: "bridge", environment: ["!=", "toykitchen7"]}
      max_samples: 100
    bridge_holdout:
      filters: {dataset_type: "bridge", environment: "toykitchen7"}
      max_samples: 100
      is_holdout: true
    language_table:
      filters: {dataset_type: "oxe", dataset_name: "language_table"}
      max_samples: 200

  # Bind strategies to buckets
  strategy_bucket_bindings:
    basic_visualization:
      buckets: all
    latent_transfer:
      buckets: [bridge_iid, bridge_holdout]
      compare_buckets: true
    action_token_scatter:
      buckets: [language_table]
    clustering:
      buckets: all

  # Configure strategies
  strategies:
    basic_visualization:
      enabled: true
      visualize_train: true
      visualize_val: true
    latent_transfer:
      enabled: true
      every_n_validations: 10
      num_pairs: 256
    action_token_scatter:
      enabled: true
      every_n_validations: 3
      num_samples: 1000
    clustering:
      enabled: true
      every_n_validations: 20
      num_clusters: 16
```

## Implementation Files

- `packages/laq/validation.py`: Strategy implementations, `ValidationCache`, `BucketConfig`
- `packages/laq/callbacks.py`: `ValidationStrategyCallback` with bucket-aware routing
- `config/training/validation.yaml`: Default validation configuration

## Key Benefits

1. **Automatic compatibility**: Strategies skip execution if data requirements not met
2. **Distribution shift analysis**: Compare mode enables IID vs OOD metric comparison
3. **Memory efficient**: Per-bucket sample limits prevent OOM
4. **Extensible**: Easy to add new strategies or datasets
5. **Heterogeneous data**: Works seamlessly across datasets with different metadata
