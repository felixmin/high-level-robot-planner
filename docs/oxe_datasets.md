# Open X-Embodiment (OXE) Datasets

## Overview

The Open X-Embodiment dataset provides access to large-scale robot learning data streamed from Google Cloud Storage. This guide covers how to use OXE datasets with the LAQ training pipeline.

## Available Datasets

### language_table
**GCS Path**: `gs://gresearch/robotics/language_table/0.0.1`

- **Episodes**: 442k
- **Task**: 2D tabletop block manipulation
- **Robot**: Simulated 2D manipulator
- **Action**: 2D (x, y) translation
- **State**: 2D effector_translation
- **Language**: Encoded tensor instructions
- **Control Frequency**: 10 Hz
- **Image Size**: 360x640 → resized to 256x256

**Use case**: Large-scale language-conditioned manipulation, diverse language instructions

### language_table_blocktorelative_oracle_sim
**GCS Path**: `gs://gresearch/robotics/language_table_blocktorelative_oracle_sim/0.0.1`

- **Episodes**: 200k
- **Task**: Same as language_table, oracle scripted agent
- **Episode Length**: 27-46 steps (longer than base language_table)
- **Action/State**: Same as language_table
- **Quality**: Higher consistency, cleaner trajectories

**Use case**: Cleaner training data with longer horizon tasks

### bridge
**GCS Path**: `gs://gresearch/robotics/bridge/0.1.0`

- **Episodes**: 25,460 train + 3,475 test
- **Task**: Kitchen object manipulation (pots, cans, towels, etc.)
- **Robot**: WidowX 250 6DOF arm
- **Action**: Dict with `world_vector` (3D xyz), `rotation_delta` (3D), `open_gripper` (bool)
  - For 2D scatter plots, uses first 2 dims of `world_vector`
- **State**: 7D robot configuration
  - For visualization, uses first 2 dims
- **Language**: String tensor natural language instructions
- **Control Frequency**: ~5 Hz
- **Image Size**: 480x640 → resized to 256x256

**Use case**: Real-world robot data, kitchen manipulation, OOD generalization testing

## Configuration

### Basic OXE Config

Create a config file in `config/data/`:

```yaml
# config/data/laq_oxe_bridge.yaml
datasets:
  - name: bridge
    train_split: "train[:90%]"
    val_split: "train[90%:]"
    offset: 5
    weight: 1.0
    size: 1031130

samples_per_episode: 0  # 0 = use all (t, t+offset) pairs in each episode; set to 1 for LAPA-style
sampling_seed: 42
image_size: 256
batch_size: 32
num_workers: 0  # tf.data handles parallelism
episode_queue_shuffle_buffer: 500
intra_episode_sample_shuffle_buffer: 0
global_stream_shuffle_buffer: 500
val_episode_queue_shuffle_buffer: 0
val_intra_episode_sample_shuffle_buffer: 0
val_global_stream_shuffle_buffer: 0
final_stream_prefetch_buffer: 4
episode_queue_prefetch_buffer: 0
num_parallel_episodes: 1
num_parallel_calls: 1
return_metadata: true  # Required for validation strategies
persistent_iterator: true
```

### TFDS Split Syntax

OXE uses TensorFlow Datasets split syntax:

```yaml
# First 1000 episodes
train_split: "train[:1000]"

# Episodes 1000-2000
train_split: "train[1000:2000]"

# First 90%
train_split: "train[:90%]"

# Last 10%
val_split: "train[90%:]"

# All test data
val_split: "test"
```

### Experiment Config

```yaml
# config/experiment/laq_oxe_bridge.yaml
defaults:
  - /model@model: laq
  - /data@data: laq_oxe_bridge
  - /training@training: laq_optimizer
  - /cluster@cluster: local_dev

experiment:
  name: laq_oxe_bridge
  description: LAQ training on Bridge OXE dataset
```

## Usage

### Training

```bash
# Language table (oracle sim)
python scripts/2_train_laq.py experiment=laq_oxe

# Bridge
python scripts/2_train_laq.py data=laq_oxe_bridge training.epochs=100

# Debug with small subset
python scripts/2_train_laq.py \
    data=laq_oxe_bridge \
    data.train_split="train[:100]" \
    data.val_split="train[100:120]"
```

### Data Inspection

```python
from common.adapters.oxe import OXEFramePairDataset, get_oxe_dataset_info

# Get dataset info
info = get_oxe_dataset_info("bridge")
print(info)

# Load samples
ds = OXEFramePairDataset(
    dataset_name="bridge",
    gcs_path=None,
    split="train[:10]",
    offset=5,
	    final_stream_prefetch_buffer=0,
    episode_queue_shuffle_buffer=0,
    intra_episode_sample_shuffle_buffer=0,
    image_size=256,
    num_parallel_calls=1,
    return_metadata=True,
    persistent_iterator=False,
    samples_per_episode=0,
    seed=None,
    precomputed_size=None,
	    episode_queue_prefetch_buffer=0,
    num_parallel_episodes=1,
)

for sample in ds:
    print(f"Frames: {sample['frames'].shape}")
    print(f"Action: {sample['action']}")
    print(f"Language: {sample['language']}")
    break
```

## Metadata Format

OXE datasets provide the following metadata when `return_metadata=True`:

```python
{
    "frames": torch.Tensor,  # [C, 2, H, W] frame pair
    "episode_id": str,  # Unique episode identifier
    "frame_idx": int,  # Starting frame index
    "offset": int,  # Frame offset between pair
    "language": str,  # Language instruction
    "dataset_type": str,  # Matches dataset_name (e.g., "bridge")
    "dataset_name": str,  # e.g., "language_table", "bridge"
    "action": List[float],  # Cumulative action between frames
    "initial_state": List[float],  # Robot state at frame_idx
}
```

**Action format**:
- **language_table**: 2D list `[x, y]`
- **bridge**: 3D list `[x, y, z]` (from `world_vector`)

## Validation Compatibility

### language_table
✅ All validation strategies work
- Action scatter strategies use 2D action directly
- State scatter strategies use 2D effector_translation

### bridge
✅ All validation strategies work
- Action scatter strategies use first 2 dims of 3D `world_vector`
- State scatter strategies use first 2 dims of 7D state

## Key Differences from Local Datasets

| Aspect | OXE | Local (Bridge/YouTube) |
|--------|-----|------------------------|
| **Storage** | GCS streaming | Local filesystem |
| **Loading** | `OXEDataModule` + `tf.data` | `LAQDataModule` + PyTorch |
| **Split syntax** | TFDS (`train[:90%]`) | Ratio/metadata filters |
| **Action data** | ✅ Built-in | ❌ Not extracted |
| **State data** | ✅ Built-in | ❌ Not extracted |
| **Language** | ✅ Built-in | ⚠️ Bridge only (text files) |
| **Caching** | No local cache needed | Reads from disk |

## Performance Tips

1. **Episode queue shuffle buffer**: Set based on dataset size to avoid OOM
   ```yaml
   # language_table (360x640 images) - can use larger buffer
   episode_queue_shuffle_buffer: 2000

   # bridge (480x640 images) - use smaller buffer
   episode_queue_shuffle_buffer: 500
   ```

2. **Prefetch**: Use 2-4 for smooth pipelining
   ```yaml
   final_stream_prefetch_buffer: 4
   ```

3. **Batch size**: OXE streaming can handle larger batches than local disk I/O
   ```yaml
   batch_size: 64  # vs 16-32 for local
   ```

4. **Workers**: Set `num_workers: 0` since `tf.data` handles parallelism internally

5. **Debug mode**: Use small splits for quick iteration
   ```yaml
   train_split: "train[:100]"
   val_split: "train[100:120]"
   ```

## Adding New OXE Datasets

To add a new OXE dataset:

1. **Find GCS path**: Check Open X-Embodiment documentation
2. **Add to registry** in `packages/common/adapters/oxe.py`:
   ```python
   OXE_DATASETS = {
       "new_dataset": OXEDatasetConfig(
           name="new_dataset",
           gcs_path="gs://path/to/dataset/version",
           image_key="rgb",  # or "image"
           instruction_key="instruction",
           state_key="state",
           image_shape=(height, width, 3),
           control_frequency_hz=10.0,
           action_dim=2,
           state_dim=2,
       ),
   }
   ```
3. **Create config** in `config/data/laq_oxe_new.yaml`
4. **Test loading**:
   ```bash
   python -c "from common.adapters.oxe import get_oxe_dataset_info; print(get_oxe_dataset_info('new_dataset'))"
   ```

## Troubleshooting

### Authentication Errors
```
All attempts to get a Google authentication bearer token failed
```
**Solution**: This is a warning, not an error. OXE datasets are public and don't require authentication.

### Dataset Not Found
```
Could not read dataset info from gs://...
```
**Solution**: Verify the GCS path is correct. Try alternate paths (e.g., `/0.0.1` vs `/1.0.0`).

### Slow Initial Loading
**Cause**: TFDS downloads dataset metadata on first access
**Solution**: Normal behavior. Subsequent runs will be faster.

### Out of Memory
**Cause**: Shuffle buffer too large
**Solution**: Reduce `episode_queue_shuffle_buffer` to 500-1000

## References

- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [TFDS Documentation](https://www.tensorflow.org/datasets)
- [RLDS Format](https://github.com/google-research/rlds)
