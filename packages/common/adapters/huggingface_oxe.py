"""
HuggingFace-based OXE adapter for streaming Open X-Embodiment datasets.

This adapter uses the HuggingFace datasets library to stream OXE data from
jxu124/OpenX-Embodiment, providing a pure PyTorch alternative to the TensorFlow-based
OXE adapter.

Key differences from TF-based adapter:
- No TensorFlow dependency
- Uses HuggingFace datasets streaming
- Directly yields PyTorch tensors

Supported datasets:
- bridge: 480x640 RGB, dict actions (world_vector), 7D state
- language_table: 640x360 RGB, 2D actions, 2D state
- rt1: 256x320 RGB, dict actions (world_vector), 3D state

Requirements:
- datasets>=3.0,<4.0  (version 4+ removed support for loading scripts)
- trust_remote_code support in datasets

Known limitations:
- num_workers must be 0 due to WebDataset sharding issues in the HF dataset
- Performance is I/O bound on network streaming (~50 samples/sec single-threaded)
"""

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


@dataclass
class HFDatasetConfig:
    """Configuration for a HuggingFace OXE dataset."""

    name: str  # Our internal name (e.g., "bridge")
    hf_name: str  # HuggingFace subset name (e.g., "bridge")
    image_key: str = "image"  # Key in observation for image
    image_shape: Tuple[int, int] = (480, 640)  # H, W expected shape
    instruction_key: str = "natural_language_instruction"  # Key for language
    state_key: str = "state"  # Key for robot state
    action_key: Optional[str] = "world_vector"  # Key in action dict (None for flat)
    action_is_dict: bool = True  # Whether action is a dict
    action_dim: int = 3  # Dims to extract from action
    state_dim: int = 2  # Dims to extract from state
    instruction_is_encoded: bool = False  # True if instruction is tokenized


# Registry of supported datasets
HF_DATASETS = {
    "bridge": HFDatasetConfig(
        name="bridge",
        hf_name="bridge",
        image_key="image",
        image_shape=(480, 640),
        instruction_key="natural_language_instruction",
        state_key="state",
        action_key="world_vector",
        action_is_dict=True,
        action_dim=3,
        state_dim=7,
    ),
    "language_table": HFDatasetConfig(
        name="language_table",
        hf_name="language_table",
        image_key="rgb",
        image_shape=(360, 640),
        instruction_key="instruction",
        state_key="effector_translation",
        action_key=None,  # Flat action
        action_is_dict=False,
        action_dim=2,
        state_dim=2,
        instruction_is_encoded=True,
    ),
    # RT-1 (fractal20220817_data) - similar to bridge
    "rt1": HFDatasetConfig(
        name="rt1",
        hf_name="fractal20220817_data",
        image_key="image",
        image_shape=(256, 320),
        instruction_key="natural_language_instruction",
        state_key="base_pose_tool_reached",
        action_key="world_vector",
        action_is_dict=True,
        action_dim=3,
        state_dim=3,
    ),
}


def decode_image(image_data: Dict) -> np.ndarray:
    """Decode image from HuggingFace format (dict with 'bytes' key)."""
    if isinstance(image_data, dict) and "bytes" in image_data:
        img = Image.open(io.BytesIO(image_data["bytes"]))
        return np.array(img)
    elif isinstance(image_data, np.ndarray):
        return image_data
    elif hasattr(image_data, "numpy"):
        return image_data.numpy()
    else:
        raise ValueError(f"Unknown image format: {type(image_data)}")


def extract_instruction(obs: Dict, config: HFDatasetConfig) -> str:
    """Extract language instruction from observation."""
    inst = obs.get(config.instruction_key, None)
    if inst is None:
        return ""
    if isinstance(inst, bytes):
        return inst.decode("utf-8")
    if isinstance(inst, str):
        return inst
    if isinstance(inst, (list, np.ndarray)):
        # Encoded instruction - return placeholder
        return "[encoded]"
    return str(inst)


def extract_action(step: Dict, config: HFDatasetConfig) -> np.ndarray:
    """Extract action from step."""
    action = step.get("action", None)
    if action is None:
        return np.zeros(config.action_dim, dtype=np.float32)

    if config.action_is_dict and config.action_key:
        action = action.get(config.action_key, action)

    if isinstance(action, (list, tuple)):
        action = np.array(action, dtype=np.float32)
    elif hasattr(action, "numpy"):
        action = action.numpy().astype(np.float32)

    # Take first action_dim values
    return action[: config.action_dim].astype(np.float32)


def extract_state(obs: Dict, config: HFDatasetConfig) -> np.ndarray:
    """Extract state from observation."""
    state = obs.get(config.state_key, None)
    if state is None:
        return np.zeros(config.state_dim, dtype=np.float32)

    if isinstance(state, (list, tuple)):
        state = np.array(state, dtype=np.float32)
    elif hasattr(state, "numpy"):
        state = state.numpy().astype(np.float32)

    # Take first state_dim values
    return state[: config.state_dim].astype(np.float32)


class HFOXEFramePairDataset(IterableDataset):
    """
    PyTorch IterableDataset that streams frame pairs from HuggingFace OXE datasets.

    Yields frame pairs (frame_t, frame_{t+offset}) from episodes, with metadata.

    Args:
        dataset_name: Name of dataset (e.g., "bridge", "language_table")
        split: HuggingFace split string (e.g., "train", "train[:1000]")
        offset: Frame offset for pairs (default 5)
        image_size: Target image size (default 256)
        shuffle_buffer: Shuffle buffer size for episodes (default 100)
        return_metadata: Whether to return action/state metadata (default True)
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        offset: int = 5,
        image_size: int = 256,
        shuffle_buffer: int = 100,
        return_metadata: bool = True,
        samples_per_episode: int = 0,  # 0 = all pairs
    ):
        super().__init__()
        if dataset_name not in HF_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(HF_DATASETS.keys())}"
            )

        self.dataset_name = dataset_name
        self.config = HF_DATASETS[dataset_name]
        self.split = split
        self.offset = offset
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.return_metadata = return_metadata
        self.samples_per_episode = samples_per_episode

        # Will be created lazily in __iter__
        self._hf_dataset = None

    def _load_hf_dataset(self):
        """Load HuggingFace dataset (lazy initialization)."""
        from datasets import load_dataset

        logger.info(
            f"Loading HuggingFace dataset: jxu124/OpenX-Embodiment/{self.config.hf_name}"
        )
        ds = load_dataset(
            "jxu124/OpenX-Embodiment",
            self.config.hf_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer)
        return ds

    def _process_image(self, image_data: Dict) -> torch.Tensor:
        """Decode and resize image to tensor."""
        img_array = decode_image(image_data)
        img = Image.fromarray(img_array)

        # Resize
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to tensor [C, H, W] and normalize to [0, 1]
        tensor = TF.to_tensor(img)
        return tensor

    def _extract_frame_pairs(
        self, episode_data: Dict, episode_id: str
    ) -> Iterator[Dict[str, Any]]:
        """Extract frame pairs from a single episode."""
        steps = episode_data.get("steps", [])
        num_steps = len(steps)

        if num_steps <= self.offset:
            return

        # Get language instruction (same for all pairs in episode)
        first_obs = steps[0].get("observation", {})
        language = extract_instruction(first_obs, self.config)

        # Generate pairs
        pairs_generated = 0
        max_pairs = num_steps - self.offset
        if self.samples_per_episode > 0:
            max_pairs = min(max_pairs, self.samples_per_episode)

        for i in range(num_steps - self.offset):
            if self.samples_per_episode > 0 and pairs_generated >= self.samples_per_episode:
                break

            step_t = steps[i]
            step_t_offset = steps[i + self.offset]

            obs_t = step_t.get("observation", {})
            obs_t_offset = step_t_offset.get("observation", {})

            # Get images
            img_key = self.config.image_key
            img_t_data = obs_t.get(img_key)
            img_t_offset_data = obs_t_offset.get(img_key)

            if img_t_data is None or img_t_offset_data is None:
                continue

            try:
                frame_t = self._process_image(img_t_data)
                frame_t_offset = self._process_image(img_t_offset_data)
            except Exception as e:
                logger.warning(f"Failed to process image: {e}")
                continue

            # Stack frames: [C, 2, H, W] to match existing format
            # Both frames are [C, H, W], stack and permute
            frames = torch.stack([frame_t, frame_t_offset], dim=0)  # [2, C, H, W]
            frames = frames.permute(1, 0, 2, 3)  # [C, 2, H, W]

            result = {
                "frames": frames,
                "episode_id": episode_id,
                "frame_idx": i,
                "dataset_name": self.dataset_name,
                "language": language,
            }

            if self.return_metadata:
                # Cumulative action over offset
                actions = []
                for j in range(i, i + self.offset):
                    actions.append(extract_action(steps[j], self.config))
                cumulative_action = np.sum(actions, axis=0)
                result["action"] = cumulative_action

                # Initial state
                result["initial_state"] = extract_state(obs_t, self.config)

            yield result
            pairs_generated += 1

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over frame pairs from all episodes."""
        # Handle worker sharding
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Load dataset
        hf_ds = self._load_hf_dataset()

        # Iterate over episodes
        for episode_idx, sample in enumerate(hf_ds):
            # Shard across workers
            if episode_idx % num_workers != worker_id:
                continue

            data = sample.get("data.pickle", sample)
            episode_id = sample.get("__key__", f"ep_{episode_idx}")

            yield from self._extract_frame_pairs(data, episode_id)


def hf_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for HuggingFace OXE datasets.

    Stacks 'frames' into a tensor, keeps other fields as lists.
    Compatible with existing oxe_collate_fn format.
    """
    if not batch:
        return {}

    # Stack frames
    frames = torch.stack([item["frames"] for item in batch], dim=0)

    result = {
        "frames": frames,
        "episode_id": [item["episode_id"] for item in batch],
        "frame_idx": [item["frame_idx"] for item in batch],
        "dataset_name": [item["dataset_name"] for item in batch],
        "language": [item["language"] for item in batch],
    }

    # Optional metadata
    if "action" in batch[0]:
        result["action"] = np.stack([item["action"] for item in batch], axis=0)
    if "initial_state" in batch[0]:
        result["initial_state"] = np.stack(
            [item["initial_state"] for item in batch], axis=0
        )

    return result


class HFMultiOXEFramePairDataset(IterableDataset):
    """
    Multi-dataset version that interleaves multiple OXE datasets with weights.

    Args:
        datasets: List of dataset configs, each with:
            - name: Dataset name (e.g., "bridge")
            - split: Split string (e.g., "train")
            - offset: Frame offset
            - weight: Sampling weight (optional, defaults to 1.0)
        image_size: Target image size
        shuffle_buffer: Shuffle buffer per dataset
        return_metadata: Whether to return action/state metadata
        samples_per_episode: Max pairs per episode (0 = all)
    """

    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        image_size: int = 256,
        shuffle_buffer: int = 100,
        return_metadata: bool = True,
        samples_per_episode: int = 0,
    ):
        super().__init__()
        self.dataset_configs = datasets
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.return_metadata = return_metadata
        self.samples_per_episode = samples_per_episode

        # Compute normalized weights
        total_weight = sum(d.get("weight", 1.0) for d in datasets)
        self.weights = [d.get("weight", 1.0) / total_weight for d in datasets]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate by randomly sampling from datasets based on weights."""
        import random

        # Create dataset iterators
        iterators = []
        for cfg in self.dataset_configs:
            ds = HFOXEFramePairDataset(
                dataset_name=cfg["name"],
                split=cfg.get("split", "train"),
                offset=cfg.get("offset", 5),
                image_size=self.image_size,
                shuffle_buffer=self.shuffle_buffer,
                return_metadata=self.return_metadata,
                samples_per_episode=self.samples_per_episode,
            )
            iterators.append(iter(ds))

        # Sample from datasets based on weights
        active = list(range(len(iterators)))
        while active:
            # Sample dataset index based on weights
            active_weights = [self.weights[i] for i in active]
            total = sum(active_weights)
            normalized = [w / total for w in active_weights]

            idx = random.choices(active, weights=normalized, k=1)[0]
            actual_idx = active.index(idx)

            try:
                yield next(iterators[idx])
            except StopIteration:
                # This dataset is exhausted, remove it
                active.remove(idx)


# DataModule for Lightning integration
try:
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader

    class HFOXEDataModule(pl.LightningDataModule):
        """
        Lightning DataModule for HuggingFace OXE datasets.

        Args:
            datasets: List of dataset configs for training
            val_datasets: List of dataset configs for validation (optional)
            image_size: Target image size
            batch_size: Batch size
            num_workers: Number of dataloader workers
            shuffle_buffer: Shuffle buffer size
            return_metadata: Whether to return action/state metadata
            samples_per_episode: Max pairs per episode
        """

        def __init__(
            self,
            datasets: List[Dict[str, Any]],
            val_datasets: Optional[List[Dict[str, Any]]] = None,
            image_size: int = 256,
            batch_size: int = 32,
            num_workers: int = 4,
            shuffle_buffer: int = 100,
            return_metadata: bool = True,
            samples_per_episode: int = 0,
            **kwargs,  # Accept extra args for compatibility
        ):
            super().__init__()
            self.datasets = datasets
            self.val_datasets = val_datasets or datasets
            self.image_size = image_size
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.shuffle_buffer = shuffle_buffer
            self.return_metadata = return_metadata
            self.samples_per_episode = samples_per_episode

            self.train_dataset = None
            self.val_dataset = None

        def setup(self, stage: Optional[str] = None):
            """Create train and val datasets."""
            if len(self.datasets) == 1:
                # Single dataset
                cfg = self.datasets[0]
                self.train_dataset = HFOXEFramePairDataset(
                    dataset_name=cfg["name"],
                    split=cfg.get("split", "train"),
                    offset=cfg.get("offset", 5),
                    image_size=self.image_size,
                    shuffle_buffer=self.shuffle_buffer,
                    return_metadata=self.return_metadata,
                    samples_per_episode=self.samples_per_episode,
                )
            else:
                # Multi-dataset
                self.train_dataset = HFMultiOXEFramePairDataset(
                    datasets=self.datasets,
                    image_size=self.image_size,
                    shuffle_buffer=self.shuffle_buffer,
                    return_metadata=self.return_metadata,
                    samples_per_episode=self.samples_per_episode,
                )

            # Validation (no shuffle, smaller buffer)
            if len(self.val_datasets) == 1:
                cfg = self.val_datasets[0]
                self.val_dataset = HFOXEFramePairDataset(
                    dataset_name=cfg["name"],
                    split=cfg.get("val_split", cfg.get("split", "train")),
                    offset=cfg.get("offset", 5),
                    image_size=self.image_size,
                    shuffle_buffer=0,  # No shuffle for val
                    return_metadata=self.return_metadata,
                    samples_per_episode=self.samples_per_episode,
                )
            else:
                # Use train configs but with val_split
                val_configs = []
                for cfg in self.val_datasets:
                    val_cfg = cfg.copy()
                    val_cfg["split"] = cfg.get("val_split", cfg.get("split", "train"))
                    val_configs.append(val_cfg)
                self.val_dataset = HFMultiOXEFramePairDataset(
                    datasets=val_configs,
                    image_size=self.image_size,
                    shuffle_buffer=0,
                    return_metadata=self.return_metadata,
                    samples_per_episode=self.samples_per_episode,
                )

        def train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=hf_collate_fn,
                pin_memory=True,
            )

        def val_dataloader(self):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=max(1, self.num_workers // 2),
                collate_fn=hf_collate_fn,
                pin_memory=True,
            )

except ImportError:
    # Lightning not available
    HFOXEDataModule = None
    logger.warning("Lightning not available, HFOXEDataModule disabled")


# Quick test
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import sys

    print("=" * 60)
    print("Testing HFOXEFramePairDataset (single dataset)...")
    print("=" * 60)

    # Test bridge dataset
    ds = HFOXEFramePairDataset(
        dataset_name="bridge",
        split="train",
        offset=5,
        image_size=256,
        shuffle_buffer=10,
        samples_per_episode=3,
    )

    loader = DataLoader(ds, batch_size=4, collate_fn=hf_collate_fn)

    print("Getting first batch...")
    batch = next(iter(loader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Frames shape: {batch['frames'].shape}")
    print(f"Episode IDs: {batch['episode_id']}")
    print(f"Languages: {batch['language']}")
    if "action" in batch:
        print(f"Action shape: {batch['action'].shape}")
    print("Single dataset test: SUCCESS!")

    # Test DataModule if available
    if HFOXEDataModule is not None:
        print("\n" + "=" * 60)
        print("Testing HFOXEDataModule...")
        print("=" * 60)

        dm = HFOXEDataModule(
            datasets=[
                {"name": "bridge", "split": "train", "offset": 5, "weight": 1.0},
            ],
            image_size=256,
            batch_size=4,
            num_workers=0,
            shuffle_buffer=10,
            samples_per_episode=3,
        )
        dm.setup()

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        print(f"DataModule batch keys: {list(batch.keys())}")
        print(f"DataModule frames shape: {batch['frames'].shape}")
        print("DataModule test: SUCCESS!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
