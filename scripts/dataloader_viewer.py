#!/usr/bin/env python3
"""
Streamlit app for visualizing dataloader batches.

This tool helps visualize what the model sees during training:
- View train/val batches with configurable batch size
- Switch between different data configs
- Navigate through samples in a batch
- Display frame pairs and metadata

Run with:
    streamlit run scripts/dataloader_viewer.py
"""

import sys
from pathlib import Path

# Add packages to path (resolve to absolute path for Hydra)
workspace_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(workspace_root / "packages"))

import streamlit as st
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, List
import logging

# Suppress TensorFlow warnings and disable GPU for faster loading
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable TF GPU - not needed for data loading

# Configure logging - show INFO for progress visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_available_configs() -> List[str]:
    """Scan config/data/ for available configs."""
    config_dir = workspace_root / "config" / "data"
    if not config_dir.exists():
        return []

    configs = []
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        # Skip non-LAQ configs
        name = yaml_file.stem
        if name.startswith("laq_") or name == "latent_labeled":
            configs.append(name)

    return configs


def load_config(config_name: str) -> Dict[str, Any]:
    """Load a data config using Hydra compose API."""
    logger.info(f"Loading config: {config_name}")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    config_dir = str(workspace_root / "config")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=[f"data={config_name}"])
        result = OmegaConf.to_container(cfg.data, resolve=True)
        logger.info(f"Config loaded: {list(result.keys())}")
        return result


def is_oxe_config(config: Dict[str, Any]) -> bool:
    """Check if config is for OXE streaming dataset."""
    return "dataset_name" in config or "datasets" in config


def create_datamodule(config: Dict[str, Any], batch_size: int, include_bridge: bool = False):
    """Create appropriate datamodule based on config."""
    from common.data import LAQDataModule, OXEDataModule

    # Override batch size
    config = config.copy()
    config["batch_size"] = batch_size

    # Remove non-datamodule keys
    config.pop("name", None)
    config.pop("task", None)

    # Ensure return_metadata is True for visualization
    config["return_metadata"] = True

    # Disable multiprocessing workers - they crash in Streamlit
    config["num_workers"] = 0

    if is_oxe_config(config):
        # Use small shuffle buffer for fast loading in viewer
        config["shuffle_buffer"] = min(config.get("shuffle_buffer", 50), 50)
        config["val_shuffle_buffer"] = 0  # No shuffling for val
        config["prefetch_buffer"] = 2
        logger.info(f"Creating OXEDataModule (shuffle_buffer={config['shuffle_buffer']})")
        datamodule = OXEDataModule(**config)
    else:
        # Limit pairs for faster loading
        config["max_pairs"] = min(config.get("max_pairs") or 500, 500)  # Small limit for viewer
        config["prefetch_factor"] = None  # Not needed with num_workers=0
        # Skip Bridge by default - scanning 50k folders is too slow for viewer
        if "sources" in config and not include_bridge:
            original_count = len(config["sources"])
            config["sources"] = [s for s in config["sources"] if s.get("type") != "bridge"]
            if len(config["sources"]) < original_count:
                logger.info("Skipping Bridge dataset (50k+ trajectories too slow for viewer)")
            if not config["sources"]:
                logger.warning("No sources left after removing Bridge!")
        logger.info(f"Creating LAQDataModule (max_pairs={config['max_pairs']})")
        datamodule = LAQDataModule(**config)

    logger.info("Calling datamodule.setup()...")
    datamodule.setup()
    logger.info("DataModule ready")
    return datamodule


def get_batch(datamodule, is_train: bool) -> Optional[Dict[str, Any]]:
    """Get a single batch from train or val loader."""
    loader = datamodule.train_dataloader() if is_train else datamodule.val_dataloader()

    try:
        batch = next(iter(loader))
        return batch
    except StopIteration:
        return None
    except Exception as e:
        st.error(f"Error loading batch: {e}")
        return None


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a [C, H, W] tensor to PIL Image."""
    # Handle different tensor formats
    if tensor.dim() == 4:
        # [B, C, H, W] - take first
        tensor = tensor[0]

    # Ensure [C, H, W]
    if tensor.shape[0] not in [1, 3]:
        # Might be [H, W, C]
        tensor = tensor.permute(2, 0, 1)

    # Convert to numpy
    img_np = tensor.cpu().numpy()

    # Normalize to 0-255
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)

    # Convert to PIL
    if img_np.shape[0] == 1:
        # Grayscale
        return Image.fromarray(img_np[0], mode="L")
    else:
        # RGB
        return Image.fromarray(np.transpose(img_np, (1, 2, 0)), mode="RGB")


def display_frame_pair(frames: torch.Tensor, idx: int):
    """Display frame pair side-by-side."""
    # frames shape: [B, C, 2, H, W] or [B, 2, C, H, W]
    sample_frames = frames[idx]

    # Determine format and extract frames
    if sample_frames.dim() == 3:
        # [C, 2, H, W] format - C=3, second dim is time
        if sample_frames.shape[1] == 2:
            frame_t = sample_frames[:, 0, :, :]  # [C, H, W]
            frame_t_offset = sample_frames[:, 1, :, :]  # [C, H, W]
        else:
            # [2, C, H, W] format
            frame_t = sample_frames[0]  # [C, H, W]
            frame_t_offset = sample_frames[1]  # [C, H, W]
    elif sample_frames.dim() == 4:
        # [C, 2, H, W]
        frame_t = sample_frames[:, 0, :, :]
        frame_t_offset = sample_frames[:, 1, :, :]
    else:
        st.error(f"Unexpected frame tensor shape: {sample_frames.shape}")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.image(tensor_to_image(frame_t), caption="Frame t", use_container_width=True)

    with col2:
        st.image(tensor_to_image(frame_t_offset), caption="Frame t+offset", use_container_width=True)


def display_metadata(batch: Dict[str, Any], idx: int):
    """Display metadata for selected sample."""
    st.subheader("Metadata")

    # Create columns for metadata display
    col1, col2 = st.columns(2)

    with col1:
        # Dataset info
        if "dataset_name" in batch:
            dataset_name = batch["dataset_name"]
            if isinstance(dataset_name, list):
                dataset_name = dataset_name[idx]
            st.markdown(f"**Dataset:** `{dataset_name}`")

        if "episode_id" in batch:
            episode_id = batch["episode_id"]
            if isinstance(episode_id, list):
                episode_id = episode_id[idx]
            st.markdown(f"**Episode ID:** `{episode_id}`")

        if "frame_idx" in batch:
            frame_idx = batch["frame_idx"]
            if isinstance(frame_idx, list):
                frame_idx = frame_idx[idx]
            elif isinstance(frame_idx, torch.Tensor):
                frame_idx = frame_idx[idx].item()
            st.markdown(f"**Frame Index:** `{frame_idx}`")

    with col2:
        if "offset" in batch:
            offset = batch["offset"]
            if isinstance(offset, list):
                offset = offset[idx]
            elif isinstance(offset, torch.Tensor):
                offset = offset[idx].item()
            st.markdown(f"**Offset:** `{offset}`")

        if "environment" in batch:
            env = batch["environment"]
            if isinstance(env, list):
                env = env[idx]
            st.markdown(f"**Environment:** `{env}`")

    # Language instruction (full width)
    if "language" in batch:
        language = batch["language"]
        if isinstance(language, list):
            language = language[idx]
        if language:
            st.markdown(f"**Language:** _{language}_")

    # Action (if available)
    if "action" in batch:
        action = batch["action"]
        if isinstance(action, list):
            action = action[idx]
        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            elif isinstance(action, np.ndarray):
                pass
            else:
                action = np.array(action)

            # Format nicely
            if action.ndim == 0:
                action_str = f"{action:.4f}"
            else:
                action_str = np.array2string(action, precision=3, suppress_small=True)
            st.markdown(f"**Action:** `{action_str}`")

    # Initial state (if available)
    if "initial_state" in batch:
        state = batch["initial_state"]
        if isinstance(state, list):
            state = state[idx]
        if state is not None:
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            elif isinstance(state, np.ndarray):
                pass
            else:
                state = np.array(state)

            state_str = np.array2string(state, precision=3, suppress_small=True)
            st.markdown(f"**Initial State:** `{state_str}`")


def display_batch_thumbnails(batch: Dict[str, Any], current_idx: int) -> Optional[int]:
    """Display thumbnails of all samples in batch. Returns clicked index if any."""
    frames = batch["frames"]
    batch_size = frames.shape[0]

    st.subheader("Batch Overview")

    # Create columns for thumbnails (max 8 per row)
    cols_per_row = min(8, batch_size)

    clicked_idx = None

    for row_start in range(0, batch_size, cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= batch_size:
                break

            sample_frames = frames[idx]
            # Get first frame for thumbnail
            if sample_frames.dim() == 3 and sample_frames.shape[1] == 2:
                thumb_frame = sample_frames[:, 0, :, :]
            elif sample_frames.dim() == 4:
                thumb_frame = sample_frames[:, 0, :, :]
            else:
                thumb_frame = sample_frames[0]

            with col:
                # Highlight current sample
                border = "2px solid #FF4B4B" if idx == current_idx else "1px solid #ddd"
                st.markdown(
                    f'<div style="border: {border}; padding: 2px; border-radius: 4px;">',
                    unsafe_allow_html=True
                )
                st.image(tensor_to_image(thumb_frame), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption(f"Sample {idx}")

    return clicked_idx


def main():
    st.set_page_config(
        page_title="Dataloader Viewer",
        page_icon="🤖",
        layout="wide"
    )

    st.title("Dataloader Viewer")
    st.markdown("Visualize what the model sees during training")

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")

        # Config selector
        configs = get_available_configs()
        if not configs:
            st.error("No data configs found in config/data/")
            return

        # Default to laq_multi_dataset or first OXE config
        if "laq_multi_dataset" in configs:
            default_idx = configs.index("laq_multi_dataset")
        elif "laq_oxe" in configs:
            default_idx = configs.index("laq_oxe")
        else:
            default_idx = 0
        config_name = st.selectbox(
            "Data Config",
            configs,
            index=default_idx,
            help="Select a data configuration from config/data/"
        )

        # Split selector
        split = st.radio(
            "Split",
            ["Train", "Val"],
            horizontal=True
        )
        is_train = split == "Train"

        # Bridge option (only for local datasets)
        include_bridge = st.checkbox(
            "Include Bridge (slow)",
            value=False,
            help="Bridge dataset has 50k+ trajectories - scanning takes minutes"
        )

        # Batch size
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=32,
            value=8,
            help="Number of samples per batch"
        )

        st.divider()

        # Load batch button
        load_button = st.button("🔄 Load Batch", use_container_width=True, type="primary")

        # Next batch button (only if batch already loaded)
        next_button = False
        if "batch" in st.session_state:
            next_button = st.button("➡️ Next Batch", use_container_width=True)

        st.divider()

        # Sample navigation
        if "batch" in st.session_state and st.session_state.batch is not None:
            current_batch_size = st.session_state.batch["frames"].shape[0]

            st.subheader("Sample Navigation")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀", use_container_width=True):
                    if st.session_state.sample_idx > 0:
                        st.session_state.sample_idx -= 1
            with col2:
                st.markdown(
                    f"<div style='text-align: center; padding: 8px;'>"
                    f"<b>{st.session_state.sample_idx + 1}</b> / {current_batch_size}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col3:
                if st.button("▶", use_container_width=True):
                    if st.session_state.sample_idx < current_batch_size - 1:
                        st.session_state.sample_idx += 1

            # Direct index input
            new_idx = st.number_input(
                "Go to sample",
                min_value=0,
                max_value=current_batch_size - 1,
                value=st.session_state.sample_idx,
                step=1
            )
            if new_idx != st.session_state.sample_idx:
                st.session_state.sample_idx = new_idx

        st.divider()

        # Config info
        if "config" in st.session_state:
            with st.expander("Config Details"):
                cfg = st.session_state.config
                st.json({
                    "type": "OXE" if is_oxe_config(cfg) else "LAQ",
                    "batch_size": cfg.get("batch_size", "N/A"),
                    "image_size": cfg.get("image_size", "N/A"),
                })

    # Initialize session state
    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = 0
    if "batch" not in st.session_state:
        st.session_state.batch = None
    if "datamodule" not in st.session_state:
        st.session_state.datamodule = None
    if "dataloader_iter" not in st.session_state:
        st.session_state.dataloader_iter = None
    if "current_config" not in st.session_state:
        st.session_state.current_config = None
    if "current_split" not in st.session_state:
        st.session_state.current_split = None
    if "current_include_bridge" not in st.session_state:
        st.session_state.current_include_bridge = None

    # Check if config or split changed
    config_changed = (
        st.session_state.current_config != config_name or
        st.session_state.current_split != is_train or
        st.session_state.current_include_bridge != include_bridge or
        st.session_state.datamodule is None
    )

    # Load batch
    if load_button or config_changed and load_button:
        with st.spinner("Loading data config..."):
            try:
                config = load_config(config_name)
                st.session_state.config = config
                st.session_state.current_config = config_name
                st.session_state.current_split = is_train
                st.session_state.current_include_bridge = include_bridge
            except Exception as e:
                st.error(f"Error loading config: {e}")
                return

        with st.spinner("Creating datamodule..."):
            try:
                datamodule = create_datamodule(config, batch_size, include_bridge)
                st.session_state.datamodule = datamodule

                # Create new iterator
                logger.info(f"Creating {'train' if is_train else 'val'} dataloader...")
                loader = datamodule.train_dataloader() if is_train else datamodule.val_dataloader()
                logger.info("Creating iterator...")
                st.session_state.dataloader_iter = iter(loader)
                logger.info("Iterator ready")
            except Exception as e:
                st.error(f"Error creating datamodule: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

        with st.spinner("Loading batch..."):
            try:
                logger.info("Fetching first batch (this may take a while for OXE)...")
                batch = next(st.session_state.dataloader_iter)
                logger.info(f"Batch loaded: {batch['frames'].shape}")
                st.session_state.batch = batch
                st.session_state.sample_idx = 0
            except StopIteration:
                st.warning("No data available in this split")
                st.session_state.batch = None
            except Exception as e:
                st.error(f"Error loading batch: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Next batch
    if next_button:
        if st.session_state.dataloader_iter is not None:
            with st.spinner("Loading next batch..."):
                try:
                    logger.info("Fetching next batch...")
                    batch = next(st.session_state.dataloader_iter)
                    logger.info(f"Batch loaded: {batch['frames'].shape}")
                    st.session_state.batch = batch
                    st.session_state.sample_idx = 0
                except StopIteration:
                    st.warning("End of dataset reached. Click 'Load Batch' to restart.")
                except Exception as e:
                    st.error(f"Error loading batch: {e}")

    # Main display area
    if st.session_state.batch is not None:
        batch = st.session_state.batch
        sample_idx = st.session_state.sample_idx

        # Frame pair display
        st.subheader(f"Sample {sample_idx + 1} of {batch['frames'].shape[0]}")
        display_frame_pair(batch["frames"], sample_idx)

        # Metadata
        display_metadata(batch, sample_idx)

        # Batch thumbnails
        st.divider()
        display_batch_thumbnails(batch, sample_idx)

        # Raw batch info (collapsed)
        with st.expander("Raw Batch Data"):
            st.write("**Batch Keys:**", list(batch.keys()))
            st.write("**Frames Shape:**", batch["frames"].shape)

            # Show sample of each key
            for key, value in batch.items():
                if key == "frames":
                    continue
                if isinstance(value, torch.Tensor):
                    st.write(f"**{key}:** Tensor shape {value.shape}")
                elif isinstance(value, list):
                    if len(value) > 0:
                        st.write(f"**{key}:** List[{type(value[0]).__name__}] len={len(value)}, first={value[0]}")
                    else:
                        st.write(f"**{key}:** Empty list")
                else:
                    st.write(f"**{key}:** {type(value).__name__}")
    else:
        # No batch loaded yet
        st.info("👈 Select a config and click 'Load Batch' to visualize data")

        # Show available configs
        st.subheader("Available Configurations")
        for cfg_name in get_available_configs():
            is_oxe = cfg_name.startswith("laq_oxe")
            cfg_type = "🌐 OXE Streaming" if is_oxe else "📁 Local Files"
            st.markdown(f"- `{cfg_name}` ({cfg_type})")


if __name__ == "__main__":
    main()
