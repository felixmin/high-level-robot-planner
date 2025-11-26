# LAPA Project: Complete Technical Specification & Implementation Plan

## Executive Summary

This document provides a **complete technical blueprint** for implementing LAPA (Latent Action Pretraining from Videos), a three-stage robot learning system. It covers architectural decisions, implementation details, and a phased task breakdown suitable for distribution to junior developers.

**System Overview:** LAPA learns robot policies from videos without action labels through:
1. **Stage 1 (LAQ)**: Transformer-based model with NSVQ that compresses frame-to-frame transitions into discrete latent codes
2. **Stage 2 (Foundation)**: 7B Vision-Language model that predicts latent actions from image + text
3. **Stage 3 (Finetuning)**: Adapts the foundation model to output continuous robot commands

**Infrastructure:** LRZ cluster (H100 GPUs, GPFS storage, Slurm scheduler)

---

## Key Architectural Updates (LAPA Paper Alignment)

This plan implements the **LAPA paper architecture** with the following key differences from standard VQ-VAE approaches:

### Stage 1: LAQ Architecture Changes

| Component | Standard VQ-VAE | LAPA (This Implementation) |
|-----------|----------------|----------------------------|
| **Encoder** | Convolutional layers + ResBlocks | Patch embedding + Spatial/Temporal Transformers |
| **Quantization** | VQ-VAE with per-position codebooks | NSVQ with **delta quantization** and single shared codebook |
| **Quantization Target** | Absolute latent features | **Frame delta** (last_frame - first_frame) |
| **Codebook Structure** | Multiple codebooks [num_tokens, vocab_size, dim] | **Single codebook** [vocab_size, dim] |
| **Straight-Through Estimator** | Standard STE with commitment loss | **Noise-substitution STE** (no extra losses) |
| **Loss Function** | MSE + codebook loss + commitment loss | **MSE only** (simpler, more stable) |
| **Decoder** | Deconvolutional layers | **Cross-attention** conditioning on action tokens |
| **Image Size** | 224×224 | **256×256** (better patch alignment) |

### Key LAPA Innovations

1. **Delta Quantization**: Quantizes the *change* between frames, not absolute features
2. **Single Shared Codebook**: All action token positions share the same 8 embeddings
3. **Noise-Substitution STE**: Gradients flow without requiring codebook/commitment losses
4. **Transformer Architecture**: Better temporal modeling than convolutional approaches
5. **Cross-Attention Decoder**: Explicitly conditions reconstruction on learned actions

### Infrastructure (Unchanged)

✅ All infrastructure decisions remain excellent and are preserved:
- Hydra configuration management
- PyTorch Lightning + Fabric hybrid approach
- WebDataset with TAR shards for LRZ GPFS
- Enroot containers + Slurm integration
- Weights & Biases tracking
- Multi-stage training pipeline

---

## Part 1: High-Level Architecture Decisions

### 1.1 Repository Architecture: Modular Monorepo

**Decision:** Single repository with installable Python packages

**Rationale:**
- Tight coupling between stages (LAQ vocabulary changes cascade to Foundation and Low-Level)
- Atomic commits prevent version skew
- Shared utilities (logging, data loading, metrics)
- Simplified dependency management

**Structure:**
```
lapa-project/
├── packages/
│   ├── common/           # Shared code (data, logging, interfaces)
│   ├── laq/              # Stage 1: Latent action quantization
│   ├── foundation/       # Stage 2: Vision-Language-Action model
│   └── low_level/        # Stage 3: Action decoding (if separate)
├── config/               # Hydra YAML configurations
├── scripts/              # Training entry points
├── slurm/                # LRZ job submission templates
├── containers/           # Enroot/Docker definitions
├── tests/                # Unit and integration tests
└── pyproject.toml        # Root dependencies
```

***

### 1.2 Training Framework Strategy

**Decision:** Hybrid approach using PyTorch Lightning ecosystem

| Stage | Framework | Rationale |
|-------|-----------|-----------|
| **Stage 1 (LAQ)** | PyTorch Lightning | Transformer-based supervised learning; auto-DDP, checkpointing, logging |
| **Stage 2 (Foundation)** | Lightning Fabric | Raw training loop control for complex FSDP; eliminates boilerplate |
| **Stage 3 (Finetuning)** | PyTorch Lightning | Small dataset; fast iteration; reuse Stage 1 patterns |

**Key Technical Insight:**
- Lightning Trainer abstracts too much for 7B multi-node FSDP training
- Fabric provides scaffolding (`fabric.launch()`, `fabric.setup()`) while preserving loop control
- Keeps codebase consistent (same ecosystem) while optimizing per-stage needs

***

### 1.3 Data Pipeline Architecture

**Decision:** WebDataset with TAR shards + Offline preprocessing

**Critical LRZ Constraint:** GPFS filesystem optimized for large sequential reads, NOT millions of small files

**Implementation Pattern:**
```
Raw Videos (100K files)
    ↓
[Preprocessing Script]
    ↓
Sharded TARs (100 archives × 1000 samples each)
    ↓
WebDataset Streaming Loader
    ↓
Training (high throughput)
```

**Key Parameters:**
- Shard size: 1-2 GB each
- Format: `.tar` containing `.jpg` + `.json` metadata
- Workers: 8 per GPU (balance CPU/I/O)
- Prefetch: 4 batches ahead

**Offline Label Generation (Critical):**
- After training LAQ, run batch inference to generate latent labels
- Cache results as new dataset: `(image, text, latent_tokens)`
- Foundation training becomes simple supervised learning (no LAQ inference overhead)

***

### 1.4 LRZ Cluster Integration

**Decision:** Enroot containers + Manual Slurm scripts (no hydra-submitit-launcher)

**Container Strategy:**
```dockerfile
# Base: NVIDIA PyTorch with CUDA 12.1
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install project packages
COPY packages/ /workspace/packages/
RUN pip install -e /workspace/packages/common && \
    pip install -e /workspace/packages/laq && \
    pip install -e /workspace/packages/foundation

# Dependencies
RUN pip install pytorch-lightning lightning-fabric hydra-core \
                webdataset wandb transformers timm
```

**Slurm Integration:**
```bash
# Universal template: slurm/train.sbatch
#!/bin/bash
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --container-image=/dss/.../lapa.sqsh
#SBATCH --container-mounts=/dss/dssfs04:/data

export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO

srun python "$@"  # Pass all args to Python script
```

**Usage:**
```bash
sbatch slurm/train.sbatch scripts/2_train_laq.py \
  experiment=laq_full cluster.num_nodes=1
```

***

### 1.5 Configuration Management

**Decision:** Hierarchical Hydra configs with composition

**Structure:**
```
config/
├── experiment/           # Top-level presets
│   ├── laq_debug.yaml
│   ├── laq_full.yaml
│   └── vla_7b.yaml
├── model/
│   ├── laq.yaml
│   ├── foundation_vla.yaml
│   └── low_level.yaml
├── data/
│   ├── openx_webdataset.yaml
│   ├── sthv2_webdataset.yaml
│   └── bridge_webdataset.yaml
├── training/
│   ├── laq_optimizer.yaml
│   └── vla_fsdp.yaml
└── cluster/
    ├── lrz_h100.yaml
    ├── lrz_a100.yaml
    └── local_dev.yaml
```

**Composition Example:**
```yaml
# config/experiment/laq_full.yaml
defaults:
  - override /model: laq
  - override /data: openx_webdataset
  - override /training: laq_optimizer
  - override /cluster: lrz_h100

experiment_name: laq_openx_v1
seed: 42
```

**CLI Overrides:**
```bash
python scripts/2_train_laq.py experiment=laq_full \
  data.batch_size=512 training.lr=5e-5
```

***

## Part 2: Detailed Component Specifications

### 2.1 Stage 1: Latent Action Quantization (LAQ)

#### 2.1.1 Model Architecture (LAPA Transformer-Based)

**Overview:**
LAPA uses a transformer-based architecture with three key components:
1. **Spatial-Temporal Encoder**: Processes frame pairs with separate spatial and temporal attention
2. **NSVQ (Noise-Substitution Vector Quantization)**: Quantizes frame deltas with single shared codebook
3. **Cross-Attention Decoder**: Reconstructs next frame conditioned on actions

**Encoder (Frame Pair → Latent Embeddings):**
```
Input: Frame pair [B, C, 2, H, W]  # C=3, H=W=256, 2 frames

Architecture:
├─ Patch Embedding (per frame):
│  ├─ frame_t: [B, 3, 1, 256, 256] → [B, 1, 64, 1024]     # 64 patches (8×8 grid)
│  └─ frame_t+1: [B, 3, 1, 256, 256] → [B, 1, 64, 1024]
│  └─ Concatenate: [B, 2, 64, 1024]  # 2 frames × 64 patches
│
├─ Spatial Transformer (8 layers):
│  ├─ Self-attention across 64 patches within each frame
│  ├─ Heads: 16, dim_head: 64 (total dim: 1024)
│  ├─ 2D Relative Position Bias for spatial relationships
│  └─ PEG (Positional Encoding Generator via 3D depthwise conv)
│
├─ Temporal Transformer (8 layers):
│  ├─ Self-attention across 2 frames for each patch position
│  ├─ Learns temporal dynamics between frame_t and frame_t+1
│  └─ Output: [B, 2, 64, 1024]
│
└─ Split tokens:
   ├─ first_frame_tokens: [B, 64, 1024]  # Represents state at t
   └─ last_frame_tokens: [B, 64, 1024]   # Represents state at t+1

Parameters:
- Image size: 256×256 (larger than 224 for better patch alignment)
- Patch size: 32×32 → 8×8 = 64 patches per frame
- Model dim: 1024
- Quantization dim: 32
- Spatial depth: 8 layers
- Temporal depth: 8 layers
- Heads: 16, dim_head: 64
```

**NSVQ Quantizer (LAPA's Key Innovation):**
```
Input: 
- first_frame_tokens: [B, 64, 1024]
- last_frame_tokens: [B, 64, 1024]

Parameters:
- code_seq_len: 4         # Action sequence length (2×2 action grid)
- codebook_size: 8        # Single shared codebook
- quant_dim: 32           # Quantization dimension

Codebook: [8, 32]         # SINGLE codebook for all positions (not per-position)

Process:
1. Project to quant space:
   ├─ first_quant = Linear(1024 → 32): [B, 64, 1024] → [B, 64, 32]
   └─ last_quant = Linear(1024 → 32): [B, 64, 1024] → [B, 64, 32]

2. Compute delta (key LAPA insight):
   delta = last_quant - first_quant  # [B, 64, 32]
   
3. CNN downsample to action grid:
   ├─ Reshape: [B, 64, 32] → [B, 32, 8, 8]  # Spatial layout
   ├─ Conv2D(32 → 32, kernel=4, stride=2): → [B, 32, 4, 4]
   ├─ Conv2D(32 → 32, kernel=4, stride=2): → [B, 32, 2, 2]
   └─ Reshape: [B, 32, 2, 2] → [B, 4, 32]  # 4 action tokens

4. Quantize with SINGLE codebook:
   ├─ For each of 4 positions, compute distances to ALL 8 embeddings
   ├─ indices = argmin(||delta[i] - codebook||²)  # [B, 4], each ∈ {0..7}
   └─ quantized = codebook[indices]  # [B, 4, 32]

5. Noise-Substitution Straight-Through Estimator (NOT standard STE):
   ├─ Forward: output = quantized
   ├─ Backward: ∇output flows to BOTH delta AND codebook
   └─ NO commitment loss or codebook loss!

6. Project back to model dim:
   quantized_actions = Linear(32 → 1024): [B, 4, 32] → [B, 4, 1024]

7. Reshape for spatial decoder:
   └─ [B, 4, 1024] → [B, 1, 2, 2, 1024]  # 2×2 action grid

Output:
- quantized_actions: [B, 1, 2, 2, 1024]
- indices: [B, 4]  # The discrete latent action codes
- perplexity: codebook usage metric
```

**Key Differences from Standard VQ-VAE:**
- ✅ Quantizes DELTA (last - first) instead of absolute features
- ✅ SINGLE codebook shared across all positions (not per-position)
- ✅ Noise-substitution STE (no codebook/commitment losses)
- ✅ Periodic unused codebook vector replacement for stability

**Loss Components:**
```
SIMPLIFIED: Only reconstruction loss!

1. Reconstruction Loss (ONLY loss):
   L_total = MSE(decoder(quantized_actions, first_frame_tokens), frame_{t+1})

NO codebook loss, NO commitment loss
This is a key LAPA simplification that improves training stability
```

**Cross-Attention Decoder (Quantized Actions → Reconstructed Frame):**
```
Input:
- Context: first_frame_tokens [B, 1, 64, 1024]  # What we see at time t
- Actions: quantized_actions [B, 1, 4, 1024]    # What action to take

Architecture:
├─ Flatten action grid: [B, 1, 2, 2, 1024] → [B, 1, 4, 1024]
│
├─ Spatial Transformer with Cross-Attention (8 layers):
│  ├─ Self-Attention on context patches
│  ├─ Cross-Attention:
│  │  ├─ Query: context patches (what we observe)
│  │  ├─ Key/Value: action tokens (what we intend to do)
│  │  └─ Attend to condition reconstruction on intended actions
│  ├─ Feedforward
│  └─ Output: [B, 1, 64, 1024]
│
├─ Patch to Pixels:
│  ├─ Reshape: [B, 1, 64, 1024] → [B, 1, 8, 8, 1024]
│  ├─ Linear: 1024 → (32 × 32 × 3) = 3072  # Patch size × channels
│  ├─ Reshape: [B, 1, 8, 8, 3072] → [B, 1, 8, 8, 32, 32, 3]
│  └─ Rearrange: → [B, 3, 1, 256, 256]
│
└─ Final activation: Tanh (normalize to [-1, 1])

Output: Reconstructed next frame [B, 3, 1, 256, 256]
```

#### 2.1.2 Data Pipeline

**Preprocessing Script (`scripts/1_videos_to_webdataset.py`):**
```
Input: Raw video files or image sequences

Process:
1. For each video:
   - Extract frames at 10 fps (or native framerate)
   - Generate consecutive pairs: (frame_t, frame_{t+1})
   - Resize to 256×256 (LAPA uses 256 for better patch alignment)
   - Save as JPEG (quality=95 for minimal loss)

2. Pack into TAR shards:
   - Target: 1000 samples per shard
   - Shard size: ~1-2 GB
   - Naming: train_shard_{00000..00099}.tar

3. TAR structure:
   train_shard_00000.tar:
   ├─ 000000_t.jpg          # frame_t
   ├─ 000000_t1.jpg         # frame_{t+1}
   ├─ 000000.json           # metadata {video_id, frame_idx}
   ├─ 000001_t.jpg
   └─ ...

Output: /dss/.../datasets/openx_frames/train_shard_*.tar
```

**DataModule (`common/data.py`):**
```python
class LAPADataModule(pl.LightningDataModule):
    Configuration:
    - shard_urls: List of TAR paths or glob pattern
    - batch_size: 256 (per GPU)
    - num_workers: 8
    - task: 'laq' | 'foundation' | 'low_level'
    
    WebDataset Pipeline:
    1. wds.SimpleShardList(shard_urls)
    2. wds.split_by_node      # Multi-node: each node gets unique shards
    3. wds.split_by_worker    # Multi-worker: each worker gets unique shards
    4. wds.tarfile_to_samples()
    5. wds.shuffle(1000)      # Shuffle buffer
    6. wds.decode("torchrgb") # JPEG → PIL → Tensor
    7. wds.map(preprocess)    # Normalize, augment
    8. wds.batched(batch_size)
    
    Returns:
    - For LAQ: {frames: [B,3,2,256,256]}  # Stacked frame pairs for LAPA
    - For Foundation: {image: [B,3,256,256], text: List[str], latents: [B,4]}
    
    Collate Function for LAQ:
    def collate_fn(batch):
        # Stack frame_t and frame_t1 into [B, C, 2, H, W] format
        frames = torch.stack([
            torch.stack([item['frame_t'], item['frame_t1']], dim=1)
            for item in batch
        ])  # [B, 3, 2, 256, 256]
        return {'frames': frames}
```

**Transforms:**
```python
Normalization: ImageNet stats
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]

Augmentations (training only):
  - RandomHorizontalFlip(p=0.5)
  - ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
  - Optional: RandomErasing(p=0.1) for robustness
```

#### 2.1.3 Training Configuration

**Model Config (`config/model/laq.yaml`):**
```yaml
model:
  name: laq_transformer  # LAPA transformer architecture
  
  # Image and patch configuration
  image_size: 256
  patch_size: 32
  channels: 3
  
  # Transformer dimensions
  dim: 1024              # Model dimension
  quant_dim: 32          # Quantization dimension
  
  # Encoder configuration
  spatial_depth: 8       # Spatial transformer layers
  temporal_depth: 8      # Temporal transformer layers
  heads: 16              # Attention heads
  dim_head: 64           # Dimension per head (heads × dim_head = dim)
  mlp_ratio: 4           # FFN expansion ratio
  
  # NSVQ configuration
  code_seq_len: 4        # 2×2 action grid = 4 tokens
  codebook_size: 8       # Single shared codebook
  noise_substitution: true  # Use NSVQ instead of standard VQ-VAE
  
  # Decoder configuration
  decoder_depth: 8       # Cross-attention decoder layers
  decoder_heads: 16
  
  # NO VQ-VAE specific parameters (no beta, no ema_decay)
  # LAPA uses only MSE loss
```

**Training Config (`config/training/laq_optimizer.yaml`):**
```yaml
training:
  epochs: 100
  
  optimizer:
    type: AdamW
    lr: 1e-4
    betas: [0.9, 0.999]
    weight_decay: 0.01
    eps: 1e-8
  
  scheduler:
    type: CosineAnnealingLR
    T_max: ${training.epochs}
    eta_min: 1e-6
    warmup_steps: 1000
  
  gradient:
    clip_val: 1.0
    clip_algorithm: norm
  
  # LAPA uses ONLY MSE loss (no VQ-specific losses)
  # No loss_weights needed - single reconstruction loss
  
  validation:
    interval_epochs: 1
    num_samples_to_log: 8  # Visualize reconstructions
```

**Cluster Config (`config/cluster/lrz_h100.yaml`):**
```yaml
cluster:
  name: lrz_h100
  
  compute:
    num_nodes: 1
    gpus_per_node: 4
    cpus_per_task: 18
    mem_gb: 500
    time_limit: "24:00:00"
  
  slurm:
    partition: mcml-hgx-h100-94x4
    qos: mcml
    account: null
  
  distributed:
    backend: nccl
    precision: bf16-mixed
    strategy: ddp  # or fsdp for very large encoders
  
  storage:
    data_root: /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/datasets
    checkpoint_dir: /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/checkpoints
    log_dir: /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/logs
```

#### 2.1.4 Lightning Module Implementation

**LAQTask (`laq/task.py`):**
```python
Key Methods:

__init__(model_cfg, optimizer_cfg):
  - Initialize LAPA model (encoder, NSVQ, decoder)
  - Store configs for optimizer/scheduler

forward(frames):
  # frames: [B, C, 2, H, W] - stacked frame pairs
  1. Encode: first_tokens, last_tokens = encoder(frames)
     # first_tokens: [B, 64, 1024], last_tokens: [B, 64, 1024]
  2. Quantize: quantized_actions, indices, perplexity = nsvq(first_tokens, last_tokens)
     # quantized_actions: [B, 1, 2, 2, 1024], indices: [B, 4]
  3. Decode: reconstructed = decoder(first_tokens, quantized_actions)
     # reconstructed: [B, 3, 1, 256, 256]
  4. Return: reconstructed, indices, perplexity

training_step(batch, batch_idx):
  1. Forward pass on stacked frames
  2. Extract target: frame_t1 = batch['frames'][:, :, 1:2, :, :]  # [B, 3, 1, 256, 256]
  3. Compute ONLY reconstruction loss: MSE(reconstructed, frame_t1)
  4. Log loss and perplexity
  5. Return loss  # Single loss, no VQ losses!

validation_step(batch, batch_idx):
  1. Same as training_step
  2. If batch_idx == 0: log reconstruction visualizations
  3. Track codebook usage statistics

predict_step(batch, batch_idx):
  # Used by script 3 (latent label generation)
  1. Forward pass
  2. Return indices only (the discrete latent codes) [B, 4]

configure_optimizers():
  1. Create AdamW optimizer from config
  2. Create cosine scheduler with warmup
  3. Return {optimizer, scheduler}
```

**Metrics to Track:**
```
Training:
- train/loss_recon (ONLY loss - no VQ losses)
- train/perplexity (codebook usage diversity)
- train/codebook_utilization (% of 8 embeddings actively used)

Validation:
- val/loss_recon
- val/perplexity
- val/codebook_utilization
- val/psnr (reconstruction quality)
- val/reconstruction_images (WandB grid)

Note: NO codebook_loss or commitment_loss (LAPA uses noise-substitution STE)
```

***

### 2.2 Stage 2: Foundation Policy (Vision-Language-Action)

#### 2.2.1 Model Architecture

**Vision Encoders (Frozen):**
```
Dual-Encoder Design:

Encoder 1: SigLIP-SO400M
├─ Purpose: General visual understanding
├─ Input: [B, 3, 224, 224]
├─ Output: [B, 576, 1152]  # 24×24 patches, 1152-dim features
└─ Weights: Load from HuggingFace, freeze

Encoder 2: DINOv2-ViT-L/14
├─ Purpose: Spatial grounding, fine-grained features
├─ Input: [B, 3, 224, 224]
├─ Output: [B, 576, 1024]  # 24×24 patches, 1024-dim features
└─ Weights: Load from HuggingFace, freeze

Why dual encoders?
- SigLIP: Trained on image-text pairs (semantic understanding)
- DINOv2: Self-supervised (better spatial features)
- Concatenating both gives richer representations
```

**Vision-Language Projector:**
```
Input: Concatenated vision features [B, 576, 2176]  # 1152 + 1024

Architecture:
├─ LayerNorm(2176)
├─ Linear(2176 → 4096)
├─ GELU activation
├─ Linear(4096 → 4096)  # Match Llama-2 embedding dim
└─ LayerNorm(4096)

Output: Visual tokens [B, 576, 4096] in language model space
```

**Language Model Backbone:**
```
Llama-2 7B (32 layers, 4096 hidden, 32 attention heads)

Input Sequence Construction:
┌────────────────────────────────────────────────────┐
│ [BOS] [IMG_0] [IMG_1] ... [IMG_575] [TEXT] [ACT]  │
│   1      576 vision tokens       N text  4 action │
└────────────────────────────────────────────────────┘

Process:
1. Text tokenization: "Pick up red block" → token_ids
2. Embed text: text_embeds = llama.embed_tokens(token_ids)
3. Concatenate: [visual_tokens, text_embeds]
4. Forward through Llama: hidden_states = llama(inputs_embeds=concat)
5. Extract last 4 positions: action_hidden = hidden_states[:, -4:, :]

Output: [B, 4, 4096]  # Hidden states for 4 action token positions
```

**Action Head:**
```
For Stage 2 (Latent Prediction):

Input: [B, 4, 4096]
Architecture:
├─ LayerNorm(4096)
├─ Linear(4096 → vocab_size)  # vocab_size = 8
└─ No activation (logits)

Output: [B, 4, 8]  # Logits for each token position

Loss: CrossEntropyLoss(logits, target_indices)
```

**For Stage 3 (Continuous Actions):**
```
Input: [B, 1, 4096]  # Single action token (not 4)

Architecture:
├─ LayerNorm(4096)
├─ Linear(4096 → 7 × 256)  # 7 DoF × 256 bins
└─ Reshape: [B, 7, 256]

Output: [B, 7, 256]  # Logits for each action dimension

Action Discretization:
- For each DoF: compute 1st and 99th percentile from training data
- Bin uniformly into 256 levels
- Map bins to unused Llama token IDs (e.g., 32000-34303)

Loss: CrossEntropyLoss(logits, binned_actions)
```

#### 2.2.2 FSDP Configuration (Critical)

**Sharding Strategy:**
```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

fsdp_config = {
    # Shard parameters AND gradients across all GPUs
    "sharding_strategy": ShardingStrategy.FULL_SHARD,
    
    # Auto-wrap each Llama layer (32 layers wrapped individually)
    "auto_wrap_policy": partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer}
    ),
    
    # BF16 training (critical for H100 + stability)
    "mixed_precision": MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    ),
    
    # Overlap communication with computation
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
    
    # H100 has 94GB, keep everything on GPU
    "cpu_offload": None,
    
    # Required for optimizer compatibility
    "use_orig_params": True,
    
    # Optional: for 13B+ models
    # "activation_checkpointing": True
}
```

**Why This Matters:**
- Without `auto_wrap_policy`: Entire 7B model treated as one unit → OOM
- With per-layer wrapping: Each GPU holds ~220MB per layer (7B / 32 layers / 4 GPUs)
- `FULL_SHARD`: Reduces memory by 4x on 4 GPUs (vs DDP)

#### 2.2.3 Data Pipeline (Latent-Labeled Dataset)

**Label Generation Script (`scripts/3_generate_latent_labels.py`):**
```python
Purpose: Run trained LAQ model on video dataset to cache latent codes

Process:
1. Load LAQ checkpoint from Stage 1
2. Load video frame dataset
3. For each (frame_t, frame_t+1) pair:
   a. Run LAQ encoder + quantizer (in eval mode)
   b. Extract latent indices: [4] discrete tokens
   c. Save to new shard with language annotation

4. Output format (new WebDataset):
   latent_labeled_shard_00000.tar:
   ├─ 000000_image.jpg       # frame_t (observation)
   ├─ 000000.txt             # "Pick up the red block"
   ├─ 000000_latent.npy      # [4] array of token indices
   ├─ 000000.json            # {video_id, frame_idx, task_id}
   └─ ...

Storage: /dss/.../datasets/openx_latent_labeled/
```

**DataModule for Foundation Training:**
```python
class FoundationDataModule:
    Returns per batch:
    {
        'image': [B, 3, 224, 224],      # RGB observation
        'text': List[str],               # Language instructions
        'latent_tokens': [B, 4],         # Target latent actions (0-7)
        'attention_mask': [B, seq_len]   # For variable-length text
    }
    
    Collation:
    - Pad text to max length in batch
    - Stack images and latent tokens as tensors
```

#### 2.2.4 Training Loop (Lightning Fabric)

**FoundationTrainer (`foundation/trainer.py`):**
```python
class FoundationTrainer:
    def __init__(self, cfg):
        # Initialize Fabric with FSDP config
        self.fabric = Fabric(
            accelerator="cuda",
            devices=cfg.cluster.gpus_per_node,
            num_nodes=cfg.cluster.num_nodes,
            strategy=FSDPStrategy(**fsdp_config),
            precision="bf16-mixed"
        )
    
    def run(self):
        # Launch all distributed processes
        self.fabric.launch()
        
        # Build model INSIDE fabric context
        with self.fabric.init_module():
            model = VLAModel(self.cfg.model)
        
        # Build optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay
        )
        
        # Fabric wraps model and optimizer (FSDP sharding happens here)
        model, optimizer = self.fabric.setup(model, optimizer)
        
        # Setup data
        datamodule = FoundationDataModule(self.cfg.data)
        train_loader = self.fabric.setup_dataloaders(
            datamodule.train_dataloader()
        )
        
        # Training loop (standard PyTorch style)
        for epoch in range(self.cfg.training.epochs):
            for batch_idx, batch in enumerate(train_loader):
                # Forward
                logits = model(
                    images=batch['image'],
                    texts=batch['text']
                )  # [B, 4, 8]
                
                # Loss
                loss = F.cross_entropy(
                    logits.view(-1, 8),           # [B*4, 8]
                    batch['latent_tokens'].view(-1)  # [B*4]
                )
                
                # Backward (Fabric handles distributed)
                self.fabric.backward(loss)
                
                # Gradient clipping
                self.fabric.clip_gradients(
                    model, optimizer,
                    max_norm=1.0
                )
                
                # Step
                optimizer.step()
                optimizer.zero_grad()
                
                # Logging (rank 0 only)
                if self.fabric.is_global_zero:
                    self.fabric.log("train/loss", loss.item())
                
                # Checkpointing
                if batch_idx % 1000 == 0:
                    self.save_checkpoint(model, optimizer, epoch, batch_idx)
    
    def save_checkpoint(self, model, optimizer, epoch, step):
        if self.fabric.is_global_zero:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": step
            }
            path = f"{self.cfg.checkpoint.dir}/checkpoint_e{epoch}_s{step}.ckpt"
            self.fabric.save(path, checkpoint)
```

**Key Differences from Lightning Trainer:**
- YOU write the `for epoch... for batch...` loop
- Fabric handles: process spawning, FSDP wrapping, distributed backward, checkpointing
- Full control over when to log, save, validate

#### 2.2.5 Training Configuration

**Model Config (`config/model/foundation_vla.yaml`):**
```yaml
model:
  name: vla_7b
  
  vision:
    siglip:
      model_name: google/siglip-so400m-patch14-384
      freeze: true
    dinov2:
      model_name: facebook/dinov2-large
      freeze: true
  
  projector:
    input_dim: 2176  # 1152 + 1024
    hidden_dim: 4096
    output_dim: 4096
    num_layers: 2
  
  llm:
    model_name: meta-llama/Llama-2-7b-hf
    freeze_embeddings: false
    freeze_layers: []  # Empty = train all
    use_flash_attention: true  # If available
  
  action_head:
    type: latent  # or 'continuous' for Stage 3
    vocab_size: 8
    num_tokens: 4
```

**FSDP Config (`config/training/vla_fsdp.yaml`):**
```yaml
training:
  epochs: 10
  effective_batch_size: 4096  # Target across all GPUs
  
  # Per-GPU batch size (adjust based on memory)
  batch_size_per_gpu: 16
  gradient_accumulation_steps: 64  # 16 × 64 × 4 GPUs = 4096
  
  optimizer:
    type: AdamW
    lr: 1e-5  # Conservative for LLM finetuning
    betas: [0.9, 0.95]
    weight_decay: 0.1
    eps: 1e-8
  
  scheduler:
    type: cosine
    warmup_steps: 2000
    min_lr: 1e-6
  
  gradient:
    clip_val: 1.0
  
  fsdp:
    sharding_strategy: FULL_SHARD
    backward_prefetch: true
    activation_checkpointing: false  # Enable for 13B+
  
  checkpoint:
    save_interval_steps: 1000
    keep_last: 5
```

***

### 2.3 Stage 3: Action Finetuning

#### 2.3.1 Architecture Modification

**Two Approaches:**

**Approach A: Modify Foundation Model (Recommended)**
```
1. Load Stage 2 checkpoint
2. Remove latent action head: Linear(4096 → 8)
3. Add continuous action head: Linear(4096 → 7×256)
4. Change token structure:
   - Before: [BOS] [IMG] [TEXT] [ACT_0] [ACT_1] [ACT_2] [ACT_3]
   - After:  [BOS] [IMG] [TEXT] [ACT]  # Single action token
5. Freeze vision encoders (keep frozen)
6. Unfreeze Llama + new action head
```

**Approach B: Separate Low-Level Policy**
```
Foundation outputs latent tokens → Small MLP decodes to actions

Architecture:
Input: [4] latent token indices (one-hot: [4, 8])
├─ Embedding layer: [4, 8] → [4, 256]
├─ Flatten: [1024]
├─ MLP: 1024 → 512 → 256 → 7
└─ Output: [7] continuous actions

Pro: Fast training, modular
Con: Extra inference step, less expressive
```

**Decision: Use Approach A** (end-to-end optimization)

#### 2.3.2 Action Discretization

**Binning Strategy:**
```python
For each action dimension (x, y, z, roll, pitch, yaw, gripper):

1. Collect all values from robot demonstration dataset
2. Compute statistics:
   - min_val = 1st percentile (robust to outliers)
   - max_val = 99th percentile
3. Create 256 uniform bins: [min_val, max_val]
4. Quantize actions:
   bin_idx = int((action - min_val) / (max_val - min_val) * 255)
   bin_idx = clip(bin_idx, 0, 255)

5. Map to token IDs:
   - Reserve Llama token IDs 32000-34303 (unused in vocab)
   - x_tokens: 32000-32255
   - y_tokens: 32256-32511
   - z_tokens: 32512-32767
   - roll_tokens: 32768-33023
   - pitch_tokens: 33024-33279
   - yaw_tokens: 33280-33535
   - gripper_tokens: 33536-33791

During inference: De-quantize
   action = min_val + (bin_idx / 255.0) * (max_val - min_val)
```

**Store Statistics:**
```python
# Save during preprocessing
action_stats = {
    'x': {'min': -0.5, 'max': 0.5, 'bins': 256},
    'y': {'min': -0.5, 'max': 0.5, 'bins': 256},
    # ... for all 7 DoF
}
torch.save(action_stats, 'action_quantization_stats.pt')
```

#### 2.3.3 Data Pipeline

**Robot Dataset Format:**
```
Expected structure per trajectory:
{
    'observations': {
        'image': [T, 3, 224, 224],
        'proprio': [T, D]  # Optional proprioception
    },
    'actions': [T, 7],  # [x, y, z, roll, pitch, yaw, gripper]
    'language': str      # "Pick up the red block"
}

WebDataset TAR format:
robot_demo_shard_00000.tar:
├─ traj_00000_step_000_image.jpg
├─ traj_00000_step_000_action.npy  # [7] array
├─ traj_00000.txt                   # Language annotation
├─ traj_00000.json                  # Metadata
└─ ...
```

**Training Returns:**
```python
{
    'image': [B, 3, 224, 224],
    'text': List[str],
    'actions': [B, 7],          # Continuous
    'action_bins': [B, 7]       # Discretized
}
```

#### 2.3.4 Training (Lightning)

**Use Same Pattern as Stage 1:**
```python
class ActionFinetuneTask(pl.LightningModule):
    def __init__(self, foundation_checkpoint, action_stats, optimizer_cfg):
        # Load Stage 2 model
        self.vla = VLAModel.load_from_checkpoint(foundation_checkpoint)
        
        # Replace action head
        self.vla.action_head = ContinuousActionHead(
            input_dim=4096,
            num_actions=7,
            num_bins=256
        )
        
        # Freeze vision encoders
        for param in self.vla.vision_siglip.parameters():
            param.requires_grad = False
        for param in self.vla.vision_dinov2.parameters():
            param.requires_grad = False
        
        self.action_stats = action_stats
    
    def training_step(self, batch, batch_idx):
        # Forward
        logits = self.vla(
            images=batch['image'],
            texts=batch['text']
        )  # [B, 7, 256]
        
        # Loss (per-dimension cross-entropy)
        loss = F.cross_entropy(
            logits.view(-1, 256),
            batch['action_bins'].view(-1)
        )
        
        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Predict and dequantize
        predicted_bins = self.predict_actions(batch)
        predicted_continuous = self.dequantize(predicted_bins)
        
        # Compute MSE in continuous space
        mse = F.mse_loss(predicted_continuous, batch['actions'])
        self.log('val/action_mse', mse)
    
    def dequantize(self, bins):
        # Convert bins [B, 7] → continuous [B, 7]
        continuous = []
        for dim_idx in range(7):
            stats = self.action_stats[dim_idx]
            vals = stats['min'] + (bins[:, dim_idx] / 255.0) * (stats['max'] - stats['min'])
            continuous.append(vals)
        return torch.stack(continuous, dim=1)
```

***

## Part 3: Project Phases & Task Breakdown

### Phase 0: Infrastructure Setup (Week 1)

**Task 0.1: Repository Setup** (1 day)
- **Owner:** Tech Lead
- **Description:** Create monorepo structure, setup `pyproject.toml`, Git repository
- **Deliverables:**
  - Git repo with directory structure
  - Initial `pyproject.toml` with base dependencies
  - `.gitignore`, `README.md`

**Task 0.2: Hydra Configuration Scaffolding** (2 days)
- **Owner:** Senior Dev
- **Description:** Create all YAML config files with placeholder values
- **Deliverables:**
  - Complete `config/` directory structure
  - Working Hydra composition (test with dummy script)
- **Files:** All `.yaml` files mentioned in Section 1.5

**Task 0.3: LRZ Account & Storage Setup** (2 days)
- **Owner:** Tech Lead
- **Description:** Get cluster access, create storage directories, test Slurm
- **Deliverables:**
  - SSH access to `cool.hpc.lrz.de`
  - Created directories: `/dss/.../datasets`, `/dss/.../checkpoints`, `/dss/.../logs`
  - Successfully run test `sbatch` job
- **Validation:** `sinfo`, `squeue` work, test GPU allocation

**Task 0.4: Container Build** (3 days)
- **Owner:** DevOps/Senior Dev
- **Description:** Create Dockerfile, build Enroot container, test on LRZ
- **Deliverables:**
  - `containers/Dockerfile`
  - `lapa.sqsh` image on LRZ storage
  - Test script runs inside container
- **Validation:** `srun --container-image=lapa.sqsh python --version` works

**Task 0.5: Logging Infrastructure** (2 days)
- **Owner:** Junior Dev 1
- **Description:** Setup WandB project, implement logging utilities
- **Deliverables:**
  - WandB project created
  - `common/logging.py` with helpers
  - Test logging from dummy training script

***

### Phase 1: LAPA Implementation (Weeks 2-4)

**Architecture Note:** This phase implements the LAPA paper's transformer-based approach with NSVQ, NOT a standard convolutional VQ-VAE. See Section 2.1.1 for detailed architecture.

**Task 1.1: Data Preprocessing Pipeline** (5 days)
- **Owner:** Junior Dev 2
- **Description:** Implement `scripts/1_videos_to_webdataset.py`
- **Requirements:**
  - Input: Directory of videos or image sequences
  - Process: Extract frames, create pairs, pack into TARs
  - Output: Sharded WebDataset on `/dss/.../datasets/`
- **Deliverables:**
  - Working preprocessing script
  - Sample dataset (1000 pairs) for testing
  - Documentation of TAR structure
- **Validation:** Load 10 random samples, verify shapes and quality

**Task 1.2: LAQ DataModule** (3 days)
- **Owner:** Junior Dev 1
- **Description:** Implement `common/data.py` with WebDataset pipeline
- **Requirements:**
  - Support LAQ task (return frame pairs)
  - Handle distributed sampling (split_by_node, split_by_worker)
  - Implement transforms and augmentations
- **Deliverables:**
  - `LAPADataModule` class
  - Unit tests for dataloader
- **Validation:** 
  - Iterate 1000 batches, measure throughput (target: >500 samples/sec on 1 GPU)
  - Test on 2 nodes × 4 GPUs (no duplicate samples)

**Task 1.3: Transformer Components** (5 days)
- **Owner:** Senior Dev
- **Description:** Implement core transformer building blocks for LAPA
- **Requirements:**
  - Transformer block with self-attention and optional cross-attention
  - PEG (Positional Encoding Generator via 3D depthwise conv)
  - ContinuousPositionBias for 2D spatial relationships
  - Support BF16 precision
- **Deliverables:**
  - `laq/models/attention.py` with:
    - `Transformer` class (self + cross attention + FFN)
    - `PEG` class (3D depthwise conv for positional encoding)
    - `ContinuousPositionBias` class (2D relative position bias)
  - Unit tests (attention shapes, gradient flow, position bias computation)
- **Validation:**
  - Transformer: Input [B,64,1024] → Output [B,64,1024]
  - PEG works with 5D tensors [B,C,T,H,W]
  - Position bias produces correct [heads, H×W, H×W] matrix

**Task 1.4: NSVQ Implementation** (6 days)
- **Owner:** Senior Dev
- **Description:** Implement NSVQ with delta quantization (Section 2.1.1)
- **Requirements:**
  - SINGLE shared codebook (not per-position like VQ-VAE)
  - Delta quantization: quantize(last_tokens - first_tokens)
  - Noise-substitution straight-through estimator
  - Periodic unused codebook vector replacement
  - CNN downsampling from 64 patches to 4 action tokens
- **Deliverables:**
  - `laq/models/nsvq.py`
  - Unit tests (delta computation, quantization, codebook usage)
- **Validation:**
  - Input: [B,64,1024] (first), [B,64,1024] (last)
  - Output: quantized [B,1,2,2,1024], indices [B,4]
  - Codebook utilization increases over dummy training
  - NO codebook/commitment losses computed

**Task 1.5: Spatial-Temporal Encoder** (4 days)
- **Owner:** Senior Dev
- **Description:** Implement patch embedding + spatial/temporal transformers
- **Requirements:**
  - Patch embedding for video frames
  - Spatial transformer (8 layers) with 2D position bias
  - Temporal transformer (8 layers)
  - Split output into first/last frame tokens
- **Deliverables:**
  - `laq/models/encoder.py`
  - Unit tests
- **Validation:**
  - Input [B,3,2,256,256] → first [B,64,1024], last [B,64,1024]
  - Forward + backward passes succeed

**Task 1.6: Cross-Attention Decoder** (4 days)
- **Owner:** Junior Dev 2 + Senior Dev
- **Description:** Implement cross-attention decoder (Section 2.1.1)
- **Requirements:**
  - Spatial transformer with cross-attention
  - Query from first_frame_tokens, K/V from action_tokens
  - Patch-to-pixel projection
- **Deliverables:**
  - `laq/models/decoder.py`
  - Unit tests
- **Validation:**
  - Input: context [B,1,64,1024], actions [B,1,4,1024]
  - Output: [B,3,1,256,256]
  - Cross-attention weights have expected patterns

**Task 1.7: LAPA Model Integration** (4 days)
- **Owner:** Senior Dev
- **Description:** Wire together all LAPA components into single model
- **Requirements:**
  - Combine patch embedding, encoder, NSVQ, decoder
  - Implement forward pass with proper tensor routing
  - Add model initialization and weight loading
- **Deliverables:**
  - `laq/models/lapa.py` (main model class)
  - End-to-end integration test
- **Validation:**
  - Input [B,3,2,256,256] → Output [B,3,1,256,256], indices [B,4]
  - Gradients flow through all components
  - Model can overfit 5 samples

**Task 1.8: LAQ Lightning Module** (3 days)
- **Owner:** Junior Dev 1
- **Description:** Implement `LAQTask` (Section 2.1.4)
- **Requirements:**
  - Wire LAPA model into Lightning module
  - Implement training/validation steps with MSE loss only
  - Add reconstruction visualization callback
  - Track perplexity and codebook utilization
- **Deliverables:**
  - `laq/task.py`
  - Integration test (overfit 10 samples)
- **Validation:**
  - Loss decreases on tiny dataset
  - Reconstructions logged to WandB
  - No VQ-specific losses in logs

**Task 1.9: Training Script** (2 days)
- **Owner:** Junior Dev 1
- **Description:** Implement `scripts/2_train_laq.py`
- **Requirements:**
  - Hydra integration
  - Lightning Trainer setup
  - Checkpoint callbacks
- **Deliverables:**
  - Working entry point
  - Slurm job script for LRZ
- **Validation:**
  - Train on 1 GPU for 5 epochs (small dataset)
  - Checkpoints save correctly

**Task 1.10: Full LAQ Training** (5 days)
- **Owner:** All team
- **Description:** Train LAPA on full dataset (OpenX or STHv2)
- **Requirements:**
  - Use 1-2 nodes (4-8 H100s)
  - Monitor training curves (only MSE loss, no VQ losses)
  - Validate reconstruction quality
- **Deliverables:**
  - Trained LAPA checkpoint (`laq_final.ckpt`)
  - Training report (MSE loss, perplexity, codebook usage, example reconstructions)
- **Success Criteria:**
  - Reconstruction PSNR > 22 dB (LAPA typically better than VQ-VAE)
  - Codebook utilization > 70% (all 8 embeddings should be used)
  - Perplexity close to codebook_size (indicating diverse usage)

**Task 1.11: Latent Label Generation** (3 days)
- **Owner:** Junior Dev 2
- **Description:** Implement `scripts/3_generate_latent_labels.py`
- **Requirements:**
  - Load LAPA checkpoint
  - Run inference on full dataset (extract indices only)
  - Save new WebDataset with latent labels
- **Deliverables:**
  - Label generation script
  - New dataset: `/dss/.../datasets/openx_latent_labeled/`
- **Validation:**
  - Load random samples, verify latent codes ∈ {0..7} for all 4 positions
  - Check file sizes (should be smaller than original)
  - Verify codebook usage distribution in generated labels

---

### Phase 2: Foundation Policy (Weeks 5-8)

**Task 2.1: Vision Encoder Integration** (3 days)
- **Owner:** Senior Dev
- **Description:** Load and test SigLIP + DINOv2 from HuggingFace
- **Requirements:**
  - Download pretrained weights
  - Test forward pass
  - Freeze parameters
- **Deliverables:**
  - `foundation/models/vision_encoder.py`
  - Unit test (dummy image → features)
- **Validation:**
  - SigLIP:  →[4][6]
  - DINOv2:  →[6][4]

**Task 2.2: Vision-Language Projector** (2 days)
- **Owner:** Junior Dev 1
- **Description:** Implement projector MLP (Section 2.2.1)
- **Requirements:**
  - Concatenate vision features
  - Project to Llama embedding space
- **Deliverables:**
  - `foundation/models/projector.py`
  - Unit test
- **Validation:**
  - Input  → Output[6]

**Task 2.3: Llama-2 Integration** (4 days)
- **Owner:** Senior Dev
- **Description:** Load Llama-2 7B, implement input construction
- **Requirements:**
  - Load from HuggingFace (`meta-llama/Llama-2-7b-hf`)
  - Concatenate visual + text tokens
  - Handle variable-length sequences
- **Deliverables:**
  - `foundation/models/llm_wrapper.py`
  - Test forward pass with dummy inputs
- **Validation:**
  - Input: , text: [1,N,4096]][6]
  - Output: [1, 576+N+4, 4096]

**Task 2.4: Action Head** (2 days)
- **Owner:** Junior Dev 2
- **Description:** Implement latent action prediction head
- **Requirements:**
  - Linear layer: 4096 → 8 (per token position)
  - Support both latent and continuous modes
- **Deliverables:**
  - `foundation/models/action_head.py`
  - Unit test
- **Validation:**
  - Latent mode: [B, 4, 4096] → [B, 4, 8]
  - Continuous mode: [B, 1, 4096] → [B, 7, 256]

**Task 2.5: VLA Model Integration** (3 days)
- **Owner:** Senior Dev
- **Description:** Combine all components into `VLAModel`
- **Requirements:**
  - Wire vision → projector → Llama → action head
  - Implement forward pass
  - Handle text tokenization
- **Deliverables:**
  - `foundation/models/vla_model.py`
  - End-to-end test
- **Validation:**
  - Input: (image, "pick up block")
  - Output: [B, 4, 8] logits

**Task 2.6: Foundation DataModule** (3 days)
- **Owner:** Junior Dev 1
- **Description:** Extend `LAPADataModule` for foundation task
- **Requirements:**
  - Load latent-labeled dataset from Task 1.9
  - Return (image, text, latent_tokens)
  - Handle text padding
- **Deliverables:**
  - Updated `common/data.py`
  - Test dataloader
- **Validation:**
  - Iterate 100 batches, verify shapes
  - Test distributed sampling

**Task 2.7: FSDP Configuration** (4 days)
- **Owner:** Senior Dev + Tech Lead
- **Description:** Implement FSDP strategy for Lightning Fabric (Section 2.2.2)
- **Requirements:**
  - Auto-wrap policy for Llama layers
  - Mixed precision (BF16)
  - Checkpoint handling
- **Deliverables:**
  - `foundation/fsdp_config.py`
  - Test on 1 node × 4 GPUs
- **Validation:**
  - Model shards correctly (check memory usage)
  - Forward + backward passes succeed
  - Gradients synchronize across GPUs

**Task 2.8: Foundation Trainer (Fabric)** (5 days)
- **Owner:** Senior Dev
- **Description:** Implement `FoundationTrainer` (Section 2.2.4)
- **Requirements:**
  - Fabric-based training loop
  - Gradient accumulation
  - Checkpoint saving/loading
  - Logging
- **Deliverables:**
  - `foundation/trainer.py`
  - Integration test (overfit 10 samples)
- **Validation:**
  - Loss decreases on tiny dataset
  - Checkpoints save/load correctly

**Task 2.9: Multi-Node Testing** (3 days)
- **Owner:** Tech Lead + Senior Dev
- **Description:** Test foundation training on 2-4 nodes
- **Requirements:**
  - Submit multi-node Slurm job
  - Monitor NCCL communication
  - Verify no hangs or deadlocks
- **Deliverables:**
  - Working Slurm script (`slurm/train_foundation.sbatch`)
  - Multi-node debugging guide
- **Validation:**
  - All ranks initialize successfully
  - Training progresses without hangs
  - Throughput scales with nodes

**Task 2.10: Full Foundation Training** (7-10 days wall-clock)
- **Owner:** All team (monitoring)
- **Description:** Train foundation policy on full dataset
- **Requirements:**
  - 4-8 nodes (16-32 H100s)
  - Target: 10 epochs on OpenX dataset
  - Monitor loss curves, accuracy
- **Deliverables:**
  - Trained foundation checkpoint (`vla_foundation.ckpt`)
  - Training report (loss, accuracy, examples)
- **Success Criteria:**
  - Latent action prediction accuracy > 60%
  - Loss converges smoothly

---

### Phase 3: Action Finetuning (Weeks 9-10)

**Task 3.1: Action Discretization** (2 days)
- **Owner:** Junior Dev 2
- **Description:** Implement binning strategy (Section 2.3.2)
- **Requirements:**
  - Compute statistics from robot dataset
  - Implement quantization/dequantization functions
  - Save action stats
- **Deliverables:**
  - `low_level/action_discretization.py`
  - `action_quantization_stats.pt`
  - Unit tests
- **Validation:**
  - Round-trip error: quantize → dequantize → error < 1%

**Task 3.2: Robot DataModule** (3 days)
- **Owner:** Junior Dev 1
- **Description:** Implement dataloader for robot demonstrations
- **Requirements:**
  - Load robot trajectories
  - Return (image, text, continuous_actions, binned_actions)
  - Handle different dataset formats (Bridge, Fractal, etc.)
- **Deliverables:**
  - Updated `common/data.py`
  - Test dataloader
- **Validation:**
  - Load from sample robot dataset
  - Verify action statistics match

**Task 3.3: Modify VLA Architecture** (2 days)
- **Owner:** Senior Dev
- **Description:** Swap action head for continuous predictions
- **Requirements:**
  - Load foundation checkpoint
  - Replace head: Linear(4096 → 7×256)
  - Modify token structure
- **Deliverables:**
  - Updated `foundation/models/vla_model.py`
  - Test forward pass
- **Validation:**
  - Input: (image, text)
  - Output: [B, 7, 256] logits

**Task 3.4: Finetuning Task (Lightning)** (3 days)
- **Owner:** Senior Dev
- **Description:** Implement `ActionFinetuneTask` (Section 2.3.4)
- **Requirements:**
  - Load foundation checkpoint
  - Freeze vision encoders
  - Implement training/validation steps with action dequantization
- **Deliverables:**
  - `low_level/task.py`
  - Integration test
- **Validation:**
  - Loss decreases on small dataset
  - Predicted actions in valid range

**Task 3.5: Full Finetuning** (3-5 days)
- **Owner:** All team
- **Description:** Finetune on robot dataset
- **Requirements:**
  - 1 node (4 H100s)
  - Monitor action prediction MSE
  - Validate on held-out trajectories
- **Deliverables:**
  - Final model checkpoint (`vla_finetuned.ckpt`)
  - Finetuning report
- **Success Criteria:**
  - Action MSE < 0.05 (normalized)
  - Visual inspection: predictions look reasonable

---

### Phase 4: Inference & Evaluation (Week 11)

**Task 4.1: Inference Pipeline** (4 days)
- **Owner:** Senior Dev
- **Description:** Implement end-to-end inference (Section 2.2 diagram)
- **Requirements:**
  - Load finetuned VLA model
  - Implement `predict()` method
  - Handle preprocessing
  - Dequantize actions
- **Deliverables:**
  - `lapa/inference/pipeline.py`
  - Example inference script
- **Validation:**
  - Single image + text → 7-DoF action in <100ms
  - Batch inference works

**Task 4.2: Evaluation Script** (3 days)
- **Owner:** Junior Dev 1
- **Description:** Implement `scripts/6_run_inference.py`
- **Requirements:**
  - Load test trajectories
  - Run inference
  - Compute metrics (MSE, success rate if labels available)
  - Generate visualizations
- **Deliverables:**
  - Evaluation script
  - Metrics report
- **Validation:**
  - Run on 100 test examples
  - Metrics align with validation during training

**Task 4.3: Robot Deployment Prep** (optional, 3 days)
- **Owner:** Tech Lead
- **Description:** Prepare for real robot deployment
- **Requirements:**
  - Convert model to TorchScript or ONNX
  - Test on CPU/edge GPU
  - Measure inference latency
- **Deliverables:**
  - Deployment guide
  - Optimized model artifact
- **Validation:**
  - Inference <50ms on target hardware

***

### Phase 5: Optimization & Documentation (Week 12)

**Task 5.1: Performance Profiling** (3 days)
- **Owner:** Senior Dev
- **Description:** Profile all training stages
- **Requirements:**
  - Use `torch.profiler`
  - Identify bottlenecks (data loading, computation, communication)
  - Document findings
- **Deliverables:**
  - Profiling report
  - Optimization recommendations
- **Validation:**
  - GPU utilization > 85% during training

**Task 5.2: Hyperparameter Tuning** (4 days)
- **Owner:** Junior Dev 2 + Senior Dev
- **Description:** Run Hydra sweeps for key hyperparameters
- **Requirements:**
  - LAQ: beta, learning rate, codebook size
  - Foundation: learning rate, batch size
  - Use WandB sweeps for tracking
- **Deliverables:**
  - Sweep configs
  - Best hyperparameter sets
- **Validation:**
  - Improved metrics over baseline

**Task 5.3: Documentation** (3 days)
- **Owner:** All team
- **Description:** Write comprehensive documentation
- **Requirements:**
  - README with quickstart
  - API documentation (Sphinx/MkDocs)
  - Training guides
  - Troubleshooting section
- **Deliverables:**
  - Complete `docs/` directory
  - Published documentation site
- **Validation:**
  - New team member can train LAQ following docs

**Task 5.4: Testing & CI** (3 days)
- **Owner:** Junior Dev 1
- **Description:** Add unit tests and CI pipeline
- **Requirements:**
  - Unit tests for all modules (target: 80% coverage)
  - Integration tests for training scripts
  - GitHub Actions or GitLab CI
- **Deliverables:**
  - Complete `tests/` directory
  - CI configuration
- **Validation:**
  - All tests pass
  - CI runs on every commit

---

## Part 4: Technical Deep Dives

### 4.1 Debugging Multi-Node Training

**Common Issues:**

**Issue 1: Training Hangs at Initialization**
```
Symptoms:
- Processes spawn but no progress
- Logs show "Waiting for all processes..."

Causes:
- MASTER_ADDR/MASTER_PORT misconfigured
- Firewall blocking communication
- Wrong NCCL_SOCKET_IFNAME

Debug Steps:
1. Check Slurm variables:
   echo $SLURM_NODELIST
   echo $MASTER_ADDR
2. Test connectivity:
   srun --nodes=2 --ntasks=2 hostname
3. Verify NCCL:
   export NCCL_DEBUG=INFO
   (Look for "Using network" in logs)

Solution:
- Use: export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
- Verify: export NCCL_SOCKET_IFNAME=ib0  # InfiniBand
```

**Issue 2: OOM on Some GPUs**
```
Symptoms:
- "CUDA out of memory" on rank 2 but not rank 0

Causes:
- Unbalanced batch sizes
- FSDP wrapping incorrect
- Gradient accumulation not synchronized

Debug Steps:
1. Log memory per rank:
   torch.cuda.max_memory_allocated(device)
2. Verify batch sizes:
   print(f"Rank {rank}, batch size: {len(batch)}")
3. Check FSDP sharding:
   model.state_dict() should be empty on non-rank-0

Solution:
- Use DistributedSampler with drop_last=True
- Verify auto_wrap_policy is set correctly
```

**Issue 3: Slow Training (Low GPU Utilization)**
```
Symptoms:
- GPU utilization <40%
- Training slower than expected

Causes:
- Data loading bottleneck
- Too many checkpoints
- CPU preprocessing expensive

Debug Steps:
1. Profile dataloader:
   with torch.profiler.profile() as prof:
       for batch in dataloader:
           ...
   print(prof.key_averages())

2. Check I/O wait:
   iostat -x 1

3. Monitor CPU:
   htop (look for 100% CPU on workers)

Solution:
- Increase num_workers (8-16)
- Use WebDataset preprocessing
- Cache preprocessed data
- Reduce checkpoint frequency
```

### 4.2 Monitoring Checklist

**During Training, Monitor:**

**System Metrics:**
- GPU utilization (target: >85%)
- GPU memory usage (should be stable)
- CPU usage (workers should be busy but not saturated)
- Disk I/O (should be high during data loading, low otherwise)
- Network bandwidth (multi-node: should see sustained traffic)

**Training Metrics:**
- Loss (should decrease smoothly)
- Learning rate (should follow schedule)
- Gradient norms (should be stable, not exploding)
- Throughput (samples/sec, should be consistent)

**Model Metrics (LAQ):**
- Reconstruction PSNR/SSIM
- Codebook utilization (% embeddings used)
- Perplexity (entropy of latent distribution)

**Model Metrics (Foundation):**
- Latent action accuracy (top-1, top-3)
- Per-token accuracy (breakdown by position 0-3)

**Model Metrics (Finetuning):**
- Per-dimension action MSE
- Max/mean absolute error
- Prediction distribution (should match data distribution)

### 4.3 Checkpoint Strategy Details

**Multi-Level Checkpointing:**

**Fast Checkpoints (Local NVMe):**
```yaml
Purpose: Recover from node failures quickly
Frequency: Every 5-10 minutes (100-200 steps)
Location: /tmp/checkpoints/ (node-local SSD)
Retention: Keep last 3 only
Format: Full state dict (simple but large)

Implementation:
if step % cfg.checkpoint.fast.interval == 0:
    path = f"/tmp/checkpoint_step_{step}.pt"
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step
    }, path)
    # Cleanup old checkpoints
    if step > 300:
        os.remove(f"/tmp/checkpoint_step_{step-300}.pt")
```

**Durable Checkpoints (Shared Storage):**
```yaml
Purpose: Long-term recovery, reproducibility
Frequency: Every 1-2 hours (1000-2000 steps)
Location: /dss/.../checkpoints/
Retention: Keep last 10
Format: Distributed checkpoint (FSDP-sharded) for large models

Implementation:
if step % cfg.checkpoint.durable.interval == 0:
    path = f"{cfg.checkpoint.dir}/checkpoint_step_{step}/"
    # FSDP distributed checkpoint
    dist_checkpoint.save_state_dict(
        state_dict={'model': model.state_dict()},
        storage_writer=FileSystemWriter(path)
    )
```

**Milestone Checkpoints:**
```yaml
Purpose: Best models, epoch boundaries
Frequency: End of epoch, best validation score
Location: /dss/.../checkpoints/milestones/
Retention: Keep all
Format: Full state dict + metadata

Implementation:
if is_best_val_loss:
    path = f"{cfg.checkpoint.dir}/best_model.ckpt"
    fabric.save(path, {
        'model': model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'config': cfg
    })
```

***

## Part 5: Testing Strategy

### 5.1 Unit Tests

**Test Coverage Requirements:**

**LAQ Models (`tests/test_laq_models.py`):**
```python
def test_encoder_shapes():
    """Verify LAPA encoder output dimensions"""
    encoder = LAPAEncoder(dim=1024, spatial_depth=8, temporal_depth=8)
    x = torch.randn(2, 3, 2, 256, 256)  # Batch, channels, 2 frames, H, W
    first_tokens, last_tokens = encoder(x)
    assert first_tokens.shape == (2, 64, 1024)  # 64 patches per frame
    assert last_tokens.shape == (2, 64, 1024)

def test_nsvq_discreteness():
    """Verify NSVQ outputs discrete codes"""
    nsvq = NSVQ(dim=1024, quant_dim=32, codebook_size=8, code_seq_len=4)
    first = torch.randn(2, 64, 1024)
    last = torch.randn(2, 64, 1024)
    quantized, indices, perplexity = nsvq(first, last)
    assert indices.shape == (2, 4)  # 4 action tokens
    assert indices.min() >= 0 and indices.max() < 8
    assert indices.dtype == torch.long

def test_nsvq_delta_quantization():
    """Verify NSVQ quantizes delta, not absolute values"""
    nsvq = NSVQ(dim=1024, quant_dim=32, codebook_size=8, code_seq_len=4)
    first = torch.randn(2, 64, 1024)
    # Last is identical to first -> delta should be near zero
    last = first.clone()
    quantized, indices, _ = nsvq(first, last)
    # With zero delta, all positions might select same embedding
    # Just verify shape and range
    assert quantized.shape == (2, 1, 2, 2, 1024)

def test_nsvq_gradient_flow():
    """Verify gradients flow through noise-substitution STE"""
    nsvq = NSVQ(dim=1024, quant_dim=32, codebook_size=8, code_seq_len=4)
    first = torch.randn(2, 64, 1024, requires_grad=True)
    last = torch.randn(2, 64, 1024, requires_grad=True)
    quantized, _, _ = nsvq(first, last)
    loss = quantized.sum()
    loss.backward()
    assert first.grad is not None
    assert last.grad is not None

def test_decoder_cross_attention():
    """Verify decoder output shape and cross-attention"""
    decoder = LAPADecoder(dim=1024, depth=8, heads=16)
    context = torch.randn(2, 1, 64, 1024)  # First frame tokens
    actions = torch.randn(2, 1, 4, 1024)   # Action tokens
    out = decoder(context, actions)
    assert out.shape == (2, 3, 1, 256, 256)
    assert out.min() >= -1 and out.max() <= 1  # Tanh output
```

**Foundation Models (`tests/test_foundation_models.py`):**
```python
def test_vision_encoder_frozen():
    """Verify vision encoders are frozen"""
    vision = VisionEncoder()
    for param in vision.parameters():
        assert not param.requires_grad

def test_vla_forward():
    """Test end-to-end VLA forward pass"""
    model = VLAModel(config)
    images = torch.randn(2, 3, 224, 224)
    texts = ["pick up block", "open drawer"]
    logits = model(images, texts)
    assert logits.shape == (2, 4, 8)  # Batch, tokens, vocab

def test_action_dequantization():
    """Test action bins → continuous conversion"""
    bins = torch.tensor([[0, 128, 255, 64, 192, 32, 128]])
    stats = load_action_stats()
    continuous = dequantize_actions(bins, stats)
    assert continuous.shape == (1, 7)
    # Check values in expected range
    for dim in range(7):
        assert stats[dim]['min'] <= continuous[0, dim] <= stats[dim]['max']
```

**Data Pipeline (`tests/test_data.py`):**
```python
def test_webdataset_loading():
    """Test WebDataset can load and decode samples"""
    dataset = LAPADataModule(task='laq')
    dataset.setup('fit')
    loader = dataset.train_dataloader()
    batch = next(iter(loader))
    assert 'frame_t' in batch
    assert batch['frame_t'].shape[1:] == (3, 224, 224)

def test_distributed_sampling():
    """Verify no duplicate samples across workers"""
    # Simulate 2 workers
    samples_worker_0 = []
    samples_worker_1 = []
    # ... load samples ...
    assert len(set(samples_worker_0) & set(samples_worker_1)) == 0
```

### 5.2 Integration Tests

**Overfitting Test:**
```python
def test_lapa_overfit():
    """Verify LAPA can overfit small dataset"""
    # Create tiny dataset (10 samples of frame pairs)
    dataset = create_dummy_dataset(num_samples=10)
    
    # Train for 100 steps
    model = LAQTask(config)
    trainer = pl.Trainer(max_steps=100, overfit_batches=10)
    trainer.fit(model, dataset)
    
    # MSE loss should be near zero (no VQ losses to check)
    assert trainer.callback_metrics['train/loss_recon'] < 0.01
    # Codebook should be utilized
    assert trainer.callback_metrics['val/codebook_utilization'] > 0.5

def test_foundation_multinode():
    """Test foundation training runs on 2 nodes"""
    # Submit Slurm job
    result = subprocess.run([
        'sbatch', '--wait', '--nodes=2', 
        'slurm/test_multinode.sbatch'
    ])
    assert result.returncode == 0
    
    # Check logs for successful completion
    logs = read_slurm_logs()
    assert 'Training completed' in logs
    assert 'All ranks synchronized' in logs
```

### 5.3 Validation Tests

**Model Quality Tests:**
```python
def test_lapa_reconstruction_quality():
    """Verify LAPA reconstructions are reasonable"""
    model = LAQTask.load_from_checkpoint('laq_final.ckpt')
    test_frames = load_test_set()  # Frame pairs
    
    reconstructions = model.predict(test_frames)
    # Compare reconstruction to second frame
    target_frames = test_frames[:, :, 1:2, :, :]
    psnr = compute_psnr(target_frames, reconstructions)
    
    assert psnr > 22, f"PSNR too low: {psnr} (LAPA should achieve >22 dB)"

def test_foundation_accuracy():
    """Verify foundation model predictions"""
    model = VLAModel.load_from_checkpoint('vla_foundation.ckpt')
    test_data = load_test_set()
    
    predictions = model.predict(test_data)
    accuracy = (predictions == test_data['latent_tokens']).float().mean()
    
    assert accuracy > 0.6, f"Accuracy too low: {accuracy}"

def test_action_distribution():
    """Verify finetuned model outputs reasonable actions"""
    model = VLAModel.load_from_checkpoint('vla_finetuned.ckpt')
    test_data = load_test_set()
    
    predicted_actions = model.predict(test_data)
    
    # Check each dimension
    for dim in range(7):
        pred_mean = predicted_actions[:, dim].mean()
        true_mean = test_data['actions'][:, dim].mean()
        assert abs(pred_mean - true_mean) < 0.1
```

***

## Part 6: Handoff to Junior Developers

### 6.1 Task Assignment Template

**For Each Task, Provide:**

```markdown
## Task X.Y: [Task Name]

**Owner:** [Developer Name]
**Dependencies:** Tasks [list]
**Estimated Time:** X days
**Priority:** High/Medium/Low

### Objective
[1-2 sentences describing what needs to be built]

### Requirements
- [ ] Requirement 1
- [ ] Requirement 2
- [ ] ...

### Technical Specifications
[Detailed description or reference to section in this document]

### Deliverables
1. File: `path/to/file.py`
2. Tests: `tests/test_file.py`
3. Documentation: Updated `docs/section.md`

### Validation Criteria
- [ ] Unit tests pass
- [ ] Integration test: [specific test]
- [ ] Code review approved
- [ ] Metrics: [specific performance target]

### Resources
- Reference implementation: [link/section]
- API documentation: [link]
- Example: [code snippet or notebook]

### Getting Started
```
# Commands to set up environment
# Commands to run initial tests
```

### Common Pitfalls
- Watch out for [specific issue]
- Remember to [specific requirement]

### Questions?
Contact: [Tech Lead] on [Slack/Email]
```

### 6.2 Developer Onboarding Checklist

**Week 1: Environment Setup**
- [ ] LRZ account created, SSH access working
- [ ] Git repository cloned
- [ ] Conda environment created, dependencies installed
- [ ] Can run test job on LRZ (Hello World GPU script)
- [ ] WandB account connected
- [ ] Read architecture overview (this document)

**Week 2: Codebase Familiarization**
- [ ] Run existing unit tests
- [ ] Implement one small feature (e.g., new transform)
- [ ] Submit first PR, go through code review
- [ ] Pair programming session with senior dev

**Ongoing:**
- [ ] Attend daily standups
- [ ] Update task tracker (GitHub Issues/Jira)
- [ ] Ask questions early and often

### 6.3 Code Review Checklist

**Before Submitting PR:**
- [ ] Code follows project style guide (PEP 8, type hints)
- [ ] Unit tests added for new functionality
- [ ] All tests pass locally
- [ ] Docstrings added for public functions
- [ ] Configuration uses Hydra (no hardcoded values)
- [ ] Changes logged in CHANGELOG.md
- [ ] GPU memory usage tested (no leaks)

**Reviewer Checks:**
- [ ] Code is readable and maintainable
- [ ] No obvious performance issues
- [ ] Error handling is appropriate
- [ ] Tests cover edge cases
- [ ] Documentation is clear
- [ ] Backward compatibility maintained

***

## Part 7: Risk Management

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **NSVQ codebook collapse** | Medium | High | Monitor codebook utilization; LAPA's periodic replacement helps; verify delta computation |
| **Foundation OOM on multi-node** | Medium | High | Start with 1 node; test FSDP carefully; use gradient checkpointing |
| **Data loading bottleneck** | High | Medium | Profile early; use WebDataset; preprocess offline |
| **Multi-node hangs** | Medium | High | Test incrementally (1→2→4 nodes); use NCCL debug logs |
| **Poor reconstruction quality** | Low | Medium | Validate on simple dataset first; tune hyperparameters |
| **Action prediction inaccurate** | Medium | Medium | Increase training data; try different binning strategies |

### 7.2 Contingency Plans

**If LAPA doesn't learn:**
1. Reduce complexity (fewer transformer layers, smaller dim)
2. Verify delta quantization is working correctly (visualize deltas)
3. Try standard VQ-VAE as baseline to isolate NSVQ issues
4. Check spatial/temporal transformer separation is correct
5. Validate on simpler video dataset (e.g., moving MNIST)

**If Foundation training is too slow:**
1. Reduce model size (use Llama-2 3B instead of 7B)
2. Freeze more layers (keep vision frozen, freeze early Llama layers)
3. Use LoRA instead of full finetuning

**If LRZ quota exceeded:**
1. Clean up old checkpoints
2. Compress datasets (use lower JPEG quality)
3. Request additional storage allocation

**If team is behind schedule:**
1. Simplify Stage 3 (use Approach B: separate low-level policy)
2. Skip hyperparameter tuning initially
3. Use pre-trained foundation model (OpenVLA) instead of training from scratch

---

## Summary

This specification provides a complete implementation plan for **LAPA (Latent Action Pretraining from Videos)** using the transformer-based architecture from the original paper, adapted to modern infrastructure.

### What This Plan Provides:

1. **High-Level Decisions:** Monorepo, hybrid training framework (Lightning + Fabric), WebDataset for LRZ cluster, Enroot containers
2. **LAPA-Aligned Architecture:** 
   - Transformer-based encoder with spatial/temporal attention
   - NSVQ with delta quantization and single shared codebook
   - Cross-attention decoder
3. **Foundation & Finetuning:** Dual vision encoders + Llama-2 7B with FSDP
4. **Implementation Details:** Complete FSDP config, data pipelines, loss functions, training loops
5. **Phased Task Breakdown:** 5 phases, 50+ tasks, ~12 weeks
6. **Testing Strategy:** Unit tests, integration tests, validation criteria
7. **Operational Guides:** Multi-node debugging, monitoring, checkpointing, risk mitigation

### Key Architectural Highlights:

- ✅ **LAPA Stage 1**: Transformer encoder + NSVQ (not conv-based VQ-VAE)
- ✅ **Delta Quantization**: Quantizes frame changes, not absolute features
- ✅ **Simplified Loss**: MSE only (no VQ-specific losses)
- ✅ **Modern Infrastructure**: Hydra configs, Lightning ecosystem, cluster-optimized data loading

### Next Steps for Tech Lead:

1. **Review** configs in `config/model/laq.yaml` - adjust `dim`, `spatial_depth`, `temporal_depth` if needed
2. **Assign Phase 0 tasks** to team (infrastructure setup)
3. **Setup monitoring**: Create WandB project, configure metric tracking
4. **Begin implementation**: Task 0.1 (repository setup)
5. **Key milestone**: End of Week 4 - trained LAPA model with >22 dB PSNR

### Implementation Priority:

**Week 1-2**: Core transformer components (`attention.py`, `nsvq.py`)  
**Week 3-4**: LAPA integration and Stage 1 training  
**Week 5-8**: Foundation policy training  
**Week 9-10**: Action finetuning  
**Week 11-12**: Evaluation and optimization  

This document serves as the **single source of truth** for the LAPA implementation. Update it as decisions change, and refer all developers to relevant sections for their tasks.