# LAPA Project: Complete Technical Specification & Implementation Plan

## Executive Summary

This document provides a **complete technical blueprint** for implementing LAPA (Latent Action Pretraining from Videos), a three-stage robot learning system. It covers architectural decisions, implementation details, and a phased task breakdown suitable for distribution to junior developers.

**System Overview:** LAPA learns robot policies from videos without action labels through:
1. **Stage 1 (LAQ)**: VQ-VAE that compresses frame-to-frame transitions into discrete latent codes
2. **Stage 2 (Foundation)**: 7B Vision-Language model that predicts latent actions from image + text
3. **Stage 3 (Finetuning)**: Adapts the foundation model to output continuous robot commands

**Infrastructure:** LRZ cluster (H100 GPUs, GPFS storage, Slurm scheduler)

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
| **Stage 1 (LAQ)** | PyTorch Lightning | Standard supervised learning; auto-DDP, checkpointing, logging |
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

#### 2.1.1 Model Architecture

**Encoder (Frame Pair → Latent Embeddings):**
```
Input: Concatenated frames [B, 6, 224, 224]  # (frame_t | frame_{t+1})

Architecture:
├─ Conv2D(6 → 64, kernel=4, stride=2, padding=1)    # → [B, 64, 112, 112]
├─ ResBlock(64)  × 2
├─ Conv2D(64 → 128, kernel=4, stride=2, padding=1)  # → [B, 128, 56, 56]
├─ ResBlock(128) × 2
├─ Conv2D(128 → 256, kernel=4, stride=2, padding=1) # → [B, 256, 28, 28]
├─ ResBlock(256) × 2
├─ Conv2D(256 → 512, kernel=4, stride=2, padding=1) # → [B, 512, 14, 14]
├─ ResBlock(512) × 2
└─ Conv2D(512 → latent_dim, kernel=3, stride=1, padding=1)

Output: [B, latent_dim, 14, 14]
```

**ResBlock Structure:**
```
Input: [B, C, H, W]
├─ GroupNorm(32 groups)
├─ SiLU activation
├─ Conv2D(C → C, kernel=3, stride=1, padding=1)
├─ GroupNorm(32 groups)
├─ SiLU activation
└─ Conv2D(C → C, kernel=3, stride=1, padding=1) + Skip connection
Output: [B, C, H, W]
```

**Vector Quantizer (Critical Component):**
```
Input: [B, latent_dim, 14, 14]

Parameters:
- num_tokens: 4           # Action sequence length
- vocab_size: 8           # Embeddings per token position
- embedding_dim: 256      # Dimension of each embedding
- beta: 0.25              # Commitment loss weight

Codebook: [4, 8, 256]     # 4 positions × 8 embeddings × 256 dims

Process:
1. Spatial pooling: [B, latent_dim, 14, 14] → [B, num_tokens, embedding_dim]
   Options:
   - Learned linear projection
   - Adaptive average pooling
   - Attention pooling

2. For each token position (0-3):
   a. Compute distances to codebook[position]: [B, 8]
   b. Select nearest: indices[position] ∈ {0,1,2,3,4,5,6,7}
   c. Replace with embedding: e = codebook[position, indices[position]]

3. Straight-through estimator:
   quantized = continuous + stop_gradient(embedding - continuous)

Output: 
- quantized: [B, num_tokens, embedding_dim]
- indices: [B, num_tokens]  # The discrete latent action
- losses: {codebook_loss, commitment_loss}
```

**Loss Components:**
```
1. Reconstruction Loss:
   L_recon = MSE(decoder(quantized), frame_{t+1})

2. Codebook Loss (update embeddings):
   L_codebook = ||stop_gradient(z_continuous) - embedding||²

3. Commitment Loss (commit encoder to codebook):
   L_commit = β × ||z_continuous - stop_gradient(embedding)||²

Total Loss:
   L_total = L_recon + L_codebook + β × L_commit
```

**Decoder (Quantized → Reconstructed Frame):**
```
Input: [B, num_tokens, embedding_dim]

Architecture:
├─ Reshape/Upsample: [B, num_tokens, embedding_dim] → [B, 512, 14, 14]
├─ ConvTranspose2D(512 → 256, kernel=4, stride=2)   # → [B, 256, 28, 28]
├─ ResBlock(256) × 2
├─ ConvTranspose2D(256 → 128, kernel=4, stride=2)   # → [B, 128, 56, 56]
├─ ResBlock(128) × 2
├─ ConvTranspose2D(128 → 64, kernel=4, stride=2)    # → [B, 64, 112, 112]
├─ ResBlock(64) × 2
├─ ConvTranspose2D(64 → 32, kernel=4, stride=2)     # → [B, 32, 224, 224]
└─ Conv2D(32 → 3, kernel=3, padding=1) + Tanh       # → [B, 3, 224, 224]

Output: Reconstructed frame [B, 3, 224, 224]
```

#### 2.1.2 Data Pipeline

**Preprocessing Script (`scripts/1_videos_to_webdataset.py`):**
```
Input: Raw video files or image sequences

Process:
1. For each video:
   - Extract frames at 10 fps (or native framerate)
   - Generate consecutive pairs: (frame_t, frame_{t+1})
   - Resize to 256×256, center crop to 224×224
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
    - For LAQ: {frame_t: [B,3,224,224], frame_t1: [B,3,224,224]}
    - For Foundation: {image: [B,3,224,224], text: List[str], latents: [B,4]}
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
  name: laq_base
  
  encoder:
    in_channels: 6
    base_channels: 64
    channel_multipliers: [1, 2, 4, 8]  # [64, 128, 256, 512]
    num_res_blocks: 2
    latent_dim: 256
  
  quantizer:
    num_tokens: 4
    vocab_size: 8
    embedding_dim: 256
    beta: 0.25
    ema_decay: 0.99  # Optional: EMA for codebook updates
  
  decoder:
    latent_dim: 256
    base_channels: 32
    channel_multipliers: [1, 2, 4, 8]
    num_res_blocks: 2
    out_channels: 3
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
  
  loss_weights:
    reconstruction: 1.0
    vq: 1.0  # Codebook + commitment combined
  
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

__init__(encoder_cfg, decoder_cfg, quantizer_cfg, optimizer_cfg):
  - Initialize encoder, decoder, quantizer
  - Store configs for optimizer/scheduler

forward(frame_t, frame_t1):
  1. Concatenate: [B,3,H,W] + [B,3,H,W] → [B,6,H,W]
  2. Encode: continuous_latents = encoder(concat)
  3. Quantize: quantized, indices, vq_losses = quantizer(continuous_latents)
  4. Decode: reconstructed = decoder(quantized)
  5. Return: reconstructed, indices, vq_losses

training_step(batch, batch_idx):
  1. Forward pass
  2. Compute reconstruction loss: MSE(reconstructed, frame_t1)
  3. Compute total loss: recon + vq_losses['codebook'] + vq_losses['commitment']
  4. Log all losses
  5. Return total_loss

validation_step(batch, batch_idx):
  1. Same as training_step
  2. If batch_idx == 0: log reconstruction visualizations

predict_step(batch, batch_idx):
  # Used by script 3 (latent label generation)
  1. Forward pass
  2. Return indices only (the discrete latent codes)

configure_optimizers():
  1. Create optimizer from config
  2. Create scheduler with warmup
  3. Return {optimizer, scheduler}
```

**Metrics to Track:**
```
Training:
- train/loss_total
- train/loss_reconstruction
- train/loss_codebook
- train/loss_commitment
- train/perplexity (codebook usage diversity)

Validation:
- val/loss_total
- val/loss_reconstruction
- val/codebook_utilization (% of embeddings used)
- val/reconstruction_images (WandB grid)
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

### Phase 1: LAQ Implementation (Weeks 2-4)

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

**Task 1.3: Encoder Network** (4 days)
- **Owner:** Senior Dev
- **Description:** Implement encoder architecture (Section 2.1.1)
- **Requirements:**
  - Modular ResBlock implementation
  - Configurable channel dimensions via Hydra
  - Support BF16 precision
- **Deliverables:**
  - `laq/models/encoder.py`
  - Unit tests (shape consistency, gradient flow)
- **Validation:**
  - Input  → Output[1][2][3]
  - Forward + backward passes succeed

**Task 1.4: Vector Quantizer** (5 days)
- **Owner:** Senior Dev
- **Description:** Implement VQ-VAE quantization layer (Section 2.1.1)
- **Requirements:**
  - Codebook management (4 positions × 8 embeddings)
  - Straight-through estimator
  - Compute codebook + commitment losses
  - Optional: EMA updates
- **Deliverables:**
  - `laq/models/quantizer.py`
  - Unit tests (quantization, loss computation)
- **Validation:**
  - Codebook utilization increases over dummy training
  - Commitment loss decreases

**Task 1.5: Decoder Network** (3 days)
- **Owner:** Junior Dev 2
- **Description:** Implement decoder (Section 2.1.1)
- **Requirements:**
  - Mirror encoder architecture (transposed convs)
  - Output normalized images
- **Deliverables:**
  - `laq/models/decoder.py`
  - Unit tests
- **Validation:**
  - Input  → Output[3][4][1]

**Task 1.6: LAQ Lightning Module** (4 days)
- **Owner:** Senior Dev
- **Description:** Implement `LAQTask` (Section 2.1.4)
- **Requirements:**
  - Wire encoder, quantizer, decoder
  - Implement training/validation steps
  - Add reconstruction visualization callback
- **Deliverables:**
  - `laq/task.py`
  - Integration test (overfit 10 samples)
- **Validation:**
  - Loss decreases on tiny dataset
  - Reconstructions logged to WandB

**Task 1.7: Training Script** (2 days)
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

**Task 1.8: Full LAQ Training** (5 days)
- **Owner:** All team
- **Description:** Train LAQ on full dataset (OpenX or STHv2)
- **Requirements:**
  - Use 1-2 nodes (4-8 H100s)
  - Monitor training curves
  - Validate reconstruction quality
- **Deliverables:**
  - Trained LAQ checkpoint (`laq_final.ckpt`)
  - Training report (losses, codebook usage, example reconstructions)
- **Success Criteria:**
  - Reconstruction PSNR > 20 dB
  - Codebook utilization > 80%

**Task 1.9: Latent Label Generation** (3 days)
- **Owner:** Junior Dev 2
- **Description:** Implement `scripts/3_generate_latent_labels.py`
- **Requirements:**
  - Load LAQ checkpoint
  - Run inference on full dataset
  - Save new WebDataset with latent labels
- **Deliverables:**
  - Label generation script
  - New dataset: `/dss/.../datasets/openx_latent_labeled/`
- **Validation:**
  - Load random samples, verify latent codes ∈[5]
  - Check file sizes (should be smaller than original)

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
    """Verify encoder output dimensions"""
    encoder = Encoder(in_channels=6, latent_dim=256)
    x = torch.randn(2, 6, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 256, 14, 14)

def test_quantizer_discreteness():
    """Verify quantizer outputs discrete codes"""
    quantizer = VectorQuantizer(num_tokens=4, vocab_size=8)
    z = torch.randn(2, 4, 256)
    quantized, indices, losses = quantizer(z)
    assert indices.min() >= 0 and indices.max() < 8
    assert indices.dtype == torch.long

def test_quantizer_gradient_flow():
    """Verify gradients flow through straight-through estimator"""
    quantizer = VectorQuantizer(num_tokens=4, vocab_size=8)
    z = torch.randn(2, 4, 256, requires_grad=True)
    quantized, _, _ = quantizer(z)
    loss = quantized.sum()
    loss.backward()
    assert z.grad is not None

def test_decoder_reconstruction():
    """Verify decoder output shape and range"""
    decoder = Decoder(latent_dim=256, out_channels=3)
    z = torch.randn(2, 4, 256)
    out = decoder(z)
    assert out.shape == (2, 3, 224, 224)
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
def test_laq_overfit():
    """Verify LAQ can overfit small dataset"""
    # Create tiny dataset (10 samples)
    dataset = create_dummy_dataset(num_samples=10)
    
    # Train for 100 steps
    model = LAQTask(config)
    trainer = pl.Trainer(max_steps=100, overfit_batches=10)
    trainer.fit(model, dataset)
    
    # Loss should be near zero
    assert trainer.callback_metrics['train/loss'] < 0.01

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
def test_laq_reconstruction_quality():
    """Verify LAQ reconstructions are reasonable"""
    model = LAQTask.load_from_checkpoint('laq_final.ckpt')
    test_images = load_test_set()
    
    reconstructions = model.predict(test_images)
    psnr = compute_psnr(test_images, reconstructions)
    
    assert psnr > 20, f"PSNR too low: {psnr}"

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
| **LAQ codebook collapse** | Medium | High | Monitor codebook utilization; add regularization; try EMA updates |
| **Foundation OOM on multi-node** | Medium | High | Start with 1 node; test FSDP carefully; use gradient checkpointing |
| **Data loading bottleneck** | High | Medium | Profile early; use WebDataset; preprocess offline |
| **Multi-node hangs** | Medium | High | Test incrementally (1→2→4 nodes); use NCCL debug logs |
| **Poor reconstruction quality** | Low | Medium | Validate on simple dataset first; tune hyperparameters |
| **Action prediction inaccurate** | Medium | Medium | Increase training data; try different binning strategies |

### 7.2 Contingency Plans

**If LAQ doesn't learn:**
1. Reduce complexity (smaller encoder, fewer tokens)
2. Try deterministic encoder (no quantization) as baseline
3. Validate on standard VQ-VAE dataset (e.g., CIFAR-10)

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

This specification provides:

1. **High-Level Decisions:** Monorepo, hybrid training framework, WebDataset, Enroot containers
2. **Detailed Architectures:** LAQ (encoder/quantizer/decoder), Foundation (vision/LLM/action head), Finetuning
3. **Implementation Details:** FSDP config, data pipelines, loss functions, training loops
4. **Phased Task Breakdown:** 5 phases, 50+ tasks, ~12 weeks
5. **Testing Strategy:** Unit tests, integration tests, validation criteria
6. **Operational Guides:** Debugging, monitoring, checkpointing, risk mitigation

**Next Steps for Tech Lead:**
1. Review and customize configs for your specific dataset/cluster
2. Assign Phase 0 tasks to team
3. Setup weekly milestones and check-ins
4. Begin with Task 0.1 (repository setup)

This document should serve as the **single source of truth** for the implementation. Update it as decisions change, and refer all developers to relevant sections for their tasks.