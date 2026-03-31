This code is build on [LAPA](https://github.com/LatentActionPretraining/LAPA).

# Robot-learning from Video


A three-stage robot learning system that learns policies from videos without action labels.

**Three Training Stages:**
1. **Stage 1 (LAM)**: VQ-VAE compressing frame-to-frame transitions into discrete latent codes
2. **Stage 2 (Policy)**: Robot policy model predicting latent actions from images + text
3. **Stage 3 (Finetuning)**: Adapting the policy model to output continuous robot commands

## Performance and resource efficiency
Performance benchmark of the training pipeline on an Desktop Nvidia 5090 GPU. With the current setup we achieve full GPU utilization.

For performance reasons we use a LeRobot v3-based dataset pipeline with weighted multi-source mixing.

<p align="center">
  <img src="docs/assets/lam_gpu_utilization.png" alt="LAM training GPU utilization" width="700">
</p>

## Deployment

For robot teleoperation and policy deployment, we use our [CRISP Controller](https://utiasdsl.github.io/crisp_controllers/) setup:
<img width="830" height="458" alt="image" src="https://github.com/user-attachments/assets/8a4fe7b9-691f-4c27-b92b-4bf3062fcb92" />


Here is a dataset view of a human demonstration:

https://github.com/user-attachments/assets/c85271ec-d35f-4310-ac96-08a0e8d20476

Here is a view of a policy deployed to Lego stacking:

https://github.com/user-attachments/assets/0cc694b1-ff8c-406d-b5a7-a6815fe8c1af

## Repository Structure

```
├── packages/              # Installable Python packages
│   ├── common/           # Shared utilities, logging, data interfaces
│   ├── lam/              # Stage 1: Latent action model (VQ-VAE)
│   ├── stage2/           # Stage 2: Robot policy model + validation
├── config/               # Hydra configurations (modular, composable)
│   ├── experiment/       # Complete experiment setups
│   ├── model/, data/, training/, cluster/  # Config components
├── scripts/              # Training entry points (numbered by stage)
├── tests/                # Unit and integration tests
├── lerobot_policy_hlrp / # Definition of the installable lerobot policy to be used with the lerobot library
├── docs/                 # Documentation (LRZ workflow guide)
└── containers/           # Enroot/Docker definitions for LRZ
```

## Getting Started

### Clone the repository

This repo uses a git submodule for `lerobot/`.
A plain `git clone` does not fetch submodules automatically.

Recommended clone:

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

After pulling new parent-repo commits, refresh the submodule with:

```bash
git pull
git submodule update --init --recursive
```

### Installation

The default LAM config uses gated DINOv3 weights (`facebook/dinov3-vits16-pretrain-lvd1689m`).
Before running Stage 1 or DINO-backed tests:

1. Request access to the DINOv3 model on Hugging Face.
2. Log in locally with `huggingface-cli login` (or `python -m huggingface_hub login`).
3. You might need to provide the token via a symlink or an environment variable to be able to use dino:
```bash 
mkdir -p ~/.huggingface
ln -s ~/.cache/huggingface/token ~/.huggingface/token
```

```bash
# Create a fresh conda environment
conda create -n hlrp python=3.12
conda activate hlrp

# Install the current PyTorch build with GPU support for your system.
# Pick the right command for your CUDA/runtime from:
# https://pytorch.org/get-started/locally/
# Validated local example for CUDA 12.8:
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# LeRobot video decoding expects torchcodec plus FFmpeg runtime libraries.
pip install torchcodec==0.10.0
conda install -n hlrp -c conda-forge ffmpeg -y

# Install local LeRobot + HLRP overlay dependencies from the repo.
# LeRobot's own dependencies are taken from lerobot/pyproject.toml.
pip install -r requirements.unified.local.txt

# Quick sanity check
python -m pytest
```

#### PyTorch / CUDA troubleshooting

On some newer GPUs, PyTorch may detect CUDA but still fail at runtime because the
installed build does not support the GPU's SM architecture yet. Typical symptoms
are warnings about unsupported compute capability such as `sm_120` or runtime
errors like `CUDA error: no kernel image is available for execution on the device`.

If that happens, reinstall PyTorch with a newer **stable** CUDA build from the
official selector. Avoid nightly builds unless you explicitly need them and can
also align all video decoding dependencies.

For this repo, if you use `torchvision>=0.24` then `torchvision.io.VideoReader`
may be unavailable; LeRobot video decoding then requires a `torchcodec` build
compatible with your exact PyTorch version.

Recommended stable reinstall pattern:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
pip install torchcodec==0.10.0
conda install -n hlrp -c conda-forge ffmpeg -y
```

If `torchcodec` fails to import with errors about missing libraries like
`libavutil.so.*` / `libavcodec.so.*`, FFmpeg runtime libraries are missing in
the environment. Install FFmpeg as shown above.

Known-working nightly matrix (example used by collaborators):

```bash
pip uninstall -y torch torchvision torchaudio torchcodec
pip install --pre torch==2.11.0.dev20260121+cu128 torchvision==0.25.0.dev20260121+cu128 torchaudio==2.11.0.dev20260122+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
pip install torchcodec==0.5
conda install -n hlrp -c conda-forge ffmpeg -y
```

Important: keep `torch`, `torchvision`, `torchaudio`, and `torchcodec` aligned.
The currently validated local/container stack is:
- `torch==2.10.0`
- `torchvision==0.25.0`
- `torchaudio==2.10.0`
- `torchcodec==0.10.0`

Mixing arbitrary nightly dates with unrelated `torchcodec` wheels can break decoding.

Check the official PyTorch install selector for the current recommended command:
https://pytorch.org/get-started/locally/

### Running Training and Rollout

```bash
# Stage 1 (LAM) local
python scripts/2_train_stage1_lam.py experiment=stage1_local

# Stage 1 (LAM) local low-memory preset
python scripts/2_train_stage1_lam.py experiment=stage1_local_lowmem

# Stage 2 (policy SmolVLA flow) local
python scripts/4_train_stage2_policy.py experiment=stage2_local

# Stage 3 (LeRobot action-only) local
python scripts/6_train_lerobot.py experiment=stage3_local

# Stage 3 rollout/eval local
python scripts/7_rollout_lerobot.py \
  experiment=stage3_rollout_local \
  lerobot_eval.policy_path=/mnt/data/.../lerobot/checkpoints/last/pretrained_model

# Cluster submit (all stages)
python scripts/submit_job.py experiment=stage3_cluster cluster=lrz_x100
```

For Stage 3 rollout, `lerobot_eval.policy_path` must point to the exported
`pretrained_model` directory inside a saved checkpoint, not to the checkpoint
root itself.

## Configuration

Uses [Hydra](https://hydra.cc) for composable configuration. Experiments compose modular config components:

```bash
# Override from CLI
python scripts/2_train_stage1_lam.py experiment=stage1_local data.loader.batch_size=32 training.optimizer.lr=5e-5
```

Machine-specific paths should go into `config/user_config/local.yaml`, not into shared experiment or profile configs.
That file is gitignored and is the right place for things like:
- local storage roots
- cache locations
- long-lived checkpoint paths you reuse across many runs, such as a preferred Stage 1 LAM checkpoint for Stage 3 multitask training

Start from `config/user_config/local.yaml.example` if you need to create or update your local file.

### Filtering Config Locations

Action-frame filtering is configured separately for Stage 1/2 and Stage 3.

- Stage 1/2 filtering lives in data configs under `config/data/*.yaml`.
- Example: `config/data/lsy_teleop_only.yaml` -> `data.filtering`.
- Stage 3 filtering lives in stage-3 dataset configs under `config/stage3_dataset/*.yaml`.
- Example: `config/stage3_dataset/lsy_teleop_full_multitask.yaml` -> `lerobot.dataset.filtering`.
- Stage 3 source list lives in `config/stage3_dataset_mix/*.yaml` and can optionally override filtering per source via `sources[i].filtering`.
- Merge order at runtime is: Stage 3 global dataset filtering, then per-source override (if present).

Implication: Stage 3 does not automatically inherit Stage 1 filtering values unless you set matching values explicitly.

For filtering semantics and cache/debugging details, see `docs/action_frame_filtering.md`.

## Dataset Sources and Downloads

- Optional predownload step for Stage 1/2:
  - `python scripts/1_download_configured_datasets.py experiment=stage1_local`
  - This script is optional. Stage 1/2 can also download on demand through the normal LeRobot/Hugging Face cache flow.
  - Use it when you want to prefetch the LeRobot-v3 sources selected by your Hydra config before starting training.
- Custom dataset root for script 1:
  - Override with `download.root=/your/path`.
  - If `download.root` is unset, script 1 writes into `HF_LEROBOT_HOME` (normally derived from `paths.cache_dir`).
  - This root stores the predownloaded LeRobot-v3 dataset contents for each configured `repo_id`, organized under `<root>/<owner>/<dataset>/`.
  - Each dataset directory contains the downloaded `meta/`, `data/`, and `videos/` trees used by the LeRobot-v3 loader.
- Workstation/local training usually uses local dataset paths:
  - Stage 1/2 LeRobot-v3 mix via Hydra data config (default in `stage1_local` / `stage2_local`)
  - LeRobot cache root defaults to `${HF_LEROBOT_HOME}` (set explicitly when needed)
  - Stage 3 Libero snapshot: `/mnt/data/workspace/hflibero/datasets--HuggingFaceVLA--libero/snapshots/<snapshot>`
- Cluster training:
  - Stage 1/2 uses LeRobot datasets and Hugging Face cache on cluster storage.
  - Stage 3 can download Libero at run start if `lerobot.dataset.root=null` and `lerobot.dataset.repo_id=HuggingFaceVLA/libero`.
- Runtime-downloaded datasets/assets:
  - `HuggingFaceVLA/libero` dataset cache (`HF_DATASETS_CACHE`)
  - LIBERO environment assets (`~/.cache/libero/assets`) if missing
  - Hugging Face model/checkpoint cache (`HF_HUB_CACHE`)
  - By default, `paths.cache_dir` is also where `HF_LEROBOT_HOME` lives, so it holds LeRobot dataset cache data unless you override `download.root` for script 1.

## Development

```bash
# Run tests
python -m pytest tests/

# Fast config sanity check
python -m pytest -q tests/test_hydra_configs.py

# Format code
python -m black packages/ scripts/ tests/

# Lint
python -m ruff check packages/ scripts/ tests/
```

## Agents

- The repo is generally designed for smooth agent usage and comes with AGENTS.md Codex CLI and CLAUDE.md for Claude Code. Generally the preferred agent is codex. In .codex skills and agents for various tasks can be found.

## Dependencies

Python 3.12. Install a current GPU-enabled PyTorch build from the official PyTorch selector, then install the repo overlay from `requirements.unified.local.txt`. The local `lerobot/` submodule is installed directly, so its dependencies come from `lerobot/pyproject.toml` rather than being duplicated in a top-level requirements file.

Key packages: lightning, transformers, webdataset, hydra-core, wandb, accelerate.



## Citation

```bibtex
@misc{hlrp2024,
  title={V-PRO: Video Pretraining for Robot Action Policies},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details
