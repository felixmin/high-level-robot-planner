# Stage 2 (Foundation): Cosmos-Reason2 (Qwen3-VL) + Action Tokens

This repo now contains a **Stage 2 “Foundation VLA” prototype** that predicts **discrete latent actions** from **(image, language)** using a **Qwen3-VL backbone loaded from Cosmos-Reason2 weights**, trained with standard LM loss on a constrained set of **action tokens**.

The high-level idea mirrors LAPA Stage 2, but implemented in PyTorch/Lightning and integrated with the existing OpenX/OXE pipeline in this repo.

## What we implemented (high level)

**Approach A: “latent actions as special tokens”**
- We represent latent actions as a fixed-length token sequence:
  - wrappers: `<ACTION> ... </ACTION>`
  - codes: `<ACT_0> ... <ACT_{K-1}>`
- The VLA model is trained by **prompt-masked cross-entropy**:
  - prompt = image + instruction (masked with `-100`)
  - completion = action tokens (loss computed only here)
- Labels are generated **online** during training by a **frozen LAQ model**:
  - LAQ encodes `(frame_t, frame_{t+offset}) → discrete code indices [B, code_seq_len]`
  - VLA learns to predict those indices from `(frame_t, instruction)` only.

**Constrained decoding for validation**
- During validation we run `generate()` and constrain outputs to follow:
  - `<ACTION> <ACT_*> x code_seq_len </ACTION> <eos>`
- We log a simple token-level accuracy (`val/token_accuracy`) on the first validation batch.

## Key design decisions (things not obvious from code)

- **Prompt masking must use the *processor with images***:
  - For Qwen3-VL, tokenizer-only prompt lengths can be wrong because multimodal tokenization depends on images.
  - We compute prompt lengths by running the **processor** on prompt-only inputs and using `attention_mask.sum()`.
- **No backwards-compat loading for LAQ checkpoints**:
  - Stage 2 expects a LAQ checkpoint that matches the current `LAQTask` module structure.
  - If you have older LAQ checkpoints, they may fail to load (missing/unexpected keys). Use a fresh LAQ run from this codebase.
- **Incremental development + tests first**:
  - We built small, unit-testable utilities (token formatting, masking logic, constrained decoding) before wiring real HF models and the local indexed-full OpenX path.

## Where the pieces live

- Token vocabulary + formatting: `packages/foundation/action_tokens.py`
- VLM-safe prompt masking / batch building:
  - `packages/foundation/vla_inputs.py`
- OXE → LAQ input conversion + LAQ code provider:
  - `packages/foundation/online_laq.py`
- Constrained decoding helper:
  - `packages/foundation/constrained_decode.py`
- OXE frame tensor → PIL images for Qwen processor:
  - `packages/foundation/image_adapters.py`
- LightningModule for Stage 2 (ties everything together):
  - `packages/foundation/vla_module.py`
- Qwen token-resize setup:
  - `packages/foundation/qwen3vl_setup.py`
- Training entrypoint:
  - `scripts/4_train_foundation.py`
- Hydra configs:
  - model: `config/model/foundation_cosmos2_tokens.yaml`
  - experiment (debug): `config/experiment/vla_cosmos2_tokens_debug.yaml`
  - training (debug): `config/training/vla_lightning_debug.yaml`

## How to run (local smoke run)

Prereqs:
- A **LAQ checkpoint** trained with this repo’s current LAQ code (Stage 1).
- Ability to stream OpenX/OXE data locally (or adjust `cfg.data.*` to whatever you can access).
- Ability to download HF weights if not cached:
  - `nvidia/Cosmos-Reason2-2B`

Command (example):
```bash
conda activate hlrp
python scripts/4_train_foundation.py \
  experiment=vla_cosmos2_tokens_debug \
  model.laq.checkpoint=/abs/path/to/laq.ckpt \
  data.loader.batch_size=2 \
  data.loader.num_workers=0
```

What to look for:
- It should print the Hydra config, then start training.
- Metrics: `train/loss`, `val/loss`, and (on first val batch) `val/token_accuracy`.

## Tests and how to iterate safely

Run fast unit tests (no HF downloads required):
```bash
conda activate hlrp
pytest -q tests/test_action_tokens.py \
         tests/test_vla_inputs_prompt_mask.py \
         tests/test_online_laq.py \
         tests/test_constrained_decode.py \
         tests/test_vla_module_cpu.py \
         tests/test_vla_generation_metrics.py \
         tests/test_image_adapters.py
```

These tests protect the tricky parts:
- correct token formatting
- correct prompt masking (processor-based, padding-safe)
- correct OXE→LAQ tensor layout conversion
- constrained decoding state machine behavior

## Current limitation / next steps

1) **Ensure you have a compatible LAQ checkpoint**
   - If Stage 2 fails loading LAQ, retrain LAQ (Stage 1) with the current code and point `model.laq.checkpoint` to the new `.ckpt`.
2) **First real local run**
   - Confirm local indexed-full OpenX data works, Cosmos weights download/caching works, and a few steps complete.
3) **Scale-up**
   - Add PEFT/LoRA knobs (if desired), tune batch size/accumulation, and then move to H100 via your preferred submission workflow (`submit_job.py`).
