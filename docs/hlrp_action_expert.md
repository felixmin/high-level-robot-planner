# HLRP Action Expert

## Purpose

This document defines what an "HLRP action expert" should look like by comparing three existing families:

- PI-style action expert (`pi0`)
- SmolVLA action expert (`smolvla`)
- Standard diffusion policy (`diffusion`)

The goal is to keep the strong language/vision conditioning of VLA models while retaining stable iterative action denoising.

---

## Are PI and Diffusion Similar?

Short answer: **partly**.

Common structure:

- Both generate action chunks iteratively from noise.
- Both condition on observation context.
- Both use receding-horizon style chunk execution.

Key difference:

- **Diffusion policy**: denoiser is usually a conditional UNet/Transformer over action trajectories.
- **PI/SmolVLA**: denoiser is an "action expert" attached to a VLM (language/image backbone), with explicit token-level fusion.

So they are in the same denoising family, but their conditioning and backbone wiring are different.

---

## Abstract Architecture Visuals

### 1) PI-style Action Expert

```text
task text -> tokenizer -> language tokens --\
images --------------------------------------> VLM prefix embeddings --\
state + noisy actions + time ----------------> expert suffix embeddings -> joint transformer pass -> v_t -> iterative update -> action chunk
```

### 2) SmolVLA-style Action Expert

```text
task text -> tokenizer -> language tokens --\
images --------------------------------------> compact VLM features --\
state ---------------------------------------> state token -----------\
noisy actions + time -------------------------------------------------> compact action expert (self/cross attention) -> v_t -> iterative update -> action chunk
```

### 3) Standard Diffusion Policy

```text
obs(state,image,env) -> obs encoder -> global_cond
noise trajectory x_T -> conditional denoiser (UNet/Transformer) + scheduler steps -> denoised actions -> action chunk
```

---

## Language Conditioning Comparison

## PI (`pi0`)

- Language is tokenized in preprocessor with `TokenizerProcessorStep`.
- Tokens are embedded with `embed_language_tokens`.
- Language embeddings are fused in prefix context before action denoising.

Main call path:

1. `make_pi0_pre_post_processors`
2. `TokenizerProcessorStep`
3. `PI0Policy.predict_action_chunk` / `PI0Policy.forward`
4. `PI0Pytorch.sample_actions` / `PI0Pytorch.forward`
5. `PI0Pytorch.embed_prefix`
6. `PaliGemmaWithExpertModel.embed_language_tokens`
7. `PaliGemmaWithExpertModel.forward`
8. `PI0Pytorch.denoise_step`

---

## SmolVLA (`smolvla`)

- Language is tokenized in preprocessor with `TokenizerProcessorStep`.
- Tokens are embedded via `SmolVLMWithExpertModel.embed_language_tokens`.
- Language, image, and state are fused into prefix sequence.
- Action expert consumes prefix cache + noisy action suffix.

Main call path:

1. `make_smolvla_pre_post_processors`
2. `TokenizerProcessorStep`
3. `SmolVLAPolicy.predict_action_chunk` / `SmolVLAPolicy.forward`
4. `VLAFlowMatching.sample_actions` / `VLAFlowMatching.forward`
5. `VLAFlowMatching.embed_prefix`
6. `SmolVLMWithExpertModel.embed_language_tokens`
7. `SmolVLMWithExpertModel.forward`
8. `VLAFlowMatching.denoise_step`

---

## Diffusion (`diffusion`)

- Current LeRobot diffusion implementation does **not** include language tokenization or language encoder path.
- Conditioning is currently state/image/env only and flattened into `global_cond`.

Main call path:

1. `make_diffusion_pre_post_processors`
2. `DiffusionPolicy.select_action` / `DiffusionPolicy.forward`
3. `DiffusionModel.generate_actions` / `DiffusionModel.compute_loss`
4. `DiffusionModel._prepare_global_conditioning`
5. `DiffusionModel.conditional_sample`
6. `DiffusionConditionalUnet1d.forward`

---

## HLRP Action Expert Recommendation

Recommended baseline:

- Start from **SmolVLA/PI-style flow-matching action expert** for stronger language conditioning.
- Keep chunked denoising loop and KV-cache prefix/suffix separation.
- Add optional abstract action token conditioning as an additional input branch.

Proposed HLRP component block:

```text
Inputs:
  - observation images
  - observation state
  - language tokens (required)
  - abstract action tokens (optional)

Backbone:
  - compact VLM prefix encoder
  - action expert suffix decoder (flow-matching)

Inference:
  - prefix cache once per chunk
  - iterative denoise/update on action suffix
  - output n_action_steps slice
```

---

## If Extending Diffusion Instead

Minimum additions needed:

1. Add tokenizer step in diffusion preprocessor.
2. Add text (and optional token) encoder modules in diffusion model.
3. Append encoded language/token features to `global_cond`.
4. Preserve language/token context in rollout queue path.

This yields language-conditioned diffusion, but with less native token-level fusion than PI/SmolVLA experts.
