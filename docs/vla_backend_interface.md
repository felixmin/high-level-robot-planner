# VLA Backend Interface (HLRP ↔ LeRobot)

This document proposes a **backend interface** that makes VLA training/inference **model-specific behind a common API**.

Goals:
- Compare different implementations cleanly (even for the *same* base model).
- Keep “model quirks” (processor, chat template, masking, special tokens, heads) behind the backend.
- Make it straightforward to reuse HLRP-pretrained models inside **LeRobot** for finetuning + benchmarking.

Terminology:
- **HLRP**: this repository (`high-level-robot-planner`)
- **LeRobot**: Hugging Face LeRobot repo (user referred to “L'Europeau”)
- **Stage 1 / LAQ**: latent action quantization teacher
- **Stage 2 / Foundation**: image + language → **LAQ** discrete latent codes (tokens), where LAQ labels come from frame pairs `(t, t+Δ)` (and may extend to `T>2` later)
- **Stage 3**: latent codes → continuous robot actions (currently TODO in HLRP)

---

## 1) HLRP: what needs to be abstracted

HLRP Stage 2 today is implemented as:
- A Hugging Face VLM backbone (currently **Qwen3-VL loaded from Cosmos-Reason2 weights** in `scripts/4_train_foundation.py`)
- A frozen LAQ teacher producing codes `[B, S]` from **frame sequences** `frames[:, :T]` (today typically a pair `(t, t+Δ)`)
- The VLM conditions on a subset of frames (today: frame `t` only) to avoid leakage
- Conversion of codes to a **token completion** (e.g. `<ACTION> <ACT_3> <ACT_0> <ACT_7> <ACT_1> </ACTION>`)
- **Prompt-masked LM loss** computed only on the completion tokens
- Inference via `generate()` + **constrained decoding** + parsing back to codes

Key observation: even if we keep “the same training objective” (predict code tokens), there are many implementation variants we want to compare:
- Chat template vs non-chat prompt formatting
- Prompt masking implementations (processor-based length vs tokenizer-only length, padding behavior)
- Different constrained decoding state machines / budgets
- Token scheme differences (separators, wrapper tokens)
- “Same model, different input packing” (e.g., different system prompts, few-shot examples)

So the abstraction should **not** force a single prompt format; it should allow multiple *backends*.

---

## 2) Proposed interface for HLRP

### 2.1 Design goals

- Make training code call **one function**: “give me loss from batch”.
- Make inference code call **one function**: “give me predicted latent representation from batch”.
- Put all model-specific behavior behind the backend:
  - processor selection
  - chat template application (or not)
  - tokenizer vocabulary edits (adding `<ACTION>`, `<ACT_i>`, etc.)
  - embedding resizing / special token ids
  - how inputs + labels are built (prompt masking)
  - how generation is constrained and parsed
- Allow multiple backends for:
  - different models (Cosmos/Qwen3-VL vs SmolVLM)
  - different prompting/masking strategies on the same model

Constraints (explicitly “forward-only”):
- No backwards-compatibility layers or fallback behavior.
- Keep checks minimal and high-signal; assume inputs are correct in normal use.

### 2.2 Suggested Python protocol (HLRP-side) (v2)

Implemented in `packages/foundation/backends/`.

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence
import torch


@dataclass(frozen=True)
class FoundationBatch:
    # Stage 2 common inputs.
    # Device expectation: `frames` are CPU tensors (uint8) because many processors
    # require PIL/NumPy conversion. Backends/teachers move tensors as needed.
    frames: torch.Tensor                 # [B, T, H, W, 3] uint8 (T>=1)
    instructions: Sequence[str]          # len B

    # Optional supervision variants:
    target_codes: torch.Tensor | None = None   # [B, S] long (if provided)
    # Optional metadata/debug:
    meta: dict[str, Any] | None = None


class BackendMode(str, Enum):
    CODES = "codes"          # discrete LAQ codes (Stage 2)
    ACTIONS = "actions"      # continuous actions (Stage 3; flow/diffusion/etc.)
    MULTITASK = "multitask"  # both (shared backbone, separate heads)


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor                   # scalar, requires_grad=True
    metrics: dict[str, Any]              # scalars or small tensors (logging)


@dataclass(frozen=True)
class LatentOutput:
    # Discrete
    logits: torch.Tensor | None = None   # [B, S, K] (optional; natural for latent-head models)
    tokens: torch.Tensor | None = None   # [B, S] long (optional; natural for generate+parse models)
    # Continuous (future)
    vector: torch.Tensor | None = None   # [B, D] or [B, S, D]
    # Debug-only (small objects)
    meta: dict[str, Any] | None = None


class VLABackend(Protocol):
    """
    Multi-mode backend (Stage 2 + Stage 3).

    The backend owns model-specific behavior:
    - underlying model + processor
    - chat template/prompting and masking rules
    - special-token/vocab edits and embedding resize
    - heads (latent tokens head, action expert head, etc.)
    - generation/parsing/constrained decoding where applicable
    """

    codebook_size: int
    code_seq_len: int

    def setup(self, *, device: torch.device) -> None:
        """Finalize tokenizer edits, resize embeddings, move modules to device, etc."""

    def loss_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LossOutput:
        """Compute training loss for the requested mode."""

    @torch.no_grad()
    def latent_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LatentOutput:
        """Return a latent representation for the requested mode."""
```

Notes:
- For Cosmos/Qwen3-VL, `tokens` is populated from constrained `generate()+parse`, while `logits=None`.
- For latent-head backends (LeRobot `latent_smol`-style), `logits` is naturally available and `tokens=argmax(logits)` is trivial.
- For future continuous latents, `vector` can be populated.

This keeps usability and comparability: downstream code can decide whether to embed tokens, use a logit distribution, etc.

### 2.4 Optional split: “model core” vs “prompting strategy”

If you want to compare “same model, different prompting/masking”, consider two layers:
- `ModelCore`: loads the weights, owns `processor`, adds tokens, exposes `forward()` / `generate()`.
- `PromptingStrategy`: builds messages, computes prompt lengths, masks labels, parses generations.

Then a backend is just `(ModelCore, PromptingStrategy)`.
This avoids duplicating model-loading logic across prompt variants.

### 2.5 Example backends to support

- **Cosmos/Qwen3-VL**
  - `Qwen3VLChatTemplateBackend` (uses `processor.apply_chat_template`)
  - `Qwen3VLPlainPromptBackend` (manual prompt string format; no chat template)
- **SmolVLM**
  - `SmolVLMChatTemplateBackend` (if supported by that processor/tokenizer)
  - `SmolVLMPlainPromptBackend` (manual prompt string format)
- **SmolVLM + latent head (HLRP reimplementation of LeRobot `latent_smol` latent mode)**
  - `SmolLatentHeadBackend` (predicts LAQ codes `[B,S,K]` from prefix hidden states; no generation required)
- **Future: PI0.5 / PI05-style**
  - This is not a “token LM loss on completions” model; it’s flow matching. It likely belongs to a *different* interface (see below).

### 2.6 Extending the interface for Stage 3 (continuous actions)

Stage 2 only predicts discrete codes. For robotics benchmarks you typically need **continuous actions**.

We recommend introducing a second interface for Stage 3 that can wrap:
- a code predictor backend (Stage 2), plus
- a “decoder” to continuous actions (LAQ decoder, MLP, flow matching head, etc.)

```python
@dataclass(frozen=True)
class ActionOutput:
    actions: torch.Tensor            # [B, T, action_dim] float (chunked)
    metrics: dict[str, Any]


class ActionPolicyBackend(Protocol):
    def setup(self, *, device: torch.device) -> None: ...
    def loss_from_batch(self, batch: Any) -> LossOutput: ...
    @torch.no_grad()
    def predict_action_chunk(self, batch: Any) -> ActionOutput: ...
```

This lets you benchmark different “Stage 3 realizations” while keeping Stage 2 fixed.

---

## 3) LeRobot alignment (practical)

LeRobot policies are expected to implement (see `lerobot/src/lerobot/policies/pretrained.py`):
- `forward(batch, ...) -> (loss, logs)` for training
- `predict_action_chunk(batch, ...) -> Tensor` for rollout/benchmarking

Mapping recommendation:
- Stage 2 code prediction (`BackendMode.CODES`) is naturally trained in HLRP.
- A thin adapter can map:
  - HLRP `VLABackend.loss_from_batch(..., mode=CODES)` to `policy.forward(...)`
  - HLRP Stage 3 `ActionPolicyBackend.predict_action_chunk(...)` (or `VLABackend` in `ACTIONS`/`MULTITASK`) to `policy.predict_action_chunk(...)`

Key correctness constraint (from LeRobot `latent_smol`):
- When supervising LAQ codes from `(t, t+Δ)`, the model must condition on **frame `t`**, not `t+Δ`, to avoid leakage.

---

## 3) How this aligns with LeRobot

LeRobot’s policy shape is essentially:
- `forward(batch) -> loss, metrics` (training)
- `predict_action_chunk(...)` / `select_action(...)` (inference)
and it has a separate processor/preprocessor pipeline.

---

## 3.0) One umbrella backend with explicit modes

You asked whether Stage 2 (code/token prediction) and Stage 3 (continuous action prediction) should be under one
umbrella (so you keep the same CLI and just switch a `mode=`), vs switching “backend objects”.

### Recommendation

Use **one umbrella object with explicit modes**, but keep the *contract* capability-based.

- A single concrete implementation can implement both capabilities:
  - Stage 2: predict discrete code tokens (LM-style)
  - Stage 3: predict continuous actions (decoder or flow-matching)
- The training/eval entrypoint selects behavior by mode:
  - `mode="codes"`
  - `mode="actions"`
  - `mode="multitask"` (compute both losses when both labels exist)

This preserves usability (“same command, switch mode”) while still allowing backends that only support one mode.

### Why not force a single universal API call?

Because supervision availability differs:
- Stage 2 data often has **no continuous actions**.
- Stage 3 robot datasets often have actions but **no LAQ codes** unless you compute them online or precompute them.
- Multi-task requires explicit choices about dataset mixing and loss weighting.

So the safest pattern is:
- one object, multiple explicit methods (or one method with a `mode` arg),
- fail fast if requested mode isn’t supported.

---

## 3.0.1) LeRobot precedent: `latent_smol` already does “modes”

LeRobot’s `latent_smol` policy is a concrete “one policy, multiple modes” example:
- `head_mode="latent"`: predicts LAQ codes via a latent head (CE loss) from **prefix/VLM hidden states**.
- `head_mode="action"`: predicts continuous actions via **flow matching** (MSE on velocity/flow) using an action expert.

Reference: `lerobot/src/lerobot/policies/latent_smol/modeling_latent_smol.py`.

What we should copy from it:
- Mode-specific preprocessing can be **critical** to avoid leakage. In latent mode, it uses frame `t=0`
  (not the future frame) for conditioning.

Where we should improve vs `latent_smol` (maintainability):
- Keep mode routing centralized (avoid sprinkling `if head_mode == ...` throughout unrelated helpers).
- Prefer composition: “core model loader” + “prompting strategy” + “heads/decoders”.
- Add contract tests per backend+mode so new backends fail early.

### 3.1 Mapping the proposed interfaces to LeRobot

**Stage 2 backend (codes-as-tokens)** maps cleanly to a LeRobot *pretraining* policy (latent supervision).
But for end-to-end robot control, LeRobot expects **continuous** actions, so you need Stage 3 somewhere.

Two practical integration patterns:

1) **LeRobot policy wraps HLRP Stage 2 + decoder**
   - LeRobot policy calls `VLABackend.latent_from_batch(..., mode=BackendMode.CODES)` to get codes.
   - Then decodes codes to continuous actions via:
     - LAQ decoder (if you have one), or
     - a small learned MLP head trained in LeRobot, or
     - a flow-matching expert conditioned on codes.

2) **LeRobot policy uses a native LeRobot action expert, but reuses HLRP-pretrained backbone**
   - This is closer to `smolvla` / `pi05` patterns in LeRobot:
     - backbone provides prefix representations
     - expert does diffusion/flow-matching for actions
   - Here, the useful abstraction is at the “prefix embedder” boundary, not at “token LM”.

### 3.2 Recommended “adapter boundary” for cross-repo reuse

To maximize reuse, design the HLRP interface so you can export/import:
- HF model weights (or safetensors)
- tokenizer additions (action tokens)
- any prompt-format config (system prompt, templates)

Then in LeRobot:
- implement a thin adapter class that:
  - loads the same HF checkpoint + tokenizer,
  - reuses the same backend logic for code prediction, and
  - provides `predict_action_chunk` by attaching a decoder.

This keeps “the model decides about templates/tokens/etc.”, but lets LeRobot own:
- dataset/normalization
- environment rollout + evaluation loops

### 3.3 Ensuring the interface can later support PI05-like models

PI05 (flow matching) is not naturally expressed as “LM loss on completion tokens”.
So:
- keep `VLABackend` for token-based predictors (Cosmos/Qwen3-VL, SmolVLM token heads)
- add `ActionPolicyBackend` for continuous-action predictors (PI05, SmolVLA, diffusion)

If you still want “one umbrella API”, make it capability-based:
- backends can implement `predict_codes` and/or `predict_action_chunk`.
But keep the *type contract* explicit so training scripts don’t silently do the wrong thing.

---

## 4) Why this abstraction makes sense (and what to watch out for)

### Makes sense because
- You can compare prompting/masking/decoding variants as separate backends (same “outer loop”).
- You can swap Cosmos vs SmolVLM without rewriting training scripts.
- You can keep experiment configs clean: `backend=qwen_chat` vs `backend=qwen_plain` vs `backend=smol_plain`.

### Watch out for
- Some models won’t support `prefix_allowed_tokens_fn` or may behave poorly with it. Keep this explicit by implementing
  a separate backend variant (no silent fallbacks) if you want to compare constrained vs unconstrained decoding.
- “Prompt masking correctness” must be backend-owned and tested; it can materially change results.
- Token additions must be reproducible across training + inference (checkpoint must bundle tokenizer state).

---

## 5) Concrete next steps (implementation plan)

### 5.1 Phase 1 (now): make HLRP Stage 2 backend-first

1) Finalize the backend contract
   - Use `BackendMode` + `LatentOutput` (tokens/logits/vector) so latent-head backends fit without special-casing.

2) Refactor HLRP training to use the backend
   - Add a backend-driven LightningModule (or retrofit existing module) so training/eval calls:
     - `backend.loss_from_batch(..., mode=BackendMode.CODES)`
     - `backend.latent_from_batch(..., mode=BackendMode.CODES)` optionally for validation metrics
   - Move Qwen3VL-specific setup out of `scripts/4_train_foundation.py` into a backend.

3) Add contract tests that every backend must pass
   - Setup must succeed (token ids, embedding resize, etc.)
   - Loss path must succeed on a minimal batch
   - Optional generation/parse path must succeed if advertised as supported

### 5.2 Phase 2: add SmolVLM family backends (prompting/masking comparisons)

4) Implement SmolVLM backends as “prompting strategies”
   - `smol_plain` (no chat template dependency)
   - `smol_chat` (uses chat template if available; otherwise fail fast)
   - Both still train on LAQ codes from frame pairs `(t, t+Δ)` (HLRP approach), not discretized real actions.

### 5.3 Phase 3: HLRP-native reimplementation of LeRobot `latent_smol` latent mode

5) Use the `LatentOutput` interface
   - Latent-head backends should return `logits` (and optionally `tokens=argmax(logits)`).

6) Implement `SmolLatentHeadBackend` in HLRP
   - Backbone: SmolVLM (HF) or an HLRP-chosen alternative
   - Head: linear projection from pooled prefix hidden → `[B,S,K]` logits
   - Supervision: LAQ codes from frame pairs `(t, t+Δ)` via HLRP’s LAQ provider
   - Mode: `mode="codes"` (no generation required)

### 5.4 Phase 4: LeRobot alignment and benchmarking

7) Provide a thin LeRobot adapter (optional but recommended)
   - Wrap an HLRP backend inside a LeRobot `PreTrainedPolicy`
   - For benchmarking, provide `predict_action_chunk()` by adding a decoder/head (Stage 3)
