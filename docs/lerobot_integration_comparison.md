# LeRobot Integration Plan: Comparison with HLRP Stage 2 Documentation

This document compares the proposed LeRobot integration plan (`LEROBOT_INTEGRATION_PLAN.md`) against the existing HLRP Stage 2 documentation (`stage2_cosmos2_tokens.md`) to identify alignment, gaps, and necessary adjustments.

---

## 1. Document Overview

| Document | Purpose |
|----------|---------|
| `stage2_cosmos2_tokens.md` | Documents the implemented HLRP Stage 2 VLA training system |
| `LEROBOT_INTEGRATION_PLAN.md` | Proposes how to wrap HLRP VLA as a LeRobot policy |

---

## 2. Architecture Alignment

### What Aligns Well

| Aspect | HLRP Stage 2 | Integration Plan | Status |
|--------|--------------|------------------|--------|
| **Token scheme** | `<ACTION>` `</ACTION>` + `<ACT_0>`...`<ACT_K>` | Same | ✅ Aligned |
| **Backbone model** | Qwen3VL (Cosmos-Reason2-2B) | Same | ✅ Aligned |
| **Prompt masking** | Uses processor with images (not tokenizer-only) | Correctly identified | ✅ Aligned |
| **Constrained decoding** | `prefix_allowed_tokens_fn` | Same approach | ✅ Aligned |
| **Online LAQ encoding** | Frozen LAQ generates labels during training | Supported as option | ✅ Aligned |
| **Key module files** | `vla_module.py`, `action_tokens.py`, `constrained_decode.py`, etc. | Referenced correctly | ✅ Aligned |

### Key Differences / Gaps

| Aspect | HLRP Stage 2 | Integration Plan | Resolution |
|--------|--------------|------------------|------------|
| **Frame pairs** | LAQ uses pairs, VLA uses single frame | Listed as open question | **Resolved**: VLA already works with single frames |
| **Stage 3 decoder** | Not implemented (TODO) | Proposes `action_decoder.py` | **New implementation needed** |
| **Proprioceptive state** | Not used | LeRobot expects it | **Optional**: Ignore initially |
| **Data format** | OXE streaming | HF datasets | **Conversion needed** |

---

## 3. Frame Pair Handling - Clarification

### HLRP Stage 2 Documentation States:

> LAQ encodes `(frame_t, frame_{t+offset}) → discrete code indices [B, code_seq_len]`
> VLA learns to predict those indices from `(frame_t, instruction)` **only**.

### Implication:

The integration plan incorrectly lists "Frame Pair Handling" as an open question. In reality:

- **Training**: LAQ uses frame pairs `(t, t+offset)` to generate target codes
- **VLA Input**: Only single frame `t` + language instruction
- **Inference**: Only single frame needed

**Action**: Remove frame pair handling from "Open Questions" in integration plan. The VLA already operates on single frames at inference time.

---

## 4. Online vs Pre-computed Latent Codes

### HLRP Stage 2 Approach:

> Labels are generated **online** during training by a **frozen LAQ model**

The training script (`4_train_foundation.py`) loads LAQ and runs it on each batch:

```python
# From vla_module.py
codes = self.code_provider.codes_from_video(video)  # [B, code_seq_len]
targets = [self.action_tokens.format_target(row.tolist()) for row in codes]
```

### Integration Plan Options:

| Option | Pros | Cons |
|--------|------|------|
| **Online encoding** (HLRP approach) | No dataset modification, always fresh | Requires LAQ model at training time, slower |
| **Pre-computed codes** (Plan recommendation) | Faster training, simpler pipeline | Requires dataset preprocessing step |

### Recommendation:

Support **both** approaches in the integration:

1. **Online mode**: For quick experiments, matches HLRP behavior
2. **Pre-computed mode**: For production training on large datasets

```python
def _get_target_codes(self, batch):
    # Option 1: Pre-computed in dataset
    if "latent_codes" in batch:
        return batch["latent_codes"]

    # Option 2: Online LAQ encoding
    video = self._prepare_video_for_laq(batch)
    return self.laq_encoder.codes_from_video(video)
```

---

## 5. Stage 3 / Action Decoder Gap

### HLRP Architecture (from CLAUDE.md):

```
Stage 1 (LAQ):        Video pairs → Discrete latent codes
Stage 2 (Foundation): Image + Language → Discrete latent codes
Stage 3 (Finetuning): Discrete codes → Continuous robot commands  ← NOT IMPLEMENTED
```

### Current State:

- Stage 1 (LAQ): ✅ Implemented
- Stage 2 (Foundation VLA): ✅ Implemented
- Stage 3 (Action Decoder): ❌ Not implemented

### Integration Plan Proposes:

`action_decoder.py` - A simplified Stage 3 that:
1. Looks up code embeddings from LAQ codebook
2. Projects to continuous actions via MLP

```python
class LAQActionDecoder(nn.Module):
    def decode(self, codes: Tensor) -> Tensor:
        embeddings = self.codebook(codes)  # [B, seq_len, embed_dim]
        flat = embeddings.flatten(1)       # [B, seq_len * embed_dim]
        return self.mlp(flat)              # [B, action_dim]
```

### Alternative (Section 14):

Use latent codes to **condition a diffusion policy** instead of direct MLP decoding:

```
Codes → Embedding → Diffusion Global Conditioning → Action Trajectory
```

### Recommendation:

1. **Start with MLP decoder** (simpler, faster)
2. **Evaluate on benchmarks**
3. **If needed, upgrade to diffusion** for smoother trajectories

---

## 6. Data Format Mismatch

### HLRP Stage 2 Data:

| Field | Format | Source |
|-------|--------|--------|
| `frames` | `[B, 2, H, W, 3]` uint8 | OXE streaming |
| `language` | `list[str]` | OXE metadata |
| Target | Generated online by LAQ | - |

### LeRobot Data:

| Field | Format | Source |
|-------|--------|--------|
| `observation.images.*` | `[B, C, H, W]` float | HF datasets |
| `observation.state` | `[B, state_dim]` | HF datasets |
| `action` | `[B, horizon, action_dim]` | HF datasets |
| `task` / `instruction` | `str` | HF datasets |

### Required Conversions:

```python
# In HLRPVLAPolicy preprocessor:

def _convert_lerobot_to_hlrp(self, batch):
    # LeRobot: [B, C, H, W] float [0, 1]
    # HLRP:    [B, H, W, C] uint8 [0, 255]

    images = batch["observation.images.top"]
    images = (images * 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1)  # BCHW → BHWC

    # Language: LeRobot may have tokens, HLRP wants strings
    if "observation.language_tokens" in batch:
        instructions = self.tokenizer.batch_decode(batch["observation.language_tokens"])
    else:
        instructions = batch.get("task", [""] * len(images))

    return images, instructions
```

---

## 7. Prompt Masking Implementation

### HLRP Stage 2 Critical Detail:

> **Prompt masking must use the *processor with images***:
> For Qwen3-VL, tokenizer-only prompt lengths can be wrong because multimodal tokenization depends on images.

### Implementation (from `vla_inputs.py`):

```python
# WRONG: Using tokenizer only
prompt_len = len(tokenizer(prompt_text)["input_ids"])

# CORRECT: Using processor with images
prompt_inputs = processor(text=prompt_texts, images=images, ...)
prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)
```

### Integration Plan Status:

The plan correctly identifies this in `_build_training_inputs()`, but should emphasize it more prominently as a **critical implementation detail**.

---

## 8. Constrained Decoding State Machine

### HLRP Implementation (`constrained_decode.py`):

```python
class ActionTokenIds:
    def next_allowed_ids(self, generated_ids: Sequence[int]) -> List[int]:
        """
        State machine:
        1. Not started → force <ACTION>
        2. After <ACTION>, codes < code_seq_len → allow <ACT_*>
        3. After code_seq_len codes → force </ACTION>
        4. After </ACTION> → force EOS
        """
```

### Integration Plan:

Implements equivalent logic in `_prefix_allowed_tokens_fn()`. Should reference the existing HLRP implementation for consistency.

**Recommendation**: Import and reuse `ActionTokenIds` from HLRP packages rather than reimplementing:

```python
from foundation.constrained_decode import ActionTokenIds, make_prefix_allowed_tokens_fn
```

---

## 9. Testing Strategy Comparison

### HLRP Stage 2 Tests:

```bash
pytest -q tests/test_action_tokens.py \
         tests/test_vla_inputs_prompt_mask.py \
         tests/test_online_laq.py \
         tests/test_constrained_decode.py \
         tests/test_vla_module_cpu.py \
         tests/test_vla_generation_metrics.py \
         tests/test_image_adapters.py
```

### Integration Plan Tests:

- `test_config_validation()`
- `test_action_token_setup()`
- `test_constrained_decoding()`
- `test_forward_loss()`
- `test_select_action_shape()`

### Gap:

Integration plan should add tests for:
- **Data format conversion** (LeRobot → HLRP)
- **Prompt masking with real processor** (not mocked)
- **End-to-end rollout** with LeRobot eval

---

## 10. Summary of Required Changes to Integration Plan

### Remove from Open Questions:

- [ ] ~~Frame pair handling~~ → VLA already uses single frames

### Add / Clarify:

- [ ] Support both online and pre-computed latent codes
- [ ] Emphasize processor-based prompt masking as critical
- [ ] Reuse HLRP's `ActionTokenIds` instead of reimplementing
- [ ] Add data format conversion layer (LeRobot ↔ HLRP)
- [ ] Add tests for format conversion and prompt masking

### Implementation Priority:

| Phase | Task | Notes |
|-------|------|-------|
| 1 | Core policy with inference only | Use pre-computed codes |
| 2 | Add training support | Online LAQ encoding |
| 3 | Add diffusion option (Section 14) | For benchmark performance |
| 4 | Benchmark evaluation | LIBERO, Aloha, etc. |

---

## 11. Code Reuse Opportunities

Instead of reimplementing, import from HLRP packages:

```python
# In lerobot/policies/hlrp_vla/modeling_hlrp_vla.py

from foundation.action_tokens import ActionTokenConfig
from foundation.constrained_decode import ActionTokenIds, make_prefix_allowed_tokens_fn
from foundation.vla_inputs import build_inputs_with_prompt_mask, build_prompt_inputs
from foundation.online_laq import LAQTaskCodeProvider
from foundation.image_adapters import oxe_first_frames_to_pil
```

This ensures:
- Consistent behavior with standalone HLRP training
- Automatic updates when HLRP code changes
- Reduced maintenance burden

---

## 12. Conclusion

The integration plan is **well-aligned** with HLRP Stage 2 implementation, with a few clarifications needed:

1. **Frame pairs are not an issue** - VLA already works with single frames
2. **Stage 3 decoder is new work** - The `action_decoder.py` is essentially implementing Stage 3
3. **Reuse HLRP code** where possible instead of reimplementing
4. **Support both online and pre-computed codes** for flexibility

The hierarchical diffusion approach (Section 14) is a valuable addition that goes beyond the original HLRP architecture and could improve benchmark performance.
