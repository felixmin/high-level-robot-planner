# LeRobot Integration Plan (Single Source of Truth)

Last updated: 2026-02-12

This document is the only active plan for LeRobot integration.

## 1. Scope and fixed decisions

1. We integrate through a LeRobot plugin distribution (Approach A): one pip package, multiple policies.
2. The plugin distribution name must start with `lerobot_policy_`.
3. We keep one install (`pip install -e .`) and expose multiple `--policy.type ...` values.
4. VLA conditioning stays on a single frame at time `t`.
5. LAQ teacher currently uses 2 frames (`t, t+offset`), but interfaces must allow `T >= 2` in the future.
6. We support both label paths for training:
   - precomputed latent codes in dataset
   - online latent code generation (frozen LAQ)
7. We will support two action-expert variants:
   - latent-conditioned expert (`image/state + latent codes -> actions`)
   - direct VLA-based expert (`image/state/language -> actions` without latent decode at inference)

## 2. Current repo status (already implemented)

Do not re-plan these; they already exist in HLRP:

- Stage-2 backend interface and types: `packages/foundation/backends/interfaces.py`
- Qwen token backend: `packages/foundation/backends/qwen3vl_chat_backend.py`
- Smol latent-head backend: `packages/foundation/backends/smol_latent_head_backend.py`
- Backend-driven Lightning module: `packages/foundation/vla_backend_module.py`
- Stage-2 training entrypoint using backends: `scripts/4_train_foundation.py`
- Existing backend tests: `tests/test_vla_backend_interface.py`, `tests/test_smol_latent_head_backend.py`

What is not implemented yet:

- A LeRobot plugin package with registered policies
- LeRobot-side pre/post processors for HLRP policies
- End-to-end LeRobot training/eval wiring for HLRP policies
- Action-expert policies in LeRobot

## 3. Target plugin architecture (Approach A)

Create one plugin package (new repo or sibling folder):

```text
lerobot_policy_felix_bundle/
  pyproject.toml
  src/
    lerobot_policy_felix_bundle/
      __init__.py
      policies/
        hlrp_vla_direct/
          __init__.py
          configuration_hlrp_vla_direct.py
          modeling_hlrp_vla_direct.py
          processor_hlrp_vla_direct.py
        hlrp_action_expert_latent/
          __init__.py
          configuration_hlrp_action_expert_latent.py
          modeling_hlrp_action_expert_latent.py
          processor_hlrp_action_expert_latent.py
        hlrp_action_expert_direct/
          __init__.py
          configuration_hlrp_action_expert_direct.py
          modeling_hlrp_action_expert_direct.py
          processor_hlrp_action_expert_direct.py
```

`pyproject.toml` (minimum):

```toml
[project]
name = "lerobot_policy_felix_bundle"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["lerobot", "torch"]

[project.optional-dependencies]
hlrp_vla_direct = []
hlrp_action_expert_latent = []
hlrp_action_expert_direct = []
all = []

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

Top-level import registration is required in `src/lerobot_policy_felix_bundle/__init__.py`:

```python
from .policies.hlrp_vla_direct.configuration_hlrp_vla_direct import HLRPVLADirectConfig  # noqa: F401
from .policies.hlrp_vla_direct.modeling_hlrp_vla_direct import HLRPVLADirectPolicy  # noqa: F401

from .policies.hlrp_action_expert_latent.configuration_hlrp_action_expert_latent import HLRPActionExpertLatentConfig  # noqa: F401
from .policies.hlrp_action_expert_latent.modeling_hlrp_action_expert_latent import HLRPActionExpertLatentPolicy  # noqa: F401

from .policies.hlrp_action_expert_direct.configuration_hlrp_action_expert_direct import HLRPActionExpertDirectConfig  # noqa: F401
from .policies.hlrp_action_expert_direct.modeling_hlrp_action_expert_direct import HLRPActionExpertDirectPolicy  # noqa: F401
```

If a policy module is not imported here, its `@PreTrainedConfig.register_subclass(...)` will not execute, and `--policy.type ...` will fail.

## 4. Policy definitions

### 4.1 `hlrp_vla_direct`

Purpose:
- Baseline integration path.
- Predict latent codes from VLA and convert to actions.

Training:
- `forward(batch)` supervises latent prediction (token CE or latent-head CE depending on backend).
- Target codes come from dataset (`latent_codes`) or online LAQ (`T>=2`, currently 2).

Inference:
- Condition VLA on a single current frame + instruction.
- Predict codes.
- Convert codes to actions using decoder head.

### 4.2 `hlrp_action_expert_latent`

Purpose:
- Action expert is conditioned on latent codes predicted by VLA.

Training:
- Inputs: observations + latent codes (precomputed or online).
- Predict continuous action chunks.

Inference:
- VLA predicts latent codes from single frame.
- Expert predicts actions conditioned on those codes.

### 4.3 `hlrp_action_expert_direct`

Purpose:
- Action expert predicts actions directly from VLA-conditioned representations.

Training:
- Inputs: single-frame observation, language, state.
- Expert head learns continuous actions directly.

Inference:
- No latent decode path required.
- One forward path from VLA + expert head to actions.

## 5. Frame handling contract

1. Public policy contract to LeRobot: single-frame observation input.
2. Internal latent-label generation contract: frame sequence `T>=2` supported.
3. Current default in configs and data adapters: `T=2`.
4. Future-proofing requirement: do not hardcode exactly 2 in model interfaces.

## 6. Implementation plan (missing work only)

### Phase 1: Plugin scaffold

Deliverables:
- Create `lerobot_policy_felix_bundle` package structure.
- Add config/model/processor stubs for 3 policies.
- Add top-level imports for registration.

Exit criteria:
- `pip install -e .` succeeds.
- `lerobot-train --help` can resolve all three policy types.

### Phase 2: `hlrp_vla_direct`

Deliverables:
- Wrap HLRP backend(s) in LeRobot `PreTrainedPolicy`.
- Implement LeRobot batch conversion to HLRP `FoundationBatch`.
- Add code-to-action decoder path.
- Add precomputed/online latent target switch.

Exit criteria:
- One short training smoke test runs.
- `predict_action_chunk()` produces valid shape/actions.

### Phase 3: `hlrp_action_expert_latent`

Deliverables:
- Add latent-conditioned action expert policy.
- Define latent-conditioning interface and queue/chunk behavior.

Exit criteria:
- LeRobot rollout loop runs with latent-conditioned policy.

### Phase 4: `hlrp_action_expert_direct`

Deliverables:
- Add direct VLA-based action-expert policy.
- Ensure language/state/image conditioning path is complete.

Exit criteria:
- LeRobot rollout loop runs with direct policy.

### Phase 5: Tests and benchmark entrypoints

Deliverables:
- Unit tests for registration/import, data conversion, shape contracts.
- Integration tests for train/eval loop per policy.
- Minimal benchmark commands documented for each policy.

Exit criteria:
- CI-level smoke tests green.
- Reproducible train/eval commands available.

## 7. Minimal registration pattern (required for every policy)

```python
from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy

@PreTrainedConfig.register_subclass("hlrp_vla_direct")
@dataclass
class HLRPVLADirectConfig(PreTrainedConfig):
    pass

class HLRPVLADirectPolicy(PreTrainedPolicy):
    config_class = HLRPVLADirectConfig
    name = "hlrp_vla_direct"
```

## 8. Commands (expected user flow)

```bash
pip install -e .

lerobot-train --policy.type hlrp_vla_direct ...
lerobot-train --policy.type hlrp_action_expert_latent ...
lerobot-train --policy.type hlrp_action_expert_direct ...
```

## 9. Open items (kept intentionally small)

1. Default decoder choice for `hlrp_vla_direct` (`mlp`, `laq`, or both).
2. Preferred expert backbone for the two action-expert policies.
3. Dataset policy for latent codes by default (`precomputed` vs `online`) per benchmark.

