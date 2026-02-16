# Foundation Legacy

This folder contains pre-shared-structure Foundation code that is still kept for
reference/migration:

- `backends/qwen3vl_chat_backend.py`
- `backends/smol_latent_head_backend.py`
- `qwen3vl_setup.py`
- `vla_module.py`

Legacy Smol flow-action variant:
- `SmolFlowActionBackend` lives in `backends/smol_latent_head_backend.py`
- selected via `model.backend=smol_flow_action` in `scripts/4_train_foundation.py`

Canonical/new path lives in `foundation.backends.smolvla_shared*`.
