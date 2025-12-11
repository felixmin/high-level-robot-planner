# LAPA vs. HLRP Reimplementation Comparison

**Verdict:** The HLRP implementation is a faithful reproduction of LAPA. The core model architecture, optimization logic, and unique validation behaviors (codebook updates) are identical. The primary differences are structural (PyTorch Lightning vs. Custom Loop) and preprocessing (Center Crop vs. Resize).

## Identical Behaviors
*   **Codebook Replacement:** Both implementations actively modify the codebook during the validation phase via `replace_unused_codebooks` inside the `forward` pass (triggered by `global_step`). This is by design in LAPA.
*   **Model Architecture:** `LatentActionQuantization`, `NSVQ`, and `Attention` modules are code-identical.
*   **Hyperparameters:** Default model configs (dim: 1024, heads: 16, codebook: 8, etc.) match LAPA's `train_sthv2.py` defaults.
*   **Optimization:** Both use `AdamW` and identical custom weight decay logic (excluding parameters with `ndim < 2` like biases/LayerNorm).

## Key Differences

### 1. Data Preprocessing (Critical)
*   **LAPA:** `T.Resize(256)` → `ToTensor()`.
    *   *Effect:* If input is non-square, this distorts/squashes the image to 256x256 (or crashes if strict sizing is asserted).
*   **HLRP:** `T.Resize(256)` → `T.CenterCrop(256)` → `ToTensor()`.
    *   *Effect:* Robustly produces square inputs by cropping the center. Preserves aspect ratio but discards peripheral information.

### 2. Framework & Loop
*   **LAPA:** Custom `while` loop with manual `accelerator` handling. Validation code is embedded in the training loop.
*   **HLRP:** PyTorch Lightning `pl.LightningModule`. Validation is a distinct hook but calls the same `forward` method, preserving LAPA's codebook update behavior.

### 3. EMA Implementation
*   **LAPA:** Uses `ema_pytorch` library.
*   **HLRP:** Uses a custom `EMACallback` with manual parameter updates (`new = old * decay + new * (1-decay)`). Functionally equivalent but lacks potential library-specific nuances (e.g., Kaiming init handling).