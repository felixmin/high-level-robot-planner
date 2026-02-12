# LAQ/LAM Token Evaluation

This folder documents the standalone offline token-evaluation workflow implemented for LAQ action-token generalization analysis.

## What Was Implemented

- Script: `scripts/eval_action_tokens.py`
- Config: `config/experiment/laq_token_analysis.yaml`
- Execution model: standalone Hydra script (manual trigger), not a Lightning `Trainer` validation loop.

The script loads a trained LAQ checkpoint, streams val data, computes token ids + RAFT-based motion summaries, and writes local plots/CSVs/JSONs.

## Runtime Model (Hydra + Lightning Checkpoint)

- Uses Hydra for config composition and overrides.
- Uses Lightning checkpoint loading for model restoration:
  - Preferred: `LAQTask.load_from_checkpoint(...)`
  - Fallback: strict state-dict load into LAQ model.
- Runs on GPU if available.

## Data + Motion Signal

Per kept sample:

1. Encode token sequence (length 4).
2. Compute GT optical flow using LAQ flow teacher (RAFT) on `(frame_t, frame_t+k)`.
3. Compute magnitude-weighted mean `(dx, dy)` with static fallback.
4. Store metadata record + optional visual subset frames.

Important:
- `gt_flow` in plots comes from RAFT teacher on real frame pairs.
- It is not flow-decoder output.

## Inspected Tokens (Cross-Analysis Reuse)

The script selects top inspected token sequences once and reuses them across analyses.

- `inspected_tokens.csv`
- `inspected_tokens.json`

This enables direct comparison of the same token across:
- examples
- consistency metrics
- latent transfer
- flow-decoder transfer
- per-token report folders

## Analyses Produced

1. `analysis_1_flow_scatter`: token-vs-motion scatter (combined + per dataset)
2. `analysis_2_consistency`: cross-dataset angular consistency + CSV/metrics/README
3. `analysis_3_examples`: larger per-token sample galleries
4. `analysis_4_transfer`: latent transfer grids (target_t, real_t+k, predicted_t+k, transfer flow)
5. `analysis_5_heatmap`: token usage by dataset (counts + percentages in overlays)
6. `analysis_6_compass`: raw and dataset-centered token mean shifts + dataset baselines
7. `analysis_7_flow_transfer`: flow-decoder transfer views:
   - same token, different images
   - different tokens, same image
   - similar-start token swap (same dataset, nearest initial state)
8. `analysis_8_token_reports`: per-token deep-dive folders with:
   - `samples.png`
   - `latent_transfer.png`
   - `flow_decoder.png`
   - `mean_shift.png`
   - `summary.json`

## Output Location

Outputs are local files (not W&B dashboards by default):

- `runs/.../output/laq_token_analysis_<timestamp>/...`

Example run path:
- `/mnt/data/workspace/runs/hlrp/2026-02-11_13-23-25_laq_token_analysis/output/laq_token_analysis_20260211_132325`

## Run Commands

Smoke test:

```bash
python scripts/eval_action_tokens.py \
  experiment=laq_token_analysis \
  analysis.samples_per_dataset=100 \
  analysis.visual_subset_size=200
```

Full run (default config):

```bash
python scripts/eval_action_tokens.py experiment=laq_token_analysis
```

All-datasets run depends on the selected Hydra data config (default is `laq_oxe_all_low_ram`).

## Reading Key Plots

- `analysis_4_transfer`:
  - Col0: target `frame_t`
  - Col1: real target `frame_t+k`
  - Col2: predicted future from transferred latent token
  - Col3: RAFT flow between Col0 and Col2
- `analysis_7_flow_transfer`:
  - `pred_flow` is direct flow-decoder output conditioned on `(frame_t, token)`
  - Compare to `gt_flow` (RAFT on real pair) for realism and direction consistency.
- `analysis_6_compass`:
  - Raw panel can be biased by dataset-level motion priors.
  - Use dataset-centered panel + `dataset_baseline.csv` for interpretation.

## Manual Trigger From Training (Recommended)

Keep this script manual/offline:

1. Train LAQ normally.
2. Pick checkpoint path.
3. Launch this script with that checkpoint override if needed:

```bash
python scripts/eval_action_tokens.py \
  experiment=laq_token_analysis \
  analysis.checkpoint_path=/path/to/your.ckpt
```

This keeps training loop lightweight while enabling deep token diagnostics on demand.
