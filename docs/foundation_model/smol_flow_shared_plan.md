# Smol Shared-Trunk Flow Plan

## Objective
Close the loop across:
1. Stage 1 LAQ pretraining on OXE/OpenX
2. Stage 2 SmolVLM training with a shared trunk and two lightweight heads
3. Stage 3 LeRobot fine-tuning/evaluation with HLRP policy plugin

Stage 2 is trained online (no offline relabeling): each OXE batch is passed through frozen LAQ to obtain latent supervision on the fly.

## Implemented Stage 2 Design

### Shared trunk
- Backbone: SmolVLM2 pooled hidden state from image+instruction (`SmolFlowActionBackend`).

### Heads
- Latent head: flow-matching head that predicts continuous LAQ codebook vectors.
  - Target: flattened LAQ codebook vectors `[B, S*D]`.
  - Loss: flow-matching MSE on velocity field.
- Real-action head: lightweight regression head for OXE cumulative action metadata.
  - Target: `batch["action"]` from OXE adapter.
  - Loss: MSE.

### Online supervision path
- `video = oxe_frames_to_laq_video(frames)`
- `(codes, vectors) = LAQTaskCodeProvider.codes_and_vectors_from_video(video)`
- `actions = extract_oxe_actions(batch)`
- Feed all targets into `FoundationBatch(..., target_latent_vectors=vectors, target_actions=actions)`.

### Backend modes
- `codes`: old token-CE mode (unchanged).
- `actions`: action-only mode.
- `multitask`: flow latent + action multitask mode (new default for this plan).

## New Configs
- Model config: `config/model/foundation_smol_flow_shared.yaml`
- Experiment config: `config/experiment/vla_smol_flow_shared.yaml`

## End-to-End Commands

### 1) Stage 1 LAQ pretrain
```bash
python scripts/submit_job.py \
  experiment=laq_oxe_cluster \
  cluster=lrz_x100 \
  experiment.name=laq_stage1_new
```

### 2) Stage 2 shared-trunk flow training
```bash
python scripts/submit_job.py \
  experiment=vla_smol_flow_shared \
  cluster=lrz_x100 \
  model.laq.checkpoint=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/<LAQ_CKPT>.ckpt \
  experiment.name=vla_smol_flow_shared_stage2
```

### 3) Stage 3 LeRobot fine-tune with HLRP policy plugin
```bash
python scripts/submit_job.py \
  experiment=lerobot_hlrp_smoke \
  cluster=lrz_x100 \
  lerobot.base_policy_ckpt=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/<STAGE2_CKPT>.ckpt \
  experiment.name=lerobot_stage3_finetune
```

## Practical Notes
- Stage 2/3 can run with dual-queue strategy:
  - submit once with `cluster=lrz_x100`
  - submit once with `cluster=mcml_x100`
  - cancel the one that starts later.
- Dataset/model downloads are cached under cluster cache paths configured by `submit_job.py`.
- For LeRobot stage, policy discovery remains editable-install based (`-e`) via `scripts/6_train_lerobot.py`.

## Open Follow-ups
- Add explicit rollout/eval script wiring after Stage 3 checkpoint export.
- Decide whether to keep action regression MSE or migrate action head to flow as well.

## Run Log
| Date | Stage | Cluster | Job ID | Config | Status |
|---|---|---|---|---|---|
| 2026-02-16 | Stage 2 smoke | LRZ X100 | 5482581 | `experiment=vla_smol_flow_shared`, `max_steps=200`, `check_interval=50`, viz off | Failed early (`extract_oxe_actions` batch format bug) |
| 2026-02-16 | Stage 2 smoke | MCML X100 | 5482582 | `experiment=vla_smol_flow_shared`, `max_steps=200`, `check_interval=50`, viz off | Canceled (duplicate queue after LRZ failure diagnosis) |
| 2026-02-16 | Stage 2 smoke retry | LRZ X100 | 5482673 | `experiment=vla_smol_flow_shared`, `max_steps=200`, `check_interval=50`, viz off | Failed early (`target_actions` dim mismatch: expected 5, got 3) |
| 2026-02-16 | Stage 2 smoke retry | MCML X100 | 5482674 | `experiment=vla_smol_flow_shared`, `max_steps=200`, `check_interval=50`, viz off | Canceled (duplicate queue after LRZ failure diagnosis) |
