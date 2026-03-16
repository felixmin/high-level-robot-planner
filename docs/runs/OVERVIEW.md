# Latent Action Model Versions

## Base checkpoint
Octo24 + Libero dataset
Pixel encoder
Pixel + flow decoder
TODO

## Randomly initialized checkpoint without training for comparisons
2026-03-15_18-57-14_stage1_cluster_test2_tiny_lam_ckpt_for_stage3_multitask

## Libero overfitting run
Only Libero dataset
Pixel encoder
Pixel + flow decoder
2026-03-14_19-27-26_stage1_local

## Custom encoder / decoder checkpoint (This was the default until 2026-03-15)
Octo24 + Libero dataset
?
?




# LIBERO Experiments

## Full libero dataset

### Baseline action only
100% action labeled: train only action head
TODO



## Baseline multitask
5% action labeled: train multitask action head and latent head at the same time




## Libero with 5% action labeled and 95% without real action labels

### Baseline only action
5% action labeled: train only action head

### Baseline multitask
Latent action model: Pretrained on octo24 + Libero (5% of the dataset)
5% action labeled: train multitask action head and latent head at the same time

### 95% for latents balanced 50/50
Latent action model: Pretrained on octo24 + Libero (5% of the dataset)
5% action labeled: train multitask action head and latent head at the same time
95% no action labeles: train latent head only
2026-03-13_15-54-35_test
Caution: here the latent loss was not underweighted! redo.

### 95% for latents balanced 50/50 random latent action model
Latent action model: Randomly initialized
5% action labeled: train multitask action head and latent head at the same time
95% no action labeles: train latent head only
Not run

### 95% for latents balanced 50/50 Libero overfitting
Latent action model: Pretrained on Libero only (2026-03-14_19-27-26_stage1_local)
5% action labeled: train multitask action head and latent head at the same time
95% no action labeles: train latent head only
2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1





