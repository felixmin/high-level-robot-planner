---
notion-id: 29620c92-0436-8049-8cb9-cb5faab82bcd
---
## Intro

VLA Training

Problem: limited robot data

Solution: internet scale video data

LAPA → unsupervised pretraining of robot foundation models

Steps

1. VQ-VAE to learn quantized latent encodings
2. Behavior cloning by pretraining a VLM to predict action latents
3. Fine-tune on small-scale robot manipulation dataset

Outperforms baselines especially in cross-environment and cross-embodiment

Outperforms OpenVLA

## Related Work

Training Robot Policies From Videos

## Method

2 stage pipeline: latent action quantization and then latent pretraining

Latent Action Quantization

Latent action pretraining

Action Finetuning