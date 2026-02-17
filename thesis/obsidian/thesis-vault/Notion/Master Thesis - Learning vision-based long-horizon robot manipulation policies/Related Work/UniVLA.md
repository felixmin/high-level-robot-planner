---
notion-id: 2b820c92-0436-808d-8dfd-f4c2c0c76234
---
What model architecture?

Data

How do they make sure the model doesnt predict motion if the data includes it ?

## Abstract

Current problem: Limitation of labeled data

Solution: derive task-centric action representations from videos

Enables: exploit extensive data across embodiments and perspectives

“To Mitigate the effect of task-irrelevant dynamics”

## Intro

Pipeline

1. Task-centric Latent Action Learning
Training a VQ-VAE on massive cross embodiment / perspective data
2. Next latent action prediction
Train auto-regressive VLM with discretized latent action tokens
→ Endow cross-embodiment planning
3. Latents decoding
> [!tip] 💡
> Here they point out difference to LAPA and IGOR

Limitations of LAPA and IGOR

Their approach separating out task-irrelevant motion allows better scaling than previously

## Related Work

VLAs

Cross-embodiment learning

Latent Action Learning

## Method
