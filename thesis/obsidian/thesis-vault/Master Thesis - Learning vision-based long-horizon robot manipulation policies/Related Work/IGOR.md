---
notion-id: 29320c92-0436-80f2-8cf8-df50f50311f4
---
## Intro

Internet scale data required for good robot policies

Currently we only have limited robotic interaction data

We need methods that leverage internet scale video data

Getting “action” latents from videos

Previous work on 2D

Question: Can we learn unified semantically consistent latent action space

> [!tip] 💡
> We could have different decoders for the latent

Latent action model

## Method

### Latent Action Model

Primary focus: label latent actions from unlabeled open-domain videos in unsupervised manner

Given: $o_{1:t+1}$

Drive: $a_t$

Challenge: Learn semantic movements consistent across varying scenarios

Inverse Dynamics Model

Forward Dynamics Model

### Foundation World Model

Continuous-time Rectified Flow Model predicts the future frames from observation history and future actions

$(o_{1:t}, a_{t:T-1}) \rightarrow (o_{t+1:T})$

Compared to the forward dynamics model this doesn’t just predict one future image but the whole sequence

Fine-tuning pre-trained Open-Sora

Components

Modifications so Open-Sora

Rectified Flow Formally

### Foundation Policy Model (2 stages or rather 2 models)

Pretraining stage (Lets call it High Level Policy (HLP) here)

Low-level policy training stage (lets call it Low Level Policy (LLP) here)