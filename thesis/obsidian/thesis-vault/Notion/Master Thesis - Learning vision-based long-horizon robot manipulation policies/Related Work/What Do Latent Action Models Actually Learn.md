---
notion-id: 2c420c92-0436-80c6-89a4-df106c7e33fa
---
> [!tip] 💡
> Extensive read (important paper)

Structure

## Abstract

- Goal of LAMs: Learning action relevant changes from unlabeled videos
- Controllable changes vs exogenous noise
- Presented here: Linear model that encapsulates essence of LAM learning
    - Insights: 
        - LAM and PCA connection
        - desiderata of data-generating policy, 
        - justification of strategies to encourage learning controllable changes using data augmentation, clearning and auxiliary action-prediciton
    - Intestigates:
        - how structure of obs, actions and noise influence LAM training

# Intro

What

- Infer controllable action changes from streams of image obs in unsupervised manner

Why

- expensive action labels

Towards Generalist Robot Learning from Internet Video: A Survey

To study what is learned: replace deep LAM with Linear model

Discovered other issues regarding overparametrization

Contributions

# Related Work

- Learning action representations as long history in RL
- Citing papers about action clustering, learning latent actions on real actions, learning latent actions on full state + action info, LAM conditioned on real actions

→ All work with real actions

Here without action labels

→ allows internet scale

- old approaches extract actions using CV techniques

From 2018 Rybkin et al LAMs with bottleneck (autoencoders) for unsupervised learning

currently

- LAMs becoming popular
- Objective, learnability, and robustness gain limited attention


# 3 Setup

## 3.2 Linear LAM

Analysis in controlled Markov process (CMP) framework ??

Evaluate latents

- if they have information about the real action
- if they dont have exogenous noise and observation

→ formalization

Typically mutual information is hard to measure

# Analysis

## 1. Linear LAM is PCA

Mathematical proof that for linear LAM the training becomes similar to PCA on (q + noise) when some assumptions hold

> [!tip] 💡
> Read Eckart-Young-Mirsky theorem

## 2. Effects of Data Collection Policy

## 3. Improvements via Data Augmentation

## 4. Improvements via Auxiliary Action Prediction

# Beyond Linear

# Conclusion

- LAMs CAN work but they currently focus on any change between the two frames
→ if the main change between the two frames is task-relevant it works
→ if the main variation is not the controllable signal then it captures noise
They thus work in low noise settings

- Lack of randomization in data collection harms learned latents

