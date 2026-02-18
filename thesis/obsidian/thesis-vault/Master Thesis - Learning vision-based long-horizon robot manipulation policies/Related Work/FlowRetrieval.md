---
notion-id: 2b620c92-0436-8010-ab55-e923ea5c7cf3
---
- What exactly is optical flow?

- “Retrieval based methods” are either
    - relying too much on exact behavior in very similar scene
    - semantic similarity of high-level language decsription, which might not be informative about low-level behavior

- Research Question: “How can we leverage motion similarity?”

- Working with optical flow instead of pure images
    - Does this include the motion inbetween and not only the future image?
    - 

- Directly predicts real action and flow latent?
→ So far assumption was to have a lower level policy that is conditioned on a latent and predicts the real action but we can predict both with the foundation policy… however this is probably slow(er) and doesnt allow us to have a faster low level policy running at higher frequency

→ Motor control might benefit from RL which is not possible like this… maybe two stage architecture is better?

## Problem Statement

IL as MDP

Goal is finding policy hat(pi) that imitates pi_expert D_target = set((s,a) | ~ pi_expert (a|s,r))

Few shot IL → D_target is small

But we have D_prior = set((s,a)| a~pi_expert(a|s,.)) which is large

Their goal: retrieve relevant data for performing tasks

> [!tip] 💡
> Difference: They use the latent to search for similar data for retrieval augmented generation, we use the latent for training the foundation policy

Their stages

1. data retrieval: construct similarity function to extract the desired data
> [!tip] 💡
> Maybe we can use a similar similarity function for evaluating our latent space?
2. policy learning: 