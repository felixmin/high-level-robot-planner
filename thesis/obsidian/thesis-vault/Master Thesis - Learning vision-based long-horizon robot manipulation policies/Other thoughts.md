---
notion-id: 29220c92-0436-80c1-874c-c40b1d7e9ffa
---
> [!note]+ Are we only grounding the manual in the real world?
> Not really because there can be steps missing and the manual maybe doesn’t make statements about the order of certain steps

Spatial reasoning VLMs

In-context hierarchical policy

Idea

Maybe the focus of the paper is not the high-level planner but rather

Idea LLM as world model

Two-level - high-level - planner

Lets write in the paper / thesis that having a high level policy makes sense because humans also dont learn this from RL… RL is only for motor skills

Using latents from IGOR (learning from video) vs using LCB (learning end to end)

- Which one is better?

→ IGOR for generalization, BUT maybe not necessary for us? Too much effort?

→ Can we think through how we would incorporate those latents if we had them?

- Supervised learning with latents from IGOR vs end to end
- Have some alternating
- Idea
    - Start with IGOR latents
→ So that we don’t have arbitrary latents that have to be aligned later (or use some distribution distance metric)
    - Continue with alternating approach

What is a symbolic planner

Can we design an action transformer that can be trained in parallel?

Argumentation Chain

For interpreting actions use

Interpreting model predictions

Supervised

Unsupervised


Can we use a variable time horizon

Uncertainty based horizon adaptation?