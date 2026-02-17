---
notion-id: 28c20c92-0436-8098-b81b-d715b8ac2585
---
[RLHF in 90 min](https://www.youtube.com/watch?v=j3BdFm_Veq4)

SFT

- Model is provided examples of (prompt, optimal response)
- Requires lots of data
- Is black and white

Better: Rank different responses


[LLM Training & Reinforcement Learning from Google Engineer | SFT + RLHF | PPO vs GRPO vs DPO](https://www.youtube.com/watch?v=aB7ddsbhhaU)

Current popular approach

1. Collect expert data and fine-tune the model (SFT)
2. RLHF
    1. Some model outputs and answers are sampled and ranked by an expert
    2. Data is used to train an reward model
    3. Generate answers and use (differentiable) reward to fine-tune the model

RL