---
notion-id: 28720c92-0436-802a-9da5-edad6510dc31
---
> [!note]+ Older ideas and approaches
> [https://docs.google.com/presentation/d/1DmvTXJtJMe3xx9VD3PoqL24nno6ARdgiKoyOnVHVc4s/edit?slide=id.g36134487619_0_0#slide=id.g36134487619_0_0](https://docs.google.com/presentation/d/1DmvTXJtJMe3xx9VD3PoqL24nno6ARdgiKoyOnVHVc4s/edit?slide=id.g36134487619_0_0#slide=id.g36134487619_0_0)
> 
> [[Chat / Gemini Research]]
> 
> [[Approaches]]
> 
> [[Which is the proper interface]]
> 
> [[Target position]]
> 
> [[PPO; GRPO; DPO; RLHF]]
> 
> [[Thoughts]]
> 
> [[Motion-Focused Selective Augmentation Research]]

## Current Todos

[[Ideas 2025-12-18]]

[[2025-12-22]]

## Literature

[[MBREUSS VLA Post]]

[[Related Work]]

[[Thesis - Paper Text]]

[[Other thoughts]]

[[Data Collection]]

## Data, Model, Training

[[Data]]

[[Latent Action Model]]

[[Latent Policy]]

[[Large scale training]]

[[Loader Learnings and Questions]]

[[Code]]

[[Model]]

## Experiments

[[ChatGPT Planning Test]]

## Thoughts

> [!note]+ Video vs Lanugage models for latent prediction
> I think i would like to continue with fine-tuning a VLA on latent actions approach
> 
> Why:
> 
> - Differentiator: most papers try to improve sota with new architectures but there is no other new "functionality"... the latent action approach can serve as a conditioning for lightweight downstream policies (like ACT or custom experiments)... then others that want to build a policy can use the latent action model as quick starting point for their experiments
> - We want long-term planning: my hypothesis is that the more fine-granular the output of a model, the more attention of the model goes into the details. that means a very abstract model like a language model does not put too much attention to the details and can focus on long term... a video model on the other hand has to predict a lot of fine-grained stuff and therefore struggles with longer term
> - Long-term planning: my hypothesis is that the output representation can serve as a regularization. Language as output regularizes long term plans so that they still make sense, while video doesnt have this and long term plans can suffer. Also the model puts more of its reasoning capacity into fine-grained details that are not relevant for the long term plan... I believe therefore language models are better for long term planning than video models and we

> [!note]+ VLAs vs Flow matching
> for fine grained control: flow matching and diffusion
> 
> for high-level planning: VLAs
> 
> see groot
> 
> ## The Emerging Winner: Hybrid Architectures
> 
> Recent SOTA systems **combine both approaches**:
> 
> **Dual-expert systems** (Nvidia Groot, Figure AI Helix):
> 
> - **System 2 (VLM)**: Slow reasoning for complex decisions and task decomposition
> - **System 1 (Diffusion)**: Fast, reactive motion generation for smooth control
> 
> **Joint learning with motion diffusion**:
> 
> - Extends VLA with a **dual-head design**: action head (autoregressive) + motion head (DiT for optical flow prediction)
> - Achieves **97.5% on LIBERO** and **+23% real-world improvement** over vanilla π₀
> - 
> - Adds motion reasoning without changing inference latency
> - 
> 
> **Performance comparison on narrow tasks**:
> 
> - **Diffusion Policy from scratch** > **Fine-tuned OpenVLA** > **Octo** (generalist baseline) for single-instruction tasks
> - But VLAs win decisively on **multi-task**, **multi-embodiment**, and **semantic generalization**
> - 
> 
> ## Bottom Line for Your Work
> 
> For embodied AI research like LAPA:
> 
> 1. **VLAs are winning on generalization and zero-shot transfer**, making them better for foundation models
> - 
> - **Diffusion policies remain superior for precision control** on known tasks, especially high-frequency manipulation
> - 
> - **The frontier is hybrid**: VLA backbone + diffusion-based motion head or dual-level systems
> - 
> - **Practical deployment favors compressed diffusion** (LightDP achieves 93× speedup on mobile devices) or optimized VLAs (OpenVLA-OFT runs at 25 Hz)
> 2. 
> 
> Given your focus on video understanding (LAPA), the **motion prediction component in hybrid VLA+diffusion systems** is directly relevant—they're essentially learning visual dynamics models alongside action policies.

> [!note]+ Maybe read andreea bobus aligning representations paper and construct theoretical underpinning for latent actions? what are theoretical requirements??


!![[Daily Todos.base]]

!![[Meetings.base]]
