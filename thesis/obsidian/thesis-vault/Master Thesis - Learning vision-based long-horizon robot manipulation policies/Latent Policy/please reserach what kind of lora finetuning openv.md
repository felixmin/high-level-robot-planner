---
notion-id: 2f520c92-0436-8084-9a42-d63f89eb11ae
---
!![[pplx-full-logo-primary-dark@2x.png]]

# please reserach what kind of lora finetuning openvla oft uses and why

Now I’ll generate the comprehensive report:

## OpenVLA LoRA Fine-Tuning and OFT: Architecture and Design Rationale

### Executive Summary

OpenVLA employs **Low-Rank Adaptation (LoRA)** for fine-tuning because it enables parameter-efficient adaptation of a 7B-parameter vision-language model to robotics tasks with limited computational resources. However, the 2025 **Optimized Fine-Tuning (OFT)** recipe moves beyond standard LoRA by combining three key innovations—parallel decoding with action chunking, continuous action representations, and L1 regression—to address fundamental speed and quality limitations of autoregressive LoRA fine-tuning. This report details the architecture choices, their empirical justification, and why this represents a significant departure from conventional VLA fine-tuning approaches.

---

### Part 1: Why OpenVLA Uses LoRA

### The Fundamental Challenge: Parameter Efficiency at Scale

OpenVLA is a 7B-parameter model built on top of Prismatic VLMs (combining Llama 2 language backbone with DINOv2 and SigLIP visual encoders). Full fine-tuning on downstream robotic tasks would require storing optimizer states for all 7 billion parameters—a prohibitively expensive operation. However, fine-tuning is essential: pretraining on 970k demonstrations from the Open X-Embodiment dataset provides strong generalist capabilities, but real-world deployment typically requires task or robot-specific adaptation.[1](about:blank#fn1)

LoRA solves this through low-rank decomposition. Instead of updating weight matrices W ∈ ℝ^(m×n) directly during adaptation, LoRA freezes W and injects trainable rank-decomposition matrices:[2](about:blank#fn2)

**h = Wx + αW_up·W_down·x**

where W_down ∈ ℝ^(m×r) and W_up ∈ ℝ^(r×n), with r << min(m,n) (typically r = 32 for OpenVLA). This reduces trainable parameters from ~1.4M (full 7B) to only ~300K per task—a ~10,000× reduction.[3](about:blank#fn3)[4](about:blank#fn4)[5](about:blank#fn5)[6](about:blank#fn6)

**Why this matters for robotics:**

- **Memory Efficiency**: Only adapter parameters occupy GPU VRAM; base weights can be loaded in lower precision (fp16/int8).[7](about:blank#fn7)[8](about:blank#fn8)
- **Data Efficiency**: Small per-task datasets (e.g., 500 demonstrations for LIBERO vs. 1M for pretraining) don’t require full retraining.[9](about:blank#fn9)
- **Catastrophic Forgetting Prevention**: Freezing pretrained weights preserves foundational vision-language knowledge. Ablation studies confirm this: removing pre-trained representations causes a 5.2% absolute success rate drop on LIBERO.[10](about:blank#fn10)
- **Task Modularity**: Different LoRA adapters can be saved and swapped for different robots/tasks without retraining.[11](about:blank#fn11)[12](about:blank#fn12)[13](about:blank#fn13)

---

### Part 2: The Autoregressive LoRA Problem and OFT’s Solution

While LoRA solved the parameter efficiency challenge, original LoRA fine-tuning exposed a critical bottleneck: **inference speed incompatibility with real-time robotic control.**

### The Speed Ceiling: Autoregressive Generation

OpenVLA’s base architecture uses **autoregressive decoding** for action generation. At each timestep, the model generates 7 discrete action tokens sequentially (3 for position, 3 for orientation, 1 for gripper), with each token requiring a separate forward pass through the decoder:[14](about:blank#fn14)[15](about:blank#fn15)

- **Single timestep latency**: 0.24 seconds (0.33s per token × 7 tokens) on an A100 GPU[16](about:blank#fn16)
- **Throughput**: 4.2 Hz (insufficient for real-time control; robots require 25-50+ Hz)[17](about:blank#fn17)[18](about:blank#fn18)

When combined with action chunking (predicting K future actions before replanning), autoregressive latency becomes prohibitive:

**Latency_chunk = 0.24s × K timesteps**

For K=8 (typical for manipulation), this yields 1.92 seconds between action chunks—far too slow for closed-loop reactive control on bimanual robots. Existing faster tokenization schemes (e.g., VQ-based compression in FAST) achieved only 2-13× speedups, still leaving 100-750ms latency between chunks.[19](about:blank#fn19)

### The Quality Ceiling: Bimanual Manipulation Failure

Beyond speed, LoRA fine-tuning with autoregressive VLAs showed poor generalization to novel robots, particularly **bimanual manipulation** (double the action dimensionality). Prior work indicated that even diffusion-based approaches struggled when autoregressive generation remained the decoding scheme.[20](about:blank#fn20)[21](about:blank#fn21)[22](about:blank#fn22)

### OFT’s Three-Part Solution

To address both speed and quality, the 2025 OpenVLA-OFT paper conducted controlled experiments across three design dimensions, each empirically justified:[23](about:blank#fn23)

| Design Dimension | Original OpenVLA | OFT Alternative | Rationale |
| --- | --- | --- | --- |
| **Action Decoding** | Autoregressive (7 sequential passes) | Parallel (1 bidirectional pass) | Enables single forward pass for entire action chunk; enables efficient chunking |
| **Action Representation** | Discrete 256-bin tokens | Continuous normalized values [-1,+1] | Higher precision; softmax suited for language, not precise control |
| **Learning Objective** | Next-token CE loss | L1 regression | Simple, fast convergence; matches diffusion quality without multi-step inference |

---

### Part 3: Technical Deep Dive—OFT Design Choices

### 1. Parallel Decoding with Action Chunking

**Implementation:**

- Replace causal attention mask (which enforces sequential generation) with bidirectional attention
- Input empty action embeddings (differing only in positional encoding) for all future timesteps simultaneously
- Predict entire action chunk in single forward pass using the VLM’s decoder

**Why this works:**
Bidirectional attention allows each output position to attend to all inputs and other output positions, eliminating dependency on previous tokens. For a 7D action space with K=8 chunk size (56 output dimensions), this changes computational flow from 56 sequential passes to 1 parallel pass.

**Empirical Results (Table II in ):**[24](about:blank#fn24)

| Configuration | Throughput (Hz) | Latency (s) | Improvement |
| --- | --- | --- | --- |
| Base OpenVLA (AR) | 4.2 | 0.240 | — |
| + Parallel Decoding | 15.9 | 0.063 | 4× speedup |
| + Parallel + Chunking (K=8) | 108.8 | 0.074 | **26× throughput, no latency degradation** |
| + Continuous + L1 | 109.7 | 0.073 | Negligible overhead from action head MLP |

**Performance Gains (Table I in ):**[25](about:blank#fn25)

Beyond efficiency, parallel decoding + chunking improved task **success rates by 14% absolute** (76.5% → 90.2% LIBERO average). This is attributed to action chunking’s ability to capture temporal dependencies and reduce compounding errors—a known benefit in imitation learning.[26](about:blank#fn26)

**Architectural Preservation:** Despite the non-causal attention, no expressivity loss was observed. The model successfully predicts multi-timestep action sequences without autoregressive supervision.

---

### 2. Continuous Actions Over Discrete Tokens

**Problem with Discretization:**

- OpenVLA’s discrete representation: 256-bin quantization per action dimension
- Quantization loss: Rounding continuous [-1, +1] motor commands to discrete bins introduces irreversible precision loss
- Task impact: Fine-grained manipulation (e.g., pinch force for grasping) is compromised

**Implementation:**
Replace the output embedding layer with a simple 4-layer MLP action head that maps final decoder hidden states directly to continuous action values in [-1, +1].

**Empirical Results (Table I in ):**[27](about:blank#fn27)

| Representation | LIBERO Avg SR | LIBERO-Long SR | Notes |
| --- | --- | --- | --- |
| Discrete tokens (PD+AC) | 90.2% | 86.5% | Baseline parallel decoding |
| Continuous L1 | **95.3%** | **90.7%** | +5.1% improvement |
| Continuous Diffusion | 95.4% | 91.1% | +5.2%, but slower inference |

The 5% gain is particularly pronounced on long-horizon tasks (LIBERO-Long: +4.2%), where accumulated precision errors from discrete quantization compound.

**Why L1 Over Diffusion:**
While diffusion modeling provides higher capacity, it requires 50 denoising steps at inference—introducing 10× latency penalty (0.792s vs. 0.073s). L1 regression achieves comparable 95.3% vs. 95.4% success with orders-of-magnitude faster inference. This aligns with the observation that large VLMs already have sufficient capacity for multi-task action distributions.[28](about:blank#fn28)

---

### 3. L1 Regression Objective

**Problem with Next-Token Prediction:**

- Original objective: Cross-entropy loss on discrete action tokens
- Limitation: Designed for language modeling; fundamentally mismatches robotics where actions are continuous physical quantities
- Poor calibration: Softmax predictions don’t correlate with action precision

**Problem with Diffusion:**

- Approach: Learn to denoise noisy action samples (similar to Diffusion Policy)[29](about:blank#fn29)
- Benefit: More expressive; handles multi-modal action distributions
- Cost: Requires 50 denoising steps → 0.792s latency vs. 0.073s with L1[30](about:blank#fn30)

**Implementation:**
Minimize mean L1 distance between predicted continuous actions **ŷ** and ground truth **y**:

**Loss = (1/D) Σ |ŷ_i - y_i|**

where D is action dimensionality (7 for single-arm, 14 for bimanual).

**Empirical Results (Table II, Convergence in ):**[31](about:blank#fn31)

| Objective | Success Rate | Latency | Train Time | Convergence |
| --- | --- | --- | --- | --- |
| Discrete CE (AR baseline) | 76.5% | 0.240s | Fast | Fast |
| Discrete CE (PD+AC) | 90.2% | 0.074s | Moderate | Moderate |
| Cont. L1 | **95.3%** | **0.073s** | Moderate | 50-150K steps |
| Cont. Diffusion | 95.4% | 0.792s | Slow | 100-250K steps |

**Why L1 Wins:**
L1 is robust to noise (learns the median response of the action distribution rather than the mean, as with MSE). For VLA fine-tuning on small datasets where noise is high, this robustness is valuable. Combined with LoRA’s ability to preserve pre-trained knowledge, L1 achieves comparable performance to diffusion without the inference overhead.

---

### Part 4: Real-World Validation and OFT+ Extension

### ALOHA Robot Deployment Challenge

To validate OFT beyond simulation, the authors fine-tuned on a **bimanual ALOHA robot** operating at 25 Hz with three camera viewpoints (two wrist-mounted, one top-down) and 14D joint angle actions. This setup differs dramatically from OpenVLA’s pretraining (single-arm, low-frequency 3-10 Hz, relative end-effector pose actions, one camera).[32](about:blank#fn32)

Results demonstrated OpenVLA-OFT outperformed state-of-the-art bimanual VLAs ([π0], RDT-1B) and from-scratch methods (ACT, Diffusion Policy) by up to 15% on real-world tasks like cloth folding and bimanual object manipulation.[33](about:blank#fn33)

### FiLM Enhancement (OFT+)

A critical observation: On multi-view ALOHA tasks, policies sometimes failed to follow language instructions (e.g., “scoop almonds” but the model ignored the ingredient specification). The root cause: with multiple camera viewpoints, spurious visual correlations overshadowed language tokens during attention.

**Solution: Feature-wise Linear Modulation (FiLM)**[34](about:blank#fn34)

Instead of modulating individual patch embeddings (which proved ineffective), FiLM applies language-guided affine transformations to entire feature map dimensions:

**FiLM(F|γ, β) = (1 + γ) ⊙ F + β**

where γ, β ∈ ℝ^(D_ViT) are learned projections of language embeddings, applied across all visual patches. This infuses language information at the feature level, forcing the model to use language for action prediction.

**Impact:**

- Without FiLM on “scoop almonds”: 35% success (near random)
- With FiLM: 100% success (perfect language following)[35](about:blank#fn35)

This demonstrates that LoRA’s parameter efficiency extends to enabling architectural innovations (like FiLM) without full retraining.

---

### Part 5: LoRA Hyperparameter Configuration

From implementation details in and OpenVLA GitHub documentation:[36](about:blank#fn36)[37](about:blank#fn37)

| Parameter | Value | Rationale |
| --- | --- | --- |
| **Rank (r)** | 32 | Balances expressivity (higher r → more capacity but less efficiency) and efficiency; empirically shown to update only 1.4% of params[38](about:blank#fn38) |
| **LoRA Alpha (α)** | 64 (2× rank) | Scaling factor; higher α increases adaptation strength; typical range 2-4× rank[39](about:blank#fn39) |
| **Target Modules** | Attention layers (Q, K, V, O projections) | Standard practice; most critical layers for task-specific knowledge[40](about:blank#fn40) |
| **Learning Rate** | 5e-4 → 5e-5 (decay) | LoRA requires higher LR than full fine-tuning; decay prevents instability[41](about:blank#fn41) |
| **Batch Size** | 64-128 on 8×A100s | Scales with GPU count; LoRA enables larger batches on consumer hardware[42](about:blank#fn42) |
| **Gradient Steps** | 50-150K | Converges faster than full fine-tuning due to reduced parameter space[43](about:blank#fn43) |
| **Quantization** | Optional 4-bit (QLoRA) | Further memory savings; freezing base model enables int4 without accuracy loss[44](about:blank#fn44)[45](about:blank#fn45) |

---

### Part 6: Comparative Analysis—Why OFT Over Alternatives

| Approach | Speed | Quality | Simplicity | Cost |
| --- | --- | --- | --- | --- |
| **Full Fine-Tuning** | Slow | Excellent | High | 100GB+ VRAM |
| **LoRA (Original AR)** | Very Slow (3-5 Hz) | Good | Medium | 27GB VRAM |
| **LoRA + Diffusion** | Moderate (10-13 Hz) | Excellent | High | 27GB VRAM + 50 steps |
| **OFT (Parallel + L1)** | **Very Fast (109 Hz)** | **Excellent (95.3%)** | **Very High** | 27GB VRAM |
| **π0 / RDT-1B (diffusion)** | Moderate (20-30 Hz) | Excellent | Complex | 27GB+ VRAM |

**Key Trade-offs:**

- **OFT vs. Diffusion VLAs**: OFT is 4-10× faster at inference but requires retraining models from OpenVLA base. Diffusion VLAs (π0, RDT-1B) benefit from larger pretraining data but are slower.
- **OFT vs. Full Fine-Tuning**: OFT reduces parameters by 10,000× while matching full fine-tuning performance, enabling consumer GPU deployment.
- **OFT vs. Original LoRA**: OFT is 26× faster than original autoregressive LoRA while improving quality (+20.6% LIBERO success).

---

### Conclusion

OpenVLA uses **LoRA fundamentally because small robotics datasets cannot justify full fine-tuning of 7B parameters**. However, the 2025 OFT recipe recognizes that standard LoRA fine-tuning—which preserves the base model’s autoregressive architecture—creates a speed barrier incompatible with real-time robotic control.

By introducing **parallel decoding + action chunking, continuous actions, and L1 regression**, OFT achieves three goals simultaneously:

1. **26-43× throughput improvement** (4.2 Hz → 109 Hz, enabling 25 Hz bimanual control)
2. **Quality gains** (+20.6% LIBERO success vs. vanilla LoRA; 97.1% vs. 76.5%)
3. **Architectural flexibility** (multi-view inputs, action chunking, language grounding via FiLM)

This represents a departure from treating LoRA as merely a “parameter-efficient wrapper” around the original model. Instead, OFT demonstrates that LoRA enables re-architecting the action generation mechanism itself, unlocking capabilities unavailable in full fine-tuning approaches on robotics-specific hardware constraints.

---

### References

OpenVLA GitHub - https://github.com/openvla/openvla[46](about:blank#fn46)
Hu et al., LoRA: Low-Rank Adaptation of Large Language Models[47](about:blank#fn47)[48](about:blank#fn48)[49](about:blank#fn49)
OpenVLA-OFT Paper - Fine-Tuning Vision-Language-Action Models[50](about:blank#fn50)[51](about:blank#fn51)
VLA-Adapter Paper[52](about:blank#fn52)[53](about:blank#fn53)[54](about:blank#fn54)
Haonan Yu blog - OpenVLA finetuning with online RL[55](about:blank#fn55)[56](about:blank#fn56)
LoRA Technical Deep Dive - ML6 Blog[57](about:blank#fn57)[58](about:blank#fn58)
Understanding OpenVLA - Hankyu Kim[59](about:blank#fn59)[60](about:blank#fn60)
Parameter-Efficient Fine-Tuning Overview[61](about:blank#fn61)[62](about:blank#fn62)
Efficient Domain Adaptation - Hyper-LoRA[63](about:blank#fn63)[64](about:blank#fn64)
LoRA Explained - Hugging Face LLM Course[65](about:blank#fn65)[66](about:blank#fn66)
Original OpenVLA Paper Abstract[67](about:blank#fn67)[68](about:blank#fn68)
OpenVLA-OFT Project Website[69](about:blank#fn69)[70](about:blank#fn70)[71](about:blank#fn71)[72](about:blank#fn72)[73](about:blank#fn73)[74](about:blank#fn74)

⁂

---

4. https://github.com/openvla/openvla[↩︎](about:blank#fnref1)
5. https://arxiv.org/html/2510.09976v1[↩︎](about:blank#fnref2)
6. https://huggingface.co/learn/llm-course/en/chapter11/4[↩︎](about:blank#fnref3)
7. https://dagshub.com/blog/streamlining-fine-tuning-with-lora-optimizing-parameter-selection-for-llms/[↩︎](about:blank#fnref4)
8. https://www.hankyukim.com/openvla[↩︎](about:blank#fnref5)
9. https://github.com/openvla/openvla[↩︎](about:blank#fnref6)
10. https://dagshub.com/blog/streamlining-fine-tuning-with-lora-optimizing-parameter-selection-for-llms/[↩︎](about:blank#fnref7)
11. https://huggingface.co/learn/llm-course/en/chapter11/4[↩︎](about:blank#fnref8)
12. https://www.azion.com/en/learning/ai/what-is-lora-fine-tuning/[↩︎](about:blank#fnref9)
13. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref10)
14. https://www.ultralytics.com/glossary/lora-low-rank-adaptation[↩︎](about:blank#fnref11)
15. https://www.ml6.eu/en/blog/low-rank-adaptation-a-technical-deep-dive[↩︎](about:blank#fnref12)
16. https://www.azion.com/en/learning/ai/what-is-lora-fine-tuning/[↩︎](about:blank#fnref13)
17. https://www.roboticsproceedings.org/rss21/p014.pdf[↩︎](about:blank#fnref14)
18. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref15)
19. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref16)
20. https://openvla-oft.github.io[↩︎](about:blank#fnref17)
21. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref18)
22. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref19)
23. https://arxiv.org/html/2509.09372v2[↩︎](about:blank#fnref20)
24. https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms[↩︎](about:blank#fnref21)
25. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref22)
26. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref23)
27. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref24)
28. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref25)
29. https://arxiv.org/abs/2406.09246[↩︎](about:blank#fnref26)
30. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref27)
31. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref28)
32. https://openreview.net/forum?id=sFO9d6XSlf[↩︎](about:blank#fnref29)
33. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref30)
34. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref31)
35. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref32)
36. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref33)
37. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref34)
38. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref35)
39. https://github.com/openvla/openvla[↩︎](about:blank#fnref36)
40. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref37)
41. https://www.hankyukim.com/openvla[↩︎](about:blank#fnref38)
42. https://huggingface.co/learn/llm-course/en/chapter11/4[↩︎](about:blank#fnref39)
43. https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms[↩︎](about:blank#fnref40)
44. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref41)
45. https://github.com/openvla/openvla[↩︎](about:blank#fnref42)
46. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref43)
47. https://huggingface.co/learn/llm-course/en/chapter11/4[↩︎](about:blank#fnref44)
48. https://dagshub.com/blog/streamlining-fine-tuning-with-lora-optimizing-parameter-selection-for-llms/[↩︎](about:blank#fnref45)
49. https://github.com/openvla/openvla[↩︎](about:blank#fnref46)
50. https://arxiv.org/html/2510.09976v1[↩︎](about:blank#fnref47)
51. https://huggingface.co/learn/llm-course/en/chapter11/4[↩︎](about:blank#fnref48)
52. https://dagshub.com/blog/streamlining-fine-tuning-with-lora-optimizing-parameter-selection-for-llms/[↩︎](about:blank#fnref49)
53. https://arxiv.org/html/2509.09372v2[↩︎](about:blank#fnref50)
54. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref51)
55. https://www.haonanyu.blog/post/openvla_rl/[↩︎](about:blank#fnref52)
56. https://www.themoonlight.io/en/review/vla-adapter-an-effective-paradigm-for-tiny-scale-vision-language-action-model[↩︎](about:blank#fnref53)
57. https://arxiv.org/html/2509.09372v2[↩︎](about:blank#fnref54)
58. https://www.haonanyu.blog/post/openvla_rl/[↩︎](about:blank#fnref55)
59. https://openreview.net/forum?id=sFO9d6XSlf[↩︎](about:blank#fnref56)
60. https://www.themoonlight.io/en/review/vla-adapter-an-effective-paradigm-for-tiny-scale-vision-language-action-model[↩︎](about:blank#fnref57)
61. https://www.ml6.eu/en/blog/low-rank-adaptation-a-technical-deep-dive[↩︎](about:blank#fnref58)
62. https://www.hankyukim.com/openvla[↩︎](about:blank#fnref59)
63. https://arxiv.org/pdf/2502.19645.pdf[↩︎](about:blank#fnref60)
64. https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-goal[↩︎](about:blank#fnref61)
65. https://softwaremind.com/blog/parameter-efficient-fine-tuning-peft-benefits-and-techniques/[↩︎](about:blank#fnref62)
66. https://vla-adapter.github.io[↩︎](about:blank#fnref63)
67. https://openreview.net/pdf?id=SEvvWs3CAL[↩︎](about:blank#fnref64)
68. https://www.alphaxiv.org/overview/2502.19645v2[↩︎](about:blank#fnref65)
69. https://huggingface.co/learn/llm-course/en/chapter11/4[↩︎](about:blank#fnref66)
70. https://www.roboticsproceedings.org/rss21/p014.pdf[↩︎](about:blank#fnref67)
71. https://arxiv.org/abs/2406.09246[↩︎](about:blank#fnref68)
72. https://github.com/OpenHelix-Team/VLA-Adapter[↩︎](about:blank#fnref69)
73. https://openvla-oft.github.io[↩︎](about:blank#fnref70)
74. https://arxiv.org/html/2510.25616v1[↩︎](about:blank#fnref71)
75. https://liner.com/review/vlaadapter-effective-paradigm-for-tinyscale-visionlanguageaction-model[↩︎](about:blank#fnref72)
76. https://www.truefoundry.com/blog/lora-fine-tuning[↩︎](about:blank#fnref73)
77. https://www.ibm.com/think/topics/lora[↩︎](about:blank#fnref74)
