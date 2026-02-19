---
notion-id: 2a920c92-0436-80e0-be42-e77e3188c9a2
---
**Core Finding**: Augmenting motion-specific/hard samples improves video model performance for motion prediction tasks.

**Key Papers & Evidence**:

1. **Motion-Focused Contrastive Learning (MCL)** - Li et al. 2021
    - Uses optical flow to identify motion-rich regions and augments those specifically
    - Temporal sampling filters static clips, spatial cropping selects high-velocity regions
    - Outperforms ImageNet supervised pretraining on UCF101 (81.91% vs 75.13%)
    - "Capitalizing on motion to achieve augmentations exhibits performance boost"
2. **Motion Coherent Augmentation (MCA)** - Zhang et al. 2024
    - Explicitly encourages models to prioritize motion over static appearance
    - Introduces appearance variation to force learning of motion patterns
    - Improves performance particularly on motion-focused datasets
3. **MAC: Mask-Augmentation for Motion-Aware Video** - Akar et al. 2022
    - Extracts foreground motion via frame differences, augments motion regions
    - Enforces learning of motion-based features vs background
    - Superior performance at low resource settings on UCF101, HMDB51
4. **When Dynamic Data Selection Meets Data Augmentation** - 2024
    - Augments **low-density samples** (underrepresented/boundary regions)
    - Low-density regions often = hard-to-predict motion boundaries
    - "Augmentation of sparse samples improves generalization across diverse data regions"
    - Achieves 94.9-96.0% on CIFAR with selective augmentation
5. **Selective Synthetic Augmentation (HistoGAN)** - Xue et al. 2020
    - Selectively augments only high-confidence hard samples
    - "Significantly outperforms arbitrary synthetic augmentation"
    - Selective > Random augmentation

**Industry Practice**:

- LocoMotion (2024): Generates motion-focused synthetic data for training
- Object Concepts from Motion (2013): Motion-based instance masks used for contrastive learning
- MotionAug (2023): Physical motion correction augmentation outperforms noise-based methods

**Why It Works**:

6. **Addresses distribution mismatch**: Overparameterized models can memorize easy static patterns, selective augmentation forces learning of motion dynamics
7. **Reduces motion prediction bias**: Models default to appearance cues when motion is hard to predict
8. **Sample efficiency**: Focusing augmentation on informative (motion-rich/boundary) samples > uniform augmentation
9. **Generalization**: Augmented motion samples = better decision boundaries for temporal features

**For Your Use Case (LEGO manipulation)**:

- Extract object motion trajectories from videos
- Augment samples where motion is complex/ambiguous (e.g., rotation-heavy placements)
- Should improve model's ability to predict precise part movements vs relying on scene context
- Particularly valuable given "overparameterized but information-rich" video data