---
notion-id: 2e620c92-0436-8068-ad07-d4a9b91bfb96
---
## Intro

par1


par2

par3

par4

## Related Work

Latent action forms

Visual encoder

Capability extension: 2 directions

## Problem setting

learning a good latent actions from videos


## Method

3 stage

1. pre train
2. distill
3. fine-tune

each corresponding to specific dataset

2 cases considered

### 1. Pseudo Supervision from Optical Flow

IDM

FDM and flow decoder

same as ours… combines reconstruction and optical flow

### 2. Learning with Sparse Action Supervision

??

### 3. Optical Flow in RGB Format

### 4. Object-Centric Optical Flow

## Experiments

conti vs discrete

optical flow better or not

no powerful VLA but extending baselines

## Discussion

LAOF: (x1,x2) → (z) ; (z,x1) → (image)(flow)

LAOF-FlowFDM: (x1,x2) → (z) ; (z,x1) →  (image,flow)

LAOF-Only (z): (x1,x2) → (z) ; (z) → (flow)

LAOF-Only (z,s): (x1,x2) → (z) ; (z,x1) → (flow)

LAOF-AE: (flow) → (z) ; (z) → (flow)

> [!tip] 💡
> This is highly unrealistic!! LAOF-Only (z) is probably not better than (z,s) why would it be??

## Limitations

Optical flow quality

LangSAM for object seg

multi-view LAOF