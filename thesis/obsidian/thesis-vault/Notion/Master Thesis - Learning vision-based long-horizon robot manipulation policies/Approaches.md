---
notion-id: 28d20c92-0436-80b4-be54-d7a2ed7e689a
---
## Simple VLM + Masking

Model

- VLM: (manual, obs) → (relevant part, action)
- Masking: (relevant part, obs) → (masked obs) 
- Output: (masked obs, action)

Training

- Pretrained VLM should work out of the box
- How to train the masking? (maybe see [LLM-Seg](/28d20c92043680a3a840e329cbe3efb2#28d20c92043680d895a9cdd21f707ff5))

Pro

- Simple 
    - Coordinate free
    - We don’t have to move anything in sim… We only work with observations and masking
- Should work more or less out of the box
- Only fine-tuning required for improvements

Cons

- Rotation of part at target position is not clear
    - Just by goal mask we don’t have the goal rotation
    - Iterative
        - We could also have a rotation action that is called if the VLM detects that the pose was not correct → This would iteratively figure out the correct pose
    - Direct prediction
        - See extension below
- Questionable how good this works without specialized models and proper training and without rotation

Self-improving

- We can train a high-level place policy along the way
    - (part current coordinates, masked obs, action) → (part goal coordinates)
- With this we can have the model run and generate fine-tuning training data along the way
→ Alternating assemble and disassemble

## Simple VLM + Masking + Rotation (extension to above)

Model

- VLM: (manual, obs) → (relevant part, action)
- Masking: (relevant part, obs) → (masked obs) 
- Rotation (current orientation) → (target orientation)
- Output: (masked obs, action, rotation)

Training

- See above
- How to train rotation? Where do we get the ground truth for the rotation?
    - manual dataset creation,
    - creating a dataset with the iterative approach, or
    - using the data from sim → we need to be able to move parts in sim (approach is not super easy anymore)
    - . However that means we don’t get renderings of how a certain orientation would look like and therefore don’t have training data.

Pro

- Simple if rotation can be solved

Cons

- Lower level model has to deduce that the target 
- Questionable how good this works

## VLM predicting Latent Actions

How to get Latent action

how to get action labeled videos for training the policy

question: how bad are the latents if we only use manual lego assembly as reference

ablate between different types of data in the training set

- robot lego manip
- human lego manip
- also other instrument for lego manip (pliers)
- combinations
- none

→ can we train a solid autoencoder with this alone?

compare

Which data would we need to collect

If training on video data how long should one action be? which is optimal? should this be variable or fixed? what if the robot varies?