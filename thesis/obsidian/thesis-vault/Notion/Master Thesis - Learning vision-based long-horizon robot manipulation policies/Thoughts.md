---
notion-id: 2c420c92-0436-800a-9642-c157281ad80f
---
Open questions

- Best representation for the output plan (Exact interface between high-level and low-level policy)
    - Plan as latent which we sample from
    - Plan as masked images with target location mask
    - Plan as masked image with target x,y location
- Model for the planning
    - Options
        - Language model
        - Diffusion model ?



Simulator

- LTRON is not maintained / adopted by the community
- BloxNet → Pybullet

Challenge

- How can we make the planning model align blocks with the LEGO grid?
→ Is this part of the high level or low level policy?
→ Very fine grained!! Hard to get without trial and error from RL!

My questions

- Does Oliver have experience with simulators?
- 

How would the training pipeline look like?

- Offline training?
- Online training?
- RL?
- Fine-tuning?


Question:

- 10 years from now… how would this problem be accomplished?
- Do we still have a model that is separately trained for following instructions or do we have one single model being able to do it? How would this model be trained so that it generalizes to our task
→ Can we do this now already or develop for future scaling



1. Identify part in the real world
2. Potentially identify coordinates
3. Identify target location
4. 


How to make it cool:

- Human in the loop
- It should generalize to arbitrary initial instruction levels
    - Give it only final image, multiple intermetiate images, detailed instruction, text instruction, final text description
- Fill gaps in instruction
    - Analyze target if there are steps missing
→ Leverage reasoning VLM?
    - Could we have a VLM with function calling that can call functions to place certain parts in certain positions?
→ Then our isaac sim executes the instruction
- Scale and continue to improve without intervention
    - We want to be able to put arbitrary completed CAD parts into sim and then work with them
        - Backgrounds should be randomly generated to generalize to new backgrounds
- Quantify uncertainty in imagination?

- Take real world perspective
    - I want to be able give high level instructions
        - Examples
            - …?
    - 



Problem

- For the model it is hard to predict the final part position out of the blue
    - Even if it is given the before model and stone position and dimensions of each part still it has to do some math to figure out the exact target position

Idea

- 

Highlight contact patches of destination and original part?

How do we get the GT → We need the placed part

5. Have the instruction and the part pile
    - We assume all parts are there → no checking if set is complete
6. Select build location
    - Consideration: Should we manually select first part location or should we plan the dimensions of the goal object beforehand?
→ Planning ahead would be nicer
7. Model options (Here we can expand in the future)
    - Select part
    - Move part (relative to current part position)
    - Place part
→ Oracle tells the model whether it got closer to the target


Should we reduce the rgb image to black and white or pointclouds or something similar to filter out noise that makes generalization harder? reduce the representation to the minimum → maybe not necessary if the robot learns that, but could be more data efficient


leverage prior knowledge

- segment anything
- **FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects**
- DINOv3
- CLIP


Grasp → masked image

Move → relative position to current position

Place → ?? 

→ Parametrize action primitives



Whats the time horizon of the tasks that we are predicting?

Fine-tuning VLAs

Pro

Cons

Question

Predicting 



High level problem

Is a model-based approach exactly what we need?

Could we ablate a model-based RL approach for the PAMDP ?