---
notion-id: 29220c92-0436-802b-ab9f-e027bce52346
---
Action oriented vs state oriented


How fine granular should the output be?

- We have different skills
    - Screw
    - Wipe
    - Take
    - …
- Each for these skills is described with a single word but also can be broken down into multiple words

→ Should which of them should the output be? Or all of them? Or a latent?

Idea: Autoencoder that is trained to create an abstract skill latent

How often does the high level planner run?

- All the time or only on demand?

What should the representation contain?

Can every action be described in text?

Idea: Maybe there is no proper grounding for something like fold the bedsheets?

> [!note]+ Why contact point prediction for the guider?
> High level is expert for what to do after another
> 
> Where to grasp? → Sounds more like a low level policy thing… You sure? Maybe it depends on the task horizon
> 
> → For lego pick and place where we grasp it might even be relevant, otherwise we have to reach around
> 

> [!note]+ We want continuous actions, not discrete primitives… why? Because each slight variation might be a new primitive… where is the differentiation? Where does the new primitive start, where does the old end?
> A single conditioned low level policy
> 

## Action oriented

> [!note]+ What does action oriented mean?
> We describe what the robot should do
> 
> E.g. grasp this, move here, let go, …

> [!note]+ Action primitives
> Pro
> 
> Con

> [!note]+ Contact points
> Pro
> 
> - Maybe we also think this way? 
> 
> Con
> 
> - Maybe sufficient for pushing, definitely not sufficient for pick and place

> [!note]+ Abstract latent
> Pro
> 
> - 
> 
> Con
> 
> - How to train?
> - Advantage? → Could be that this could better carry the abstract meaning of moving the part through space?

## State oriented

> [!note]+ What does state oriented mean?


Goal states





Text description


