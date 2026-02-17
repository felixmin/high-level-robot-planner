---
notion-id: 2b820c92-0436-80ba-bc26-e94e9f5e6272
---
# What is being put in/out ?

Options

- RGB
- Motion Track
- Masks
- Depth

If we dont have depth we will struggle with detecting depth wise movement

→ Do we have depth wise movement? If yes we have to analyze

# Architecture

Positional embeddings

## Ideas

> [!note]+ Make the **reconstruction loss** **invariant **to slight** camera movements**
> If the image shifts by a few pixels loss should still be 0… only significant movements should be captured
> 
> How can we encode that in our loss?
> 
> Is that already achieved by using embeddings like DINOv3?
> 
> > [!tip] 💡
> > Check if dinov3 embeddings change completely upon minimal camera shift
> 
> With basic reconstruction everything goes to trash upon slight camera shift but with our method there should only be a minimal loss increase
> 
> Maybe dinov3 is sufficient but maybe we need more advanced loss technique here
> 
> > [!tip] 💡
> > We first have to analyze whether this is an issue in the first place at all!

> [!note]+ Avoid having to reconstruct the full image to focus on the parts with movement
> Instead of outputting the full image we take the original image and add the output of the LAM
> 
> With this the LAM can focus on the moving parts
> 
> → But this of course does not work with camera movement etc… maybe this limits our ability to generalize and scale??
> 

> [!note]+ Compare performance of different architectures
> Start with linear model and then move to more advanced models like VQ-VAE and spatio-temporal transformers


# Training

- Use this gradient clipping?
`gradient_clip_val=1.0`
`algorithm="norm"`

# Evaluation of the Latents


# How to filter out task-irrelevant dynamics?

## Through the representation

Don’t reconstruct the future image but rather the motion track 

## Learn a separate camera movement model

Have 2 models

1. Predict next camera view from camera transition
2. Predict next state from latent action

Encoder

Decoder

## Focus on the actions

It must somehow be possible to leverage the text labels to differentiate task relevant and task irrelevant motion… there is a correlation between labels and the relevant motion in the image and no correlation between label and irrelevant motion → somehow use this correlation

- If we have language labels for the videos leverage them somehow
    - Add a contrastive component between 

Can we have “motion tokens” in the latent space? how can we make the motion tokens learn only motion and the task tokens learn only tasks?

→ If we want to do this we dont only have IDM and FDM anymore… our models not only understand dynamics but also semantics because the acquire an understanding of what the relevant actor does and what is irrelevant… the model understands where to focus on

→ maybe focus is the keyword here. where to focus your attention

I think the following paper investigates that

[[Related Work]] 


# Thoughts

> [!note]+ Is the bottleneck required and central? Is that why autoencoders are a must? Are other approaches possible?
> Read in 