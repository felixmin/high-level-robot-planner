---
notion-id: 29e20c92-0436-80e4-a732-f1f10dfb1080
---
## Lego Videos

[https://www.youtube.com/@PureBuilds/videos](https://www.youtube.com/@PureBuilds/videos)

[https://www.youtube.com/@brickbuilder23/videos](https://www.youtube.com/@brickbuilder23/videos)

[https://www.youtube.com/@AustrianBrickFan/videos](https://www.youtube.com/@AustrianBrickFan/videos)

[https://www.youtube.com/@MADABOUTLEGO/videos](https://www.youtube.com/@MADABOUTLEGO/videos)

## Labels in my dataset

gripper: hand or robot

part: lego or other item

camera: static or moving

bimanual

## Features in my dataset

images 256 x 256

masks of hand, masks of lego 

motion tracks

check again optical flow!

Regarding the optical flow i was also thinking:
For us it is not really relevant if we have dense or sparse optical flow
-> We are using optical flow because we are predicting motion and optical flow contains all this motion... the pixels are not relevant for our movement predictions
-> Neighboring pixels move in almost the same way, only with quasi-dense or sparse optical flow they differ significantly
-> As our action tokens are really high-level, we


→ Maybe provide image latents to the model instead of images (could be processed in the flow)

## Use cases

### VAE

encoder

decoder