- check what the last run was... i think i prompted the model to start a only real action head but in the run command in wandb it looks like latent head weight was 1 https://wandb.ai/felixmin/lerobot/runs/tu3legsv/overview?nw=nwuserfelixmin
  - check if the script is wrongly assigning this
  - check if the model just did wrong
- check the params from the last run in detail and compare against smolvla paper.. is everything corrrect? also check stuff like image dimensions
- check the difference between policy ation chunks and steps or something
- do some more evals of this last run on the different lerobot environments
- 


- check whether research about when interpolation between training samples works and when they are "too far" is a promising direction
  - what was done for robotics or other directions
