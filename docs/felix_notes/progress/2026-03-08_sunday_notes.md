- check what the last run was... i think i prompted the model to start a only real action head but in the run command in wandb it looks like latent head weight was 1 https://wandb.ai/felixmin/lerobot/runs/tu3legsv/overview?nw=nwuserfelixmin
  - check if the script is wrongly assigning this
  - check if the model just did wrong
- check the params from the last run in detail and compare against smolvla paper.. is everything corrrect? also check stuff like image dimensions
- check the difference between policy ation chunks and steps or something
- do some more evals of this last run on the different lerobot environments
- check if batches are currently always clean


- check whether research about when interpolation between training samples works and when they are "too far" is a promising direction
  - what was done for robotics or other directions


- rollouts are shit... check why they are shit... i suppose there must be an error somehwere... script 7 taking different normalization or whatever than script 6


- in progress
    - make downloader script _1 and ensure the paths are set properly. make downloader also set the env etc, just like the other scripts


    
    - autodownload failed to to some camera selectoin issue in lerobot_v3_source... check whats going on here. 
 
 conda run -n lerobot python scripts/7_rollout_lerobot.py \
    experiment=stage3_rollout_local \
    lerobot_eval.policy_path=/mnt/data/workspace/runs_root/runs/2026-03-08_00-31-29_stage3_hlrp_libero_action_only_eval10k_local/lerobot/checkpoints/last/
  pretrained_model \
    experiment.name=stage3_rollout_action_only_paper_libero