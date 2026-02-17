# lerobot_policy_hlrp

LeRobot plugin package for HLRP policies.

This package currently includes two installable smoke-test policies:

- `hlrp_smoke`
- `hlrp_smoke_b`

## Quickstart

```bash
pip install -e /path/to/lerobot
pip install -e /path/to/high-level-robot-planner/lerobot_policy_hlrp
python /path/to/high-level-robot-planner/lerobot_policy_hlrp/scripts/check_available.py
```

If discovery and registration work, both policy types should instantiate via LeRobot.


## First working run with our smoke policy
HF_HOME=/mnt/data/tmp/hf     HF_HUB_CACHE=/mnt/data/tmp/hf/hub     HF_DATASETS_CACHE=/mnt/data/tmp/hf/datasets     lerobot-train       --policy.type hlrp_smoke       --policy.device cpu       --env.type libero       --dataset.repo_id HuggingFaceVLA/libero       --steps 500000       --batch_size 64       --save_freq 10000       --log_freq 100       --eval_freq 5000       --policy.push_to_hub false       --dataset.video_backend pyav       --env.task libero_object       --eval.n_episodes 10       --eval.batch_size 2       --wandb.enable true       --job_name hlrp_smoke_libero_test_cpu