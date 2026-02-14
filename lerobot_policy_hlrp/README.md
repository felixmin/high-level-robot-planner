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
