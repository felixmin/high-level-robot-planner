from lerobot_policy_hlrp.policies.hlrp_smoke.modeling_hlrp_smoke import HLRPSmokePolicy
from lerobot_policy_hlrp.policies.hlrp_smoke_b.configuration_hlrp_smoke_b import HLRPSmokeBConfig


class HLRPSmokeBPolicy(HLRPSmokePolicy):
    """Second smoke policy variant that reuses the same implementation core."""

    config_class = HLRPSmokeBConfig
    name = "hlrp_smoke_b"
