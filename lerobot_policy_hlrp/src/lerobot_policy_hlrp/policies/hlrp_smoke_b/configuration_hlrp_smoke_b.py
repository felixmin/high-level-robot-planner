from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from lerobot_policy_hlrp.policies.hlrp_smoke.configuration_hlrp_smoke import HLRPSmokeConfig


@PreTrainedConfig.register_subclass("hlrp_smoke_b")
@dataclass
class HLRPSmokeBConfig(HLRPSmokeConfig):
    """Alternative smoke policy config to validate multi-policy bundles."""

    hidden_dim: int = 64
