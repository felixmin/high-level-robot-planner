"""LeRobot plugin package containing HLRP policy implementations."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use lerobot_policy_hlrp."
    ) from exc

from lerobot_policy_hlrp.policies.hlrp_smoke.configuration_hlrp_smoke import HLRPSmokeConfig
from lerobot_policy_hlrp.policies.hlrp_smoke.modeling_hlrp_smoke import HLRPSmokePolicy
from lerobot_policy_hlrp.policies.hlrp_smoke.processor_hlrp_smoke import (
    make_hlrp_smoke_pre_post_processors,
)

from lerobot_policy_hlrp.policies.hlrp_smoke_b.configuration_hlrp_smoke_b import HLRPSmokeBConfig
from lerobot_policy_hlrp.policies.hlrp_smoke_b.modeling_hlrp_smoke_b import HLRPSmokeBPolicy
from lerobot_policy_hlrp.policies.hlrp_smoke_b.processor_hlrp_smoke_b import (
    make_hlrp_smoke_b_pre_post_processors,
)
from lerobot_policy_hlrp.policies.hlrp_smolvla_shared.configuration_hlrp_smolvla_shared import (
    HLRPSmolVLASharedConfig,
)
from lerobot_policy_hlrp.policies.hlrp_smolvla_shared.modeling_hlrp_smolvla_shared import (
    HLRPSmolVLASharedPolicy,
)
from lerobot_policy_hlrp.policies.hlrp_smolvla_shared.processor_hlrp_smolvla_shared import (
    make_hlrp_smolvla_shared_pre_post_processors,
)

__all__ = [
    "HLRPSmokeConfig",
    "HLRPSmokePolicy",
    "make_hlrp_smoke_pre_post_processors",
    "HLRPSmokeBConfig",
    "HLRPSmokeBPolicy",
    "make_hlrp_smoke_b_pre_post_processors",
    "HLRPSmolVLASharedConfig",
    "HLRPSmolVLASharedPolicy",
    "make_hlrp_smolvla_shared_pre_post_processors",
]
