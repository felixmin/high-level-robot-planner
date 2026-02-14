import torch

from lerobot_policy_hlrp.policies.hlrp_smoke.processor_hlrp_smoke import (
    make_hlrp_smoke_pre_post_processors,
)
from lerobot_policy_hlrp.policies.hlrp_smoke_b.configuration_hlrp_smoke_b import HLRPSmokeBConfig


def make_hlrp_smoke_b_pre_post_processors(
    config: HLRPSmokeBConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    return make_hlrp_smoke_pre_post_processors(config=config, dataset_stats=dataset_stats)
