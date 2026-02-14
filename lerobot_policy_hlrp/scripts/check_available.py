"""Validate plugin discovery and smoke-policy instantiation via LeRobot."""

import importlib
import logging

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE


def _register_plugins() -> None:
    """Register third-party plugins for lerobot versions with/without auto discovery."""
    try:
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
    except (ImportError, AttributeError):
        # Older LeRobot versions may not expose register_third_party_plugins.
        # Importing the plugin package directly still executes registration decorators.
        import lerobot_policy_hlrp  # noqa: F401


def _resolve_policy_class(policy_type: str):
    config_cls = PreTrainedConfig.get_choice_class(policy_type)
    config_cls_name = config_cls.__name__
    model_name = config_cls_name.removesuffix("Config")
    if model_name == config_cls_name:
        raise ValueError(
            f"Config class '{config_cls_name}' does not end with 'Config'."
        )

    module_path = config_cls.__module__.replace("configuration_", "modeling_")
    module = importlib.import_module(module_path)
    return getattr(module, model_name + "Policy")


def _resolve_processors(policy_type: str, cfg, dataset_stats=None):
    function_name = f"make_{policy_type}_pre_post_processors"
    module_path = cfg.__class__.__module__.replace("configuration_", "processor_")
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function(cfg, dataset_stats=dataset_stats)


def _run_smoke_test(policy_type: str) -> None:
    cfg_cls = PreTrainedConfig.get_choice_class(policy_type)
    cfg = cfg_cls()

    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    cfg.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }

    policy_cls = _resolve_policy_class(policy_type)
    policy = policy_cls(config=cfg)

    _resolve_processors(policy_type=policy_type, cfg=cfg, dataset_stats=None)

    batch = {
        OBS_STATE: torch.zeros(2, 8),
        ACTION: torch.zeros(2, 4),
    }
    loss, info = policy.forward(batch)
    action = policy.select_action({OBS_STATE: torch.zeros(2, 8)})

    assert action.shape == (2, 4), f"Unexpected action shape for {policy_type}: {tuple(action.shape)}"
    logging.info("%s ok | loss=%s | info=%s", policy_type, float(loss.detach().cpu()), info)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    _register_plugins()

    for policy_type in ("hlrp_smoke", "hlrp_smoke_b"):
        _run_smoke_test(policy_type)

    logging.info("All smoke policy checks passed.")


if __name__ == "__main__":
    main()
