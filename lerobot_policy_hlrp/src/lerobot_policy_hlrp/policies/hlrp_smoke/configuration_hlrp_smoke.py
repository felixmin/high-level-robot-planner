from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig


@PreTrainedConfig.register_subclass("hlrp_smoke")
@dataclass
class HLRPSmokeConfig(PreTrainedConfig):
    """Minimal policy config for LeRobot plugin smoke tests."""

    n_obs_steps: int = 1
    horizon: int = 1
    n_action_steps: int = 1

    hidden_dim: int = 128

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {self.n_obs_steps}.")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}.")
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {self.n_action_steps}.")

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        start = 1 - self.n_obs_steps
        return list(range(start, start + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        if self.action_feature is None:
            raise ValueError("Policy requires an output ACTION feature named 'action'.")

        if self.robot_state_feature is None and self.env_state_feature is None:
            raise ValueError(
                "Policy requires either 'observation.state' or 'observation.environment_state'."
            )
