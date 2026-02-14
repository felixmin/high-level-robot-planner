from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from lerobot_policy_hlrp.policies.hlrp_smoke.configuration_hlrp_smoke import HLRPSmokeConfig


class HLRPSmokePolicy(PreTrainedPolicy):
    """Tiny MLP policy used to validate plugin installation and wiring."""

    config_class = HLRPSmokeConfig
    name = "hlrp_smoke"

    def __init__(
        self,
        config: HLRPSmokeConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
        dataset_meta=None,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()

        self.config = config
        self.dataset_stats = dataset_stats
        self.dataset_meta = dataset_meta

        self._obs_key = OBS_STATE if config.robot_state_feature is not None else OBS_ENV_STATE
        self._obs_dim = self._infer_dim(config, self._obs_key)
        self._action_dim = self._infer_dim(config, ACTION)

        self.backbone = nn.Sequential(
            nn.Linear(self._obs_dim * self.config.n_obs_steps, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self._action_dim),
        )

        self._queues: dict[str, deque[torch.Tensor]] = {}
        self.reset()

    @staticmethod
    def _infer_dim(config: HLRPSmokeConfig, key: str) -> int:
        if key == ACTION:
            feature = config.action_feature
        elif key == OBS_STATE:
            feature = config.robot_state_feature
        elif key == OBS_ENV_STATE:
            feature = config.env_state_feature
        else:
            feature = None

        if feature is None:
            raise ValueError(f"Missing feature for key '{key}'.")

        dim = 1
        for s in feature.shape:
            dim *= s
        return int(dim)

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        self._queues = {
            self._obs_key: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _prepare_obs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._obs_key not in batch:
            raise KeyError(f"Expected '{self._obs_key}' in batch.")

        obs = batch[self._obs_key]
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        elif obs.ndim != 3:
            raise ValueError(f"Expected 2D or 3D obs tensor, got shape {tuple(obs.shape)}")

        if obs.shape[1] < self.config.n_obs_steps:
            missing = self.config.n_obs_steps - obs.shape[1]
            obs = torch.cat([obs[:, :1].expand(-1, missing, -1), obs], dim=1)
        elif obs.shape[1] > self.config.n_obs_steps:
            obs = obs[:, -self.config.n_obs_steps :, :]

        return obs.reshape(obs.shape[0], -1)

    @staticmethod
    def _prepare_action(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if ACTION not in batch:
            raise KeyError("Expected 'action' in batch.")

        action = batch[ACTION]
        if action.ndim == 3:
            action = action[:, 0, :]
        elif action.ndim != 2:
            raise ValueError(
                f"Expected 2D or 3D action tensor, got shape {tuple(action.shape)}"
            )
        return action

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        obs = self._prepare_obs(batch)
        target = self._prepare_action(batch)

        pred = self.backbone(obs)
        loss = F.mse_loss(pred, target)

        return loss, {"loss": float(loss.detach().cpu())}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        obs = self._prepare_obs(batch)
        pred = self.backbone(obs)
        return pred.unsqueeze(1).expand(-1, self.config.n_action_steps, -1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        obs = batch[self._obs_key]
        if obs.ndim == 3:
            obs = obs[:, -1, :]
        elif obs.ndim != 2:
            raise ValueError(
                f"Expected 2D or 3D observation in select_action, got shape {tuple(obs.shape)}"
            )

        if len(self._queues[self._obs_key]) == 0:
            for _ in range(self.config.n_obs_steps):
                self._queues[self._obs_key].append(obs)
        else:
            self._queues[self._obs_key].append(obs)

        if len(self._queues[ACTION]) == 0:
            stacked_obs = torch.stack(list(self._queues[self._obs_key]), dim=1)
            chunk = self.predict_action_chunk({self._obs_key: stacked_obs})
            for idx in range(self.config.n_action_steps):
                self._queues[ACTION].append(chunk[:, idx, :])

        return self._queues[ACTION].popleft()
