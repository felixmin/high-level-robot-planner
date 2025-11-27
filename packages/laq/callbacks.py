"""
PyTorch Lightning callbacks for LAQ training.

Includes:
- ReconstructionVisualizationCallback: Logs reconstruction grids to WandB
"""

from typing import Optional

import torch
from torchvision.utils import make_grid
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from einops import rearrange

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ReconstructionVisualizationCallback(Callback):
    """
    Visualize LAQ reconstructions during training and/or validation.

    Matches LAPA visualization style:
    - Grid with 3 columns: [frame_t, frame_t+offset, reconstruction]
    - Logs to WandB as image

    Args:
        num_samples: Number of samples to visualize (default: 8)
        log_every_n_epochs: Visualization frequency (default: 1)
        visualize_train: Whether to visualize training reconstructions (default: False)
        visualize_val: Whether to visualize validation reconstructions (default: True)
    """

    def __init__(
        self,
        num_samples: int = 8,
        log_every_n_epochs: int = 1,
        visualize_train: bool = False,
        visualize_val: bool = True,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.visualize_train = visualize_train
        self.visualize_val = visualize_val

    def _get_wandb_logger(self, trainer: pl.Trainer):
        """Get WandB logger from trainer if available."""
        if not WANDB_AVAILABLE:
            return None

        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                return logger
        return None

    def _visualize_reconstructions(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_getter_method: str,
        log_key: str,
    ) -> None:
        """
        Common visualization logic for train and val.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            batch_getter_method: Name of method to get batch ('get_validation_batch' or 'get_training_batch')
            log_key: WandB key for logging ('val/reconstructions' or 'train/reconstructions')
        """
        # Check if we should visualize this epoch
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # Get WandB logger
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None:
            return

        # Get batch from module
        batch = getattr(pl_module, batch_getter_method)()
        if batch is None:
            return

        # Limit to num_samples
        batch = batch[:self.num_samples]

        # Generate reconstructions
        recons = pl_module.generate_reconstructions(batch)

        # Create visualization grid
        # Input: [B, C, 2, H, W] where 2 = [frame_t, frame_t+offset]
        # Output: grid with 3 columns per sample [frame_t, frame_t+offset, recons]
        frame_t = batch[:, :, 0]      # [B, C, H, W]
        frame_t_plus = batch[:, :, 1]  # [B, C, H, W]
        recons = recons.cpu()              # [B, C, H, W]

        # Stack: [frame_t, frame_t+offset, recons]
        imgs_and_recons = torch.stack([frame_t, frame_t_plus, recons], dim=0)  # [3, B, C, H, W]
        imgs_and_recons = rearrange(imgs_and_recons, 'r b c h w -> (b r) c h w')  # [B*3, C, H, W]

        # Clamp to valid range
        imgs_and_recons = imgs_and_recons.clamp(0.0, 1.0)

        # Create grid (3 columns per sample)
        grid = make_grid(
            imgs_and_recons,
            nrow=3,
            normalize=False,  # Already normalized
            value_range=(0, 1),
        )

        # Log to WandB
        wandb_logger.log_image(
            key=log_key,
            images=[grid],
            caption=[f"Epoch {trainer.current_epoch}"],
        )

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Called at the end of training epoch.

        Generates and logs training reconstruction visualizations.
        """
        if not self.visualize_train:
            return

        self._visualize_reconstructions(
            trainer=trainer,
            pl_module=pl_module,
            batch_getter_method='get_training_batch',
            log_key='train/reconstructions',
        )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Called at the end of validation epoch.

        Generates and logs validation reconstruction visualizations.
        """
        if not self.visualize_val:
            return

        self._visualize_reconstructions(
            trainer=trainer,
            pl_module=pl_module,
            batch_getter_method='get_validation_batch',
            log_key='val/reconstructions',
        )


class EMACallback(Callback):
    """
    Exponential Moving Average callback.

    Maintains EMA of model weights during training (LAPA style).

    Args:
        decay: EMA decay rate (default: 0.999)
        update_every: Update EMA every N steps (default: 1)
        update_after_step: Start EMA updates after N steps (default: 0)
    """

    def __init__(
        self,
        decay: float = 0.999,
        update_every: int = 1,
        update_after_step: int = 0,
    ):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.ema_model = None
        self.num_updates = 0

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Initialize EMA model."""
        # Clone model for EMA
        self.ema_model = type(pl_module.model)(
            **pl_module.model_config
        ).to(pl_module.device)
        self.ema_model.load_state_dict(pl_module.model.state_dict())
        self.ema_model.eval()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Update EMA weights after training step."""
        if trainer.global_step < self.update_after_step:
            return

        if trainer.global_step % self.update_every != 0:
            return

        # Update EMA weights
        self.num_updates += 1
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                pl_module.model.parameters(),
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1 - self.decay
                )

    def state_dict(self):
        """Save EMA state."""
        return {
            "ema_model": self.ema_model.state_dict() if self.ema_model else None,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict):
        """Load EMA state."""
        if state_dict["ema_model"] is not None and self.ema_model is not None:
            self.ema_model.load_state_dict(state_dict["ema_model"])
        self.num_updates = state_dict["num_updates"]
