"""
Common Lightning callbacks shared across stages.
"""

from __future__ import annotations

from typing import Any, Optional

from lightning.pytorch.callbacks import Callback


class ProgressLoggerCallback(Callback):
    """
    Log training progress to stdout for cluster jobs where tqdm doesn't work well
    in log files.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = int(log_every_n_steps)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.log_every_n_steps <= 0:
            return
        if (int(trainer.global_step) + 1) % self.log_every_n_steps != 0:
            return

        loss: Optional[float] = None
        try:
            loss_obj = outputs.get("loss") if isinstance(outputs, dict) else outputs
            loss = float(loss_obj) if loss_obj is not None else None
        except Exception:
            loss = None

        lr: Optional[float] = None
        try:
            if trainer.optimizers:
                lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        except Exception:
            lr = None

        # Use print() for progress output (captured by WandB and unified logging).
        msg = f"[Step {int(trainer.global_step) + 1}]"
        if loss is not None:
            msg += f" loss={loss:.4f},"
        if lr is not None:
            msg += f" lr={lr:.2e},"
        msg += f" epoch={int(trainer.current_epoch)}"
        print(msg)

    def on_validation_end(self, trainer, pl_module) -> None:
        metrics = {k: v for k, v in trainer.callback_metrics.items() if "val" in k}
        if not metrics:
            return
        try:
            metrics_str = ", ".join(f"{k}={float(v):.4f}" for k, v in metrics.items())
        except Exception:
            return
        print(f"[Validation] step={int(trainer.global_step)}, {metrics_str}")

