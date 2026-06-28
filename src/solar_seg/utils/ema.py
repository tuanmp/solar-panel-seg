"""Exponential Moving Average (EMA) callback for Lightning."""

from __future__ import annotations

import copy
from typing import Any

import lightning as L
import torch


class EMACallback(L.Callback):
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of the model parameters, updated each training
    step with the EMA rule: shadow = decay * shadow + (1 - decay) * current.

    Swaps to shadow weights during validation, testing, and checkpoint saving.
    """

    def __init__(self, decay: float = 0.999, use_num_updates: bool = True) -> None:
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self.num_updates: int = 0

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.shadow = {
            name: param.data.clone().detach()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.num_updates += 1
        if self.use_num_updates:
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay

        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if not param.requires_grad:
                    continue
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def _swap_shadow(self, pl_module: L.LightningModule, to_shadow: bool) -> None:
        src = self.shadow if to_shadow else self.backup
        dst = self.backup if to_shadow else None
        for name, param in pl_module.named_parameters():
            if not param.requires_grad or name not in src:
                continue
            if to_shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(src[name])
            else:
                param.data.copy_(src[name])

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_shadow(pl_module, to_shadow=True)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_shadow(pl_module, to_shadow=False)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_shadow(pl_module, to_shadow=True)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_shadow(pl_module, to_shadow=False)

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        self._swap_shadow(pl_module, to_shadow=True)
        checkpoint["state_dict"] = copy.deepcopy(pl_module.state_dict())
        self._swap_shadow(pl_module, to_shadow=False)
