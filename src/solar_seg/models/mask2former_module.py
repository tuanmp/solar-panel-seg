from __future__ import annotations

from typing import Any

import lightning as L
import torch
from transformers import Mask2FormerForUniversalSegmentation


class Mask2FormerModule(L.LightningModule):
    """LightningModule wrapping HuggingFace Mask2Former for panoptic segmentation."""

    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-base-coco-panoptic",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(pixel_values=pixel_values)

    def _prepare_labels(
        self, semantic_mask: torch.Tensor, instance_mask: torch.Tensor | None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Convert dataset masks to Mask2Former label format.

        Each image in batch produces:
        - mask_labels[i]: tensor of shape (K_i, H, W) — stacked binary instance masks
        - class_labels[i]: tensor of shape (K_i,) — class ID per instance
        """
        batch_size = semantic_mask.shape[0]
        mask_labels: list[torch.Tensor] = []
        class_labels: list[torch.Tensor] = []

        for b in range(batch_size):
            sem = semantic_mask[b]  # (H, W)
            inst = instance_mask[b] if instance_mask is not None else None

            batch_masks: list[torch.Tensor] = []
            batch_classes: list[int] = []

            if inst is not None:
                unique_ids = torch.unique(inst)
                unique_ids = unique_ids[unique_ids > 0]
                for uid in unique_ids:
                    mask = (inst == uid).float()  # (H, W)
                    class_id = int(sem[mask > 0].mode().values.item()) if mask.sum() > 0 else 0
                    class_id = max(0, min(class_id, self.hparams.num_labels - 1))
                    batch_masks.append(mask)
                    batch_classes.append(class_id)
            else:
                # Fallback: treat each connected component in semantic mask as an instance
                unique_classes = torch.unique(sem)
                for cls_id in unique_classes:
                    if cls_id == 0:
                        continue
                    mask = (sem == cls_id).float()
                    if mask.sum() > 0:
                        batch_masks.append(mask)
                        batch_classes.append(int(cls_id))

            if batch_masks:
                mask_labels.append(torch.stack(batch_masks))  # (K, H, W)
                class_labels.append(torch.tensor(batch_classes, dtype=torch.long, device=sem.device))
            else:
                # No instances — provide an empty tensor with correct spatial dims
                h, w = sem.shape
                mask_labels.append(torch.zeros((0, h, w), dtype=torch.float, device=sem.device))
                class_labels.append(torch.zeros((0,), dtype=torch.long, device=sem.device))

        return mask_labels, class_labels

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        semantic_mask = batch.get("semantic_mask")
        instance_mask = batch.get("instance_mask")

        mask_labels, class_labels = self._prepare_labels(semantic_mask, instance_mask)

        outputs = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        loss = outputs.loss
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(current_step: int) -> float:
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            return max(
                0.0,
                (1.0 - current_step / max(1, self.trainer.estimated_stepping_batches))
                ** 1.0,
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
