from __future__ import annotations

from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation

from solar_seg.evaluation.visualization import overlay_mask

MAX_VAL_VIZ_SAMPLES = 4


class Mask2FormerModule(L.LightningModule):
    """LightningModule wrapping HuggingFace Mask2Former for panoptic segmentation."""

    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-base-coco-panoptic",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        num_labels: int = 2,
        loss_ce_weight: float = 2.0,
        loss_mask_weight: float = 5.0,
        loss_dice_weight: float = 5.0,
        freeze_backbone: bool = False,
        unfreeze_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        self.model.weight_dict = {
            "loss_cross_entropy": loss_ce_weight,
            "loss_mask": loss_mask_weight,
            "loss_dice": loss_dice_weight,
        }
        self.model.criterion.weight_dict = self.model.weight_dict

        self._backbone_unfrozen = not freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()

        self._val_viz_batch: dict[str, torch.Tensor] | None = None

    def _freeze_backbone(self) -> None:
        backbone = self.model.model.pixel_level_module.encoder
        for param in backbone.parameters():
            param.requires_grad = False
        self._backbone_unfrozen = False

    def _unfreeze_backbone(self) -> None:
        backbone = self.model.model.pixel_level_module.encoder
        for param in backbone.parameters():
            param.requires_grad = True
        self._backbone_unfrozen = True

    def on_train_epoch_start(self) -> None:
        if (
            not self._backbone_unfrozen
            and self.hparams.unfreeze_epoch > 0
            and self.current_epoch >= self.hparams.unfreeze_epoch
        ):
            self._unfreeze_backbone()
            print(f"Epoch {self.current_epoch}: backbone unfrozen")

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
            sem = semantic_mask[b]
            inst = instance_mask[b] if instance_mask is not None else None

            batch_masks: list[torch.Tensor] = []
            batch_classes: list[int] = []

            if inst is not None:
                unique_ids = torch.unique(inst)
                unique_ids = unique_ids[unique_ids > 0]
                for uid in unique_ids:
                    mask = (inst == uid).float()
                    class_id = int(sem[mask > 0].mode().values.item()) if mask.sum() > 0 else 0
                    class_id = max(0, min(class_id, self.hparams.num_labels - 1))
                    batch_masks.append(mask)
                    batch_classes.append(class_id)
            else:
                unique_classes = torch.unique(sem)
                for cls_id in unique_classes:
                    if cls_id == 0:
                        continue
                    mask = (sem == cls_id).float()
                    if mask.sum() > 0:
                        batch_masks.append(mask)
                        batch_classes.append(int(cls_id))

            if batch_masks:
                mask_labels.append(torch.stack(batch_masks))
                class_labels.append(torch.tensor(batch_classes, dtype=torch.long, device=sem.device))
            else:
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

        # Log total loss
        total_loss = outputs.loss
        self.log(
            f"{stage}/loss",
            total_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        # Log individual loss components
        loss_dict = self.model.get_loss_dict(
            outputs.masks_queries_logits,
            outputs.class_queries_logits,
            mask_labels,
            class_labels,
            outputs.auxiliary_logits,
        )
        for key, value in loss_dict.items():
            self.log(
                f"{stage}/{key}",
                value,
                on_epoch=True,
                on_step=False,
            )

        return total_loss

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
        # Store first batch for visualization at epoch end
        if self._val_viz_batch is None:
            self._val_viz_batch = {
                k: v.detach().cpu().clone()
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        if self._val_viz_batch is None:
            return

        try:
            self._log_prediction_plots()
        except Exception as e:
            print(f"Warning: prediction plot generation failed: {e}")
        finally:
            self._val_viz_batch = None

    def _log_prediction_plots(self) -> None:
        """Generate and log prediction plots for the first validation batch."""
        batch = self._val_viz_batch
        pixel_values = batch["pixel_values"].to(self.device)

        # Run inference (no labels)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        masks_logits = outputs.masks_queries_logits  # (B, Q, H_pred, W_pred)
        class_logits = outputs.class_queries_logits  # (B, Q, num_labels)

        # Post-process: get binary semantic prediction
        # Mask2Former predicts per-query masks and classes; we take argmax over class logits
        pred_classes = class_logits.softmax(dim=-1)
        # For each query, get its most likely class and confidence
        pred_scores, pred_class = pred_classes.max(dim=-1)  # (B, Q)

        n_viz = min(MAX_VAL_VIZ_SAMPLES, pixel_values.shape[0])
        fig, axes = plt.subplots(n_viz, 4, figsize=(16, 4 * n_viz), squeeze=False)

        for i in range(n_viz):
            img = pixel_values[i].cpu()
            # Unnormalize (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_vis = img * std + mean
            img_vis = img_vis.permute(1, 2, 0).numpy()
            img_vis = np.clip(img_vis, 0, 1)

            # Ground truth semantic mask
            gt_sem = batch.get("semantic_mask", None)
            gt_sem_i = gt_sem[i].cpu().numpy() if gt_sem is not None else None

            # Ground truth instance mask
            gt_inst = batch.get("instance_mask", None)
            gt_inst_i = gt_inst[i].cpu().numpy() if gt_inst is not None else None

            # Predicted semantic mask: combine query masks for class=1 with score > 0.5
            bg_score = pred_scores[i]
            is_panel = pred_class[i] == 1
            panel_queries = torch.where(is_panel)[0]

            H, W = img_vis.shape[:2]
            pred_sem = np.zeros((H, W), dtype=np.uint8)
            pred_inst = np.zeros((H, W), dtype=np.uint16)

            inst_id = 1
            for q in panel_queries:
                score = bg_score[q].item()
                if score < 0.3:
                    continue
                mask = masks_logits[i, q].sigmoid()
                # Resize mask from prediction resolution to image resolution
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8)
                pred_sem[mask_np > 0] = 255
                pred_inst[mask_np > 0] = inst_id
                inst_id += 1

            # Column 0: Input image
            axes[i, 0].imshow(img_vis)
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            # Column 1: Ground truth semantic
            if gt_sem_i is not None:
                axes[i, 1].imshow(img_vis)
                gt_overlay = np.zeros_like(img_vis)
                gt_overlay[gt_sem_i > 0] = [0, 1, 0]
                axes[i, 1].imshow(gt_overlay, alpha=0.5)
                axes[i, 1].set_title(f"GT Semantic ({int(gt_sem_i.max())} panels)")
            else:
                axes[i, 1].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[i, 1].axis("off")

            # Column 2: Predicted semantic
            axes[i, 2].imshow(img_vis)
            pred_overlay = np.zeros_like(img_vis)
            pred_overlay[pred_sem > 0] = [0, 1, 0]
            axes[i, 2].imshow(pred_overlay, alpha=0.5)
            axes[i, 2].set_title(f"Pred Semantic ({inst_id-1} panels)")
            axes[i, 2].axis("off")

            # Column 3: Predicted instance mask (color-coded)
            axes[i, 3].imshow(pred_inst, cmap="tab20", vmin=0, vmax=20)
            axes[i, 3].set_title(f"Pred Instances")
            axes[i, 3].axis("off")

        plt.tight_layout()

        # Log to MLflow
        if isinstance(self.logger, L.pytorch.loggers.MLFlowLogger):
            self.logger.experiment.log_figure(
                self.logger.run_id,
                fig,
                f"val_predictions_epoch_{self.current_epoch:03d}.png",
            )
            plt.close(fig)

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
