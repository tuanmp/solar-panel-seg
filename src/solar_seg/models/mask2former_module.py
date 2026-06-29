from __future__ import annotations

from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation

from solar_seg.evaluation.metrics import mean_iou, panoptic_quality

MAX_VAL_VIZ_SAMPLES = 4
MAX_METRIC_SAMPLES = 500


class Mask2FormerModule(L.LightningModule):
    """LightningModule wrapping HuggingFace Mask2Former for panoptic segmentation."""

    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-base-coco-panoptic",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        num_labels: int = 2,
        source_names: list[str] | None = None,
        loss_ce_weight: float = 2.0,
        loss_mask_weight: float = 5.0,
        loss_dice_weight: float = 5.0,
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

        self._source_names = source_names or ["default"]
        self._val_viz_batches: dict[str, dict[str, torch.Tensor]] = {}
        self._metric_preds: dict[str, list[dict]] = {}

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(pixel_values=pixel_values)

    def _prepare_labels(
        self, semantic_mask: torch.Tensor, instance_mask: torch.Tensor | None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
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

    def _decode_predictions(
        self,
        class_logits: torch.Tensor,
        masks_logits: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Convert Mask2Former logits to per-image (semantic, instance) masks.

        Args:
            class_logits: (B, Q, num_labels)
            masks_logits: (B, Q, H_pred, W_pred) — at 1/4 input resolution
            target_hw: (H, W) of target mask to upscale to

        Returns:
            List of (pred_semantic_uint8, pred_instance_uint16) per image,
            with values 0/1 for semantic and 0/1/2/... for instance.
        """
        B = class_logits.shape[0]
        pred_classes = class_logits.softmax(dim=-1)
        pred_scores, pred_class = pred_classes.max(dim=-1)
        tH, tW = target_hw

        results: list[tuple[np.ndarray, np.ndarray]] = []
        for b in range(B):
            is_panel = pred_class[b] == 1
            panel_queries = torch.where(is_panel)[0]

            pred_sem = np.zeros((tH, tW), dtype=np.uint8)
            pred_inst = np.zeros((tH, tW), dtype=np.uint16)
            inst_id = 1

            for q in panel_queries:
                score = pred_scores[b, q].item()
                if score < 0.3:
                    continue
                mask = masks_logits[b, q].sigmoid()
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(tH, tW),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8)
                pred_sem[mask_np > 0] = 1
                pred_inst[mask_np > 0] = inst_id
                inst_id += 1

            results.append((pred_sem, pred_inst))

        return results

    def _compute_and_log_metrics(
        self, source_name: str, preds: list[dict]
    ) -> None:
        """Compute mIoU and panoptic quality from accumulated predictions."""
        prefix = f"val/{source_name}"
        total_pq, total_sq, total_rq = 0.0, 0.0, 0.0
        total_iou = 0.0
        count = 0

        for item in preds:
            gt_sem = item["gt_semantic"]
            gt_inst = item["gt_instance"]
            pred_sem = item["pred_semantic"]
            pred_inst = item["pred_instance"]

            try:
                iou = mean_iou(
                    torch.from_numpy(pred_sem),
                    torch.from_numpy(gt_sem),
                    num_classes=self.hparams.num_labels,
                )
                total_iou += iou

                pq_metrics = panoptic_quality(
                    torch.from_numpy(pred_sem),
                    torch.from_numpy(pred_inst),
                    torch.from_numpy(gt_sem),
                    torch.from_numpy(gt_inst),
                    num_classes=self.hparams.num_labels,
                )
                total_pq += pq_metrics["pq"]
                total_sq += pq_metrics["sq"]
                total_rq += pq_metrics["rq"]
                count += 1
            except Exception:
                continue

        if count == 0:
            return

        for metric_name, value in [
            ("miou", total_iou / count),
            ("pq", total_pq / count),
            ("sq", total_sq / count),
            ("rq", total_rq / count),
        ]:
            self.log(
                f"{prefix}/{metric_name}",
                value,
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
        source_name: str = "",
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

        prefix = stage if not source_name else f"{stage}/{source_name}"

        total_loss = outputs.loss
        self.log(
            f"{prefix}/loss",
            total_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            add_dataloader_idx=False,
        )

        loss_dict = self.model.get_loss_dict(
            outputs.masks_queries_logits,
            outputs.class_queries_logits,
            mask_labels,
            class_labels,
            outputs.auxiliary_logits,
        )
        for key, value in loss_dict.items():
            self.log(
                f"{prefix}/{key}",
                value,
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )

        # Accumulate predictions for metric computation
        if (stage in ("val", "test")) and source_name:
            key = source_name
            if key not in self._metric_preds:
                self._metric_preds[key] = []
            if len(self._metric_preds[key]) < MAX_METRIC_SAMPLES:
                # Decode predictions at target resolution
                gt_H, gt_W = semantic_mask.shape[1:]
                decoded = self._decode_predictions(
                    outputs.class_queries_logits,
                    outputs.masks_queries_logits,
                    target_hw=(gt_H, gt_W),
                )
                for b in range(semantic_mask.shape[0]):
                    if len(self._metric_preds[key]) >= MAX_METRIC_SAMPLES:
                        break
                    gt_sem_np = semantic_mask[b].cpu().numpy().astype(np.uint8)
                    inst = instance_mask[b] if instance_mask is not None else None
                    gt_inst_np = inst.cpu().numpy().astype(np.int32) if inst is not None else np.zeros_like(gt_sem_np, dtype=np.int32)
                    self._metric_preds[key].append({
                        "gt_semantic": gt_sem_np,
                        "gt_instance": gt_inst_np,
                        "pred_semantic": decoded[b][0],
                        "pred_instance": decoded[b][1],
                    })

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
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        source = self._source_names[dataloader_idx]

        if source not in self._val_viz_batches:
            self._val_viz_batches[source] = {
                k: v.detach().cpu().clone()
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

        return self._shared_step(batch, "val", source_name=source)

    def on_validation_epoch_end(self) -> None:
        # Compute and log per-source metrics
        for source, preds in self._metric_preds.items():
            self._compute_and_log_metrics(source, preds)
        self._metric_preds.clear()

        # Generate prediction plots
        if self._val_viz_batches:
            try:
                for source, batch in self._val_viz_batches.items():
                    self._log_prediction_plots(batch, source)
            except Exception as e:
                print(f"Warning: prediction plot generation failed: {e}")
            finally:
                self._val_viz_batches.clear()

    def _log_prediction_plots(self, batch: dict[str, torch.Tensor], source: str) -> None:
        pixel_values = batch["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        masks_logits = outputs.masks_queries_logits
        class_logits = outputs.class_queries_logits

        pred_classes = class_logits.softmax(dim=-1)
        pred_scores, pred_class = pred_classes.max(dim=-1)

        n_viz = min(MAX_VAL_VIZ_SAMPLES, pixel_values.shape[0])
        fig, axes = plt.subplots(n_viz, 4, figsize=(16, 4 * n_viz), squeeze=False)

        for i in range(n_viz):
            img = pixel_values[i].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_vis = img * std + mean
            img_vis = img_vis.permute(1, 2, 0).numpy()
            img_vis = np.clip(img_vis, 0, 1)

            gt_sem = batch.get("semantic_mask", None)
            gt_sem_i = gt_sem[i].cpu().numpy() if gt_sem is not None else None

            is_panel = pred_class[i] == 1
            panel_queries = torch.where(is_panel)[0]

            H, W = img_vis.shape[:2]
            pred_sem = np.zeros((H, W), dtype=np.uint8)
            pred_inst = np.zeros((H, W), dtype=np.uint16)

            inst_id = 1
            for q in panel_queries:
                score = pred_scores[i, q].item()
                if score < 0.3:
                    continue
                mask = masks_logits[i, q].sigmoid()
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

            axes[i, 0].imshow(img_vis)
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            if gt_sem_i is not None:
                axes[i, 1].imshow(img_vis)
                gt_overlay = np.zeros_like(img_vis)
                gt_overlay[gt_sem_i > 0] = [0, 1, 0]
                axes[i, 1].imshow(gt_overlay, alpha=0.5)
                axes[i, 1].set_title(f"GT Semantic ({int(gt_sem_i.max())} panels)")
            else:
                axes[i, 1].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(img_vis)
            pred_overlay = np.zeros_like(img_vis)
            pred_overlay[pred_sem > 0] = [0, 1, 0]
            axes[i, 2].imshow(pred_overlay, alpha=0.5)
            axes[i, 2].set_title(f"Pred Semantic ({inst_id - 1} panels)")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(pred_inst, cmap="tab20", vmin=0, vmax=20)
            axes[i, 3].set_title("Pred Instances")
            axes[i, 3].axis("off")

        fig.suptitle(f"{source} — Epoch {self.current_epoch:03d}", fontsize=12, fontweight="bold")
        plt.tight_layout()

        if isinstance(self.logger, L.pytorch.loggers.MLFlowLogger):
            self.logger.experiment.log_figure(
                self.logger.run_id,
                fig,
                f"val_predictions_{source}_epoch_{self.current_epoch:03d}.png",
            )
        plt.close(fig)

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        source = self._source_names[dataloader_idx]
        return self._shared_step(batch, "test", source_name=source)

    def on_test_epoch_end(self) -> None:
        for source, preds in self._metric_preds.items():
            self._compute_and_log_metrics(source, preds)
        self._metric_preds.clear()

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
