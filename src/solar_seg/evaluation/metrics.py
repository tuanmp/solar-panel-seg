from __future__ import annotations

import numpy as np
import torch


def panoptic_quality(
    pred_semantic: torch.Tensor,
    pred_instance: torch.Tensor,
    target_semantic: torch.Tensor,
    target_instance: torch.Tensor,
    num_classes: int = 2,
) -> dict[str, float]:
    """Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and
    Recognition Quality (RQ).

    Simplified implementation following Kirillov et al. "Panoptic
    Segmentation" (CVPR 2019).

    Args:
        pred_semantic: Predicted semantic labels (H, W).
        pred_instance: Predicted instance IDs (H, W).
        target_semantic: Ground truth semantic labels (H, W).
        target_instance: Ground truth instance IDs (H, W).

    Returns:
        dict with keys 'pq', 'sq', 'rq'.
    """
    pred_semantic = pred_semantic.cpu().numpy()
    pred_instance = pred_instance.cpu().numpy()
    target_semantic = target_semantic.cpu().numpy()
    target_instance = target_instance.cpu().numpy()

    pred_ids = pred_instance.astype(np.int32)
    target_ids = target_instance.astype(np.int32)

    pq_sum = 0.0
    sq_sum = 0.0
    rq_sum = 0.0
    matched_pairs = 0
    total_pred = 0
    total_target = 0

    for class_id in range(num_classes):
        pred_class_mask = pred_semantic == class_id
        target_class_mask = target_semantic == class_id

        pids = np.unique(pred_ids[pred_class_mask])
        tids = np.unique(target_ids[target_class_mask])
        pids = pids[pids > 0]
        tids = tids[tids > 0]

        total_pred += len(pids)
        total_target += len(tids)

        if len(pids) == 0 and len(tids) == 0:
            continue

        ious = np.zeros((len(pids), len(tids)))
        for i, pid in enumerate(pids):
            p_mask = pred_ids == pid
            for j, tid in enumerate(tids):
                t_mask = target_ids == tid
                intersection = np.logical_and(p_mask, t_mask).sum()
                union = np.logical_or(p_mask, t_mask).sum()
                ious[i, j] = intersection / max(union, 1)

        matched = 0
        for _ in range(min(len(pids), len(tids))):
            if ious.size == 0:
                break
            idx = ious.argmax()
            i, j = np.unravel_index(idx, ious.shape)
            if ious[i, j] < 0.5:
                break
            pq_sum += ious[i, j]
            sq_sum += ious[i, j]
            matched += 1
            matched_pairs += 1
            ious[i, :] = -1
            ious[:, j] = -1

    if matched_pairs == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0}

    sq = sq_sum / max(matched_pairs, 1)
    rq = matched_pairs / max((total_pred + total_target) / 2.0, 1)
    pq = sq_sum / max((total_pred + total_target) / 2.0, 1)

    return {"pq": float(pq), "sq": float(sq), "rq": float(rq)}


def mean_iou(
    pred_semantic: torch.Tensor,
    target_semantic: torch.Tensor,
    num_classes: int = 2,
    smooth: float = 1e-6,
) -> float:
    """Compute mean Intersection over Union for semantic segmentation."""
    pred = pred_semantic.cpu().numpy().flatten()
    target = target_semantic.cpu().numpy().flatten()

    ious = []
    for c in range(num_classes):
        intersection = np.logical_and(pred == c, target == c).sum()
        union = np.logical_or(pred == c, target == c).sum()
        ious.append((intersection + smooth) / (union + smooth))

    return float(np.mean(ious))
