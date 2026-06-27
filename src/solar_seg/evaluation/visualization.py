from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def overlay_mask(
    image: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Overlay a binary mask on an image with transparency.

    Args:
        image: RGB image (H, W, 3) uint8 or float [0,1].
        mask: Binary mask (H, W) uint8 or float.
        alpha: Mask transparency (0=invisible, 1=opaque).
        color: RGB color for the mask overlay.

    Returns:
        Overlay image as uint8 numpy array (H, W, 3).
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    if mask.dtype in (np.float32, np.float64):
        mask_bool = mask > 0.5
    else:
        mask_bool = mask > 0

    overlay = image.copy()
    color_arr = np.array(color, dtype=np.uint8)
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + color_arr * alpha
    ).astype(np.uint8)
    return overlay


def plot_predictions(
    image: np.ndarray | torch.Tensor,
    pred_semantic: np.ndarray | torch.Tensor,
    target_semantic: np.ndarray | torch.Tensor,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """Side-by-side: input, ground truth overlay, prediction overlay."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(pred_semantic, torch.Tensor):
        pred_semantic = pred_semantic.cpu().numpy()
    if isinstance(target_semantic, torch.Tensor):
        target_semantic = target_semantic.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image if image.ndim == 3 else image.transpose(1, 2, 0))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    gt_overlay = overlay_mask(image, target_semantic)
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    pred_overlay = overlay_mask(image, pred_semantic)
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
