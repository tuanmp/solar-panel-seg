#!/usr/bin/env python3
"""Generate a showcase figure of model predictions for slides.

Loads a checkpoint, runs inference on a few test images, and renders a grid:
    Input image | Ground-truth instances | Predicted instances

Instances are color-coded; panels are blended over the (faded) aerial image.

Usage:
    uv run python scripts/make_prediction_figure.py \
        --ckpt 736168525124488930/.../best-epoch=13-val/loss=7.9167.ckpt \
        --data-root data/processed/bdappv \
        --output figures/predictions_showcase.png \
        --n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from solar_seg.data.preprocessing.dataset import _SplitDataset
from solar_seg.models.mask2former_module import Mask2FormerModule

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def colorize(inst: np.ndarray, cmap_name: str = "tab20") -> np.ndarray:
    """Map integer instance IDs to distinct RGB colors (0 = transparent)."""
    cmap = plt.get_cmap(cmap_name)
    out = np.zeros((*inst.shape, 3), dtype=float)
    ids = [u for u in np.unique(inst) if u > 0]
    for i, uid in enumerate(ids):
        out[inst == uid] = cmap(i % 20)[:3]
    return out


def blend(img: np.ndarray, inst: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Blend colorized instances over the image, with a light background wash."""
    colored = colorize(inst)
    mask = (inst > 0)[..., None]
    faded = 0.72 * img + 0.28 * np.ones_like(img)  # keep aerial context visible
    out = np.where(mask, (1 - alpha) * img + alpha * colored, faded)
    # white outline around each instance
    edges = cv2.Canny((inst > 0).astype(np.uint8) * 255, 50, 150) > 0
    out[edges] = [1.0, 1.0, 1.0]
    return np.clip(out, 0, 1)


def denormalize(pixel_values: torch.Tensor) -> np.ndarray:
    img = pixel_values.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/bdappv"))
    parser.add_argument("--output", type=Path, default=Path("figures/predictions_showcase.png"))
    parser.add_argument("--n", type=int, default=3, help="Number of examples (rows)")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # --- Build the test split (last val_split of sorted paths) ---
    img_dir = args.data_root / "images"
    sem_dir = args.data_root / "semantic_masks"
    inst_dir = args.data_root / "instance_masks"
    paths = sorted(img_dir.glob("*.png"))
    n_val = max(1, int(len(paths) * args.val_split))
    test_paths = paths[len(paths) - n_val :]

    ds = _SplitDataset(
        image_paths=test_paths,
        semantic_mask_dir=sem_dir,
        instance_mask_dir=inst_dir if inst_dir.exists() else None,
        transform=None,
    )

    # --- Pick images with panels, spread across the test set ---
    rng = np.random.default_rng(args.seed)
    candidates = []
    for idx in rng.permutation(len(ds)):
        item = ds[int(idx)]
        inst = item.get("instance_mask")
        if inst is not None and int(inst.max()) >= 1:
            candidates.append(int(idx))
        if len(candidates) >= args.n * 20:
            break
    # Sort by instance count for a spread of easy -> crowded scenes
    candidates.sort(key=lambda i: int(ds[i]["instance_mask"].max()))
    step = max(1, len(candidates) // args.n)
    chosen = candidates[::step][: args.n]

    # --- Load model ---
    from solar_seg.data.preprocessing.augmentations import validation_transforms

    tf = validation_transforms()
    model = Mask2FormerModule.load_from_checkpoint(
        str(args.ckpt), source_names=["bdappv"], max_metric_samples=0,
    )
    model.eval()
    model.freeze()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    n = len(chosen)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.6 * n), squeeze=False)

    col_titles = ["Input image", "Ground truth", "Prediction"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=14, fontweight="bold", pad=8)

    for i, idx in enumerate(chosen):
        item = ds[idx]
        img = denormalize(item["pixel_values"])
        gt_inst = item["instance_mask"].numpy()

        # Inference
        pv = item["pixel_values"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.model(pixel_values=pv)
        gt_H, gt_W = gt_inst.shape
        pred_sem, pred_inst = model._decode_predictions(
            out.class_queries_logits, out.masks_queries_logits, (gt_H, gt_W)
        )[0]

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(blend(img, gt_inst))
        axes[i, 2].imshow(blend(img, pred_inst))

        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_visible(False)
        axes[i, 1].set_xlabel(f"GT: {int(gt_inst.max())} panels", fontsize=10)
        axes[i, 2].set_xlabel(f"Pred: {int(pred_inst.max())} panels", fontsize=10)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(args.output), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}  ({n} examples)")


if __name__ == "__main__":
    main()
