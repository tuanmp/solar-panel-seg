from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def training_transforms(
    hflip_prob: float = 0.3,
    vflip_prob: float = 0.0,
    color_jitter_p: float = 0.8,
    gaussian_blur_p: float = 0.1,
) -> A.Compose:
    """Training augmentation pipeline.

    Args:
        hflip_prob: Probability of horizontal flip.
        vflip_prob: Probability of vertical flip.
        color_jitter_p: Probability of color jitter.
        gaussian_blur_p: Probability of Gaussian blur.
    """
    transforms = []
    if hflip_prob > 0:
        transforms.append(A.HorizontalFlip(p=hflip_prob))
    if vflip_prob > 0:
        transforms.append(A.VerticalFlip(p=vflip_prob))
    if color_jitter_p > 0:
        transforms.append(A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=color_jitter_p,
        ))
    if gaussian_blur_p > 0:
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=gaussian_blur_p))
    transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transforms.append(ToTensorV2())

    return A.Compose(
        transforms,
        additional_targets={
            "mask": "mask",
            "instance_mask": "mask",
        },
    )


def validation_transforms() -> A.Compose:
    """Validation/test augmentation pipeline (no random ops)."""
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={
            "mask": "mask",
            "instance_mask": "mask",
        },
    )
