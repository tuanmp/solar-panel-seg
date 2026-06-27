from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def training_transforms(image_size: int = 384) -> A.Compose:
    """Training augmentation pipeline.

    Args:
        image_size: Output spatial size (square crop).
    """
    return A.Compose(
        [
            A.RandomCrop(image_size, image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.8,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
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
