import numpy as np
from solar_seg.data.preprocessing.augmentations import (
    training_transforms,
    validation_transforms,
)


def test_training_augmentation_shapes():
    image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (400, 400), dtype=np.uint8)

    train_aug = training_transforms()
    result = train_aug(image=image, mask=mask)
    assert result["image"].shape == (3, 400, 400)
    assert result["mask"].shape == (400, 400)


def test_validation_augmentation_shapes():
    image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (400, 400), dtype=np.uint8)

    val_aug = validation_transforms()
    result = val_aug(image=image, mask=mask)
    assert result["image"].shape == (3, 400, 400)
