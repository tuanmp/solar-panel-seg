import numpy as np
import torch

from solar_seg.data.preprocessing.augmentations import (
    training_transforms,
    validation_transforms,
)


def test_training_transforms_output():
    aug = training_transforms()
    image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (400, 400), dtype=np.uint8)
    instance_mask = np.random.randint(0, 5, (400, 400), dtype=np.int32)

    result = aug(image=image, mask=mask, instance_mask=instance_mask)
    assert isinstance(result["image"], torch.Tensor)
    assert result["image"].shape == (3, 400, 400)
    assert isinstance(result["mask"], torch.Tensor)
    assert result["mask"].shape == (400, 400)
    assert isinstance(result["instance_mask"], torch.Tensor)


def test_validation_transforms_output():
    aug = validation_transforms()
    image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (400, 400), dtype=np.uint8)

    result = aug(image=image, mask=mask)
    assert result["image"].shape == (3, 400, 400)
