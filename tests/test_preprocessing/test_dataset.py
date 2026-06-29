import numpy as np
import cv2

from solar_seg.data.preprocessing.dataset import SolarSegDataset, SolarSegDataModule


def _create_dummy_data(root):
    img_dir = root / "images"
    sem_dir = root / "semantic_masks"
    inst_dir = root / "instance_masks"
    for d in [img_dir, sem_dir, inst_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{i}.png"), img)

        sem = np.random.randint(0, 2, (400, 400), dtype=np.uint8) * 255
        cv2.imwrite(str(sem_dir / f"{i}_semantic.png"), sem)

        inst = np.random.randint(0, 3, (400, 400), dtype=np.int32)
        cv2.imwrite(str(inst_dir / f"{i}_instance.png"), inst)


def test_dataset_len(tmp_path):
    _create_dummy_data(tmp_path)
    ds = SolarSegDataset(
        image_dir=tmp_path / "images",
        semantic_mask_dir=tmp_path / "semantic_masks",
        instance_mask_dir=tmp_path / "instance_masks",
    )
    assert len(ds) == 5


def test_dataset_getitem(tmp_path):
    _create_dummy_data(tmp_path)
    ds = SolarSegDataset(
        image_dir=tmp_path / "images",
        semantic_mask_dir=tmp_path / "semantic_masks",
        instance_mask_dir=tmp_path / "instance_masks",
    )
    item = ds[0]
    assert "pixel_values" in item
    assert "semantic_mask" in item
    assert "instance_mask" in item
    assert item["pixel_values"].shape[0] == 3


def test_datamodule_setup(tmp_path):
    _create_dummy_data(tmp_path)
    dm = SolarSegDataModule(data_roots={"test": tmp_path}, batch_size=2, val_split=0.2)
    dm.setup("fit")
    assert dm._train_ds is not None
    assert len(dm._val_ds_list) == 1
    assert len(dm._train_ds) >= 3


def test_dataset_empty_dir(tmp_path):
    import pytest
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    sem_dir = tmp_path / "semantic_masks"
    sem_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No .png images found"):
        SolarSegDataset(
            image_dir=empty_dir,
            semantic_mask_dir=sem_dir,
        )
