from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L


class _SplitDataset(Dataset):
    """Internal dataset that loads from a specific list of image paths."""

    def __init__(
        self,
        image_paths: list[Path],
        semantic_mask_dir: Path,
        instance_mask_dir: Path | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.semantic_mask_dir = semantic_mask_dir
        self.instance_mask_dir = instance_mask_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        stem = img_path.stem
        sem_path = self.semantic_mask_dir / f"{stem}_semantic.png"

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sem_mask = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
        if sem_mask is None:
            raise FileNotFoundError(f"Missing semantic mask: {sem_path}")
        sem_mask = (sem_mask > 0).astype(np.uint8)

        inst_mask = None
        if self.instance_mask_dir:
            inst_path = self.instance_mask_dir / f"{stem}_instance.png"
            if inst_path.exists():
                inst_mask = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
                inst_mask = np.ascontiguousarray(inst_mask)

        if self.transform:
            kwargs = {"image": image, "mask": sem_mask}
            if inst_mask is not None:
                kwargs["instance_mask"] = inst_mask
            transformed = self.transform(**kwargs)
            image = transformed["image"]
            sem_mask = transformed["mask"]
            inst_mask = transformed.get("instance_mask")
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            sem_mask = torch.from_numpy(sem_mask).long()

        result = {
            "pixel_values": image if isinstance(image, torch.Tensor) else torch.from_numpy(np.array(image)),
            "semantic_mask": sem_mask.long() if isinstance(sem_mask, torch.Tensor) else torch.from_numpy(sem_mask).long(),
        }
        if inst_mask is not None:
            result["instance_mask"] = inst_mask.long() if isinstance(inst_mask, torch.Tensor) else torch.from_numpy(inst_mask).long()
        result["id"] = stem
        return result


class SolarSegDataset(Dataset):
    """Dataset for solar panel panoptic segmentation.

    Expected directory structure:
        data_root/
            images/          {id}.png
            semantic_masks/  {id}_semantic.png
            instance_masks/  {id}_instance.png
    """

    def __init__(
        self,
        image_dir: Path,
        semantic_mask_dir: Path,
        instance_mask_dir: Path | None = None,
        transform: Callable | None = None,
    ) -> None:
        image_dir = Path(image_dir)
        semantic_mask_dir = Path(semantic_mask_dir)
        instance_mask_dir = Path(instance_mask_dir) if instance_mask_dir else None

        image_paths = sorted(image_dir.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No .png images found in {image_dir}")

        self._impl = _SplitDataset(
            image_paths=image_paths,
            semantic_mask_dir=semantic_mask_dir,
            instance_mask_dir=instance_mask_dir,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self._impl)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._impl[idx]


class SolarSegDataModule(L.LightningDataModule):
    """LightningDataModule for solar panel segmentation."""

    def __init__(
        self,
        data_root: Path,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        val_split: float = 0.15,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.val_split = val_split

        self.image_dir = self.data_root / "images"
        self.semantic_dir = self.data_root / "semantic_masks"
        self.instance_dir = self.data_root / "instance_masks"

        self.train_ds: _SplitDataset | None = None
        self.val_ds: _SplitDataset | None = None
        self.test_ds: _SplitDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        all_paths = sorted(self.image_dir.glob("*.png"))
        if not all_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        n_val = max(1, int(len(all_paths) * self.val_split))
        n_train = len(all_paths) - n_val

        train_paths = all_paths[:n_train]
        val_paths = all_paths[n_train : n_train + n_val]

        if stage in (None, "fit"):
            self.train_ds = self._make_dataset(train_paths, self.train_transform)
            self.val_ds = self._make_dataset(val_paths, self.val_transform)
        if stage in (None, "test"):
            self.test_ds = self._make_dataset(val_paths, self.val_transform)

    def _make_dataset(
        self, paths: list[Path], transform: Callable | None
    ) -> _SplitDataset:
        return _SplitDataset(
            image_paths=paths,
            semantic_mask_dir=self.semantic_dir,
            instance_mask_dir=self.instance_dir if self.instance_dir.exists() else None,
            transform=transform,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Call setup('fit') before requesting train_dataloader")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Call setup('fit') before requesting val_dataloader")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("Call setup('test') before requesting test_dataloader")
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
