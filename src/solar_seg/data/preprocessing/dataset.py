from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
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
    """LightningDataModule for multi-source solar panel segmentation.

    Supports any number of data sources.  Each source is a directory root with
    images/, semantic_masks/, and (optionally) instance_masks/ subdirectories.

    * Training: all source train splits are concatenated into a single loader.
    * Validation: returns **one loader per source** so the LightningModule
      can log per-source metrics.
    * Testing: same as validation (one loader per source).
    """

    def __init__(
        self,
        data_roots: dict[str, Path],
        batch_size: int = 8,
        num_workers: int = 4,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        val_split: float = 0.15,
    ) -> None:
        super().__init__()
        self._data_roots = {name: Path(root) for name, root in data_roots.items()}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.val_split = val_split

        self._source_names = list(self._data_roots.keys())

        self._train_ds: ConcatDataset | None = None
        self._val_ds_list: list[_SplitDataset] = []
        self._test_ds_list: list[_SplitDataset] = []

    @property
    def source_names(self) -> list[str]:
        """Ordered list of source names (matches dataloader order)."""
        return list(self._source_names)

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "test"):
            return

        all_train: list[Dataset] = []
        self._val_ds_list = []
        self._test_ds_list = []

        for name, root in self._data_roots.items():
            img_dir = root / "images"
            sem_dir = root / "semantic_masks"
            inst_dir = root / "instance_masks"
            if not inst_dir.exists():
                inst_dir = None

            paths = sorted(img_dir.glob("*.png"))
            if not paths:
                print(f"Warning: no images found in {img_dir} — skipping source '{name}'")
                continue

            n_val = max(1, int(len(paths) * self.val_split))
            n_train = len(paths) - n_val
            train_paths = paths[:n_train]
            val_paths = paths[n_train:]

            if stage in (None, "fit"):
                src_train = _SplitDataset(
                    image_paths=train_paths,
                    semantic_mask_dir=sem_dir,
                    instance_mask_dir=inst_dir,
                    transform=self.train_transform,
                )
                src_val = _SplitDataset(
                    image_paths=val_paths,
                    semantic_mask_dir=sem_dir,
                    instance_mask_dir=inst_dir,
                    transform=self.val_transform,
                )
                all_train.append(src_train)
                self._val_ds_list.append(src_val)

            if stage in (None, "test"):
                src_test = _SplitDataset(
                    image_paths=val_paths,
                    semantic_mask_dir=sem_dir,
                    instance_mask_dir=inst_dir,
                    transform=self.val_transform,
                )
                self._test_ds_list.append(src_test)

        if stage in (None, "fit"):
            if not all_train:
                raise RuntimeError("No training data found for any source")
            self._train_ds = ConcatDataset(all_train) if len(all_train) > 1 else all_train[0]
            # source_names only includes sources that actually produced data
            self._source_names = [name for name in self._data_roots
                                  if (self._data_roots[name] / "images").is_dir()]

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("Call setup('fit') before requesting train_dataloader")
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> list[DataLoader]:
        if not self._val_ds_list:
            raise RuntimeError("Call setup('fit') before requesting val_dataloader")
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            for ds in self._val_ds_list
        ]

    def test_dataloader(self) -> list[DataLoader]:
        if not self._test_ds_list:
            raise RuntimeError("Call setup('test') before requesting test_dataloader")
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            for ds in self._test_ds_list
        ]
