from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L

from solar_seg.data.preprocessing.dataset import SolarSegDataModule
from solar_seg.data.preprocessing.augmentations import (
    training_transforms,
    validation_transforms,
)
from solar_seg.models.mask2former_module import Mask2FormerModule


def run_ablation(
    config_path: str,
    overrides: list[str],
    output_dir: Path = Path("results/ablations"),
) -> dict:
    """Run a single ablation experiment and return metrics."""
    with hydra.initialize_config_dir(config_dir="configs/experiment"):
        cfg = hydra.compose(
            config_name=config_path,
            overrides=overrides,
        )

    data_roots_raw = OmegaConf.to_object(cfg.data.data_roots)
    data_roots: dict[str, Path] = {
        name: Path(root).resolve() for name, root in data_roots_raw.items()
    }

    datamodule = SolarSegDataModule(
        data_roots=data_roots,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_transform=training_transforms(),
        val_transform=validation_transforms(),
        val_split=cfg.data.val_split,
    )
    datamodule.setup("fit")
    source_names = datamodule.source_names

    model = Mask2FormerModule(
        model_name=cfg.model.model_name,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        num_labels=cfg.model.num_labels,
        source_names=source_names,
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_progress_bar=False,
        logger=False,
    )

    trainer.fit(model=model, datamodule=datamodule)
    results = trainer.test(model=model, datamodule=datamodule)

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = "_".join(overrides).replace("/", "-").replace("=", "_")
    report = {
        "config": cfg.experiment_name,
        "overrides": overrides,
        "test_metrics": [{k: str(v) for k, v in r.items()} for r in results],
    }
    (output_dir / f"{fname}.json").write_text(json.dumps(report, indent=2))

    return report
