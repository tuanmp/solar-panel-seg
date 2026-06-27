from __future__ import annotations

import hydra
from omegaconf import DictConfig
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger

from solar_seg.data.preprocessing.dataset import SolarSegDataModule
from solar_seg.data.preprocessing.augmentations import (
    training_transforms,
    validation_transforms,
)
from solar_seg.models.mask2former_module import Mask2FormerModule


@hydra.main(
    config_path="../../configs",
    config_name=None,
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    L.seed_everything(seed=cfg.seed, workers=True)

    data_cfg = cfg.data
    model_cfg = cfg.model
    trainer_cfg = cfg.trainer

    datamodule = SolarSegDataModule(
        data_root=Path(data_cfg.data_root),
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        train_transform=training_transforms(),
        val_transform=validation_transforms(),
        val_split=data_cfg.val_split,
    )

    model = Mask2FormerModule(
        model_name=model_cfg.model_name,
        learning_rate=model_cfg.learning_rate,
        weight_decay=model_cfg.weight_decay,
        warmup_steps=model_cfg.warmup_steps,
        num_labels=model_cfg.num_labels,
    )

    logger = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        tracking_uri=cfg.mlflow_tracking_uri,
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        ),
        EarlyStopping(monitor="val_loss", mode="min", patience=5),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        deterministic=trainer_cfg.deterministic,
        precision=trainer_cfg.precision,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
        enable_progress_bar=trainer_cfg.enable_progress_bar,
        limit_train_batches=trainer_cfg.limit_train_batches,
        limit_val_batches=trainer_cfg.limit_val_batches,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
