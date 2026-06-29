from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
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


def _get_checkpoint_dir(base_dir: str = "lightning_logs") -> str:
    """Return a job-specific checkpoint directory.

    Uses SLURM_JOB_ID for batch jobs; falls back to a timestamp for interactive runs.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    qos = os.environ.get("SLURM_JOB_QOS", "")
    if job_id and "interactive" not in qos and "jupyter" not in qos:
        return os.path.join(base_dir, job_id)
    return os.path.join(base_dir, datetime.now().strftime("%Y-%m-%d--%H-%M"))


@hydra.main(
    config_path="../../configs",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    L.seed_everything(seed=cfg.seed, workers=True)

    data_cfg = cfg.data
    model_cfg = cfg.model
    trainer_cfg = cfg.trainer

    # Resolve data_roots to absolute paths (Hydra may have interpolated CWD)
    data_roots_raw = OmegaConf.to_object(data_cfg.data_roots)
    data_roots: dict[str, Path] = {
        name: Path(root).resolve() for name, root in data_roots_raw.items()
    }

    aug_kwargs = {}
    if hasattr(data_cfg, "aug") and data_cfg.aug is not None:
        aug_kwargs = dict(data_cfg.aug)

    datamodule = SolarSegDataModule(
        data_roots=data_roots,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        train_transform=training_transforms(**aug_kwargs),
        val_transform=validation_transforms(),
        val_split=data_cfg.val_split,
    )

    # Setup early to discover source names
    datamodule.setup("fit")
    source_names = datamodule.source_names

    model = Mask2FormerModule(
        model_name=model_cfg.model_name,
        learning_rate=model_cfg.learning_rate,
        weight_decay=model_cfg.weight_decay,
        warmup_steps=model_cfg.warmup_steps,
        num_labels=model_cfg.num_labels,
        source_names=source_names,
        loss_ce_weight=float(model_cfg.get("loss_ce_weight", 2.0)),
        loss_mask_weight=float(model_cfg.get("loss_mask_weight", 5.0)),
        loss_dice_weight=float(model_cfg.get("loss_dice_weight", 5.0)),
    )

    logger = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        tracking_uri=cfg.mlflow_tracking_uri,
        log_model="all",
    )

    # Monitor the first source's validation loss for checkpointing
    primary_source = source_names[0]
    monitor_metric = f"val/{primary_source}/loss"

    callbacks = [
        ModelCheckpoint(
            monitor=monitor_metric,
            mode="min",
            save_top_k=1,
            filename=f"best-{{epoch:02d}}-{{val/{primary_source}/loss:.4f}}",
        ),
        EarlyStopping(monitor=monitor_metric, mode="min", patience=5),
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
        default_root_dir=_get_checkpoint_dir(),
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
