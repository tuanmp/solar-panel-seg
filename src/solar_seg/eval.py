#!/usr/bin/env python3
"""Evaluate a trained Mask2Former checkpoint on one or more datasets.

Loads a Lightning checkpoint, sets up the test dataloaders (one per data
source), and reports per-source loss, mIoU, and panoptic quality metrics.

Usage:
    # Single dataset
    uv run python -m solar_seg.eval \
        --ckpt lightning_logs/12345/abc/checkpoints/best.ckpt \
        --data-roots bdappv=data/processed/bdappv

    # Multiple datasets — metrics reported separately per source
    uv run python -m solar_seg.eval \
        --ckpt lightning_logs/12345/abc/checkpoints/best.ckpt \
        --data-roots bdappv=data/processed/bdappv bradbury=data/processed/bradbury
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch

from solar_seg.data.preprocessing.augmentations import validation_transforms
from solar_seg.data.preprocessing.dataset import SolarSegDataModule
from solar_seg.models.mask2former_module import Mask2FormerModule


def _parse_data_roots(pairs: list[str]) -> dict[str, Path]:
    """Parse ['name=path', ...] into {name: Path}."""
    roots: dict[str, Path] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"Invalid --data-roots entry '{pair}'. Expected NAME=PATH."
            )
        name, path = pair.split("=", 1)
        roots[name] = Path(path).resolve()
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained solar panel segmentation checkpoint"
    )
    parser.add_argument(
        "--ckpt", type=Path, required=True,
        help="Path to a Lightning .ckpt file",
    )
    parser.add_argument(
        "--data-roots", type=str, nargs="+", required=True,
        metavar="NAME=PATH",
        help="One or more datasets as NAME=PATH (e.g. bdappv=data/processed/bdappv)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction held out for the test split (default: 0.15)")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--limit-batches", type=float, default=1.0,
                        help="Fraction of test batches to run (default: 1.0)")
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    data_roots = _parse_data_roots(args.data_roots)

    datamodule = SolarSegDataModule(
        data_roots=data_roots,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_transform=validation_transforms(),
        val_split=args.val_split,
    )
    datamodule.setup("test")
    source_names = datamodule.source_names
    print(f"Evaluating on sources: {source_names}")

    # Override source_names from the checkpoint with the current run's ordering
    model = Mask2FormerModule.load_from_checkpoint(
        str(args.ckpt),
        source_names=source_names,
    )

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
        limit_test_batches=args.limit_batches,
    )

    results = trainer.test(model=model, datamodule=datamodule)

    # Merge results (each dataloader returns the same aggregated metric set)
    merged: dict[str, float] = {}
    for result in results:
        for key, value in result.items():
            if isinstance(value, (int, float, torch.Tensor)):
                merged[key] = float(value)

    # Group by source: "test/{source}/{metric}" -> {source: {metric: value}}
    by_source: dict[str, dict[str, float]] = {}
    for key, value in merged.items():
        parts = key.split("/")
        if len(parts) >= 3:
            source = parts[1]
            metric = "/".join(parts[2:])
            by_source.setdefault(source, {})[metric] = value

    print("\n=== Evaluation results ===")
    if not by_source:
        for key, value in sorted(merged.items()):
            print(f"  {key}: {value:.4f}")
    for source, metrics in by_source.items():
        print(f"\n[{source}]")
        for metric, value in sorted(metrics.items()):
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
