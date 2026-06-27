#!/usr/bin/env python3
"""Process BDAPPV raw dataset into training-ready tiles.

BDAPPV provides 400×400 image/mask pairs under two providers (Google, IGN).
Masks are binary PNGs (white=panel, black=background) — no instance separation.
This script runs connected-components labeling to derive instance masks.

Pipeline:
  1. Scan data/raw/bdappv/bdappv/{google,ign}/mask/ for binary mask PNGs
  2. For each mask, find the corresponding image in {provider}/img/
  3. Convert binary mask → semantic (0/255) + instance (connected-components) masks
  4. Organize into the dataset-expected layout:
     data/processed/bdappv/{images,semantic_masks,instance_masks}/

Usage:
  uv run python scripts/process_bdappv.py
  uv run python scripts/process_bdappv.py --limit 100  # quick test
"""

import argparse
import shutil
import sys
from pathlib import Path

from solar_seg.data.preprocessing.mask_converter import bdappv_mask_to_labelmaps

# Directory layout inside the extracted bdappv.zip
PROVIDERS = ["google", "ign"]


def find_image(mask_stem: str, provider_dir: Path) -> Path | None:
    """Find the corresponding image PNG for a given mask stem."""
    img_path = provider_dir / "img" / f"{mask_stem}.png"
    return img_path if img_path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process BDAPPV binary masks into semantic + instance label maps"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw/bdappv/bdappv"),
        help="Directory containing google/ and ign/ subdirectories",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/bdappv"),
        help="Output directory for processed tiles and masks",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only N masks (0 = all)",
    )
    parser.add_argument(
        "--include-providers", type=str, nargs="+", default=PROVIDERS,
        help="Which providers to include (default: google ign)",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    output_dir = args.output_dir

    # Create output subdirectories
    img_dir = output_dir / "images"
    sem_dir = output_dir / "semantic_masks"
    inst_dir = output_dir / "instance_masks"
    for d in [img_dir, sem_dir, inst_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all mask files
    mask_files: list[tuple[Path, str]] = []  # (mask_path, provider_name)
    for provider in args.include_providers:
        mask_glob = raw_dir / provider / "mask"
        if not mask_glob.exists():
            print(f"Warning: {mask_glob} not found — skipping provider '{provider}'")
            continue
        found = list(mask_glob.glob("*.png"))
        mask_files.extend((p, provider) for p in found)
        print(f"Found {len(found)} masks in {provider}/mask/")

    if not mask_files:
        print("No mask files found. Run scripts/download_bdappv.py first.")
        sys.exit(1)

    total = len(mask_files)
    if args.limit:
        mask_files = mask_files[: args.limit]
        print(f"Limited to {len(mask_files)} masks")

    # Create a temp output dir for the converter, then sort into subdirs
    tmp_dir = output_dir  # bdappv_mask_to_labelmaps writes directly here

    processed = 0
    skipped_missing_image = 0
    errors = 0

    for mask_path, provider in mask_files:
        stem = mask_path.stem

        img_path = find_image(stem, raw_dir / provider)
        if img_path is None:
            skipped_missing_image += 1
            continue

        try:
            # Convert mask → {stem}_semantic.png + {stem}_instance.png (in tmp_dir)
            bdappv_mask_to_labelmaps(mask_path, tmp_dir)

            # Sort into expected subdirectories
            sem_src = tmp_dir / f"{stem}_semantic.png"
            inst_src = tmp_dir / f"{stem}_instance.png"
            sem_src.rename(sem_dir / f"{stem}_semantic.png")
            inst_src.rename(inst_dir / f"{stem}_instance.png")

            # Copy image
            shutil.copy2(str(img_path), str(img_dir / f"{stem}.png"))

            processed += 1
        except Exception as e:
            print(f"  Error processing {mask_path.name}: {e}")
            errors += 1

        if processed % 2000 == 0:
            print(f"  {processed}/{total} tiles ...")

    print(f"\nDone: {processed} tiles written to {output_dir}")
    if skipped_missing_image:
        print(f"  {skipped_missing_image} masks skipped (missing image)")
    if errors:
        print(f"  {errors} errors")


if __name__ == "__main__":
    main()
