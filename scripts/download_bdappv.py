#!/usr/bin/env python3
"""Download BDAPPV dataset (Kasmi et al., 2023) from Zenodo.

Training data (bdappv.zip):  ~8.2 GB  — images + masks + metadata
Crowdsourcing data (data.zip): ~17 MB — raw annotations + replication

Usage:
  # Full training dataset (~8.2 GB)
  uv run python scripts/download_bdappv.py

  # Crowdsourcing/raw data only (~17 MB)
  uv run python scripts/download_bdappv.py --raw-only

  # Both
  uv run python scripts/download_bdappv.py --all
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import urllib.request

BDAPPV_URL = "https://zenodo.org/api/records/7358126/files/bdappv.zip/content"
DATA_URL = "https://zenodo.org/api/records/7358126/files/data.zip/content"


def download_file(url: str, dest: Path, label: str = "") -> None:
    if dest.exists():
        print(f"Skipping {dest.name} (exists)")
        return
    print(f"Downloading {label or dest.name} ... ", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"done ({dest.stat().st_size / 1e6:.0f} MB)")


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    print(f"Extracting {zip_path.name} to {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BDAPPV solar panel dataset from Zenodo"
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Download only crowdsourcing/raw data (data.zip, ~17 MB)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download both training data and raw data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/bdappv"),
        help="Output directory (default: data/raw/bdappv)",
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    download_training = args.all or not args.raw_only
    download_raw = args.all or args.raw_only

    if download_training:
        zip_path = output_dir / "bdappv.zip"
        download_file(BDAPPV_URL, zip_path, "bdappv.zip (training data)")
        extract_zip(zip_path, output_dir)

    if download_raw:
        zip_path = output_dir / "data.zip"
        download_file(DATA_URL, zip_path, "data.zip (crowdsourcing data)")
        extract_zip(zip_path, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
