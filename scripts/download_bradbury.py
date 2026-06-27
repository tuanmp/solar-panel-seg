#!/usr/bin/env python3
"""Download Bradbury et al. (2016) solar panel dataset from Figshare.

Annotation data (~115 MB):
  - polygonDataExceptVertices.csv
  - polygonVertices_LatitudeLongitude.csv
  - polygonVertices_PixelCoordinates.csv
  - SolarArrayPolygons.geojson
  - SolarArrayPolygons.json

Imagery (~45 GB total, USGS orthoimagery 5000×5000 px TIFs):
  - Fresno:  412 TIFs, ~30 GB
  - Stockton:  94 TIFs,  ~7 GB
  - Modesto:   20 TIFs, ~1.5 GB
  - Oxnard:    75 TIFs,  ~5 GB

Usage:
  # Annotations only (fast, ~115 MB)
  uv run python scripts/download_bradbury.py --annotations-only

  # Annotations + one city
  uv run python scripts/download_bradbury.py --city fresno

  # Annotations + all imagery (~45 GB)
  uv run python scripts/download_bradbury.py --city all
"""

import argparse
import os
import sys
from pathlib import Path

import urllib.request

ANNOTATION_FILES = [
    ("https://ndownloader.figshare.com/files/24115682", "polygonDataExceptVertices.csv"),
    ("https://ndownloader.figshare.com/files/24115685", "polygonVertices_LatitudeLongitude.csv"),
    ("https://ndownloader.figshare.com/files/24115688", "polygonVertices_PixelCoordinates.csv"),
    ("https://ndownloader.figshare.com/files/24115691", "SolarArrayPolygons.geojson"),
    ("https://ndownloader.figshare.com/files/24115694", "SolarArrayPolygons.json"),
]

CITY_ARTICLES = {
    "fresno": 3385828,
    "stockton": 3385804,
    "modesto": 3385789,
    "oxnard": 3385807,
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Skipping {dest.name} (exists)")
        return
    print(f"  Downloading {dest.name} ... ", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"done ({dest.stat().st_size / 1e6:.1f} MB)")


def download_annotations(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading Bradbury annotations ...")
    for url, fname in ANNOTATION_FILES:
        download_file(url, output_dir / fname)
    print("Annotations done.\n")


def download_city_imagery(article_id: int, city_name: str, output_dir: Path) -> None:
    city_dir = output_dir / city_name
    city_dir.mkdir(parents=True, exist_ok=True)

    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    import json

    print(f"Fetching file list for {city_name} ...")
    data = json.load(urllib.request.urlopen(api_url))
    files = data.get("files", [])

    total_mb = sum(f["size"] for f in files) / 1e6
    print(f"  {len(files)} files, {total_mb:.0f} MB total")

    for f in files:
        fname = f["name"]
        fsize = f["size"] / 1e6
        download_file(f["download_url"], city_dir / fname)

    print(f"{city_name.title()} done.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Bradbury solar panel dataset from Figshare"
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download only annotation files (JSON, CSV) — ~115 MB",
    )
    parser.add_argument(
        "--city",
        choices=["none", "all"] + list(CITY_ARTICLES),
        default="none",
        help="Download USGS orthoimagery for specific city (large downloads)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/bradbury"),
        help="Output directory (default: data/raw/bradbury)",
    )
    args = parser.parse_args()

    output_dir = args.output

    if args.city == "none" and not args.annotations_only:
        # Default: download annotations
        download_annotations(output_dir)
    elif args.annotations_only:
        download_annotations(output_dir)
    else:
        download_annotations(output_dir)

        if args.city == "all":
            for city_name, article_id in CITY_ARTICLES.items():
                download_city_imagery(article_id, city_name, output_dir)
        else:
            download_city_imagery(CITY_ARTICLES[args.city], args.city, output_dir)


if __name__ == "__main__":
    main()
