#!/usr/bin/env python3
"""Process Bradbury et al. (2016) raw annotation data into training-ready tiles.

Pipeline:
  1. Load polygon metadata CSV (centroids, image names, Jaccard scores)
  2. Load polygon vertices CSV (pixel coordinates per polygon)
  3. For each image TIF file:
     a. Read all polygons associated with that image
     b. For each polygon:
        - Extract 400x400 tile centered on the centroid
        - Convert polygon vertices to tile-local coordinates
        - Rasterize to binary + instance masks
        - Save tile image and masks
  4. Output: data/processed/bradbury/{images,semantic_masks,instance_masks}/

Usage:
  # Annotations already in data/raw/bradbury/
  # Imagery TIFs expected under data/raw/bradbury/<city>/

  uv run python scripts/process_bradbury.py \
      --raw-dir data/raw/bradbury \
      --output-dir data/processed/bradbury \
      --min-jaccard 0.5 \
      --min-area 20
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage import measure

from solar_seg.data.preprocessing.mask_converter import polygons_to_mask, polygons_to_instance_mask
from solar_seg.data.preprocessing.tile_extractor import extract_tile_around_centroid


TILE_SIZE = 400
CITY_IMAGE_DIRS = ["fresno", "stockton", "modesto", "oxnard"]


def load_polygon_metadata(csv_path: Path) -> list[dict]:
    """Load polygonDataExceptVertices.csv into list of dicts."""
    rows = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "polygon_id": int(row["polygon_id"]),
                    "centroid_x": float(row["centroid_longitude_pixels"]),
                    "centroid_y": float(row["centroid_latitude_pixels"]),
                    "image_name": row["image_name"].strip(),
                    "city": row["city"].strip().lower(),
                    "jaccard": float(row["jaccard_index"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def load_polygon_vertices(csv_path: Path) -> dict[int, np.ndarray]:
    """Load polygonVertices_PixelCoordinates.csv into {polygon_id: (N,2) array}."""
    vertices = {}
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = int(row["polygon_id"])
                n_verts = int(row["number_vertices"])
                pts = []
                for i in range(1, n_verts + 1):
                    x_str = row.get(f"lon{i}", "")
                    y_str = row.get(f"lat{i}", "")
                    if x_str.strip() == "" or y_str.strip() == "":
                        break
                    pts.append((float(x_str), float(y_str)))
                if len(pts) == n_verts and len(pts) >= 3:
                    vertices[pid] = np.array(pts, dtype=np.float64)
            except (ValueError, KeyError):
                continue
    return vertices


def polygon_to_tile_local(
    vertices: np.ndarray, centroid_x: int, centroid_y: int
) -> list[list[tuple[float, float]]]:
    """Convert polygon vertices from full-image coords to tile-local coords."""
    half = TILE_SIZE // 2
    local = []
    for x, y in vertices:
        local.append((x - centroid_x + half, y - centroid_y + half))
    return [local]


def find_image_tif(image_name: str, raw_dir: Path) -> Path | None:
    """Find a TIF file by image_name in any city subdirectory."""
    for city in CITY_IMAGE_DIRS:
        # USGS images: e.g. 11ska460890.tif or 11ska460890.tif.xml or .tif without .xml
        candidates = [
            raw_dir / city / f"{image_name}.tif",
            raw_dir / city / f"{image_name}.tif.xml",
        ]
        for cand in candidates:
            if cand.suffix == ".tif" and cand.exists():
                return cand
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process Bradbury annotations into training tiles"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw/bradbury"),
        help="Directory containing annotation CSVs and city imagery subdirectories",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/bradbury"),
        help="Output directory for processed tiles and masks",
    )
    parser.add_argument(
        "--min-jaccard", type=float, default=0.5,
        help="Minimum Jaccard index to include (default: 0.5, 99.4%% above this)",
    )
    parser.add_argument(
        "--min-area", type=int, default=20,
        help="Minimum polygon area in pixels to include",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only N images (0 = all)",
    )
    args = parser.parse_args()

    meta_path = args.raw_dir / "polygonDataExceptVertices.csv"
    vert_path = args.raw_dir / "polygonVertices_PixelCoordinates.csv"

    if not meta_path.exists():
        print(f"Error: {meta_path} not found. Run scripts/download_bradbury.py first.")
        sys.exit(1)
    if not vert_path.exists():
        print(f"Error: {vert_path} not found. Run scripts/download_bradbury.py first.")
        sys.exit(1)

    print("Loading annotation data ...")
    all_polygons = load_polygon_metadata(meta_path)
    all_vertices = load_polygon_vertices(vert_path)
    print(f"  {len(all_polygons)} polygons loaded")

    # Filter by Jaccard
    all_polygons = [p for p in all_polygons if p["jaccard"] >= args.min_jaccard]
    print(f"  {len(all_polygons)} polygons after min_jaccard={args.min_jaccard}")

    # Group by image_name
    polygons_by_image: dict[str, list[dict]] = defaultdict(list)
    for p in all_polygons:
        polygons_by_image[p["image_name"]].append(p)

    print(f"  {len(polygons_by_image)} unique images")

    # Output directories
    img_dir = args.output_dir / "images"
    sem_dir = args.output_dir / "semantic_masks"
    inst_dir = args.output_dir / "instance_masks"
    for d in [img_dir, sem_dir, inst_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Process
    images_processed = 0
    tiles_written = 0
    images_skipped = 0

    for image_name, polygons in sorted(polygons_by_image.items()):
        if args.limit and images_processed >= args.limit:
            break

        tif_path = find_image_tif(image_name, args.raw_dir)
        if tif_path is None:
            images_skipped += 1
            continue

        img = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
        if img is None:
            images_skipped += 1
            continue

        images_processed += 1
        h, w = img.shape[:2]

        for poly_meta in polygons:
            pid = poly_meta["polygon_id"]
            cx = int(round(poly_meta["centroid_x"]))
            cy = int(round(poly_meta["centroid_y"]))

            if pid not in all_vertices:
                continue

            vertices_full = all_vertices[pid]
            area = cv2.contourArea(vertices_full.astype(np.int32).reshape((-1, 1, 2)))
            if area < args.min_area:
                continue

            # Extract tile
            result = extract_tile_around_centroid(
                tif_path, cx, cy, TILE_SIZE, img_dir, f"{image_name}_{pid}"
            )
            if result["image"] is None:
                continue

            # Convert polygon to tile-local coords
            local_poly = polygon_to_tile_local(vertices_full, cx, cy)

            tile_shape = (TILE_SIZE, TILE_SIZE)
            bin_mask = polygons_to_mask(local_poly, tile_shape)
            inst_mask = polygons_to_instance_mask(local_poly, tile_shape)

            # Save masks
            cv2.imwrite(str(sem_dir / f"{image_name}_{pid}_semantic.png"), bin_mask * 255)
            cv2.imwrite(str(inst_dir / f"{image_name}_{pid}_instance.png"), inst_mask.astype(np.uint16))

            tiles_written += 1

        if images_processed % 50 == 0:
            print(f"  Processed {images_processed} images, {tiles_written} tiles ...")

    print(f"\nDone: {images_processed} images processed, {tiles_written} tiles written")
    if images_skipped:
        print(f"  {images_skipped} images skipped (TIF not found)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
