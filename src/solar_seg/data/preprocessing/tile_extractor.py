from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_tile_around_centroid(
    image_path: Path,
    centroid_x: int,
    centroid_y: int,
    tile_size: int = 400,
    output_dir: Path = Path("data/processed/tiles"),
    prefix: str = "tile",
) -> dict[str, Path]:
    """Extract a fixed-size tile from a large image centered on a centroid.

    Pads with zeros if the tile extends beyond image boundaries.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    half = tile_size // 2

    x_start = centroid_x - half
    y_start = centroid_y - half
    x_end = x_start + tile_size
    y_end = y_start + tile_size

    tile = np.zeros((tile_size, tile_size, 3), dtype=img.dtype)
    src_x_start = max(0, x_start)
    src_y_start = max(0, y_start)
    src_x_end = min(w, x_end)
    src_y_end = min(h, y_end)

    dst_x_start = max(0, -x_start)
    dst_y_start = max(0, -y_start)

    tile[
        dst_y_start : dst_y_start + (src_y_end - src_y_start),
        dst_x_start : dst_x_start + (src_x_end - src_x_start),
    ] = img[src_y_start:src_y_end, src_x_start:src_x_end]

    output_dir.mkdir(parents=True, exist_ok=True)
    tile_path = output_dir / f"{prefix}_x{centroid_x}_y{centroid_y}.png"
    cv2.imwrite(str(tile_path), tile)

    return {"image": tile_path}


def filter_tile_by_panel_fraction(
    mask_path: Path,
    min_fraction: float = 0.05,
) -> bool:
    """Return True if the mask has at least min_fraction panel pixels."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    return float((mask > 0).sum()) / mask.size >= min_fraction


def batch_extract_centroids(
    image_path: Path,
    centroids: list[tuple[int, int]],
    output_dir: Path,
    tile_size: int = 400,
    prefix: str = "tile",
) -> list[dict[str, Path | None]]:
    """Extract tiles for a list of centroids from a single large image."""
    results = []
    for idx, (cx, cy) in enumerate(centroids):
        result = extract_tile_around_centroid(
            image_path, cx, cy, tile_size, output_dir, f"{prefix}_{idx}"
        )
        results.append(result)
    return results
