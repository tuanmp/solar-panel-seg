from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage import measure


def polygons_to_mask(
    polygons: list[list[tuple[float, float]]],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Convert list of polygon vertex lists to a binary mask.

    Args:
        polygons: Each polygon is a list of (x, y) pixel coordinate tuples.
        image_shape: (height, width) of the output mask.

    Returns:
        Binary mask of shape image_shape: 1 = panel, 0 = background.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in polygons:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask


def polygons_to_instance_mask(
    polygons: list[list[tuple[float, float]]],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Convert polygons to an instance label map.

    Each polygon gets a unique integer ID (1, 2, 3, ...). Background is 0.
    """
    label_map = np.zeros(image_shape, dtype=np.int32)
    for idx, polygon in enumerate(polygons, start=1):
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(label_map, [pts], idx)
    return label_map


def mask_to_polygons(mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    """Convert a binary mask to list of polygon vertex arrays.

    Uses OpenCV contours. Filters out polygons below min_area (in pixels).
    """
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            polygon = contour.squeeze(axis=1)
            if polygon.ndim == 2 and polygon.shape[1] == 2:
                polygons.append(polygon)
    return polygons


def bdappv_mask_to_labelmaps(
    mask_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Convert BDAPPV binary mask .png to semantic + instance label maps."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    binary = (mask > 0).astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)

    semantic_path = output_dir / f"{mask_path.stem}_semantic.png"
    cv2.imwrite(str(semantic_path), binary * 255)

    labeled = measure.label(binary, connectivity=2)
    instance_path = output_dir / f"{mask_path.stem}_instance.png"
    cv2.imwrite(str(instance_path), labeled.astype(np.int32))

    return {"semantic": semantic_path, "instance": instance_path}
