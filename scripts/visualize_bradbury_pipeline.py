#!/usr/bin/env python3
"""Generate a figure demonstrating the Bradbury preprocessing pipeline.

Shows: raw TIF → polygon annotation → 400×400 tile → semantic mask → instance mask.

Usage:
  uv run python scripts/visualize_bradbury_pipeline.py
  uv run python scripts/visualize_bradbury_pipeline.py --output figures/pipeline.png
"""

import argparse
import sys
from pathlib import Path

# Allow importing from scripts/ (not a package)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from process_bradbury import (
    load_polygon_metadata,
    load_polygon_vertices,
    find_image_tif,
)

TILE_SIZE = 400
RAW_DIR = Path("data/raw/bradbury")
DEFAULT_OUTPUT = Path("figures/bradbury_pipeline.png")


def polygon_to_tile_local(vertices, cx, cy):
    half = TILE_SIZE // 2
    return [(x - cx + half, y - cy + half) for x, y in vertices]


def rasterize_polygon(vertices, shape):
    pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def pick_best_sample(meta, verts, raw_dir, n=5):
    """Pick samples with large, high-confidence polygons near image center."""
    scored = []
    for p in meta:
        if p["polygon_id"] not in verts:
            continue
        v = verts[p["polygon_id"]]
        area = cv2.contourArea(v.astype(np.int32).reshape((-1, 1, 2)))
        scored.append((area * p["jaccard"], p))
    scored.sort(key=lambda x: x[0], reverse=True)
    for _, p in scored:
        tif = find_image_tif(p["image_name"], raw_dir)
        if tif:
            return p
    return scored[0][1] if scored else meta[0]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Bradbury preprocessing pipeline"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    meta_path = args.raw_dir / "polygonDataExceptVertices.csv"
    vert_path = args.raw_dir / "polygonVertices_PixelCoordinates.csv"
    meta = load_polygon_metadata(meta_path)
    meta = [m for m in meta if m["jaccard"] >= 0.5]
    verts = load_polygon_vertices(vert_path)

    sample = pick_best_sample(meta, verts, args.raw_dir)
    pid = sample["polygon_id"]
    image_name = sample["image_name"]
    tif_path = find_image_tif(image_name, args.raw_dir)

    if tif_path is None:
        print(f"Error: TIF not found for {image_name}. Download imagery first.")
        return

    img = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    cx = int(round(sample["centroid_x"]))
    cy = int(round(sample["centroid_y"]))
    vertices_full = verts[pid]

    # Extract tile
    half = TILE_SIZE // 2
    tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    sx1, sy1 = max(0, cx - half), max(0, cy - half)
    sx2, sy2 = min(w, cx + half), min(h, cy + half)
    dx1, dy1 = max(0, half - cx), max(0, half - cy)
    tile_src = img_rgb[sy1:sy2, sx1:sx2]
    th, tw = tile_src.shape[:2]
    tile[dy1 : dy1 + th, dx1 : dx1 + tw] = tile_src

    # Convert polygon to tile-local
    local_poly = polygon_to_tile_local(vertices_full, cx, cy)
    sem_mask = rasterize_polygon(local_poly, (TILE_SIZE, TILE_SIZE)) * 255
    inst_mask = rasterize_polygon(local_poly, (TILE_SIZE, TILE_SIZE))

    # == Build figure ==
    fig = plt.figure(figsize=(14, 6), facecolor="white")

    # Panel A: Full orthoimage with polygon annotation
    ax1 = fig.add_subplot(2, 3, (1, 2))
    ax1.imshow(img_rgb)
    # Draw the polygon
    poly_local = vertices_full.copy()
    poly_closed = np.vstack([poly_local, poly_local[:1]])
    ax1.plot(poly_closed[:, 0], poly_closed[:, 1], "lime", linewidth=2)
    ax1.plot(cx, cy, "ro", markersize=6)
    # Draw tile bounding box
    rect = mpatches.Rectangle(
        (cx - half, cy - half), TILE_SIZE, TILE_SIZE,
        fill=False, edgecolor="red", linewidth=2, linestyle="--"
    )
    ax1.add_patch(rect)
    ax1.set_title(
        f"USGS Orthoimage ({image_name}.tif, {w}×{h} px)\n"
        f"Polygon #{pid} · Jaccard {sample['jaccard']:.3f}",
        fontsize=10,
    )
    ax1.axis("off")

    # Panel B: Zoomed overview
    ax2 = fig.add_subplot(2, 3, 3)
    margin = 300
    zx1, zx2 = max(0, cx - margin), min(w, cx + margin)
    zy1, zy2 = max(0, cy - margin), min(h, cy + margin)
    zoom = img_rgb[zy1:zy2, zx1:zx2]
    ax2.imshow(zoom)
    ax2.plot(
        poly_closed[:, 0] - zx1, poly_closed[:, 1] - zy1,
        "lime", linewidth=2,
    )
    rect = mpatches.Rectangle(
        (cx - half - zx1, cy - half - zy1), TILE_SIZE, TILE_SIZE,
        fill=False, edgecolor="red", linewidth=2, linestyle="--",
    )
    ax2.add_patch(rect)
    ax2.set_title(f"Zoomed · {2*margin}×{2*margin}px region", fontsize=10)
    ax2.axis("off")

    # Panel C: 400×400 tile
    ax3 = fig.add_subplot(2, 3, 4)
    ax3.imshow(tile)
    ax3.set_title("400×400 Tile (model input)", fontsize=10)
    ax3.axis("off")

    # Panel D: Semantic mask
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.imshow(sem_mask, cmap="gray", vmin=0, vmax=255)
    ax4.set_title("Semantic Mask (0/255)", fontsize=10)
    ax4.axis("off")

    # Panel E: Instance mask
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.imshow(inst_mask, cmap="tab20", vmin=0, vmax=20)
    ax5.set_title("Instance Mask (0/1)", fontsize=10)
    ax5.axis("off")

    plt.suptitle(
        "Bradbury Dataset — Preprocessing Pipeline",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(args.output), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
