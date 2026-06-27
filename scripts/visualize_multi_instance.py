#!/usr/bin/env python3
"""Visualize a Bradbury image with multiple distinct instances.

Shows: full orthoimage with all polygon annotations, then the extracted
400×400 tile + semantic + instance mask for each panel individually.

Usage:
  uv run python scripts/visualize_multi_instance.py
  uv run python scripts/visualize_multi_instance.py --image 11ska445680 --output figures/multi_instance.png
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from process_bradbury import load_polygon_vertices, find_image_tif

TILE_SIZE = 400
RAW_DIR = Path("data/raw/bradbury")
DEFAULT_OUTPUT = Path("figures/multi_instance.png")
DEFAULT_IMAGE = "11ska400710"  # 7 well-distributed panels


def load_metadata(csv_path: Path) -> list[dict]:
    with csv_path.open("r") as f:
        return [
            r
            for r in csv.DictReader(f)
            if float(r["jaccard_index"]) >= 0.5
        ]


def rasterize_polygon(vertices, shape):
    pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def extract_tile(img_rgb, cx, cy):
    h, w = img_rgb.shape[:2]
    half = TILE_SIZE // 2
    tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    sx1, sy1 = max(0, cx - half), max(0, cy - half)
    sx2, sy2 = min(w, cx + half), min(h, cy + half)
    dx1, dy1 = max(0, half - cx), max(0, half - cy)
    src = img_rgb[sy1:sy2, sx1:sx2]
    th, tw = src.shape[:2]
    tile[dy1 : dy1 + th, dx1 : dx1 + tw] = src
    return tile


def main():
    parser = argparse.ArgumentParser(description="Visualize multi-instance Bradbury")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--max-panels", type=int, default=12,
                        help="Max panels to show (avoids overcrowding)")
    args = parser.parse_args()

    meta_path = args.raw_dir / "polygonDataExceptVertices.csv"
    vert_path = args.raw_dir / "polygonVertices_PixelCoordinates.csv"

    meta = load_metadata(meta_path)
    verts = load_polygon_vertices(vert_path)

    # Filter to target image
    img_rows = [r for r in meta if r["image_name"] == args.image]
    if not img_rows:
        print(f"No polygons found for image '{args.image}'")
        return

    # Get city from first row
    city = img_rows[0]["city"].strip().lower()
    tif_path = find_image_tif(args.image, args.raw_dir)
    if tif_path is None:
        print(f"TIF not found for {args.image}")
        return

    img = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load {tif_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Limit panels
    img_rows = img_rows[: args.max_panels]
    n = len(img_rows)

    # Layout: full image (big) on top left, then tiles in a grid
    n_cols = min(4, n)
    n_rows = 1 + (n + n_cols - 1) // n_cols  # full-image row + tile rows
    total_rows = n_rows
    total_cols = n_cols

    fig = plt.figure(figsize=(n_cols * 4 + 2, total_rows * 4), facecolor="white")

    # -- Full image panel --
    ax_full = fig.add_subplot(total_rows, total_cols, (1, total_cols))
    ax_full.imshow(img_rgb)
    cmap = plt.cm.tab20

    for idx, r in enumerate(img_rows):
        pid = int(float(r["polygon_id"]))
        cx, cy = int(round(float(r["centroid_longitude_pixels"]))), int(
            round(float(r["centroid_latitude_pixels"]))
        )
        v = verts.get(pid)
        if v is None:
            continue
        color = cmap(idx % 20)
        # Polygon outline
        poly_closed = np.vstack([v, v[:1]])
        ax_full.plot(poly_closed[:, 0], poly_closed[:, 1],
                     color=color, linewidth=2)
        # Centroid
        ax_full.plot(cx, cy, "o", color=color, markersize=6, mec="white", mew=0.5)
        # Label
        ax_full.text(cx + 10, cy - 10, str(idx + 1), fontsize=8,
                     color=color, fontweight="bold")
        # Tile extent
        half = TILE_SIZE // 2
        rect = mpatches.Rectangle(
            (cx - half, cy - half), TILE_SIZE, TILE_SIZE,
            fill=False, edgecolor=color, linewidth=1.5, linestyle="--",
        )
        ax_full.add_patch(rect)

    ax_full.set_title(
        f"{args.image} ({w}×{h} px) — {n} panels\n"
        f"City: {city.title()}",
        fontsize=11,
    )
    ax_full.axis("off")

    # -- Per-panel tiles --
    for idx, r in enumerate(img_rows):
        pid = int(float(r["polygon_id"]))
        cx, cy = int(round(float(r["centroid_longitude_pixels"]))), int(
            round(float(r["centroid_latitude_pixels"]))
        )
        v = verts.get(pid)
        if v is None:
            continue

        # Extract tile
        tile = extract_tile(img_rgb, cx, cy)
        local_poly = [(x - cx + TILE_SIZE // 2, y - cy + TILE_SIZE // 2)
                      for x, y in v]
        sem_mask = rasterize_polygon(local_poly, (TILE_SIZE, TILE_SIZE)) * 255
        inst_mask = rasterize_polygon(local_poly, (TILE_SIZE, TILE_SIZE))

        # Subplot row, col
        row = 1 + idx // n_cols
        col = idx % n_cols
        subplot_idx = row * total_cols + col + 1

        ax_tile = fig.add_subplot(total_rows, total_cols, subplot_idx)
        ax_tile.imshow(tile)
        # Overlay polygon outline on tile
        poly_closed = local_poly + [local_poly[0]]
        xs, ys = zip(*poly_closed)
        ax_tile.plot(xs, ys, color=cmap(idx % 20), linewidth=2)

        area = float(r["area_pixels"])
        jac = float(r["jaccard_index"])
        ax_tile.set_title(f"Panel {idx+1} | {area:.0f}px | J={jac:.2f}",
                          fontsize=8)
        ax_tile.axis("off")

    plt.suptitle(
        f"Bradbury ({city.title()}) — Multi-Instance Example\n"
        f"Full orthoimage (left) and individual 400×400 tiles for each panel",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(args.output), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {args.output} ({n} panels from {args.image})")


if __name__ == "__main__":
    main()
