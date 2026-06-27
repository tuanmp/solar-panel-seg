import numpy as np
import cv2

from solar_seg.data.preprocessing.tile_extractor import (
    extract_tile_around_centroid,
    filter_tile_by_panel_fraction,
)


def test_extract_tile_around_centroid(tmp_path):
    img_path = tmp_path / "test.png"
    img = np.ones((5000, 5000, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(img_path), img)

    result = extract_tile_around_centroid(img_path, 2500, 2500, tile_size=400, output_dir=tmp_path)
    assert result["image"].exists()

    tile = cv2.imread(str(result["image"]))
    assert tile.shape == (400, 400, 3)


def test_extract_near_edge(tmp_path):
    img_path = tmp_path / "edge.png"
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(img_path), img)

    result = extract_tile_around_centroid(img_path, 10, 10, tile_size=400, output_dir=tmp_path)
    assert result["image"].exists()
    tile = cv2.imread(str(result["image"]))
    assert tile.shape == (400, 400, 3)


def test_filter_tile_by_panel_fraction(tmp_path):
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    cv2.imwrite(str(mask_path), mask)

    assert filter_tile_by_panel_fraction(mask_path, min_fraction=0.03)
    assert not filter_tile_by_panel_fraction(mask_path, min_fraction=0.10)


def test_filter_missing_file(tmp_path):
    assert not filter_tile_by_panel_fraction(tmp_path / "nonexistent.png")
