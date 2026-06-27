import numpy as np

from solar_seg.data.preprocessing.mask_converter import (
    polygons_to_mask,
    polygons_to_instance_mask,
    mask_to_polygons,
)


def test_polygons_to_mask():
    polygons = [[(10, 10), (90, 10), (90, 90), (10, 90)]]
    mask = polygons_to_mask(polygons, (100, 100))
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert mask[50, 50] == 1
    assert mask[0, 0] == 0


def test_polygons_to_instance_mask():
    polygons = [
        [(10, 10), (40, 10), (40, 40), (10, 40)],
        [(60, 60), (90, 60), (90, 90), (60, 90)],
    ]
    label_map = polygons_to_instance_mask(polygons, (100, 100))
    assert label_map[20, 20] == 1
    assert label_map[75, 75] == 2
    assert label_map[0, 0] == 0


def test_mask_to_polygons():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 1
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 1
    assert polygons[0].shape[1] == 2


def test_roundtrip():
    polygons = [[(10, 10), (90, 10), (90, 90), (10, 90)]]
    mask = polygons_to_mask(polygons, (100, 100))
    recovered = mask_to_polygons(mask)
    assert len(recovered) == 1
