from solar_seg.data.acquisition.geo_utils import BBox, bbox_from_center, gsd_to_zoom


def test_bbox_center():
    bbox = BBox(lat_min=37.0, lon_min=-122.0, lat_max=38.0, lon_max=-121.0)
    assert bbox.center == (37.5, -121.5)


def test_bbox_from_center():
    bbox = bbox_from_center(37.77, -122.42, size_km=1.0)
    assert bbox.lat_min < 37.77 < bbox.lat_max
    assert bbox.lon_min < -122.42 < bbox.lon_max
    assert abs(bbox.height_km - 1.0) < 0.01


def test_gsd_to_zoom():
    assert gsd_to_zoom(0.1) == 21
    assert gsd_to_zoom(0.5) == 18
    assert gsd_to_zoom(10.0) == 14


def test_bbox_to_list():
    bbox = BBox(37.0, -122.0, 38.0, -121.0)
    assert bbox.to_list() == [-122.0, 37.0, -121.0, 38.0]
