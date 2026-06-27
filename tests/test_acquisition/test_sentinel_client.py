from solar_seg.data.acquisition.geo_utils import BBox
from solar_seg.data.acquisition.sentinel_client import SentinelHubClient


def test_sentinel_client_init():
    client = SentinelHubClient()
    assert client._config is not None


def test_sentinel_client_bbox_conversion():
    client = SentinelHubClient()
    bbox = BBox(37.77, -122.42, 37.78, -122.41)
    sh_bbox = client._to_sh_bbox(bbox)
    assert "4326" in sh_bbox.crs.value
    assert abs(sh_bbox.min_x + 122.42) < 0.001
