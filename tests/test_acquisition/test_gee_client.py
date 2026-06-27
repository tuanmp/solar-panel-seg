from pathlib import Path
from unittest.mock import patch

import pytest

from solar_seg.data.acquisition.geo_utils import BBox
from solar_seg.data.acquisition.gee_client import GEEClient


def test_gee_client_init():
    client = GEEClient(project="test-project")
    assert client.project == "test-project"
    assert not client._initialized


def test_gee_client_unsupported_source():
    client = GEEClient()
    with pytest.raises(ValueError, match="Unsupported source"):
        client._get_collection("unknown", BBox(0, 0, 1, 1), "2020-01-01", "2024-01-01")
