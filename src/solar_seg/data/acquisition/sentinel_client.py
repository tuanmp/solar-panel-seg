from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from sentinelhub import (
    BBox as SHBBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
)

from solar_seg.data.acquisition.geo_utils import BBox


@dataclass
class SentinelHubClient:
    """Client for Sentinel Hub OGC/WMS API.

    Requires OAuth2 credentials configured via Sentinel Hub CLI or env vars.
    """

    instance_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    _config: SHConfig = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._config = SHConfig()
        if self.instance_id:
            self._config.instance_id = self.instance_id
        if self.client_id:
            self._config.sh_client_id = self.client_id
        if self.client_secret:
            self._config.sh_client_secret = self.client_secret

    def _to_sh_bbox(self, bbox: BBox) -> SHBBox:
        return SHBBox(
            [bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max],
            crs=CRS.WGS84,
        )

    def download_tile(
        self,
        bbox: BBox,
        output_dir: Path = Path("data/acquired"),
        prefix: str = "tile",
        resolution: int = 10,
        max_cloud_cover: float = 20.0,
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
    ) -> dict[str, str]:
        """Download a single Sentinel-2 tile for the given bounding box."""
        sh_bbox = self._to_sh_bbox(bbox)
        width = int(bbox.width_km * 1000 / resolution)
        height = int(bbox.height_km * 1000 / resolution)

        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04"],
                output: { bands: 3, sampleType: "FLOAT32" }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
        """

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(start_date, end_date),
                    maxcc=max_cloud_cover / 100.0,
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            bbox=sh_bbox,
            size=(width, height),
            config=self._config,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"{prefix}_sentinel2_{bbox.center[0]:.4f}_{bbox.center[1]:.4f}.tif"
        )
        output_path = output_dir / fname

        image_data = request.get_data()[0]
        arr = (np.clip(image_data, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).save(str(output_path))

        return {
            "source": "sentinel2",
            "bbox": bbox.to_list(),
            "crs": "EPSG:4326",
            "resolution_m": str(resolution),
            "output_path": str(output_path),
        }
