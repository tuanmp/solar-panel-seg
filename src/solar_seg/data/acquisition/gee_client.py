from __future__ import annotations

from pathlib import Path
from typing import Any

from solar_seg.data.acquisition.geo_utils import BBox


class GEEClient:
    """Google Earth Engine client for satellite/aerial image acquisition.

    Requires Earth Engine authentication. Supports NAIP (1m, US) and
    Sentinel-2 (10m, global) imagery sources.

    Usage:
        client = GEEClient()
        client.export_tile(
            bbox=BBox(37.77, -122.42, 37.78, -122.41),
            source="naip",
            output_dir=Path("data/acquired"),
        )
    """

    SUPPORTED_SOURCES = {"naip", "sentinel2"}

    def __init__(self, project: str | None = None) -> None:
        self.project = project
        self._initialized = False

    def initialize(self) -> None:
        """Authenticate and initialize the Earth Engine API."""
        try:
            import ee
            if self.project:
                ee.Initialize(project=self.project)
            else:
                ee.Initialize()
            self._initialized = True
        except Exception as e:
            raise RuntimeError(
                f"Earth Engine initialization failed. "
                f"Run `earthengine authenticate` first. Error: {e}"
            ) from e

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def _get_collection(self, source: str, bbox: BBox, start_date: str, end_date: str):
        """Get the appropriate image collection filtered by bounds and date."""
        import ee

        collections = {
            "naip": "USDA/NAIP/DOQQ",
            "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
        }
        if source not in collections:
            raise ValueError(
                f"Unsupported source '{source}'. Choose from {self.SUPPORTED_SOURCES}"
            )

        self._ensure_initialized()
        region = ee.Geometry.Rectangle(bbox.to_list())

        collection = (
            ee.ImageCollection(collections[source])
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .sort("CLOUD_COVER")
        )
        return collection, region

    def export_tile(
        self,
        bbox: BBox,
        source: str = "naip",
        output_dir: Path = Path("data/acquired"),
        prefix: str = "tile",
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
        scale: float | None = None,
    ) -> dict[str, Any]:
        """Export a single tile from GEE to a GeoTIFF."""
        import ee
        import urllib.request

        collection, region = self._get_collection(source, bbox, start_date, end_date)
        image = collection.first()
        if image is None:
            raise RuntimeError(f"No images found for {source} at {bbox}")

        scale = scale or {"naip": 1.0, "sentinel2": 10.0}.get(source, 10.0)
        url = image.getDownloadURL(
            {
                "region": region,
                "scale": scale,
                "format": "GeoTIFF",
                "crs": "EPSG:4326",
            }
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = str(image.get("system:time_start").getInfo())
        fname = f"{prefix}_{source}_{bbox.center[0]:.4f}_{bbox.center[1]:.4f}.tif"
        output_path = output_dir / fname

        urllib.request.urlretrieve(url, output_path)

        return {
            "source": source,
            "bbox": bbox.to_list(),
            "crs": "EPSG:4326",
            "scale": scale,
            "timestamp": timestamp,
            "output_path": str(output_path),
        }

    def list_available(
        self,
        bbox: BBox,
        source: str = "naip",
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
    ) -> dict[str, Any]:
        """List available images matching the query without downloading."""
        import ee

        collection, _ = self._get_collection(source, bbox, start_date, end_date)
        count = collection.size().getInfo()
        info = collection.limit(10).getInfo()
        images = []
        for feat in info.get("features", []):
            props = feat.get("properties", {})
            images.append({
                "id": feat.get("id"),
                "date": props.get("system:time_start"),
                "cloud_cover": props.get("CLOUD_COVER", "N/A"),
            })
        return {"count": count, "samples": images}
