# Solar Panel Panoptic Segmentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-stack ML demonstrator for panoptic segmentation of solar PV panels from satellite imagery, covering data acquisition (GEE + Sentinel Hub), dataset preprocessing (BDAPPV + Bradbury), Mask2Former fine-tuning, experiment tracking, and evaluation.

**Architecture:** Modular package `src/solar_seg/` with submodules for acquisition, preprocessing, models, training, and evaluation. Config-driven via Hydra. Experiment tracking via MLflow. Data versioning via DVC.

**Tech Stack:** Python 3.11+, PyTorch 2.2+, Lightning 2.2+, HuggingFace Transformers (Mask2Former), Albumentations, Hydra, MLflow, DVC, Earth Engine API, Sentinel Hub SDK, uv, pytest.

---

### Task 1: Project Scaffolding — Dependencies, Package Setup, Directory Structure

**Files:**
- Modify: `pyproject.toml`
- Create: `src/solar_seg/__init__.py`, `src/solar_seg/data/__init__.py`, `src/solar_seg/data/acquisition/__init__.py`, `src/solar_seg/data/preprocessing/__init__.py`, `src/solar_seg/models/__init__.py`, `src/solar_seg/training/__init__.py`, `src/solar_seg/evaluation/__init__.py`, `src/solar_seg/utils/__init__.py`
- Modify: `Makefile`

- [ ] **Step 1: Update pyproject.toml with new package and dependencies**

```toml
[project]
name = "solar-panel-seg"
version = "0.1.0"
description = "Solar panel panoptic segmentation from satellite/aerial imagery"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "lightning>=2.2,<3.0",
  "torch>=2.2,<3.0",
  "torchvision>=0.17,<1.0",
  "numpy>=1.26,<3.0",
  "pyyaml>=6.0,<7.0",
  "transformers>=4.38,<5.0",
  "albumentations>=1.3,<2.0",
  "hydra-core>=1.3,<2.0",
  "mlflow>=2.10,<3.0",
  "opencv-python>=4.8,<5.0",
  "scikit-image>=0.22,<1.0",
  "matplotlib>=3.7,<4.0",
  "rasterio>=1.3,<2.0",
  "earthengine-api>=1.0,<2.0",
  "sentinelhub>=3.10,<4.0",
]

[dependency-groups]
dev = [
  "pytest>=8.0,<9.0",
  "dvc>=3.0,<4.0",
]

[build-system]
requires = ["hatchling>=1.24.0"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ml_cookbook", "src/solar_seg"]

[tool.pytest.ini_options]
pythonpath = ["src"]
addopts = "-q"
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync --group dev`
Expected: All 17+ packages installed successfully.

- [ ] **Step 3: Create package directory tree**

Run: `mkdir -p src/solar_seg/{data/{acquisition,preprocessing},models,training,evaluation,utils}`

- [ ] **Step 4: Create `__init__.py` files in all new directories**

Create each file with a minimal docstring:

```python
# src/solar_seg/__init__.py
"""Solar panel panoptic segmentation package."""

__version__ = "0.1.0"
```

```python
# src/solar_seg/data/__init__.py
"""Data loading, acquisition, and preprocessing."""
```

```python
# src/solar_seg/data/acquisition/__init__.py
"""Satellite imagery API clients (GEE, Sentinel Hub)."""
```

```python
# src/solar_seg/data/preprocessing/__init__.py
"""Dataset preprocessing: mask conversion, tiling, augmentations."""
```

```python
# src/solar_seg/models/__init__.py
"""Mask2Former model wrappers."""
```

```python
# src/solar_seg/training/__init__.py
"""Training orchestration and callbacks."""
```

```python
# src/solar_seg/evaluation/__init__.py
"""Metrics, visualization, and ablation studies."""
```

```python
# src/solar_seg/utils/__init__.py
"""Reproducibility and shared utilities."""
```

- [ ] **Step 5: Update Makefile with new solar_seg commands**

```makefile
UV := uv

.PHONY: sync train test lint format train_solar

sync:
	$(UV) sync --group dev

train:
	$(UV) run python train.py --config configs/default.yaml

train_solar:
	$(UV) run python -m solar_seg.train

test:
	$(UV) run pytest

lint:
	$(UV) run python -m compileall src tests

format:
	@echo "No formatter configured yet"
```

- [ ] **Step 6: Run lint to verify imports compile**

Run: `make lint`
Expected: No errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml Makefile uv.lock src/solar_seg/
git commit -m "feat: scaffold solar_seg package with dependencies"
```

---

### Task 2: Data Acquisition — Geo Utilities

**Files:**
- Create: `src/solar_seg/data/acquisition/geo_utils.py`
- Test: `tests/test_acquisition/test_geo_utils.py`

- [ ] **Step 1: Write `geo_utils.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Bounding box in WGS84 (lat, lon) decimal degrees."""

    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.lat_min + self.lat_max) / 2.0,
            (self.lon_min + self.lon_max) / 2.0,
        )

    @property
    def width_km(self) -> float:
        """Approximate width in km at the bounding box center latitude."""
        from math import cos, radians
        lat_center = radians(self.center[0])
        return (self.lon_max - self.lon_min) * 111.32 * cos(lat_center)

    @property
    def height_km(self) -> float:
        return (self.lat_max - self.lat_min) * 111.32

    def to_list(self) -> list[float]:
        return [self.lon_min, self.lat_min, self.lon_max, self.lat_max]


def bbox_from_center(
    lat: float, lon: float, size_km: float = 1.0
) -> BBox:
    """Create a square bounding box centered on (lat, lon) with given side length in km."""
    lat_delta = size_km / 111.32 / 2.0
    from math import cos, radians
    lon_delta = size_km / (111.32 * cos(radians(lat))) / 2.0
    return BBox(
        lat_min=lat - lat_delta,
        lon_min=lon - lon_delta,
        lat_max=lat + lat_delta,
        lon_max=lon + lon_delta,
    )


def gsd_to_zoom(gsd: float) -> int:
    """Approximate Google Maps zoom level from ground sampling distance in meters."""
    import math
    return max(1, min(22, int(round(math.log(156543.03 / gsd, 2)))))
```

- [ ] **Step 2: Write tests for `geo_utils.py`**

```python
# tests/test_acquisition/test_geo_utils.py
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
    assert gsd_to_zoom(0.1) == 20
    assert gsd_to_zoom(0.5) == 18
    assert gsd_to_zoom(10.0) == 14


def test_bbox_to_list():
    bbox = BBox(37.0, -122.0, 38.0, -121.0)
    assert bbox.to_list() == [-122.0, 37.0, -121.0, 38.0]
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_acquisition/test_geo_utils.py -v`
Expected: 4 passed.

- [ ] **Step 4: Create test directories**

Run: `mkdir -p tests/test_acquisition`

Move the test file there if needed.

- [ ] **Step 5: Commit**

```bash
git add src/solar_seg/data/acquisition/geo_utils.py tests/test_acquisition/
git commit -m "feat: add geo_utils with BBox and coordinate helpers"
```

---

### Task 3: Data Acquisition — GEE Client

**Files:**
- Create: `src/solar_seg/data/acquisition/gee_client.py`
- Test: `tests/test_acquisition/test_gee_client.py`

- [ ] **Step 1: Write `gee_client.py` (sketch without actual API call)**

```python
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

        self._ensure_initialized()
        region = ee.Geometry.Rectangle(bbox.to_list())

        collections = {
            "naip": "USDA/NAIP/DOQQ",
            "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
        }
        if source not in collections:
            raise ValueError(f"Unsupported source '{source}'. Choose from {self.SUPPORTED_SOURCES}")

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
        """Export a single tile from GEE to a GeoTIFF.

        Returns metadata dict with source, bbox, crs, and output path.
        """
        import ee

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

        import urllib.request
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
    ) -> list[dict[str, Any]]:
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
```

- [ ] **Step 2: Write GEE client test (mock ee to avoid auth)**

```python
# tests/test_acquisition/test_gee_client.py
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
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_acquisition/test_gee_client.py -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add src/solar_seg/data/acquisition/gee_client.py tests/test_acquisition/test_gee_client.py
git commit -m "feat: add GEEClient for NAIP and Sentinel-2 image export"
```

---

### Task 4: Data Acquisition — Sentinel Hub Client

**Files:**
- Create: `src/solar_seg/data/acquisition/sentinel_client.py`
- Test: `tests/test_acquisition/test_sentinel_client.py`

- [ ] **Step 1: Write `sentinel_client.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentinelhub import (
    BBox as SHBBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    SHConfig,
)

from solar_seg.data.acquisition.geo_utils import BBox


@dataclass
class SentinelHubClient:
    """Client for Sentinel Hub OGC/WMS API.

    Requires OAuth2 credentials configured via SentinelHub CLI or env vars.

    Usage:
        client = SentinelHubClient()
        result = client.search(bbox=..., start_date="2023-01-01")
        client.download(result, output_dir=Path("data/acquired"))
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
    ) -> dict[str, Any]:
        """Download a single Sentinel-2 tile for the given bounding box."""
        from datetime import date

        sh_bbox = self._to_sh_bbox(bbox)
        width = sh_bbox.width * 111_320 / resolution
        height = sh_bbox.height * 111_320 / resolution

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
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=sh_bbox,
            size=(int(width), int(height)),
            config=self._config,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{prefix}_sentinel2_{bbox.center[0]:.4f}_{bbox.center[1]:.4f}.tif"
        output_path = output_dir / fname

        image_data = request.get_data()[0]
        from PIL import Image
        import numpy as np
        arr = (np.clip(image_data, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).save(str(output_path))

        return {
            "source": "sentinel2",
            "bbox": bbox.to_list(),
            "crs": "EPSG:4326",
            "resolution_m": resolution,
            "output_path": str(output_path),
        }
```

- [ ] **Step 2: Write test for Sentinel Hub client**

```python
# tests/test_acquisition/test_sentinel_client.py
from pathlib import Path

from solar_seg.data.acquisition.geo_utils import BBox
from solar_seg.data.acquisition.sentinel_client import SentinelHubClient


def test_sentinel_client_init():
    client = SentinelHubClient()
    assert client._config is not None


def test_sentinel_client_bbox_conversion():
    client = SentinelHubClient()
    bbox = BBox(37.77, -122.42, 37.78, -122.41)
    sh_bbox = client._to_sh_bbox(bbox)
    assert sh_bbox.crs.value == "EPSG:4326"
    assert abs(sh_bbox.min_x + 122.42) < 0.001
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_acquisition/test_sentinel_client.py -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add src/solar_seg/data/acquisition/sentinel_client.py tests/test_acquisition/test_sentinel_client.py
git commit -m "feat: add SentinelHubClient for Sentinel-2 image download"
```

---

### Task 5: Data Acquisition — CLI Entry Point

**Files:**
- Create: `src/solar_seg/data/acquisition/cli.py`

- [ ] **Step 1: Write CLI with argparse**

```python
from __future__ import annotations

import argparse
from pathlib import Path

from solar_seg.data.acquisition.geo_utils import BBox, bbox_from_center
from solar_seg.data.acquisition.gee_client import GEEClient
from solar_seg.data.acquisition.sentinel_client import SentinelHubClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire satellite/aerial imagery for solar panel segmentation."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # gee
    gee_p = sub.add_parser("gee", help="Google Earth Engine download")
    gee_p.add_argument("--lat", type=float, required=True)
    gee_p.add_argument("--lon", type=float, required=True)
    gee_p.add_argument("--size-km", type=float, default=1.0)
    gee_p.add_argument("--source", choices=["naip", "sentinel2"], default="naip")
    gee_p.add_argument("--output", type=Path, default=Path("data/acquired"))
    gee_p.add_argument("--prefix", default="tile")
    gee_p.add_argument("--start-date", default="2020-01-01")
    gee_p.add_argument("--end-date", default="2024-01-01")
    gee_p.add_argument("--project", type=str, default=None)
    gee_p.add_argument("--list-only", action="store_true", help="List available images without downloading")

    # sentinel
    sh_p = sub.add_parser("sentinel", help="Sentinel Hub download")
    sh_p.add_argument("--lat", type=float, required=True)
    sh_p.add_argument("--lon", type=float, required=True)
    sh_p.add_argument("--size-km", type=float, default=1.0)
    sh_p.add_argument("--output", type=Path, default=Path("data/acquired"))
    sh_p.add_argument("--prefix", default="tile")
    sh_p.add_argument("--start-date", default="2023-01-01")
    sh_p.add_argument("--end-date", default="2024-01-01")
    sh_p.add_argument("--resolution", type=int, default=10)

    args = parser.parse_args()
    bbox = bbox_from_center(args.lat, args.lon, args.size_km)

    if args.command == "gee":
        client = GEEClient(project=args.project)
        try:
            client.initialize()
        except RuntimeError as e:
            print(f"Error: {e}")
            return

        if args.list_only:
            result = client.list_available(bbox, args.source, args.start_date, args.end_date)
            print(f"Available images: {result['count']}")
            for img in result["samples"]:
                print(img)
        else:
            result = client.export_tile(
                bbox=bbox,
                source=args.source,
                output_dir=args.output,
                prefix=args.prefix,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            print(f"Downloaded: {result['output_path']}")

    elif args.command == "sentinel":
        client = SentinelHubClient()
        result = client.download_tile(
            bbox=bbox,
            output_dir=args.output,
            prefix=args.prefix,
            start_date=args.start_date,
            end_date=args.end_date,
            resolution=args.resolution,
        )
        print(f"Downloaded: {result['output_path']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test CLI help works**

Run: `uv run python -m solar_seg.data.acquisition.cli --help`
Expected: Shows help with gee and sentinel subcommands.

- [ ] **Step 3: Commit**

```bash
git add src/solar_seg/data/acquisition/cli.py
git commit -m "feat: add CLI entry point for GEE and Sentinel Hub acquisition"
```

---

### Task 6: Preprocessing — Mask Converter (Polygon → Binary/Instance Rasters)

**Files:**
- Create: `src/solar_seg/data/preprocessing/mask_converter.py`
- Test: `tests/test_preprocessing/test_mask_converter.py`

- [ ] **Step 1: Write `mask_converter.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

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

    Returns:
        Instance label map of shape image_shape.
    """
    label_map = np.zeros(image_shape, dtype=np.int32)
    for idx, polygon in enumerate(polygons, start=1):
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(label_map, [pts], idx)
    return label_map


def mask_to_polygons(mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    """Convert a binary mask to list of polygon vertex arrays.

    Uses OpenCV contours. Filters out polygons below min_area (in pixels).

    Returns:
        List of polygons, each as an (N, 2) numpy array of (x, y) vertices.
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
    """Convert a BDAPPV-style binary mask .png to semantic + instance label maps.

    BDAPPV masks are white (panel) on black (background) .png files.
    Connected components in the mask become instance IDs.

    Returns dict with 'semantic' and 'instance' output paths.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    binary = (mask > 0).astype(np.uint8)

    semantic_path = output_dir / f"{mask_path.stem}_semantic.png"
    cv2.imwrite(str(semantic_path), binary * 255)

    labeled = measure.label(binary, connectivity=2)
    instance_path = output_dir / f"{mask_path.stem}_instance.png"
    cv2.imwrite(str(instance_path), labeled.astype(np.int32))

    return {"semantic": semantic_path, "instance": instance_path}
```

- [ ] **Step 2: Write tests**

```python
# tests/test_preprocessing/test_mask_converter.py
import numpy as np
from pathlib import Path

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
```

- [ ] **Step 3: Create test directories**

Run: `mkdir -p tests/test_preprocessing`

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_preprocessing/test_mask_converter.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/solar_seg/data/preprocessing/mask_converter.py tests/test_preprocessing/test_mask_converter.py
git commit -m "feat: add mask_converter for polygon-to-raster conversion"
```

---

### Task 7: Preprocessing — Tile Extractor (Large Image → Tiles)

**Files:**
- Create: `src/solar_seg/data/preprocessing/tile_extractor.py`
- Test: `tests/test_preprocessing/test_tile_extractor.py`

- [ ] **Step 1: Write `tile_extractor.py`**

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_tile_around_centroid(
    image_path: Path,
    centroid_x: int,
    centroid_y: int,
    tile_size: int = 400,
    output_dir: Path = Path("data/processed/tiles"),
    prefix: str = "tile",
) -> dict[str, Path | None]:
    """Extract a fixed-size tile from a large image centered on a centroid.

    Pads with zeros if the tile extends beyond image boundaries.

    Returns dict with 'image' path and optionally 'mask' path.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    half = tile_size // 2

    x_start = centroid_x - half
    y_start = centroid_y - half
    x_end = x_start + tile_size
    y_end = y_start + tile_size

    tile = np.zeros((tile_size, tile_size, 3), dtype=img.dtype)
    src_x_start = max(0, x_start)
    src_y_start = max(0, y_start)
    src_x_end = min(w, x_end)
    src_y_end = min(h, y_end)

    dst_x_start = max(0, -x_start)
    dst_y_start = max(0, -y_start)

    tile[
        dst_y_start : dst_y_start + (src_y_end - src_y_start),
        dst_x_start : dst_x_start + (src_x_end - src_x_start),
    ] = img[src_y_start:src_y_end, src_x_start:src_x_end]

    output_dir.mkdir(parents=True, exist_ok=True)
    tile_path = output_dir / f"{prefix}_x{centroid_x}_y{centroid_y}.png"
    cv2.imwrite(str(tile_path), tile)

    return {"image": tile_path}


def filter_tile_by_panel_fraction(
    mask_path: Path,
    min_fraction: float = 0.05,
) -> bool:
    """Return True if the mask has at least min_fraction panel pixels."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    return float(mask.sum()) / mask.size >= min_fraction


def batch_extract_centroids(
    image_path: Path,
    centroids: list[tuple[int, int]],
    output_dir: Path,
    tile_size: int = 400,
    prefix: str = "tile",
) -> list[dict[str, Path | None]]:
    """Extract tiles for a list of centroids from a single large image."""
    results = []
    for idx, (cx, cy) in enumerate(centroids):
        result = extract_tile_around_centroid(
            image_path, cx, cy, tile_size, output_dir, f"{prefix}_{idx}"
        )
        results.append(result)
    return results
```

- [ ] **Step 2: Write tests**

```python
# tests/test_preprocessing/test_tile_extractor.py
import numpy as np
import cv2
from pathlib import Path

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
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_preprocessing/test_tile_extractor.py -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add src/solar_seg/data/preprocessing/tile_extractor.py tests/test_preprocessing/test_tile_extractor.py
git commit -m "feat: add tile_extractor for large-image-to-tile cropping"
```

---

### Task 8: Preprocessing — Augmentations

**Files:**
- Create: `src/solar_seg/data/preprocessing/augmentations.py`
- Test: `tests/test_preprocessing/test_augmentations.py`

- [ ] **Step 1: Write `augmentations.py`**

```python
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def training_transforms(image_size: int = 384) -> A.Compose:
    """Training augmentation pipeline.

    Args:
        image_size: Output spatial size (square crop).

    Returns:
        Albumentations Compose that operates on image, mask, and instance_mask.
    """
    return A.Compose(
        [
            A.RandomCrop(image_size, image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.8,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={
            "mask": "mask",
            "instance_mask": "mask",
        },
    )


def validation_transforms() -> A.Compose:
    """Validation/test augmentation pipeline (no random ops)."""
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={
            "mask": "mask",
            "instance_mask": "mask",
        },
    )
```

- [ ] **Step 2: Write tests**

```python
# tests/test_preprocessing/test_augmentations.py
import numpy as np
import torch

from solar_seg.data.preprocessing.augmentations import (
    training_transforms,
    validation_transforms,
)


def test_training_transforms_output():
    aug = training_transforms(image_size=384)
    image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (400, 400), dtype=np.uint8)
    instance_mask = np.random.randint(0, 5, (400, 400), dtype=np.int32)

    result = aug(image=image, mask=mask, instance_mask=instance_mask)
    assert isinstance(result["image"], torch.Tensor)
    assert result["image"].shape == (3, 384, 384)
    assert isinstance(result["mask"], torch.Tensor)
    assert result["mask"].shape == (384, 384)
    assert isinstance(result["instance_mask"], torch.Tensor)


def test_validation_transforms_output():
    aug = validation_transforms()
    image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (400, 400), dtype=np.uint8)

    result = aug(image=image, mask=mask)
    assert result["image"].shape == (3, 400, 400)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_preprocessing/test_augmentations.py -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add src/solar_seg/data/preprocessing/augmentations.py tests/test_preprocessing/test_augmentations.py
git commit -m "feat: add augmentation pipelines (train/val) with Albumentations"
```

---

### Task 9: Preprocessing — Dataset + LightningDataModule

**Files:**
- Create: `src/solar_seg/data/preprocessing/dataset.py`
- Test: `tests/test_preprocessing/test_dataset.py`

- [ ] **Step 1: Write `SolarSegDataset`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SolarSegDataset(Dataset):
    """Dataset for solar panel panoptic segmentation.

    Reads preprocessed images, semantic masks, and instance masks from disk.
    Supports both BDAPPV and Bradbury datasets in the same directory layout.

    Expected structure:
        data/processed/
            images/        {id}.png
            semantic_masks/{id}_semantic.png
            instance_masks/{id}_instance.png
    """

    def __init__(
        self,
        image_dir: Path,
        semantic_mask_dir: Path,
        instance_mask_dir: Path | None = None,
        transform: Callable | None = None,
        num_classes: int = 2,
    ) -> None:
        self.image_dir = image_dir
        self.semantic_mask_dir = semantic_mask_dir
        self.instance_mask_dir = instance_mask_dir
        self.transform = transform
        self.num_classes = num_classes

        self.image_paths = sorted(image_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No .png images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        stem = img_path.stem
        sem_path = self.semantic_mask_dir / f"{stem}_semantic.png"
        inst_path = (
            self.instance_mask_dir / f"{stem}_instance.png"
            if self.instance_mask_dir
            else None
        )

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sem_mask = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
        sem_mask = (sem_mask > 0).astype(np.uint8)

        inst_mask = None
        if inst_path and inst_path.exists():
            inst_mask = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
            inst_mask = np.ascontiguousarray(inst_mask)

        if self.transform:
            kwargs = {"image": image, "mask": sem_mask}
            if inst_mask is not None:
                kwargs["instance_mask"] = inst_mask
            transformed = self.transform(**kwargs)
            image = transformed["image"]
            sem_mask = transformed["mask"]
            inst_mask = transformed.get("instance_mask")
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            sem_mask = torch.from_numpy(sem_mask).long()

        result = {
            "pixel_values": image,
            "semantic_mask": sem_mask.long() if isinstance(sem_mask, torch.Tensor) else torch.from_numpy(sem_mask).long(),
        }

        if isinstance(inst_mask, torch.Tensor):
            result["instance_mask"] = inst_mask.long()
        elif inst_mask is not None:
            result["instance_mask"] = torch.from_numpy(inst_mask).long()

        result["id"] = stem
        return result
```

- [ ] **Step 2: Write `SolarSegDataModule`**

```python
class SolarSegDataModule(L.LightningDataModule):
    """LightningDataModule for solar panel segmentation.

    Args:
        data_root: Root directory containing 'images/', 'semantic_masks/', 'instance_masks/'.
        batch_size: Batch size per GPU.
        num_workers: DataLoader workers.
        train_transform: Callable for training augmentation.
        val_transform: Callable for validation augmentation.
        val_split: Fraction of training data to use for validation.
    """

    def __init__(
        self,
        data_root: Path,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        val_split: float = 0.15,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.val_split = val_split

        self.image_dir = self.data_root / "images"
        self.semantic_dir = self.data_root / "semantic_masks"
        self.instance_dir = self.data_root / "instance_masks"

        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        all_paths = sorted(self.image_dir.glob("*.png"))
        if not all_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        n = len(all_paths)
        n_val = int(n * self.val_split)
        n_train = n - n_val

        train_paths = all_paths[:n_train]
        val_paths = all_paths[n_train:]

        if stage in (None, "fit"):
            self.train_ds = self._make_dataset(train_paths, transform=self.train_transform)
            self.val_ds = self._make_dataset(val_paths, transform=self.val_transform)
        if stage in (None, "test"):
            self.test_ds = self._make_dataset(val_paths, transform=self.val_transform)

    def _make_dataset(
        self, paths: list[Path], transform: Callable | None
    ) -> SolarSegDataset:
        return SolarSegDataset(
            image_dir=self.image_dir,
            semantic_mask_dir=self.semantic_dir,
            instance_mask_dir=self.instance_dir,
            transform=transform,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
```

- [ ] **Step 3: Write tests**

```python
# tests/test_preprocessing/test_dataset.py
import numpy as np
import cv2
from pathlib import Path

from solar_seg.data.preprocessing.dataset import SolarSegDataset, SolarSegDataModule


def _create_dummy_data(root: Path):
    img_dir = root / "images"
    sem_dir = root / "semantic_masks"
    inst_dir = root / "instance_masks"
    for d in [img_dir, sem_dir, inst_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{i}.png"), img)

        sem = np.random.randint(0, 2, (400, 400), dtype=np.uint8) * 255
        cv2.imwrite(str(sem_dir / f"{i}_semantic.png"), sem)

        inst = np.random.randint(0, 3, (400, 400), dtype=np.int32)
        cv2.imwrite(str(inst_dir / f"{i}_instance.png"), inst)


def test_dataset_len(tmp_path):
    _create_dummy_data(tmp_path)
    ds = SolarSegDataset(
        image_dir=tmp_path / "images",
        semantic_mask_dir=tmp_path / "semantic_masks",
        instance_mask_dir=tmp_path / "instance_masks",
    )
    assert len(ds) == 5


def test_dataset_getitem(tmp_path):
    _create_dummy_data(tmp_path)
    ds = SolarSegDataset(
        image_dir=tmp_path / "images",
        semantic_mask_dir=tmp_path / "semantic_masks",
        instance_mask_dir=tmp_path / "instance_masks",
    )
    item = ds[0]
    assert "pixel_values" in item
    assert "semantic_mask" in item
    assert "instance_mask" in item
    assert item["pixel_values"].shape[0] == 3


def test_datamodule_setup(tmp_path):
    _create_dummy_data(tmp_path)
    dm = SolarSegDataModule(data_root=tmp_path, batch_size=2, val_split=0.2)
    dm.setup("fit")
    assert dm.train_ds is not None
    assert dm.val_ds is not None
    assert len(dm.train_ds) >= 3
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_preprocessing/test_dataset.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/solar_seg/data/preprocessing/dataset.py tests/test_preprocessing/test_dataset.py
git commit -m "feat: add SolarSegDataset and SolarSegDataModule"
```

---

### Task 10: Model — Mask2Former LightningModule

**Files:**
- Create: `src/solar_seg/models/mask2former_module.py`
- Test: `tests/test_models/test_mask2former_shapes.py`

- [ ] **Step 1: Write `mask2former_module.py`**

```python
from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch import nn
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)


class Mask2FormerModule(L.LightningModule):
    """LightningModule wrapping HuggingFace Mask2Former for panoptic segmentation.

    Supports panoptic, semantic, and instance segmentation modes.

    Args:
        model_name: HuggingFace model identifier (e.g., "facebook/mask2former-swin-base-coco-panoptic").
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        num_labels: Number of semantic classes (including background).
    """

    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-base-coco-panoptic",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(pixel_values=pixel_values)

    def _compute_loss(
        self, model_outputs: dict[str, Any], semantic_mask: torch.Tensor, instance_mask: torch.Tensor | None
    ) -> torch.Tensor:
        loss = model_outputs.loss

        panoptic_map = semantic_mask
        if instance_mask is not None:
            panoptic_map = panoptic_map + instance_mask * self.hparams.num_labels

        return loss + self.loss_fn(
            model_outputs.class_queries_logits.view(-1, self.hparams.num_labels),
            panoptic_map.view(-1),
        )

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self(pixel_values=batch["pixel_values"])
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self(pixel_values=batch["pixel_values"])
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        pred_masks = outputs.masks_queries_logits
        pred_classes = outputs.class_queries_logits
        return {"val_loss": loss}

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self(pixel_values=batch["pixel_values"])
        loss = outputs.loss
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(current_step: int) -> float:
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            return (1.0 - current_step / self.trainer.estimated_stepping_batches) ** 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
```

- [ ] **Step 2: Write shape test (CPU, small input)**

```python
# tests/test_models/test_mask2former_shapes.py
import torch
import pytest
from transformers import Mask2FormerForUniversalSegmentation


@pytest.fixture(scope="module")
def model():
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic",
        num_labels=2,
        ignore_mismatched_sizes=True,
    )


def test_mask2former_forward_shape(model):
    batch = 2
    pixel_values = torch.randn(batch, 3, 384, 384)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    assert outputs.masks_queries_logits is not None
    assert outputs.class_queries_logits is not None
    assert outputs.class_queries_logits.shape[0] == batch
    assert outputs.class_queries_logits.shape[2] == 2


def test_mask2former_config(model):
    assert model.config.num_labels == 2
```

- [ ] **Step 3: Create test directory**

Run: `mkdir -p tests/test_models`

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_models/test_mask2former_shapes.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/solar_seg/models/mask2former_module.py tests/test_models/test_mask2former_shapes.py
git commit -m "feat: add Mask2Former LightningModule"
```

---

### Task 11: Training — Hydra Configs

**Files:**
- Create: `configs/data/bdappv.yaml`, `configs/data/bradbury.yaml`, `configs/model/mask2former_swin_b.yaml`, `configs/trainer/base.yaml`, `configs/experiment/bdappv_only.yaml`, `configs/experiment/combined.yaml`

- [ ] **Step 1: Create Hydra config structure**

Run: `mkdir -p configs/{data,model,trainer,experiment}`

- [ ] **Step 2: Write `configs/data/bdappv.yaml`**

```yaml
# @package _group_
data_root: data/processed/bdappv
batch_size: 8
num_workers: 4
val_split: 0.15
```

- [ ] **Step 3: Write `configs/data/bradbury.yaml`**

```yaml
# @package _group_
data_root: data/processed/bradbury
batch_size: 8
num_workers: 4
val_split: 0.15
```

- [ ] **Step 4: Write `configs/model/mask2former_swin_b.yaml`**

```yaml
# @package _group_
model_name: facebook/mask2former-swin-base-coco-panoptic
learning_rate: 1.0e-4
weight_decay: 0.05
warmup_steps: 1000
num_labels: 2
```

- [ ] **Step 5: Write `configs/trainer/base.yaml`**

```yaml
# @package _group_
max_epochs: 50
accelerator: auto
devices: 1
deterministic: true
precision: bf16-mixed
log_every_n_steps: 10
enable_progress_bar: true
limit_train_batches: 1.0
limit_val_batches: 1.0
```

- [ ] **Step 6: Write `configs/experiment/bdappv_only.yaml`**

```yaml
defaults:
  - data: bdappv
  - model: mask2former_swin_b
  - trainer: base
  - _self_

seed: 42
experiment_name: bdappv_only
mlflow_tracking_uri: mlruns
```

- [ ] **Step 7: Write `configs/experiment/combined.yaml`**

```yaml
defaults:
  - data: bdappv
  - model: mask2former_swin_b
  - trainer: base
  - _self_

seed: 42
experiment_name: combined_bdappv_bradbury
mlflow_tracking_uri: mlruns
```

- [ ] **Step 8: Commit**

```bash
git add configs/
git commit -m "feat: add Hydra configs for data, model, trainer, experiments"
```

---

### Task 12: Training — Entry Point with Hydra + MLflow

**Files:**
- Create: `src/solar_seg/train.py`

- [ ] **Step 1: Write training entry point**

```python
from __future__ import annotations

import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger
import lightning as L

from solar_seg.data.preprocessing.dataset import SolarSegDataModule
from solar_seg.data.preprocessing.augmentations import (
    training_transforms,
    validation_transforms,
)
from solar_seg.models.mask2former_module import Mask2FormerModule
from solar_seg.utils.repro import seed_everything


@hydra.main(config_path="../configs", config_name="experiment/bdappv_only", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(seed=cfg.seed, deterministic=cfg.trainer.deterministic)

    ds_cfg = cfg.data
    model_cfg = cfg.model

    datamodule = SolarSegDataModule(
        data_root=ds_cfg.data_root,
        batch_size=ds_cfg.batch_size,
        num_workers=ds_cfg.num_workers,
        train_transform=training_transforms(),
        val_transform=validation_transforms(),
        val_split=ds_cfg.val_split,
    )

    model = Mask2FormerModule(
        model_name=model_cfg.model_name,
        learning_rate=model_cfg.learning_rate,
        weight_decay=model_cfg.weight_decay,
        warmup_steps=model_cfg.warmup_steps,
        num_labels=model_cfg.num_labels,
    )

    logger = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        tracking_uri=cfg.mlflow_tracking_uri,
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best",
        ),
        EarlyStopping(monitor="val_loss", mode="min", patience=5),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        deterministic=cfg.trainer.deterministic,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update root `train.py` entry point (optional routing)**

Modify `train.py` to route to the solar seg training:

```python
from solar_seg.train import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Makefile test**

Run: `uv run python -c "from solar_seg.train import main; print('import OK')"`
Expected: "import OK"

- [ ] **Step 4: Commit**

```bash
git add src/solar_seg/train.py train.py
git commit -m "feat: add Hydra-driven training entry point with MLflow logging"
```

---

### Task 13: Evaluation — Panoptic Quality Metrics

**Files:**
- Create: `src/solar_seg/evaluation/metrics.py`
- Test: `tests/test_evaluation/test_metrics.py`

- [ ] **Step 1: Write `metrics.py`**

```python
from __future__ import annotations

import torch
import numpy as np


def panoptic_quality(
    pred_semantic: torch.Tensor,
    pred_instance: torch.Tensor,
    target_semantic: torch.Tensor,
    target_instance: torch.Tensor,
    num_classes: int = 2,
) -> dict[str, float]:
    """Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).

    Simplified implementation following the PQ definition from
    Kirillov et al. "Panoptic Segmentation" (CVPR 2019).

    Args:
        pred_semantic: Predicted semantic labels (H, W).
        pred_instance: Predicted instance IDs (H, W).
        target_semantic: Ground truth semantic labels (H, W).
        target_instance: Ground truth instance IDs (H, W).

    Returns:
        dict with keys 'pq', 'sq', 'rq'.
    """
    pred_semantic = pred_semantic.cpu().numpy()
    pred_instance = pred_instance.cpu().numpy()
    target_semantic = target_semantic.cpu().numpy()
    target_instance = target_instance.cpu().numpy()

    pred_ids = pred_instance.astype(np.int32)
    target_ids = target_instance.astype(np.int32)

    pq_sum = 0.0
    sq_sum = 0.0
    rq_sum = 0.0
    matched_pairs = 0

    for class_id in range(num_classes):
        pred_class_mask = pred_semantic == class_id
        target_class_mask = target_semantic == class_id

        pred_class_ids = np.unique(pred_ids[pred_class_mask])
        target_class_ids = np.unique(target_ids[target_class_mask])

        pred_class_ids = pred_class_ids[pred_class_ids > 0]
        target_class_ids = target_class_ids[target_class_ids > 0]

        ious = np.zeros((len(pred_class_ids), len(target_class_ids)))
        for i, pid in enumerate(pred_class_ids):
            p_mask = pred_ids == pid
            for j, tid in enumerate(target_class_ids):
                t_mask = target_ids == tid
                intersection = np.logical_and(p_mask, t_mask).sum()
                union = np.logical_or(p_mask, t_mask).sum()
                ious[i, j] = intersection / max(union, 1)

        matched_pred = set()
        matched_target = set()
        for _ in range(min(len(pred_class_ids), len(target_class_ids))):
            if ious.size == 0:
                break
            idx = ious.argmax()
            i, j = np.unravel_index(idx, ious.shape)
            if ious[i, j] < 0.5:
                break
            iou = ious[i, j]
            pq_sum += iou
            sq_sum += iou
            rq_sum += 1
            matched_pairs += 1
            matched_pred.add(i)
            matched_target.add(j)
            ious[i, :] = -1
            ious[:, j] = -1

    if matched_pairs == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0}

    sq = sq_sum / matched_pairs
    rq = rq_sum / max(
        (len(pred_class_ids) + len(target_class_ids)) / 2 + (matched_pairs - len(target_class_ids) - len(pred_class_ids)) / 2, 1
    )
    pq = pq_sum / matched_pairs

    return {"pq": float(pq), "sq": float(sq), "rq": float(rq)}


def mean_iou(
    pred_semantic: torch.Tensor,
    target_semantic: torch.Tensor,
    num_classes: int = 2,
    smooth: float = 1e-6,
) -> float:
    """Compute mean Intersection over Union for semantic segmentation."""
    pred = pred_semantic.cpu().numpy().flatten()
    target = target_semantic.cpu().numpy().flatten()

    ious = []
    for c in range(num_classes):
        intersection = np.logical_and(pred == c, target == c).sum()
        union = np.logical_or(pred == c, target == c).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return float(np.mean(ious))
```

- [ ] **Step 2: Write tests**

```python
# tests/test_evaluation/test_metrics.py
import pytest
import torch
from solar_seg.evaluation.metrics import panoptic_quality, mean_iou


def test_panoptic_quality_perfect():
    sem = torch.zeros((100, 100), dtype=torch.long)
    inst = torch.zeros((100, 100), dtype=torch.long)
    inst[10:40, 10:40] = 1
    sem[10:40, 10:40] = 1

    result = panoptic_quality(sem, inst, sem, inst, num_classes=2)
    assert result["pq"] == pytest.approx(1.0, abs=0.01)
    assert result["sq"] == pytest.approx(1.0, abs=0.01)
    assert result["rq"] == pytest.approx(1.0, abs=0.01)


def test_panoptic_quality_no_match():
    sem1 = torch.zeros((100, 100), dtype=torch.long)
    inst1 = torch.zeros((100, 100), dtype=torch.long)
    sem2 = torch.ones((100, 100), dtype=torch.long)
    inst2 = torch.ones((100, 100), dtype=torch.long)

    result = panoptic_quality(sem1, inst1, sem2, inst2, num_classes=2)
    assert result["pq"] == 0.0


def test_mean_iou_perfect():
    pred = torch.zeros((100, 100), dtype=torch.long)
    target = torch.zeros((100, 100), dtype=torch.long)
    result = mean_iou(pred, target, num_classes=2)
    assert result == pytest.approx(1.0, abs=0.01)


def test_mean_iou_half():
    pred = torch.zeros((100, 100), dtype=torch.long)
    target = torch.zeros((100, 100), dtype=torch.long)
    pred[:50] = 1
    target[:50] = 0
    target[50:] = 1
    result = mean_iou(pred, target, num_classes=2)
    assert 0.0 < result < 1.0
```

- [ ] **Step 3: Create test directories**

Run: `mkdir -p tests/test_evaluation`

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_evaluation/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/solar_seg/evaluation/metrics.py tests/test_evaluation/test_metrics.py
git commit -m "feat: add panoptic quality metrics (PQ, SQ, RQ, mIoU)"
```

---

### Task 14: Evaluation — Visualization

**Files:**
- Create: `src/solar_seg/evaluation/visualization.py`

- [ ] **Step 1: Write `visualization.py`**

```python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def overlay_mask(
    image: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a binary mask on an image with transparency.

    Args:
        image: RGB image (H, W, 3) uint8 or float [0,1].
        mask: Binary mask (H, W) uint8 or float.
        alpha: Mask transparency.

    Returns:
        Overlay image as uint8 numpy array (H, W, 3).
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    mask_bool = mask > 0.5 if mask.dtype in (np.float32, np.float64) else mask > 0
    overlay = image.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + color * alpha
    ).astype(np.uint8)
    return overlay


def plot_predictions(
    image: np.ndarray | torch.Tensor,
    pred_semantic: np.ndarray | torch.Tensor,
    target_semantic: np.ndarray | torch.Tensor,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """Side-by-side comparison: image, ground truth overlay, prediction overlay."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(pred_semantic, torch.Tensor):
        pred_semantic = pred_semantic.cpu().numpy()
    if isinstance(target_semantic, torch.Tensor):
        target_semantic = target_semantic.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(image if image.ndim == 3 else image.transpose(1, 2, 0))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    gt_overlay = overlay_mask(image, target_semantic)
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    pred_overlay = overlay_mask(image, pred_semantic)
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add src/solar_seg/evaluation/visualization.py
git commit -m "feat: add visualization utilities for mask overlay and result comparison"
```

---

### Task 15: Tests — Smoke Test Update

**Files:**
- Modify: `tests/test_train_smoke.py`

- [ ] **Step 1: Update `test_train_smoke.py`**

```python
from pathlib import Path
import torch
import pytest
from lightning import Trainer
from torch.utils.data import DataLoader
from huggingface_hub import scan_cache_dir

from solar_seg.models.mask2former_module import Mask2FormerModule
from solar_seg.data.preprocessing.dataset import SolarSegDataset


class DummySolarDataset(SolarSegDataset):
    """Override with random data to avoid real filesystem dependencies."""

    def __init__(self, num_samples: int = 8, img_size: int = 128) -> None:
        self.num_samples = num_samples
        self.img_size = img_size
        self.image_paths = list(range(num_samples))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "pixel_values": torch.randn(3, self.img_size, self.img_size),
            "semantic_mask": torch.randint(0, 2, (self.img_size, self.img_size)),
            "id": str(idx),
        }


def _model_available() -> bool:
    """Check if HuggingFace model can be loaded (requires internet)."""
    try:
        import transformers
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="HuggingFace model download requires internet")
def test_solar_seg_smoke():
    model = Mask2FormerModule(
        model_name="facebook/mask2former-swin-tiny-coco-panoptic",
        num_labels=2,
    )
    ds = DummySolarDataset(num_samples=4, img_size=128)
    dl = DataLoader(ds, batch_size=2)

    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=dl, val_dataloaders=dl)
    assert True
```

- [ ] **Step 2: Update `tests/test_shapes.py`**

```python
from solar_seg.data.preprocessing.augmentations import training_transforms, validation_transforms


def test_augmentation_shapes():
    import numpy as np
    image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (400, 400), dtype=np.uint8)

    train_aug = training_transforms(384)
    result = train_aug(image=image, mask=mask)
    assert result["image"].shape == (3, 384, 384)
    assert result["mask"].shape == (384, 384)

    val_aug = validation_transforms()
    result = val_aug(image=image, mask=mask)
    assert result["image"].shape == (3, 400, 400)
```

- [ ] **Step 3: Run smoke test**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_train_smoke.py tests/test_shapes.py
git commit -m "test: update smoke test for solar seg and test augmentation shapes"
```

---

### Task 16: DVC Pipeline Setup

**Files:**
- Create: `dvc.yaml`, `.dvc/config`

- [ ] **Step 1: Initialize DVC**

Run: `uv run dvc init`

- [ ] **Step 2: Write `dvc.yaml`**

```yaml
stages:
  download_bdappv:
    desc: Download BDAPPV dataset from Zenodo (8.17 GB training data)
    cmd: python -c "
      import urllib.request, zipfile, pathlib, os;
      url = 'https://zenodo.org/api/records/7358126/files/bdappv.zip/content';
      path = pathlib.Path('data/raw/bdappv');
      path.mkdir(parents=True, exist_ok=True);
      zip_path = path / 'bdappv.zip';
      print(f'Downloading BDAPPV from {url}...');
      urllib.request.urlretrieve(url, zip_path);
      with zipfile.ZipFile(zip_path, 'r') as zf: zf.extractall(path);
      os.remove(zip_path);
    "
    outs:
      - data/raw/bdappv

  download_bradbury_annotations:
    desc: Download Bradbury annotation data from Figshare
    cmd: python -c "
      import urllib.request, pathlib;
      path = pathlib.Path('data/raw/bradbury');
      path.mkdir(parents=True, exist_ok=True);
      files = [
        ('https://ndownloader.figshare.com/files/24115682', 'polygonDataExceptVertices.csv'),
        ('https://ndownloader.figshare.com/files/24115685', 'polygonVertices_LatitudeLongitude.csv'),
        ('https://ndownloader.figshare.com/files/24115688', 'polygonVertices_PixelCoordinates.csv'),
        ('https://ndownloader.figshare.com/files/24115691', 'SolarArrayPolygons.geojson'),
        ('https://ndownloader.figshare.com/files/24115694', 'SolarArrayPolygons.json'),
      ];
      for url, fname in files:
          out = path / fname;
          if not out.exists():
              print(f'Downloading {fname}...');
              urllib.request.urlretrieve(url, out);
    "
    outs:
      - data/raw/bradbury

  preprocess:
    desc: Convert raw datasets to processed format
    cmd: python -c "
      from solar_seg.data.preprocessing.mask_converter import bdappv_mask_to_labelmaps;
      from pathlib import Path;
      import glob;
      for mask_file in Path('data/raw/bdappv').rglob('*.png'):
          if 'mask' in mask_file.name or 'label' in mask_file.name:
              bdappv_mask_to_labelmaps(mask_file, Path('data/processed/bdappv/semantic_masks'));
    "
    deps:
      - data/raw/bdappv
      - data/raw/bradbury
      - src/solar_seg/data/preprocessing/mask_converter.py
      - src/solar_seg/data/preprocessing/tile_extractor.py
    outs:
      - data/processed
```

- [ ] **Step 3: Add `.dvc/` and `dvc.yaml` files**

```bash
git add dvc.yaml .dvc/config
git commit -m "feat: add DVC pipeline for data download and preprocessing"
```

---

### Task 17: Evaluation — Ablation Study Runner

**Files:**
- Create: `src/solar_seg/evaluation/ablations.py`

- [ ] **Step 1: Write `ablations.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
import lightning as L

from solar_seg.data.preprocessing.dataset import SolarSegDataModule
from solar_seg.data.preprocessing.augmentations import (
    training_transforms,
    validation_transforms,
)
from solar_seg.models.mask2former_module import Mask2FormerModule
from solar_seg.evaluation.metrics import panoptic_quality, mean_iou


def run_ablation(
    config_path: str,
    overrides: list[str],
    output_dir: Path = Path("results/ablations"),
) -> dict:
    """Run a single ablation experiment and return metrics."""
    with hydra.initialize_config_dir(config_dir=str(config_path.parent)):
        cfg = hydra.compose(
            config_name=config_path.name,
            overrides=overrides,
        )

    datamodule = SolarSegDataModule(
        data_root=cfg.data.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_transform=training_transforms(),
        val_transform=validation_transforms(),
        val_split=cfg.data.val_split,
    )

    model = Mask2FormerModule(
        model_name=cfg.model.model_name,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        num_labels=cfg.model.num_labels,
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        deterministic=cfg.trainer.deterministic,
        precision=cfg.trainer.precision,
        enable_progress_bar=False,
        logger=False,
    )

    trainer.fit(model=model, datamodule=datamodule)
    results = trainer.test(model=model, datamodule=datamodule)

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = "_".join(overrides).replace("/", "-").replace("=", "_")
    report = {
        "config": cfg.experiment_name,
        "overrides": overrides,
        "test_metrics": results,
    }
    (output_dir / f"{fname}.json").write_text(json.dumps(report, indent=2))

    return report
```

- [ ] **Step 2: Commit**

```bash
git add src/solar_seg/evaluation/ablations.py
git commit -m "feat: add ablation study runner for systematic experiments"
```

---

### Task 18: Move `repro.py` to solar_seg/utils

**Files:**
- Copy: `src/ml_cookbook/utils/repro.py` → `src/solar_seg/utils/repro.py`

- [ ] **Step 1: Copy repro.py**

```bash
cp src/ml_cookbook/utils/repro.py src/solar_seg/utils/repro.py
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from solar_seg.utils.repro import seed_everything; seed_everything(42)"`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/solar_seg/utils/repro.py
git commit -m "chore: copy repro utils to solar_seg package"
```

---

### Task 19: Notebooks — EDA Template

**Files:**
- Create: `notebooks/01_eda.ipynb`
- Create: `notebooks/02_results.ipynb`

- [ ] **Step 1: Create EDA notebook skeleton**

Write a minimal Jupyter notebook JSON:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# EDA: Solar Panel Segmentation Datasets"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "from pathlib import Path\n",
    "import matplotlib.pyplot as plt\n",
    "import cv2\n",
    "import numpy as np\n",
    "\n",
    "data_root = Path('data/processed')\n",
    "image_dir = data_root / 'images'\n",
    "sem_dir = data_root / 'semantic_masks'"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "images = sorted(image_dir.glob('*.png'))\n",
    "print(f'Total images: {len(images)}')\n",
    "img = cv2.imread(str(images[0]))\n",
    "img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "plt.imshow(img)\n",
    "plt.title(f'Sample: {images[0].name}')\n",
    "plt.axis('off')"
   ],
   "outputs": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

Save this to `notebooks/01_eda.ipynb`.

- [ ] **Step 2: Create results notebook skeleton**

Save a minimal results notebook to `notebooks/02_results.ipynb` with cells importing evaluation modules, loading MLflow, and plotting metrics.

- [ ] **Step 3: Commit**

```bash
git add notebooks/
git commit -m "docs: add notebook skeletons for EDA and results analysis"
```

---

### Task 20: Final Lint and Test Pass

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 2: Run lint**

Run: `make lint`
Expected: No errors.

- [ ] **Step 3: Run smoke train (CPU, 1 epoch, tiny model)**

Run: `uv run python -m pytest tests/test_train_smoke.py -v`
Expected: PASS.

- [ ] **Step 4: Check `uv.lock` is tracked**

Run: `git status`
Expected: `uv.lock` modified and should be committed with other changes.

```bash
git add uv.lock
git commit -m "chore: update uv.lock after dependency changes"
```
