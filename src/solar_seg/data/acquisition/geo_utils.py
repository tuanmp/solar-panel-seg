from __future__ import annotations

from dataclasses import dataclass
from math import cos, log, radians


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
        lat_center = radians(self.center[0])
        return (self.lon_max - self.lon_min) * 111.32 * cos(lat_center)

    @property
    def height_km(self) -> float:
        return (self.lat_max - self.lat_min) * 111.32

    def to_list(self) -> list[float]:
        """Return bounding box as [lon_min, lat_min, lon_max, lat_max] (GeoJSON order)."""
        return [self.lon_min, self.lat_min, self.lon_max, self.lat_max]


def bbox_from_center(
    lat: float, lon: float, size_km: float = 1.0
) -> BBox:
    """Create a square bounding box centered on (lat, lon)."""
    lat_delta = size_km / 111.32 / 2.0
    lon_delta = size_km / (111.32 * cos(radians(lat))) / 2.0
    return BBox(
        lat_min=lat - lat_delta,
        lon_min=lon - lon_delta,
        lat_max=lat + lat_delta,
        lon_max=lon + lon_delta,
    )


def gsd_to_zoom(gsd: float) -> int:
    """Approximate Google Maps zoom level from ground sampling distance in meters."""
    return max(1, min(22, int(round(log(156543.03 / gsd, 2)))))
