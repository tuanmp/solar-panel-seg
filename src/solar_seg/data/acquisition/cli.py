from __future__ import annotations

import argparse
from pathlib import Path

from solar_seg.data.acquisition.geo_utils import bbox_from_center
from solar_seg.data.acquisition.gee_client import GEEClient
from solar_seg.data.acquisition.sentinel_client import SentinelHubClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire satellite/aerial imagery for solar panel segmentation."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gee_p = sub.add_parser("gee", help="Google Earth Engine download")
    gee_p.add_argument("--lat", type=float, required=True)
    gee_p.add_argument("--lon", type=float, required=True)
    gee_p.add_argument("--size-km", type=float, default=1.0)
    gee_p.add_argument(
        "--source", choices=["naip", "sentinel2"], default="naip"
    )
    gee_p.add_argument("--output", type=Path, default=Path("data/acquired"))
    gee_p.add_argument("--prefix", default="tile")
    gee_p.add_argument("--start-date", default="2020-01-01")
    gee_p.add_argument("--end-date", default="2024-01-01")
    gee_p.add_argument("--project", type=str, default=None)
    gee_p.add_argument(
        "--list-only",
        action="store_true",
        help="List available images without downloading",
    )

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
            result = client.list_available(
                bbox, args.source, args.start_date, args.end_date
            )
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
        try:
            result = client.download_tile(
                bbox=bbox,
                output_dir=args.output,
                prefix=args.prefix,
                start_date=args.start_date,
                end_date=args.end_date,
                resolution=args.resolution,
            )
            print(f"Downloaded: {result['output_path']}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
