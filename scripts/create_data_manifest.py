"""Create a checksummed manifest for the public example data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from jpdfusion.config import load_config, source_key
from jpdfusion.io import list_date_rasters, sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write Data/data_manifest.csv with SHA-256 checksums")
    parser.add_argument("--config", default="config.example.yml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(cfg["project_root"]) / "Data" / "data_manifest.csv"
    rows = []
    for collection, path_key in (("HLS", "hls"), ("SAR", "sar"), ("MCD43A4", "mcd43a4")):
        for item_date, path in list_date_rasters(cfg["paths"][path_key]):
            rows.append(
                {
                    "collection": collection,
                    "date": item_date.isoformat(),
                    "relative_path": path.relative_to(cfg["project_root"]).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("collection", "date", "relative_path", "size_bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    print(output)

