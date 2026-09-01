"""Create tiny GeoTIFFs for a smoke test; these are not the paper's real data."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def write_stack(path: Path, array: np.ndarray, nodata: int = -20000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[2],
        height=array.shape[1],
        count=array.shape[0],
        dtype="int16",
        crs="EPSG:32649",
        transform=from_origin(500000, 3000000, 30, 30),
        nodata=nodata,
        compress="deflate",
    ) as writer:
        writer.write(array.astype(np.int16))


def create(root: Path, force: bool = False) -> None:
    if root.exists() and any(root.rglob("*.tif")) and not force:
        raise FileExistsError(f"GeoTIFFs already exist under {root}; use --force only for disposable smoke data")
    rng = np.random.default_rng(42)
    rows, cols = np.mgrid[:24, :30]
    start = date(2020, 1, 1)
    for index in range(9):
        item_date = start + timedelta(days=5 * index)
        seasonal = 700 + 25 * index + 120 * np.sin(rows / 6) + 90 * np.cos(cols / 7)
        hls = seasonal + rng.normal(0, 12, size=seasonal.shape)
        if index in {2, 5}:
            hls[6:14, 9:20] = -20000
        write_stack(root / "HLS" / f"HLS_{item_date:%Y-%m-%d}.tif", hls[None])

        vv = -1300 + 0.30 * seasonal + rng.normal(0, 20, seasonal.shape)
        vh = -1900 + 0.20 * seasonal + rng.normal(0, 20, seasonal.shape)
        rvi = 2500 + 0.12 * seasonal + rng.normal(0, 15, seasonal.shape)
        sar = np.stack([vv, vh, rvi])
        sar[:, 3:5, 4:9] = -20000
        write_stack(root / "SAR" / f"SAR_{item_date:%Y-%m-%d}.tif", sar)

        mcd_bands = np.stack(
            [seasonal * factor + offset + rng.normal(0, 10, seasonal.shape) for factor, offset in zip((0.8, 1.2, 0.9, 1.4, 1.1, 0.7), (0, 150, 50, 250, 100, 20))]
        )
        mcd_bands[:, 17:20, 12:18] = -20000
        write_stack(root / "MCD43A4" / f"MCD43A4_{item_date:%Y-%m-%d}.tif", mcd_bands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("smoke_data/Data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    create(args.root, args.force)
    print(args.root.resolve())

