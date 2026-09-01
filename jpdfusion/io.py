"""GeoTIFF and time-series helpers shared by all pipeline stages."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window


DATE_PATTERNS = (
    re.compile(r"(?<!\d)((?:19|20)\d{2})[-_]?(\d{2})[-_]?(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{3})(?!\d)"),
)


def parse_date(path_or_name: str | Path) -> date:
    name = Path(path_or_name).stem
    match = DATE_PATTERNS[0].search(name)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = DATE_PATTERNS[1].search(name)
    if match:
        return datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%j").date()
    raise ValueError(f"No YYYY-MM-DD, YYYYMMDD, or YYYYDDD date found in: {name}")


def list_date_rasters(folder: str | Path) -> list[tuple[date, Path]]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    mapping: dict[date, Path] = {}
    candidates = sorted({*root.glob("*.tif"), *root.glob("*.tiff")})
    for path in candidates:
        try:
            item_date = parse_date(path)
        except ValueError:
            continue
        if item_date in mapping:
            raise ValueError(f"Duplicate date {item_date}: {mapping[item_date]} and {path}")
        mapping[item_date] = path.resolve()
    if not mapping:
        raise FileNotFoundError(f"No dated GeoTIFFs were found directly under: {root}")
    return sorted(mapping.items())


def date_map(folder: str | Path) -> dict[date, Path]:
    return dict(list_date_rasters(folder))


def iter_windows(width: int, height: int, block_size: int) -> Iterable[Window]:
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            yield Window(col, row, min(block_size, width - col), min(block_size, height - row))


def tile_windows(width: int, height: int, tile_width: int, tile_height: int) -> Iterable[Window]:
    yield from iter_windows(width, height, max(tile_width, tile_height)) if tile_width == tile_height else _rect_windows(width, height, tile_width, tile_height)


def _rect_windows(width: int, height: int, tile_width: int, tile_height: int) -> Iterable[Window]:
    for row in range(0, height, tile_height):
        for col in range(0, width, tile_width):
            yield Window(col, row, min(tile_width, width - col), min(tile_height, height - row))


def padded_window(window: Window, pad: int, width: int, height: int) -> tuple[Window, int, int]:
    row0 = max(0, int(window.row_off) - pad)
    col0 = max(0, int(window.col_off) - pad)
    row1 = min(height, int(window.row_off + window.height) + pad)
    col1 = min(width, int(window.col_off + window.width) + pad)
    padded = Window(col0, row0, col1 - col0, row1 - row0)
    return padded, int(window.row_off) - row0, int(window.col_off) - col0


def read_bands_as_nan(dataset: rasterio.io.DatasetReader, bands: Sequence[int], window: Window | None = None) -> np.ndarray:
    array = dataset.read(list(bands), window=window, masked=True).astype(np.float32)
    values = np.asarray(array.filled(np.nan), dtype=np.float32)
    for band_index, band in enumerate(bands):
        nodata = dataset.nodatavals[band - 1]
        if nodata is not None and np.isfinite(nodata):
            values[band_index][values[band_index] == nodata] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values


def assert_same_grid(paths: Sequence[str | Path], minimum_bands: int = 1) -> dict[str, object]:
    if not paths:
        raise ValueError("No rasters supplied for grid validation")
    with rasterio.open(paths[0]) as reference:
        expected = {
            "width": reference.width,
            "height": reference.height,
            "crs": reference.crs,
            "transform": reference.transform,
            "profile": reference.profile.copy(),
        }
        if reference.count < minimum_bands:
            raise ValueError(f"{paths[0]} has {reference.count} bands; need {minimum_bands}")
    for path in paths[1:]:
        with rasterio.open(path) as dataset:
            actual = (dataset.width, dataset.height, dataset.crs, dataset.transform)
            target = (expected["width"], expected["height"], expected["crs"], expected["transform"])
            if actual != target:
                raise ValueError(f"Grid mismatch: {path} does not match {paths[0]}")
            if dataset.count < minimum_bands:
                raise ValueError(f"{path} has {dataset.count} bands; need {minimum_bands}")
    return expected


def output_profile(reference_profile: Mapping[str, object], count: int, dtype: str, nodata: float | int, compression: str) -> dict[str, object]:
    profile = dict(reference_profile)
    profile.update(
        driver="GTiff",
        count=count,
        dtype=dtype,
        nodata=nodata,
        compress=compression,
        BIGTIFF="IF_SAFER",
    )
    if dtype in {"float32", "float64"}:
        profile.update(predictor=3)
    elif dtype != "uint8":
        profile.update(predictor=2)
    return profile


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

