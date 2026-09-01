"""Step 5: temporal-background extraction and HLS residual compensation."""

from __future__ import annotations

import logging
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import Mapping

import numpy as np
import rasterio
from scipy import ndimage as ndi

from .config import output_dir
from .io import assert_same_grid, date_map, iter_windows, list_date_rasters, output_profile, read_bands_as_nan, write_json
from .preprocessing import linear_interpolate_and_edge_fill

LOGGER = logging.getLogger(__name__)


def temporal_nanmedian(values: np.ndarray, window_length: int) -> np.ndarray:
    """Centered, edge-clipped NaN-aware temporal median for T-by-N arrays."""
    radius = int(window_length) // 2
    output = np.full_like(values, np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for index in range(values.shape[0]):
            first = max(0, index - radius)
            last = min(values.shape[0], index + radius + 1)
            output[index] = np.nanmedian(values[first:last], axis=0)
    return output


def temporal_nanmean(values: np.ndarray, window_length: int) -> np.ndarray:
    """NaN-aware moving temporal mean for T-by-N arrays."""
    size = max(1, int(window_length))
    valid = np.isfinite(values).astype(np.float32)
    filled = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    total = ndi.uniform_filter1d(filled, size=size, axis=0, mode="nearest") * size
    count = ndi.uniform_filter1d(valid, size=size, axis=0, mode="nearest") * size
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0).astype(np.float32)


def compensate_residuals(
    q50: np.ndarray,
    hls: np.ndarray,
    time_coordinates: np.ndarray,
    background_window: int,
    smooth_window: int,
    preserve_observed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return temporal background, filled residual, and final reflectance."""
    background = temporal_nanmedian(q50, background_window)
    observed_valid = np.isfinite(hls) & np.isfinite(background)
    observed_residual = np.where(observed_valid, hls - background, np.nan).astype(np.float32)
    interpolated, _ = linear_interpolate_and_edge_fill(
        observed_residual,
        observed_valid,
        np.asarray(time_coordinates, dtype=np.float64),
    )
    smoothed = temporal_nanmean(interpolated, smooth_window)
    supported = observed_valid.any(axis=0)
    filled_residual = np.where(supported[None, :], smoothed, 0.0).astype(np.float32)
    if preserve_observed:
        filled_residual[observed_valid] = observed_residual[observed_valid]
    final = background + filled_residual
    final[~np.isfinite(background)] = np.nan
    return background, filled_residual, final.astype(np.float32)


def residual_source(cfg: Mapping[str, object], source: str, overwrite: bool = False) -> Path:
    source = source.upper()
    copula_items = list_date_rasters(output_dir(cfg, "03_copula", source))
    dates = [item[0] for item in copula_items]
    copula_paths = [item[1] for item in copula_items]
    hls_map = date_map(cfg["paths"]["hls"])
    grid = assert_same_grid(copula_paths, minimum_bands=4)
    assert_same_grid([copula_paths[0], *hls_map.values()], minimum_bands=1)
    with rasterio.open(copula_paths[0]) as reference:
        profile = output_profile(
            reference.profile,
            1,
            "float32",
            float(cfg["raster"]["output_nodata"]),
            str(cfg["raster"]["compression"]),
        )

    stage = output_dir(cfg, "04_fusion", source)
    background_dir = stage / "background"
    residual_dir = stage / "residual"
    final_dir = stage / "final"
    for folder in (background_dir, residual_dir, final_dir):
        folder.mkdir(parents=True, exist_ok=True)
    expected = [
        folder / f"{item_date:%Y-%m-%d}.tif"
        for folder in (background_dir, residual_dir, final_dir)
        for item_date in dates
    ]
    manifest_path = stage / "residual_compensation_manifest.json"
    if manifest_path.exists() and all(path.exists() for path in expected) and not overwrite:
        LOGGER.info("Step 5 %s already complete; reusing %s", source, final_dir)
        return final_dir

    time_coordinates = np.asarray([(item - dates[0]).days for item in dates], dtype=np.float64)
    residual_cfg = cfg["residual"]
    low_raw, high_raw = [float(item) for item in cfg["raster"]["hls_valid_range_raw"]]
    hls_scale = float(cfg["raster"]["hls_scale"])
    clip_low, clip_high = [float(item) for item in residual_cfg["clip_reflectance"]]
    nodata = float(cfg["raster"]["output_nodata"])

    with ExitStack() as stack:
        copula_readers = [stack.enter_context(rasterio.open(path)) for path in copula_paths]
        hls_readers = {
            item_date: stack.enter_context(rasterio.open(path))
            for item_date, path in hls_map.items()
            if item_date in set(dates)
        }
        background_writers = [
            stack.enter_context(rasterio.open(background_dir / f"{item_date:%Y-%m-%d}.tif", "w", **profile))
            for item_date in dates
        ]
        residual_writers = [
            stack.enter_context(rasterio.open(residual_dir / f"{item_date:%Y-%m-%d}.tif", "w", **profile))
            for item_date in dates
        ]
        final_writers = [
            stack.enter_context(rasterio.open(final_dir / f"{item_date:%Y-%m-%d}.tif", "w", **profile))
            for item_date in dates
        ]
        for writer in background_writers:
            writer.set_band_description(1, "copula_temporal_background_reflectance")
        for writer in residual_writers:
            writer.set_band_description(1, "temporally_filled_hls_residual")
        for writer in final_writers:
            writer.set_band_description(1, "jpdfusion_final_reflectance")

        for window in iter_windows(int(grid["width"]), int(grid["height"]), int(cfg["raster"]["block_size"])):
            height, width = int(window.height), int(window.width)
            q50 = np.stack([read_bands_as_nan(reader, [2], window)[0] for reader in copula_readers]).reshape(len(dates), -1)
            hls = np.full_like(q50, np.nan, dtype=np.float32)
            for index, item_date in enumerate(dates):
                if item_date not in hls_readers:
                    continue
                raw = read_bands_as_nan(hls_readers[item_date], [1], window)[0]
                raw[(raw < low_raw) | (raw > high_raw)] = np.nan
                hls[index] = (raw * hls_scale).reshape(-1)

            background, residual, final = compensate_residuals(
                q50,
                hls,
                time_coordinates,
                int(residual_cfg["background_median_window"]),
                int(residual_cfg["smooth_window"]),
                bool(residual_cfg["preserve_observed_hls"]),
            )
            final = np.clip(final, clip_low, clip_high)
            for index in range(len(dates)):
                for values, writer in (
                    (background[index], background_writers[index]),
                    (residual[index], residual_writers[index]),
                    (final[index], final_writers[index]),
                ):
                    array = values.reshape(height, width).astype(np.float32)
                    array[~np.isfinite(array)] = nodata
                    writer.write(array, 1, window=window)

    write_json(
        manifest_path,
        {
            "source": source,
            "dates": [str(item) for item in dates],
            "background_median_window": int(residual_cfg["background_median_window"]),
            "residual_smooth_window": int(residual_cfg["smooth_window"]),
            "preserve_observed_hls": bool(residual_cfg["preserve_observed_hls"]),
            "output_units": "physical surface reflectance",
            "output_directories": {
                "background": background_dir,
                "residual": residual_dir,
                "final": final_dir,
            },
        },
    )
    return final_dir


def residual_all(cfg: Mapping[str, object], sources: list[str] | None = None, overwrite: bool = False) -> dict[str, Path]:
    selected = sources or list(cfg["sources"])
    return {source: residual_source(cfg, source, overwrite=overwrite) for source in selected}
