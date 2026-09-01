"""Step 2: auxiliary time-series interpolation and Savitzky-Golay filtering."""

from __future__ import annotations

import logging
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import rasterio
from scipy.signal import savgol_filter

from .config import output_dir, source_key
from .io import assert_same_grid, iter_windows, list_date_rasters, output_profile, read_bands_as_nan, write_json

LOGGER = logging.getLogger(__name__)


def linear_interpolate_and_edge_fill(
    values: np.ndarray,
    valid: np.ndarray,
    time_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill each T-by-N series by linear interpolation and nearest edge values.

    Observed samples are unchanged. Pixels with no valid sample remain NaN. The
    returned edge mask marks leading/trailing values copied from the nearest
    observation so SG filtering can leave those boundary fills unchanged.
    """
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("values and valid must both have shape (time, pixels)")
    t_len, n_pixels = values.shape
    coordinates = np.asarray(time_coordinates, dtype=np.float64)
    if coordinates.shape != (t_len,):
        raise ValueError(f"Expected {t_len} time coordinates, got {coordinates.shape}")

    indices = np.broadcast_to(np.arange(t_len, dtype=np.int32)[:, None], values.shape)
    previous = np.maximum.accumulate(np.where(valid, indices, -1), axis=0)
    following = np.minimum.accumulate(np.where(valid, indices, t_len)[::-1], axis=0)[::-1]
    supported = valid.any(axis=0)
    missing = ~valid
    internal = missing & (previous >= 0) & (following < t_len) & (following > previous)
    leading = missing & (previous < 0) & (following < t_len)
    trailing = missing & (previous >= 0) & (following >= t_len)
    completed = values.astype(np.float32, copy=True)

    for t_index in range(t_len):
        cols = np.flatnonzero(internal[t_index])
        if cols.size:
            p = previous[t_index, cols]
            q = following[t_index, cols]
            alpha = (coordinates[t_index] - coordinates[p]) / (coordinates[q] - coordinates[p])
            completed[t_index, cols] = values[p, cols] + alpha.astype(np.float32) * (
                values[q, cols] - values[p, cols]
            )

        cols = np.flatnonzero(leading[t_index])
        if cols.size:
            q = following[t_index, cols]
            completed[t_index, cols] = values[q, cols]

        cols = np.flatnonzero(trailing[t_index])
        if cols.size:
            p = previous[t_index, cols]
            completed[t_index, cols] = values[p, cols]

    completed[:, ~supported] = np.nan
    return completed, leading | trailing


def observed_constrained_sg(
    completed: np.ndarray,
    observed_values: np.ndarray,
    observed_valid: np.ndarray,
    edge_filled: np.ndarray,
    window_length: int,
    polyorder: int,
    preserve_observations: bool = True,
) -> np.ndarray:
    """Apply SG filtering while retaining observations and nearest edge fills."""
    supported = observed_valid.any(axis=0)
    filtered_input = completed.copy()
    filtered_input[:, ~supported] = 0.0
    filtered_input[~np.isfinite(filtered_input)] = 0.0

    effective_window = min(int(window_length), completed.shape[0])
    if effective_window % 2 == 0:
        effective_window -= 1
    if effective_window <= polyorder or effective_window < 3:
        filtered = completed.copy()
    else:
        filtered = savgol_filter(
            filtered_input,
            window_length=effective_window,
            polyorder=min(int(polyorder), effective_window - 1),
            axis=0,
            mode="interp",
        ).astype(np.float32)

    if preserve_observations:
        filtered[observed_valid] = observed_values[observed_valid]
    filtered[edge_filled] = completed[edge_filled]
    filtered[:, ~supported] = np.nan
    return filtered.astype(np.float32, copy=False)


def _encode(array: np.ndarray, dtype: np.dtype, nodata: float | int) -> np.ndarray:
    result = array.copy()
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        result = np.rint(result)
        result = np.clip(result, info.min, info.max)
    result[~np.isfinite(result)] = nodata
    return result.astype(dtype)


def preprocess_source(cfg: Mapping[str, object], source: str, overwrite: bool = False) -> Path:
    """Preprocess one auxiliary source and return its Linear_SG directory."""
    source = source.upper()
    input_dir = Path(cfg["paths"][source_key(source)])
    if not bool(cfg["preprocessing"]["enabled"]):
        LOGGER.info("Step 2 %s disabled; using raw regularized inputs from %s", source, input_dir)
        return input_dir
    items = list_date_rasters(input_dir)
    dates = [item[0] for item in items]
    paths = [item[1] for item in items]
    grid = assert_same_grid(paths)
    pre = cfg["preprocessing"]
    raster_cfg = cfg["raster"]
    block_size = int(raster_cfg["block_size"])
    compression = str(raster_cfg["compression"])

    with rasterio.open(paths[0]) as reference:
        band_count = reference.count
        dtype = np.dtype(reference.dtypes[0])
        output_nodata = reference.nodata
        if output_nodata is None:
            raise ValueError(f"Auxiliary GeoTIFF must declare NoData metadata: {paths[0]}")
        profile = output_profile(reference.profile, band_count, dtype.name, output_nodata, compression)

    stage_root = output_dir(cfg, "01_preprocessed", source)
    linear_dir = stage_root / "Linear"
    sg_dir = stage_root / "Linear_SG"
    if bool(pre["write_linear"]):
        linear_dir.mkdir(parents=True, exist_ok=True)
    if bool(pre["write_linear_sg"]):
        sg_dir.mkdir(parents=True, exist_ok=True)

    outputs_linear = [linear_dir / f"{item_date:%Y-%m-%d}.tif" for item_date in dates]
    outputs_sg = [sg_dir / f"{item_date:%Y-%m-%d}.tif" for item_date in dates]
    requested_outputs = []
    if bool(pre["write_linear"]):
        requested_outputs.extend(outputs_linear)
    if bool(pre["write_linear_sg"]):
        requested_outputs.extend(outputs_sg)
    manifest_path = stage_root / "preprocessing_manifest.json"
    if requested_outputs and manifest_path.exists() and all(path.exists() for path in requested_outputs) and not overwrite:
        LOGGER.info("Step 2 %s already complete; reusing %s", source, sg_dir)
        return sg_dir if bool(pre["write_linear_sg"]) else linear_dir

    LOGGER.info("Step 2 %s: %d dates, %d bands, %dx%d", source, len(dates), band_count, grid["width"], grid["height"])
    time_coordinates = np.asarray([(item - dates[0]).days for item in dates], dtype=np.float64)

    with ExitStack() as stack:
        readers = [stack.enter_context(rasterio.open(path)) for path in paths]
        linear_writers = []
        sg_writers = []
        if bool(pre["write_linear"]):
            linear_writers = [stack.enter_context(rasterio.open(path, "w", **profile)) for path in outputs_linear]
        if bool(pre["write_linear_sg"]):
            sg_writers = [stack.enter_context(rasterio.open(path, "w", **profile)) for path in outputs_sg]

        for window in iter_windows(int(grid["width"]), int(grid["height"]), block_size):
            height, width = int(window.height), int(window.width)
            raw = np.stack([read_bands_as_nan(reader, range(1, band_count + 1), window) for reader in readers])
            for band_index in range(band_count):
                observed = raw[:, band_index].reshape(len(dates), -1)
                valid = np.isfinite(observed)
                linear, edge_mask = linear_interpolate_and_edge_fill(observed, valid, time_coordinates)
                sg = observed_constrained_sg(
                    linear,
                    observed,
                    valid,
                    edge_mask,
                    int(pre["sg_window_length"]),
                    int(pre["sg_polyorder"]),
                    bool(pre["preserve_observations"]),
                )
                for index in range(len(dates)):
                    if linear_writers:
                        linear_writers[index].write(
                            _encode(linear[index].reshape(height, width), dtype, output_nodata),
                            band_index + 1,
                            window=window,
                        )
                    if sg_writers:
                        sg_writers[index].write(
                            _encode(sg[index].reshape(height, width), dtype, output_nodata),
                            band_index + 1,
                            window=window,
                        )

    write_json(
        manifest_path,
        {
            "source": source,
            "input_directory": input_dir,
            "dates": [str(item) for item in dates],
            "bands": band_count,
            "sg_window_length": int(pre["sg_window_length"]),
            "sg_polyorder": int(pre["sg_polyorder"]),
            "preserve_observations": bool(pre["preserve_observations"]),
            "edge_fill": "nearest observation",
        },
    )
    return sg_dir if bool(pre["write_linear_sg"]) else linear_dir


def preprocessed_input_dir(cfg: Mapping[str, object], source: str) -> Path:
    if not bool(cfg["preprocessing"]["enabled"]):
        return Path(cfg["paths"][source_key(source.upper())])
    product = str(cfg["clustering"]["input_product"])
    return output_dir(cfg, "01_preprocessed", source) / product


def preprocess_all(cfg: Mapping[str, object], sources: list[str] | None = None, overwrite: bool = False) -> dict[str, Path]:
    selected = sources or list(cfg["sources"])
    return {source: preprocess_source(cfg, source, overwrite=overwrite) for source in selected}
